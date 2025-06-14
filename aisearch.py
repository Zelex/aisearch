import os
import re
import sys
import ast
import argparse
import concurrent.futures
import time
from typing import List, Dict, Any, Set, Tuple, Optional, Callable, Union
import threading

# Third-party imports
import anthropic
import openai
from openai import AzureOpenAI
# Try to import pyre2 as regex if available, otherwise use standard regex
try:
    import re2 as regex
    print("Using Google's RE2 regex engine for improved performance")
except ImportError:
    import regex
from tqdm import tqdm

# Cache for file lists when directory and extensions don't change
_file_cache = {}

# Cache for file contents to avoid re-reading files
_content_cache = {}
_content_cache_stats = {"hits": 0, "misses": 0, "max_size": 50000}

# Maps file extensions to programming languages
LANGUAGE_MAP = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.ts': 'TypeScript',
    '.jsx': 'React JSX',
    '.tsx': 'React TSX',
    '.java': 'Java',
    '.c': 'C',
    '.cpp': 'C++',
    '.cs': 'C#',
    '.go': 'Go',
    '.rb': 'Ruby',
    '.php': 'PHP',
    '.swift': 'Swift',
    '.kt': 'Kotlin',
    '.rs': 'Rust'
}

# Default directories to skip when searching
DEFAULT_SKIP_DIRS = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', 'build', 'dist', 'obj', 'bin'}

def get_ai_client(provider: str = "anthropic") -> Union[anthropic.Anthropic, openai.OpenAI, AzureOpenAI]:
    """
    Get the appropriate AI client based on provider.
    
    Args:
        provider: AI provider to use ("anthropic", "openai", or "azure")
        
    Returns:
        Configured client for the specified provider
        
    Raises:
        ValueError: If API key is missing or provider is unsupported
    """
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return anthropic.Anthropic()
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return openai.OpenAI()
    elif provider == "azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            raise ValueError("Azure OpenAI API key not found. Please set AZURE_OPENAI_API_KEY environment variable.")
        if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            raise ValueError("Azure OpenAI endpoint not found. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        if not os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"):
            raise ValueError("Azure OpenAI deployment name not found. Please set AZURE_OPENAI_DEPLOYMENT_NAME environment variable.")
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        )
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def extensions_to_languages(extensions: List[str]) -> List[str]:
    """
    Convert file extensions to their corresponding programming languages.
    
    Args:
        extensions: List of file extensions (with or without leading dot)
        
    Returns:
        List of programming language names
    """
    languages = []
    if not extensions:
        return languages
    
    for ext in extensions:
        ext = ext if ext.startswith('.') else f'.{ext}'
        if ext in LANGUAGE_MAP:
            languages.append(LANGUAGE_MAP[ext])
    
    return languages

def format_extension_info(extensions: List[str]) -> str:
    """
    Format file extension and language information for AI prompts.
    
    Args:
        extensions: List of file extensions to include
        
    Returns:
        Formatted string with extension and language information
    """
    if not extensions:
        return ""
    
    languages = extensions_to_languages(extensions)
    ext_list = ', '.join(extensions)
    
    if languages:
        lang_list = ', '.join(languages)
        return f"\nLook specifically in {ext_list} files ({lang_list}). Tailor search terms to these languages using their specific syntax patterns."
    else:
        return f"\nLook specifically in {ext_list} files. Tailor search terms to these file types."

def get_search_terms_from_prompt(prompt: str, max_terms: int = 10, 
                                extensions: Optional[List[str]] = None, 
                                provider: str = "anthropic",
                                multiline: bool = True) -> Tuple[List[str], List[str]]:
    """
    Generate search terms and anti-patterns from a natural language prompt using AI.
    
    Args:
        prompt: Natural language prompt describing what to search for
        max_terms: Maximum number of search terms to generate
        extensions: List of file extensions to focus on
        provider: AI provider to use ("anthropic", "openai", or "azure")
        multiline: Whether to enable multi-line regex patterns (default: True)
        
    Returns:
        Tuple of (search_terms, anti_patterns): Lists of regex patterns
    """
    client = get_ai_client(provider)
    
    system_message = """You are a code search expert. Generate search terms that will effectively find relevant code patterns.
Focus on practical patterns that would appear in actual code.
Generate proper regex patterns that can be used with Python's re module, not just literal strings.
Include regex syntax for more powerful searches (e.g., \b for word boundaries, .* for any characters, etc.).

You must generate TWO distinct sets of patterns:
1. Search patterns - regex patterns to find matches for the user's query
2. Anti-patterns - regex patterns to EXCLUDE matches that are irrelevant or false positives

IMPORTANT: Think about how your search terms will work TOGETHER to discover the underlying search intent.
Create a diverse set of patterns that capture different aspects of the search and complement each other.
Some terms should be specific, others more general to ensure good coverage of potential matches.

For anti-patterns, think carefully about what might cause false positives and create patterns to exclude those.
"""

    if multiline:
        system_message += """
This search supports multi-line patterns with re.MULTILINE and re.DOTALL flags enabled.
You CAN and SHOULD create regex patterns that match across multiple lines when appropriate.
Use patterns like `function\\s+name.*?\\{.*?\\}` to match entire function blocks, or `class.*?\\{.*?\\}` for class definitions.
"""
    else:
        system_message += """
IMPORTANT: Each line is processed individually, so DO NOT create multi-line regex patterns. All patterns must match on a single line.
"""

    system_message += """
IMPORTANT EFFICIENCY GUIDELINES:
- Avoid nested .* or .*? patterns which cause catastrophic backtracking
- Limit pattern complexity - simpler is faster and more reliable
- Use atomic groups (?>...) where possible
- Prefer bounded quantifiers {0,100} instead of * or +
- Prefer character classes [a-z] over wildcard .
- Add anchors (^ $) when appropriate to limit search space
- For function/class definitions, use more specific start/end markers rather than greedy matches

YOUR OUTPUT FORMAT MUST BE:
Return your answer as a Python list of two lists, where the first list contains the search patterns and the second list contains the anti-patterns.
For example:
[
    ['pattern1', 'pattern2', 'pattern3'],
    ['anti-pattern1', 'anti-pattern2']
]

Make sure your output can be parsed directly by Python's ast.literal_eval() function.
DO NOT include any explanation or additional text - ONLY the list structure above.
"""
    
    extensions_info = format_extension_info(extensions)
    
    if provider == "anthropic":
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4096,
            temperature=1,
            system=system_message,
            thinking={
                "type": "enabled",
                "budget_tokens": 1024
            },
            messages=[{
                "role": "user",
                "content": f"Generate up to {max_terms} effective regex search patterns and anti-patterns for finding: '{prompt}'{extensions_info}"
            }]
        )
        raw_text = response.content[1].text.strip()
    elif provider == "azure":
        deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate up to {max_terms} effective regex search patterns and anti-patterns for finding: '{prompt}'{extensions_info}"}
            ],
            temperature=1,
            max_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
    else:  # OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate up to {max_terms} effective regex search patterns and anti-patterns for finding: '{prompt}'{extensions_info}"}
            ],
            temperature=1,
            max_completion_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
    
    return parse_ai_response_with_antipatterns(raw_text, max_terms)


def parse_ai_response_with_antipatterns(raw_text: str, max_terms: int) -> Tuple[List[str], List[str]]:
    """
    Parse and clean AI response into usable search terms and anti-patterns.
    
    Args:
        raw_text: Raw text response from AI
        max_terms: Maximum number of terms to return
        
    Returns:
        Tuple of (search_terms, anti_patterns)
    """
    search_terms = []
    anti_patterns = []
    
    # Special case for the specific format we're encountering:
    # - ['pattern1', 'pattern2', ...]
    # - ['anti-pattern1', 'anti-pattern2', ...]
    dash_list_pattern = r'-\s*\[(.*?)\]'
    dash_lists = re.findall(dash_list_pattern, raw_text, re.DOTALL)
    
    if len(dash_lists) >= 2:
        print("DEBUG: Found dash-list format, parsing directly")
        try:
            # Try to parse the first list (search patterns)
            search_list_str = dash_lists[0]
            search_list_str = '[' + search_list_str + ']'  # Add brackets back
            search_list = ast.literal_eval(search_list_str)
            search_terms = [str(s) for s in search_list]
            
            # Try to parse the second list (anti-patterns)
            anti_list_str = dash_lists[1]
            anti_list_str = '[' + anti_list_str + ']'  # Add brackets back
            anti_list = ast.literal_eval(anti_list_str)
            anti_patterns = [str(s) for s in anti_list]
            
            print(f"DEBUG: Successfully parsed dash-list format")
            print(f"DEBUG: search_terms: {search_terms[:3]}...")
            print(f"DEBUG: anti_patterns: {anti_patterns[:3]}...")
            
            # Return early since we successfully parsed the format
            return search_terms[:max_terms], anti_patterns[:max_terms]
        except Exception as e:
            print(f"DEBUG: Error parsing dash-list format: {e}")
            # Continue with other parsing methods
    
    # If the above parsing failed, try standard methods
    try:
        # First, let's check if the response looks like it contains Python list literals
        if raw_text.strip().startswith('[') and raw_text.strip().endswith(']'):
            import ast
            
            try:
                # Try to parse using ast.literal_eval which is safe for literals
                parsed_data = ast.literal_eval(raw_text)
                
                # Check if this is a list of lists (search patterns and anti-patterns)
                if isinstance(parsed_data, list) and len(parsed_data) >= 2:
                    # First element is likely search patterns
                    for item in parsed_data[0]:
                        if isinstance(item, str) and item not in search_terms:
                            search_terms.append(item)
                    
                    # Second element is likely anti-patterns
                    for item in parsed_data[1]:
                        if isinstance(item, str) and item not in anti_patterns:
                            anti_patterns.append(item)
                    
                    # Successfully parsed, return early
                    return (
                        list(search_terms[:max_terms]) if search_terms else [],
                        list(anti_patterns[:max_terms]) if anti_patterns else []
                    )
                
                # If we get here, it might be a single list of patterns
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    for item in parsed_data:
                        if isinstance(item, str) and item not in search_terms:
                            search_terms.append(item)
                    
                    # Successfully parsed, return early with empty anti-patterns
                    return (
                        list(search_terms[:max_terms]) if search_terms else [],
                        []
                    )
            except:
                # If ast.literal_eval fails, continue with regex-based parsing
                pass
        
        # If AST parsing didn't succeed, try to extract list literals from the text
        list_pattern = r'\[(.*?)\]'
        list_matches = re.findall(list_pattern, raw_text, re.DOTALL)
        
        if len(list_matches) >= 2:
            # First match is likely search patterns
            search_match = list_matches[0]
            anti_match = list_matches[1]
            
            # Extract patterns from search_match
            pattern_matches = re.findall(r'[\'"]([^\'"]+)[\'"]', search_match)
            for pattern in pattern_matches:
                if pattern and pattern not in search_terms:
                    search_terms.append(pattern)
            
            # Extract patterns from anti_match
            pattern_matches = re.findall(r'[\'"]([^\'"]+)[\'"]', anti_match)
            for pattern in pattern_matches:
                if pattern and pattern not in anti_patterns:
                    anti_patterns.append(pattern)
            
            # If we found patterns, return early
            if search_terms or anti_patterns:
                return (
                    list(search_terms[:max_terms]) if search_terms else [],
                    list(anti_patterns[:max_terms]) if anti_patterns else []
                )
        
        # If we didn't find list literals or they didn't contain patterns,
        # fall back to standard parsing
        parts = re.split(r'(?i)^\s*ANTI.?PATTERNS\s*:?\s*$', raw_text, flags=re.MULTILINE)
        
        if len(parts) >= 2:
            search_part = parts[0]
            anti_part = parts[1]
            
            # Further refine the search part if it contains a header
            search_header_match = re.search(r'(?i)^\s*SEARCH.?PATTERNS\s*:?\s*$', search_part, re.MULTILINE)
            if search_header_match:
                search_part = search_part[search_header_match.end():]
            
            # Parse search terms
            for line in search_part.splitlines():
                line = line.strip()
                # Skip empty lines or lines that look like explanations/headers
                if not line or line.startswith('#') or line.startswith('-') or line.startswith('SEARCH') or len(line) > 120:
                    continue
                # Remove any prefix numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and line not in search_terms:
                    search_terms.append(line)
            
            # Parse anti-patterns
            for line in anti_part.splitlines():
                line = line.strip()
                # Skip empty lines or lines that look like explanations/headers
                if not line or line.startswith('#') or line.startswith('-') or len(line) > 120:
                    continue
                # Remove any prefix numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and line not in anti_patterns:
                    anti_patterns.append(line)
        else:
            # Fallback if no clear separation found - assume all are search terms
            for line in raw_text.splitlines():
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-') or len(line) > 120:
                    continue
                # Remove any prefix numbering (1., 2., etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line and line not in search_terms:
                    search_terms.append(line)
    
    except Exception as e:
        print(f"Warning: Error parsing AI response: {e}")
        print("Falling back to simple parsing...")
        # Simplified fallback parsing
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or len(line) > 120:
                continue
            if "anti" in line.lower():
                continue  # Skip any line with "anti" to avoid mixing search/anti-patterns
            if line and line not in search_terms:
                search_terms.append(line)
    
    # Ensure we're returning proper lists and not exceeding max_terms
    return (
        list(search_terms[:max_terms]) if search_terms else [], 
        list(anti_patterns[:max_terms]) if anti_patterns else []
    )


def sanitize_regex_pattern(pattern: str) -> str:
    """
    Sanitize a potentially problematic regex pattern.
    Handles common regex issues that might cause 'unterminated character set' errors.
    
    Args:
        pattern: Regular expression pattern to sanitize
        
    Returns:
        Sanitized pattern that can be safely compiled
    """
    # Balance brackets
    for left, right in [('[', ']'), ('{', '}'), ('(', ')')]:
        if pattern.count(left) != pattern.count(right):
            pattern = pattern.replace(left, f'\\{left}').replace(right, f'\\{right}')
    
    # Escape other problematic characters if they're not already escaped
    problematic_chars = ['*', '+', '?', '|', '^', '$', '.']
    for char in problematic_chars:
        # Only escape if not already escaped
        pattern = regex.sub(r'(?<!\\)' + regex.escape(char), f'\\{char}', pattern)
    
    return pattern


def compile_regex(pattern: str, flags: int) -> regex.Pattern:
    """
    Compile a regex pattern using the regex module which offers better features.
    
    Args:
        pattern: Regular expression pattern to compile
        flags: Regex flags to use
        
    Returns:
        Compiled regex pattern
    """
    # Convert re flags to regex flags
    regex_flags = 0
    if flags & re.IGNORECASE:
        regex_flags |= regex.IGNORECASE
    if flags & re.MULTILINE:
        regex_flags |= regex.MULTILINE
    if flags & re.DOTALL:
        regex_flags |= regex.DOTALL
    
    # Add performance optimizations for the regex module
    # These can help with GIL release and performance
    try:
        return regex.compile(pattern, regex_flags | regex.VERSION1)
    except regex.error:
        # Fallback to basic compilation if VERSION1 fails
        return regex.compile(pattern, regex_flags)


def highlight_match(line: str, term: str, pattern: Optional[regex.Pattern], 
                   color_output: bool, case_sensitive: bool, use_regex: bool) -> str:
    """
    Highlight matching text in console output.
    
    Args:
        line: Line of text to highlight
        term: Search term that matched
        pattern: Compiled regex pattern (if using regex)
        color_output: Whether to use color in output
        case_sensitive: Whether search is case sensitive
        use_regex: Whether using regex or literal string search
        
    Returns:
        Line with matches highlighted using ANSI color codes
    """
    if not color_output:
        return line

    color_start = '\033[91m'  # Red
    color_end = '\033[0m'  # Reset

    def replace(match):
        return f"{color_start}{match.group(0)}{color_end}"

    if use_regex and pattern:
        return pattern.sub(replace, line)
    else:
        flags = 0 if case_sensitive else regex.IGNORECASE
        return regex.sub(f'({regex.escape(term)})', f'{color_start}\\1{color_end}', line, flags=flags)


def is_comment(line: str, file_ext: str) -> bool:
    """
    Determine if a line is a comment based on file extension.
    
    Args:
        line: Line of text to check
        file_ext: File extension (including dot)
        
    Returns:
        True if line is a comment, False otherwise
    """
    file_ext = file_ext.lower()
    line = line.strip()
    
    # C-style comments
    if file_ext in ['.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.jsx', '.tsx', '.cs', '.php', '.swift']:
        if line.startswith('//') or line.startswith('/*') or line.endswith('*/') or regex.match(r'^\s*\*', line):
            return True
    
    # Python, Ruby, Shell comments
    if file_ext in ['.py', '.rb', '.sh', '.bash'] and line.startswith('#'):
        return True
    
    # HTML/XML comments
    if file_ext in ['.html', '.xml', '.svg'] and ('<!--' in line or '-->' in line):
        return True
        
    # CSS comments
    if file_ext in ['.css', '.scss', '.less'] and ('/*' in line or '*/' in line):
        return True
        
    # Lua comments
    if file_ext == '.lua' and (line.startswith('--') or (line.startswith('--[[') or line.endswith(']]'))):
        return True
        
    # SQL comments
    if file_ext in ['.sql'] and (line.startswith('--') or line.startswith('/*')):
        return True
    
    return False


def fast_walk(directory: str, skip_dirs: Optional[Set[str]] = None) -> Tuple[str, List[str], List[str]]:
    """
    Ultra-fast parallel directory traversal using recursive threading.
    
    Args:
        directory: Directory to walk
        skip_dirs: Set of directory names to skip
        
    Yields:
        Tuples of (current_dir, subdirectories, files)
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_SKIP_DIRS
    
    # Enable long path support on Windows
    if sys.platform == 'win32' and not directory.startswith('\\\\?\\'):
        directory = os.path.abspath(directory)
        if not directory.startswith('\\\\'):
            directory = '\\\\?\\' + directory

    import concurrent.futures
    from collections import deque
    
    results = []
    results_lock = threading.Lock()
    
    def scan_directory_parallel(dir_path: str, executor: concurrent.futures.ThreadPoolExecutor) -> List[concurrent.futures.Future]:
        """Scan a single directory and submit subdirectories for parallel processing."""
        futures = []
        
        try:
            subdirs = []
            files = []
            
            # Use scandir for efficient directory scanning
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    try:
                        name = entry.name
                        
                        # Skip hidden files/dirs and specified skip dirs
                        if (sys.platform == 'win32' and name.startswith('.')) or name in skip_dirs:
                            continue
                        
                        # Use cached stat info from scandir
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            files.append(name)
                            
                    except (PermissionError, OSError, FileNotFoundError):
                        continue
            
            # Add current directory results
            with results_lock:
                results.append((dir_path, [os.path.basename(d) for d in subdirs], files))
            
            # Submit subdirectories for parallel processing
            for subdir in subdirs:
                future = executor.submit(scan_directory_parallel, subdir, executor)
                futures.append(future)
                
        except (PermissionError, OSError):
            pass
            
        return futures
    
    # Use a large thread pool for directory scanning
    max_workers = min(64, os.cpu_count() * 8)  # More aggressive for I/O bound directory scanning
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start with the root directory
        initial_futures = scan_directory_parallel(directory, executor)
        
        # Process all futures as they complete
        all_futures = deque(initial_futures)
        
        while all_futures:
            # Process completed futures in batches
            batch_size = min(100, len(all_futures))
            current_batch = [all_futures.popleft() for _ in range(batch_size)]
            
            for future in concurrent.futures.as_completed(current_batch):
                try:
                    new_futures = future.result()
                    all_futures.extend(new_futures)
                except Exception:
                    pass  # Skip any failed directory scans
    
    # Yield all collected results
    for result in results:
        yield result


def search_file(path: str, file_ext: str, search_terms: List[str], 
               case_sensitive: bool, use_regex: bool, color_output: bool, 
               context_lines: int, ignore_comments: bool, 
               sanitized_patterns: Dict[str, str], 
               compiled_patterns: Optional[List[regex.Pattern]] = None,
               multiline: bool = True,
               anti_patterns: Optional[List[regex.Pattern]] = None) -> List[Dict[str, Any]]:
    """
    Search a single file for all terms and return matches.
    
    Args:
        path: Path to file to search
        file_ext: File extension (including dot)
        search_terms: List of search terms (regex patterns or literal strings)
        case_sensitive: Whether search is case sensitive
        use_regex: Whether to use regex or literal string search
        color_output: Whether to use color in output
        context_lines: Number of lines of context to include before/after match
        ignore_comments: Whether to ignore matches in comments
        sanitized_patterns: Dictionary of sanitized regex patterns
        compiled_patterns: List of pre-compiled regex patterns
        multiline: Whether to use multi-line regex mode (default: True)
        anti_patterns: List of regex patterns to exclude matches (default: None)
        
    Returns:
        List of match dictionaries
    """
    file_matches = []
    
    try:
        # Use cached content if available
        lines = _read_file_with_cache(path)
        if lines is None:
            return file_matches  # Return empty list if file can't be read

        # Pre-process content for better performance
        if multiline and use_regex:
            # Process the entire file as a single string for multiline mode
            file_content = ''.join(lines)
            line_offsets = [0]  # Track start positions of each line
            current_pos = 0
            for line in lines:
                current_pos += len(line)
                line_offsets.append(current_pos)
            
            # Optimized binary search function to find line index from character offset
            def find_line_index(offset, offsets):
                left, right = 0, len(offsets) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if offsets[mid] <= offset and (mid == len(offsets) - 1 or offsets[mid + 1] > offset):
                        return mid
                    elif offsets[mid] > offset:
                        right = mid - 1
                    else:
                        left = mid + 1
                return 0  # Default to first line if not found
            
            # Optimized function to find matches using regex module's efficient search
            def find_matches(pattern, text):
                try:
                    # Use finditer which is more memory efficient than findall
                    return list(pattern.finditer(text))
                except Exception as e:
                    return []
            
            # Process each search term
            for i, term in enumerate(search_terms):
                # Use MULTILINE and DOTALL flags for multiline mode
                flags = re.MULTILINE | re.DOTALL
                if not case_sensitive:
                    flags |= re.IGNORECASE
                
                if compiled_patterns and i < len(compiled_patterns):
                    pattern = compiled_patterns[i]
                else:
                    try:
                        pattern = compile_regex(term, flags)
                    except regex.error:
                        # Try to sanitize and compile the pattern
                        if term in sanitized_patterns:
                            sanitized_term = sanitized_patterns[term]
                        else:
                            sanitized_term = sanitize_regex_pattern(term)
                            sanitized_patterns[term] = sanitized_term
                        
                        try:
                            pattern = compile_regex(sanitized_term, flags)
                        except regex.error:
                            continue  # Skip this term if it still can't be compiled
                
                # Find all matches in the file content
                for match in find_matches(pattern, file_content):
                    match_start, match_end = match.span()
                    
                    # Find which line contains the start and end of the match using binary search
                    start_line_idx = find_line_index(match_start, line_offsets)
                    end_line_idx = find_line_index(match_end - 1 if match_end > 0 else 0, line_offsets)
                    
                    # Extract the actual matched text
                    matched_text = match.group(0)
                    
                    # Check against anti-patterns if provided
                    if anti_patterns:
                        should_exclude = False
                        for anti_pattern in anti_patterns:
                            if anti_pattern.search(matched_text):
                                should_exclude = True
                                break
                        if should_exclude:
                            continue  # Skip this match if it matches an anti-pattern
                    
                    # Ensure we get the entire multi-line match plus context
                    context_start = max(0, start_line_idx - context_lines)
                    context_end = min(len(lines), end_line_idx + context_lines + 1)
                    context = "".join(lines[context_start:context_end])
                    
                    # Get the highlighted text (may span multiple lines)
                    highlighted_text = match.group(0)
                    if color_output:
                        color_start = '\033[91m'  # Red
                        color_end = '\033[0m'  # Reset
                        highlighted = f"{color_start}{highlighted_text}{color_end}"
                    else:
                        highlighted = highlighted_text
                    
                    file_matches.append({
                        "file": path,
                        "line": start_line_idx + 1,
                        "term": term,
                        "context": context.strip(),
                        "highlighted": highlighted,
                        "multiline_match": True,
                        "match_lines": (start_line_idx + 1, end_line_idx + 1)
                    })
        else:
            # Optimized single-line search mode
            # Pre-compile all patterns for better performance
            patterns_to_use = []
            for i, term in enumerate(search_terms):
                flags = 0 if case_sensitive else re.IGNORECASE
                
                if use_regex:
                    if compiled_patterns and i < len(compiled_patterns):
                        pattern = compiled_patterns[i]
                    else:
                        try:
                            pattern = compile_regex(term, flags)
                        except regex.error:
                            # Try to sanitize and compile the pattern
                            if term in sanitized_patterns:
                                sanitized_term = sanitized_patterns[term]
                            else:
                                sanitized_term = sanitize_regex_pattern(term)
                                sanitized_patterns[term] = sanitized_term
                            
                            try:
                                pattern = compile_regex(sanitized_term, flags)
                            except regex.error:
                                continue  # Skip this term if it still can't be compiled
                    patterns_to_use.append((term, pattern))
                else:
                    patterns_to_use.append((term, None))

            # Process lines in batches for better performance
            for line_idx, line in enumerate(lines):
                line_to_check = line.rstrip()
                
                # Skip comments if enabled
                if ignore_comments and is_comment(line_to_check, file_ext):
                    continue
                
                # Check all patterns against this line
                for term, pattern in patterns_to_use:
                    if use_regex:
                        match = pattern.search(line_to_check)
                    else:
                        # Regular string containment
                        match = (term in line_to_check if case_sensitive else 
                                term.lower() in line_to_check.lower())
                    
                    if match:
                        # For regular expression matches, check against anti-patterns
                        if use_regex and anti_patterns:
                            matched_text = line_to_check if not isinstance(match, regex.Match) else match.group(0)
                            should_exclude = False
                            for anti_pattern in anti_patterns:
                                if anti_pattern.search(matched_text):
                                    should_exclude = True
                                    break
                            if should_exclude:
                                continue  # Skip this match if it matches an anti-pattern
                        
                        start = max(0, line_idx - context_lines)
                        end = min(len(lines), line_idx + context_lines + 1)
                        context = "".join(lines[start:end])
                        
                        highlighted = highlight_match(
                            line_to_check, term, pattern if use_regex else None, 
                            color_output, case_sensitive, use_regex
                        )
                        
                        file_matches.append({
                            "file": path,
                            "line": line_idx + 1,
                            "term": term,
                            "context": context.strip(),
                            "highlighted": highlighted
                        })

    except Exception as e:
        # Just return empty list if file can't be read
        pass
        
    return file_matches


def search_code(directory: str, search_terms: List[str], 
               extensions: Optional[List[str]] = None, 
               case_sensitive: bool = False, color_output: bool = True, 
               context_lines: int = 3, ignore_comments: bool = True, 
               max_workers: Optional[int] = None, 
               stop_requested: Optional[Callable[[], bool]] = None,
               multiline: bool = True,
               anti_regex: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Search code in directory for specified terms.
    
    Args:
        directory: Directory to search in
        search_terms: List of search terms (regex patterns or literal strings)
        extensions: List of file extensions to include (with or without leading dot)
        case_sensitive: Whether search is case sensitive
        color_output: Whether to use color in output
        context_lines: Number of lines of context to include before/after match
        ignore_comments: Whether to ignore matches in comments
        max_workers: Number of parallel workers (default: 2x CPU count, max 8 on Windows)
        stop_requested: Callable that returns True if search should be stopped
        multiline: Whether to use multi-line regex mode (default: True)
        anti_regex: List of regex patterns to exclude matches that match them
        
    Returns:
        List of match dictionaries
    """
    # Normalize extensions
    if extensions:
        extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions}

    # Prepare for regex pattern sanitization
    sanitized_patterns = {}
    
    # Precompile regex patterns
    compiled_patterns = []
    flags = 0 if case_sensitive else re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE | re.DOTALL
        
    for term in search_terms:
        try:
            pattern = compile_regex(term, flags)
        except regex.error:
            sanitized_term = sanitize_regex_pattern(term)
            sanitized_patterns[term] = sanitized_term
            try:
                pattern = compile_regex(sanitized_term, flags)
            except regex.error:
                # If compilation still fails, use a pattern that won't match anything
                pattern = regex.compile(r'(?!x)x')  # This pattern always fails to match
        compiled_patterns.append(pattern)
    
    # Compile anti-regex patterns if provided
    anti_patterns = None
    if anti_regex:
        anti_patterns = []
        for term in anti_regex:
            try:
                pattern = compile_regex(term, flags)
                anti_patterns.append(pattern)
            except regex.error:
                sanitized_term = sanitize_regex_pattern(term)
                try:
                    pattern = compile_regex(sanitized_term, flags)
                    anti_patterns.append(pattern)
                except regex.error:
                    print(f"Warning: Could not compile anti-regex pattern: '{term}'")
                    continue
    
    # Setup tracking variables
    total_files = 0
    processed_files = 0
    displayed_locations = set()  # Track displayed file:line combinations to avoid duplicates
    unique_matches = {}  # Map to store unique matches by location (file:line)
    context_ranges = {}  # Map of file -> list of (start_line, end_line) tuples for context ranges

    # Set appropriate number of workers
    if max_workers is None:
        # Use more aggressive threading for regex-heavy workloads
        # The regex module and file I/O can release the GIL more effectively
        max_workers = os.cpu_count() * 4  # Increased from 2x to 4x
        max_workers = max(1, min(max_workers, 16 if sys.platform == 'win32' else 32))
    
    print(f"Using {max_workers} worker threads for parallel regex processing")
    
    # Check for cached file list - create a hashable key
    # Note: We don't include anti_regex in the cache key as it only affects results, not file list
    cache_key = (os.path.abspath(directory), frozenset(extensions) if extensions else None)
    cached_files = _file_cache.get(cache_key)
    
    # Process search in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        pbar = tqdm(desc="Searching", unit="files")
        
        # Define function to process completed futures
        def process_completed_futures(block=False):
            nonlocal processed_files
            
            # Check if search should be stopped
            if stop_requested and stop_requested():
                for future in futures:
                    future.cancel()
                futures.clear()
                return True
            
            # Use as_completed with timeout=0 for non-blocking or None for blocking
            timeout = None if block else 0
            
            try:
                # Process any completed futures
                for future in concurrent.futures.as_completed(list(futures.keys()), timeout=timeout):
                    file_path = futures.pop(future)
                    try:
                        file_matches = future.result()
                        if file_matches:
                            # Process matches and deduplicate
                            for match in file_matches:
                                # Handle multiline matches differently
                                if multiline and match.get("multiline_match"):
                                    # Use file and match start-end lines as identifier
                                    start_line, end_line = match.get("match_lines", (match["line"], match["line"]))
                                    location = f"{match['file']}:{start_line}-{end_line}"
                                else:
                                    location = f"{match['file']}:{match['line']}"
                                
                                # Skip if we've already seen this exact location
                                if location in displayed_locations:
                                    continue
                                
                                # Check if this match is contained within the context of another match
                                file_path = match['file']
                                match_line = match['line']
                                
                                # For multiline matches, check overlap with any part of the match
                                if multiline and match.get("multiline_match"):
                                    start_line, end_line = match.get("match_lines", (match_line, match_line))
                                    is_contained = False
                                    
                                    # Check if any part of this match overlaps with existing context ranges
                                    if file_path in context_ranges:
                                        for ctx_start, ctx_end in context_ranges[file_path]:
                                            # If any line of the match is within an existing context, skip it
                                            if (start_line <= ctx_end and end_line >= ctx_start):
                                                is_contained = True
                                                break
                                    
                                    if is_contained:
                                        continue
                                    
                                    # Add this match's context range to our tracking
                                    context_start = max(1, start_line - context_lines)
                                    context_end = end_line + context_lines
                                    if file_path not in context_ranges:
                                        context_ranges[file_path] = []
                                    context_ranges[file_path].append((context_start, context_end))
                                else:
                                    # Single line match - check if it falls within an existing context range
                                    is_contained = False
                                    if file_path in context_ranges:
                                        for ctx_start, ctx_end in context_ranges[file_path]:
                                            if ctx_start <= match_line <= ctx_end:
                                                is_contained = True
                                                break
                                    
                                    if is_contained:
                                        continue
                                    
                                    # Add this match's context range to our tracking
                                    context_start = max(1, match_line - context_lines)
                                    context_end = match_line + context_lines
                                    if file_path not in context_ranges:
                                        context_ranges[file_path] = []
                                    context_ranges[file_path].append((context_start, context_end))
                                
                                # If we reach here, this is a new unique match
                                displayed_locations.add(location)
                                unique_matches[location] = match
                                print(f"{location}: {match['highlighted']}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}", file=sys.stderr)
                    finally:
                        processed_files += 1
                        pbar.update(1)
            except concurrent.futures.TimeoutError:
                # No futures completed within the timeout
                pass
            
            return False
        
        try:
            if cached_files:
                # Use cached file list if available
                print(f"Using cached file list ({len(cached_files)} files)")
                total_files = len(cached_files)
                pbar.total = total_files
                
                # Process files from cache
                for file_info in cached_files:
                    # Check for stop request
                    if stop_requested and stop_requested():
                        break
                    
                    path, file_ext = file_info
                    
                    # Submit the file for processing
                    future = executor.submit(
                        search_file, path, file_ext, search_terms, 
                        case_sensitive, True, color_output, 
                        context_lines, ignore_comments, 
                        sanitized_patterns, compiled_patterns, multiline, anti_patterns
                    )
                    futures[future] = path
                    
                    # Periodically process completed futures
                    # Use larger batch sizes for better throughput with more threads
                    if len(futures) > 200 or (sys.platform == 'win32' and len(futures) > 100):
                        if process_completed_futures(block=False):
                            break  # Stop requested
            else:
                # No cached files, collect them first using fast parallel directory traversal
                print("Scanning directories in parallel...")
                start_time = time.time()
                file_list = []
                
                # Collect all files first using the optimized fast_walk
                for root, dirs, files in fast_walk(directory, DEFAULT_SKIP_DIRS):
                    for file in files:
                        # Get extension efficiently
                        file_lower = file.lower()
                        dot_index = file_lower.rfind('.')
                        
                        if dot_index != -1:
                            file_ext = file_lower[dot_index:]
                            if extensions and file_ext not in extensions:
                                continue
                        elif extensions:
                            continue
                        else:
                            file_ext = ''
                        
                        path = os.path.join(root, file)
                        file_list.append((path, file_ext))
                
                scan_time = time.time() - start_time
                total_files = len(file_list)
                print(f"Found {total_files} files in {scan_time:.2f} seconds ({total_files/scan_time:.0f} files/sec)")
                pbar.total = total_files
                
                # Process files in optimized batches
                batch_size = max_workers * 4  # Process multiple batches per worker
                for i in range(0, len(file_list), batch_size):
                    if stop_requested and stop_requested():
                        break
                        
                    batch = file_list[i:i + batch_size]
                    
                    # Submit batch for processing
                    for path, file_ext in batch:
                        if stop_requested and stop_requested():
                            break
                        
                        future = executor.submit(
                            search_file, path, file_ext, search_terms, 
                            case_sensitive, True, color_output, 
                            context_lines, ignore_comments, 
                            sanitized_patterns, compiled_patterns, multiline, anti_patterns
                        )
                        futures[future] = path
                    
                    # Process completed futures when we have enough
                    if len(futures) > max_workers * 2:
                        if process_completed_futures(block=False):
                            break  # Stop requested
                
                # Cache the file list for future searches if not stopped
                if not (stop_requested and stop_requested()):
                    _file_cache[cache_key] = file_list
            
            # Process all remaining futures if not stopped
            if not (stop_requested and stop_requested()):
                process_completed_futures(block=True)
            
        finally:
            pbar.close()
    
    # Report warnings for sanitized patterns
    for term, sanitized_term in sanitized_patterns.items():
        print(f"Warning: Fixed problematic regex pattern: '{term}' → '{sanitized_term}'")
    
    # Convert unique_matches map to list
    all_matches = list(unique_matches.values())
    
    # Print summary
    if stop_requested and stop_requested():
        print(f"\nSearch stopped. Processed {processed_files} files, found {len(all_matches)} unique matches")
    else:
        print(f"\nProcessed {processed_files} files, found {len(all_matches)} unique matches")
    
    return all_matches


def chat_about_matches(matches: List[Dict[str, Any]], 
                      original_prompt: str, 
                      provider: str = "anthropic") -> None:
    """
    Start an interactive chat session with AI about search results.
    
    Args:
        matches: List of match dictionaries
        original_prompt: Original search prompt
        provider: AI provider to use ("anthropic", "openai", or "azure")
    """
    client = get_ai_client(provider)
    
    # Prepare context for the AI
    context_sections = []
    for i, m in enumerate(matches[:20]):  # Limit to first 20 matches
        # Handle multiline matches differently
        if m.get("multiline_match"):
            start_line, end_line = m.get("match_lines", (m["line"], m["line"]))
            context_sections.append(
                f"{m['file']}:{start_line}-{end_line} (Match #{i+1})\n"
                f"Matched term: {m['term']}\n{m['context']}"
            )
        else:
            context_sections.append(
                f"{m['file']}:{m['line']} (Match #{i+1})\n"
                f"Matched term: {m['term']}\n{m['context']}"
            )
    
    combined_contexts = "\n\n---\n\n".join(context_sections)
    
    system_message = """You are an expert code analyst helping to interpret search results from a codebase.
Focus on explaining:
1. How the found code works and its purpose
2. Potential security implications or bugs if relevant 
3. Connections between different matches
4. Clear, factual analysis based only on the provided code

IMPORTANT CONSTRAINTS:
- DO NOT provide code fixes or solutions
- DO NOT generate new code
- DO NOT suggest code changes
- Only analyze and explain the existing code
- If you identify issues, explain them but do not propose fixes

When referring to matches, ALWAYS use the FULL file path and line number (e.g., '/path/to/file.cpp:123') rather than match numbers or just the filename.
Keep your responses concise and to the point.

IMPORTANT: Always directly address the user's questions. If they ask a specific question about the code or results, make sure to answer that question directly rather than continuing your previous analysis.
"""

    print(f"\nEntering chat mode. {len(matches)} matches found for: '{original_prompt}'")
    print("Ask questions about the search results or type 'exit' to quit.\n")

    # Store the initial search results as a fixed context that will always be included
    search_context = f"These are code search results for: '{original_prompt}'\n\n{combined_contexts}"
    
    # Keep track of user-assistant exchanges separately from the fixed context
    exchanges = []

    # Initial analysis request
    initial_prompt = "Please analyze these findings."
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        # Add user message to exchanges
        exchanges.append({"role": "user", "content": user_input})
        
        # Limit exchanges to prevent context overflow
        recent_exchanges = exchanges[-12:] if len(exchanges) > 12 else exchanges
        
        # Construct messages: first the context message, then recent exchanges
        messages = [{"role": "user", "content": search_context}]
        
        # Add a separator if we have exchanges
        if recent_exchanges:
            messages.append({"role": "assistant", "content": "I'll analyze these code search results. What specific aspects would you like me to focus on?"})
            messages.extend(recent_exchanges)
        else:
            # If this is the first exchange, add the initial analysis request
            messages.append({"role": "user", "content": initial_prompt})
        
        if provider == "anthropic":
            # Use streaming API for Anthropic
            print("\nClaude: ", end="", flush=True)
            full_response = ""
            
            with client.messages.stream(
                model="claude-3-7-sonnet-latest",
                max_tokens=4096,
                temperature=1,
                system=system_message,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2048
                },
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
        elif provider == "azure":
            # Use streaming API for Azure OpenAI
            print("\nAzure OpenAI: ", end="", flush=True)
            full_response = ""
            
            deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
            stream = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "system", "content": system_message}] + messages,
                temperature=1,
                max_tokens=4096,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_response += text
        else:  # OpenAI
            # Use streaming API for OpenAI
            print("\nGPT-4: ", end="", flush=True)
            full_response = ""
            
            stream = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "system", "content": system_message}] + messages,
                temperature=1,
                max_completion_tokens=4096,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_response += text
        
        print()  # Add a newline after the streamed response
        
        # Add assistant response to exchanges
        exchanges.append({"role": "assistant", "content": full_response})


def clear_file_cache() -> None:
    """Clear the file list cache."""
    global _file_cache
    _file_cache.clear()


def clear_content_cache() -> None:
    """Clear the file content cache."""
    global _content_cache, _content_cache_stats
    _content_cache.clear()
    _content_cache_stats["hits"] = 0
    _content_cache_stats["misses"] = 0


def get_content_cache_stats() -> Dict[str, Any]:
    """Get content cache statistics."""
    return {
        "cache_size": len(_content_cache),
        "hits": _content_cache_stats["hits"],
        "misses": _content_cache_stats["misses"],
        "hit_rate": _content_cache_stats["hits"] / max(1, _content_cache_stats["hits"] + _content_cache_stats["misses"]),
        "max_size": _content_cache_stats["max_size"]
    }


def set_content_cache_max_size(max_size: int) -> None:
    """Set the maximum number of files to cache in memory."""
    global _content_cache_stats
    _content_cache_stats["max_size"] = max_size
    _evict_content_cache_if_needed()


def _evict_content_cache_if_needed() -> None:
    """Evict oldest entries from content cache if it exceeds max size."""
    global _content_cache
    max_size = _content_cache_stats["max_size"]
    
    if len(_content_cache) > max_size:
        # Remove oldest entries (simple FIFO eviction)
        items_to_remove = len(_content_cache) - max_size
        keys_to_remove = list(_content_cache.keys())[:items_to_remove]
        for key in keys_to_remove:
            del _content_cache[key]


def _get_file_cache_key(path: str) -> Tuple[str, float]:
    """
    Generate a cache key for a file based on path and modification time.
    
    Args:
        path: File path
        
    Returns:
        Tuple of (absolute_path, modification_time)
    """
    try:
        abs_path = os.path.abspath(path)
        mtime = os.path.getmtime(path)
        return (abs_path, mtime)
    except (OSError, FileNotFoundError):
        # If we can't get mtime, use current time to force cache miss
        return (os.path.abspath(path), time.time())


def _read_file_with_cache(path: str) -> Optional[List[str]]:
    """
    Read file contents with caching support.
    
    Args:
        path: Path to file to read
        
    Returns:
        List of lines from the file, or None if file cannot be read
    """
    global _content_cache, _content_cache_stats
    
    cache_key = _get_file_cache_key(path)
    
    # Check if we have cached content for this file
    if cache_key in _content_cache:
        _content_cache_stats["hits"] += 1
        return _content_cache[cache_key]
    
    # Cache miss - read the file
    _content_cache_stats["misses"] += 1
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Cache the content if we're under the size limit
        if len(_content_cache) < _content_cache_stats["max_size"]:
            _content_cache[cache_key] = lines
        else:
            # Evict old entries and add new one
            _evict_content_cache_if_needed()
            _content_cache[cache_key] = lines
        
        return lines
    except Exception:
        return None


def get_refined_search_terms(prompt: str, matches: List[Dict[str, Any]], 
                            max_terms: int = 10, extensions: Optional[List[str]] = None, 
                            context_lines: int = 3, provider: str = "anthropic",
                            multiline: bool = True) -> Tuple[List[str], List[str]]:
    """
    Generate refined search terms and anti-patterns based on initial matches.
    
    Args:
        prompt: Original search prompt
        matches: List of match dictionaries from initial search
        max_terms: Maximum number of search terms to generate
        extensions: List of file extensions to focus on
        context_lines: Number of lines of context around matches (default: 3)
        provider: AI provider to use ("anthropic" or "openai")
        multiline: Whether to enable multi-line regex patterns (default: True)
        
    Returns:
        Tuple of (search_terms, anti_patterns)
    """
    client = get_ai_client(provider)
    
    # Prepare context from current matches
    context_sections = []
    for i, m in enumerate(matches[:10]):  # Use first 10 matches for context
        # Split context into lines and get the middle
        context_lines_list = m['context'].split('\n')
        middle_line_index = len(context_lines_list) // 2
        
        # Add file and line info
        if m.get("multiline_match"):
            start_line, end_line = m.get("match_lines", (m["line"], m["line"]))
            context_sections.append(f"File: {m['file']}\nLines: {start_line}-{end_line}\nMatched term: {m['term']}")
        else:
            context_sections.append(f"File: {m['file']}\nLine: {m['line']}\nMatched term: {m['term']}")
        
        # Add context with the matched line highlighted
        context_sections.append("Context:")
        for j, line in enumerate(context_lines_list):
            if j == middle_line_index:
                context_sections.append(f">>> {line}")  # Highlight the matched line
            else:
                context_sections.append(f"    {line}")
    
    combined_contexts = "\n".join(context_sections)
    
    # System prompt for refined search terms
    system_message = """You are a code search expert. Analyze the provided code matches and generate refined search terms.
Focus on:
1. Patterns that appear in successful matches
2. Related code patterns that might be relevant
3. Language-specific syntax patterns
4. Practical regex patterns that can be used with Python's re module

You must generate TWO distinct sets of patterns:
1. Search patterns - regex patterns to find matches for the user's query
2. Anti-patterns - regex patterns to EXCLUDE matches that are irrelevant or false positives

IMPORTANT: Think about how your search terms will work TOGETHER to discover the underlying search intent.
Create a diverse set of patterns that capture different aspects of the search and complement each other.
Some terms should be specific to match known patterns, others more general to discover related code.

For anti-patterns, analyze the current matches to identify false positives or irrelevant matches that should be excluded.
"""
    
    if multiline:
        system_message += "\nYou can create multi-line regex patterns using the re.MULTILINE and re.DOTALL flags support."
    else:
        system_message += "\nIMPORTANT: Each line is processed individually, so DO NOT create multi-line regex patterns. All patterns must match on a single line."
    
    system_message += """
IMPORTANT EFFICIENCY GUIDELINES:
- Avoid nested .* or .*? patterns which cause catastrophic backtracking
- Limit pattern complexity - simpler is faster and more reliable
- Use atomic groups (?>...) where possible
- Prefer bounded quantifiers {0,100} instead of * or +
- Prefer character classes [a-z] over wildcard .
- Add anchors (^ $) when appropriate to limit search space
- For function/class definitions, use more specific start/end markers rather than greedy matches

Your response MUST be structured in two clearly separated sections:

SEARCH PATTERNS:
(list your search patterns here, one per line)

ANTI-PATTERNS:
(list your anti-patterns here, one per line)

Keep each section brief - just list the patterns, with no additional text, explanations, or numbering.
"""
    
    extensions_info = format_extension_info(extensions)
    
    if provider == "anthropic":
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4096,
            temperature=1,
            system=system_message,
            thinking={
                "type": "enabled",
                "budget_tokens": 2048
            },
            messages=[{
                "role": "user",
                "content": f"Original search prompt: '{prompt}'\n\nCurrent matches:\n{combined_contexts}\n\nGenerate up to {max_terms} refined search patterns and anti-patterns based on these matches.{extensions_info}"
            }]
        )
        raw_text = response.content[1].text.strip()
    else:  # OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Original search prompt: '{prompt}'\n\nCurrent matches:\n{combined_contexts}\n\nGenerate up to {max_terms} refined search patterns and anti-patterns based on these matches.{extensions_info}"}
            ],
            temperature=1,
            max_completion_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
    
    return parse_ai_response_with_antipatterns(raw_text, max_terms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search code and chat with AI about results.")
    parser.add_argument("directory", help="Directory to search in")
    parser.add_argument("--prompt", help="Natural language prompt to generate search terms")
    parser.add_argument("-e", "--extensions", nargs='+', help="File extensions to include (e.g., .py .js)")
    parser.add_argument("-i", "--insensitive", action="store_true", help="Case-insensitive search")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--no-chat", action="store_true", help="Skip chat mode")
    parser.add_argument("--include-comments", action="store_true", help="Include comments in search results")
    parser.add_argument("--terms", type=int, default=10, help="Number of search terms to generate")
    parser.add_argument("--context", type=int, default=6, help="Lines of context before/after match")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--provider", choices=["anthropic", "openai", "azure"], default="anthropic", help="AI provider to use")
    parser.add_argument("--single-line", action="store_true", help="Disable multi-line regex mode (uses single-line mode)")
    parser.add_argument("--cache-size", type=int, default=50000, help="Maximum number of files to cache in memory (default: 50000)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear both file list and content caches before searching")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics after search")
    args = parser.parse_args()
    
    try:
        # Handle cache management
        if args.clear_cache:
            clear_file_cache()
            clear_content_cache()
            print("Cleared file list and content caches")
        
        # Set content cache size
        set_content_cache_max_size(args.cache_size)
        
        # Get search terms and anti-patterns
        search_terms, ai_anti_patterns = get_search_terms_from_prompt(
            args.prompt, 
            args.terms, 
            args.extensions, 
            args.provider,
            not args.single_line  # Invert the single-line flag to get multiline
        )
        
        # Debug output
        print(f"DEBUG: search_terms type before processing: {type(search_terms)}")
        
        # Convert string patterns to lists of strings if needed
        if isinstance(search_terms, str):
            search_terms = [search_terms]
        elif not isinstance(search_terms, list):
            try:
                search_terms = list(search_terms)
            except:
                search_terms = []
                
        if isinstance(ai_anti_patterns, str):
            ai_anti_patterns = [ai_anti_patterns]
        elif not isinstance(ai_anti_patterns, list):
            try:
                ai_anti_patterns = list(ai_anti_patterns)
            except:
                ai_anti_patterns = []
        
        # Ensure all items in lists are strings
        search_terms = [str(term) for term in search_terms]
        ai_anti_patterns = [str(pattern) for pattern in ai_anti_patterns]
        
        # Print generated patterns
        print("Generated search patterns:")
        for term in search_terms:
            print(f"  {term}")
        
        print("\nGenerated anti-patterns:")
        for pattern in ai_anti_patterns:
            print(f"  {pattern}")
        
        # Debug output
        print(f"DEBUG: Final search_terms type: {type(search_terms)}")
        print(f"DEBUG: Final ai_anti_patterns type: {type(ai_anti_patterns)}")
        
        # Search code
        matches = search_code(
            directory=args.directory,
            search_terms=search_terms,
            extensions=args.extensions,
            case_sensitive=not args.insensitive,
            color_output=not args.no_color,
            context_lines=args.context,
            ignore_comments=not args.include_comments,
            max_workers=args.workers,
            multiline=not args.single_line,  # Invert the single-line flag to get multiline
            anti_regex=ai_anti_patterns
        )
        
        # Chat about results if enabled
        if not args.no_chat and matches:
            chat_about_matches(matches, args.prompt, args.provider)
        
        # Show cache statistics if requested
        if args.cache_stats:
            stats = get_content_cache_stats()
            print(f"\nContent Cache Statistics:")
            print(f"  Cache size: {stats['cache_size']} files")
            print(f"  Cache hits: {stats['hits']}")
            print(f"  Cache misses: {stats['misses']}")
            print(f"  Hit rate: {stats['hit_rate']:.2%}")
            print(f"  Max cache size: {stats['max_size']}")
    
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
