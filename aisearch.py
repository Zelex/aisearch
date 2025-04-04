import os
import re
import sys
import argparse
import concurrent.futures
from typing import List, Dict, Any, Set, Tuple, Optional, Callable, Union

# Third-party imports
import anthropic
import openai
from tqdm import tqdm

# Cache for file lists when directory and extensions don't change
_file_cache = {}

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

def get_ai_client(provider: str = "anthropic") -> Union[anthropic.Anthropic, openai.OpenAI]:
    """
    Get the appropriate AI client based on provider.
    
    Args:
        provider: AI provider to use ("anthropic" or "openai")
        
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
                                provider: str = "anthropic") -> List[str]:
    """
    Generate search terms from a natural language prompt using AI.
    
    Args:
        prompt: Natural language prompt describing what to search for
        max_terms: Maximum number of search terms to generate
        extensions: List of file extensions to focus on
        provider: AI provider to use ("anthropic" or "openai")
        
    Returns:
        List of search terms (regex patterns)
    """
    client = get_ai_client(provider)
    
    system_message = """You are a code search expert. Generate search terms that will effectively find relevant code patterns.
Focus on practical patterns that would appear in actual code.
Generate proper regex patterns that can be used with Python's re module, not just literal strings.
Include regex syntax for more powerful searches (e.g., \b for word boundaries, .* for any characters, etc.).
Keep your response VERY brief - just list the search terms, one per line.
Respond with ONLY the search terms, with no additional text, explanations, or numbering.
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
                "content": f"Generate up to {max_terms} effective regex search patterns for finding: '{prompt}'{extensions_info}"
            }]
        )
        raw_text = response.content[1].text.strip()
    else:  # OpenAI
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate up to {max_terms} effective regex search patterns for finding: '{prompt}'{extensions_info}"}
            ],
            temperature=1,
            max_completion_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
    
    return parse_ai_response(raw_text, max_terms)


def parse_ai_response(raw_text: str, max_terms: int) -> List[str]:
    """
    Parse and clean AI response into usable search terms.
    
    Args:
        raw_text: Raw text response from AI
        max_terms: Maximum number of terms to return
        
    Returns:
        List of cleaned search terms
    """
    terms = []
    for line in raw_text.splitlines():
        line = line.strip()
        # Skip empty lines or lines that look like explanations/headers
        if not line or line.startswith('#') or line.startswith('-') or len(line) > 60:
            continue
        # Remove any prefix numbering (1., 2., etc.)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if line:
            terms.append(line)
    
    return terms[:max_terms]  # Ensure we don't exceed max_terms


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
        pattern = re.sub(r'(?<!\\)' + re.escape(char), f'\\{char}', pattern)
    
    return pattern


def highlight_match(line: str, term: str, pattern: Optional[re.Pattern], 
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
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.sub(f'({re.escape(term)})', f'{color_start}\\1{color_end}', line, flags=flags)


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
        if line.startswith('//') or line.startswith('/*') or line.endswith('*/') or re.match(r'^\s*\*', line):
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
    Faster alternative to os.walk using scandir with Windows-specific optimizations.
    
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
        # Convert to absolute path with \\?\ prefix to support long paths
        directory = os.path.abspath(directory)
        if not directory.startswith('\\\\'):  # Not a UNC path
            directory = '\\\\?\\' + directory
    
    # Use a stack instead of a queue for better memory locality
    dirs = [directory]
    while dirs:
        current = dirs.pop()  # Depth-first is more memory efficient
        try:
            # Get subdirectories and files in one pass
            subdirs = []
            files = []
            
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        name = entry.name
                        # Skip hidden files and directories on Windows
                        if sys.platform == 'win32' and name.startswith('.'):
                            continue
                            
                        # Quick name check before expensive is_dir call
                        if name in skip_dirs:
                            continue
                            
                        # Use stat attributes directly when possible
                        if hasattr(entry, 'is_dir') and hasattr(entry, 'is_file'):
                            if entry.is_dir(follow_symlinks=False):
                                subdirs.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                files.append(name)
                        else:
                            # Fallback for systems without DirEntry attributes
                            if os.path.isdir(entry.path):
                                subdirs.append(entry.path)
                            elif os.path.isfile(entry.path):
                                files.append(name)
                    except (PermissionError, OSError, FileNotFoundError):
                        # Skip entries we can't access
                        continue
            
            # Yield the current directory, subdirectory basenames, and files
            yield current, [os.path.basename(d) for d in subdirs], files
            
            # Add subdirs to the stack in reverse order to maintain expected traversal order
            dirs.extend(reversed(subdirs))
        except (PermissionError, OSError, Exception):
            # Skip any directories we can't access
            continue


def search_file(path: str, file_ext: str, search_terms: List[str], 
               case_sensitive: bool, use_regex: bool, color_output: bool, 
               context_lines: int, ignore_comments: bool, 
               sanitized_patterns: Dict[str, str], 
               compiled_patterns: Optional[List[re.Pattern]] = None) -> List[Dict[str, Any]]:
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
        
    Returns:
        List of match dictionaries
    """
    file_matches = []
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for i, term in enumerate(search_terms):
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if use_regex:
                if compiled_patterns and i < len(compiled_patterns):
                    pattern = compiled_patterns[i]
                else:
                    try:
                        pattern = re.compile(term, flags)
                    except re.error:
                        # Try to sanitize and compile the pattern
                        if term in sanitized_patterns:
                            sanitized_term = sanitized_patterns[term]
                        else:
                            sanitized_term = sanitize_regex_pattern(term)
                            sanitized_patterns[term] = sanitized_term
                        
                        try:
                            pattern = re.compile(sanitized_term, flags)
                        except re.error:
                            continue  # Skip this term if it still can't be compiled
            else:
                pattern = None

            for line_idx, line in enumerate(lines):
                line_to_check = line.rstrip()
                
                # Skip comments if enabled
                if ignore_comments and is_comment(line_to_check, file_ext):
                    continue
                    
                if use_regex:
                    match = pattern.search(line_to_check)
                else:
                    # Regular string containment
                    match = (term in line_to_check if case_sensitive else 
                            term.lower() in line_to_check.lower())
                
                if match:
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

    except Exception:
        # Just return empty list if file can't be read
        pass
        
    return file_matches


def search_code(directory: str, search_terms: List[str], 
               extensions: Optional[List[str]] = None, 
               case_sensitive: bool = False, color_output: bool = True, 
               context_lines: int = 3, ignore_comments: bool = True, 
               max_workers: Optional[int] = None, 
               stop_requested: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
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
    for term in search_terms:
        try:
            pattern = re.compile(term, flags)
        except re.error:
            sanitized_term = sanitize_regex_pattern(term)
            sanitized_patterns[term] = sanitized_term
            try:
                pattern = re.compile(sanitized_term, flags)
            except re.error:
                # If compilation still fails, use a pattern that won't match anything
                pattern = re.compile(r'(?!x)x')  # This pattern always fails to match
        compiled_patterns.append(pattern)
    
    # Setup tracking variables
    total_files = 0
    processed_files = 0
    displayed_locations = set()  # Track displayed file:line combinations to avoid duplicates
    unique_matches = {}  # Map to store unique matches by location (file:line)

    # Set appropriate number of workers
    if max_workers is None:
        max_workers = os.cpu_count() * 2
        max_workers = max(1, min(max_workers, 8 if sys.platform == 'win32' else max_workers))
    
    # Check for cached file list
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
                                location = f"{match['file']}:{match['line']}"
                                # If we haven't seen this location before
                                if location not in displayed_locations:
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
                        sanitized_patterns, compiled_patterns
                    )
                    futures[future] = path
                    
                    # Periodically process completed futures
                    if len(futures) > 100 or (sys.platform == 'win32' and len(futures) > 50):
                        if process_completed_futures(block=False):
                            break  # Stop requested
            else:
                # No cached files, collect them first
                file_list = []
                
                # Walk the directory tree using fast_walk and submit tasks as we find files
                for root, dirs, files in fast_walk(directory, DEFAULT_SKIP_DIRS):
                    # Check for stop request
                    if stop_requested and stop_requested():
                        break
                    
                    for file in files:
                        # Check for stop request
                        if stop_requested and stop_requested():
                            break
                            
                        # Get extension - more efficient than splitext
                        file_lower = file.lower()
                        dot_index = file_lower.rfind('.')
                        
                        if dot_index != -1:
                            file_ext = file_lower[dot_index:]
                            # Filter by extension if specified
                            if extensions and file_ext not in extensions:
                                continue
                        elif extensions:
                            # Skip files without extensions if extensions specified
                            continue
                        else:
                            file_ext = ''
                        
                        path = os.path.join(root, file)
                        total_files += 1
                        
                        # Add to file list for caching
                        file_list.append((path, file_ext))
                        
                        # Submit the file for processing
                        future = executor.submit(
                            search_file, path, file_ext, search_terms, 
                            case_sensitive, True, color_output, 
                            context_lines, ignore_comments, 
                            sanitized_patterns, compiled_patterns
                        )
                        futures[future] = path
                        
                        # Periodically process completed futures
                        batch_size = 50 if sys.platform == 'win32' else 100
                        if len(futures) > batch_size:
                            if process_completed_futures(block=False):
                                break  # Stop requested
                        
                        # Update progress bar
                        pbar.total = total_files
                        pbar.refresh()
                    
                    # Process completed futures after each directory
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
        provider: AI provider to use ("anthropic" or "openai")
    """
    client = get_ai_client(provider)
    
    # Prepare context for the AI
    context_sections = []
    for i, m in enumerate(matches[:20]):  # Limit to first 20 matches
        context_sections.append(
            f"{m['file']}:{m['line']} (Match #{i+1})\n"
            f"Matched term: {m['term']}\n{m['context']}"
        )
    
    combined_contexts = "\n\n---\n\n".join(context_sections)
    
    system_message = """You are an expert code analyst helping to interpret search results from a codebase.
Focus on explaining:
1. How the found code works
2. Potential security implications or bugs if relevant 
3. Connections between different matches
4. Clear, factual analysis based only on the provided code

When referring to matches, ALWAYS use the FULL file path and line number (e.g., '/path/to/file.cpp:123') rather than match numbers or just the filename.
Keep your responses concise and to the point."""

    print(f"\nEntering chat mode. {len(matches)} matches found for: '{original_prompt}'")
    print("Ask questions about the search results or type 'exit' to quit.\n")

    history = [{
        "role": "user",
        "content": f"These are code search results for: '{original_prompt}'\n\n{combined_contexts}\n\nPlease analyze these findings."
    }]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        history.append({"role": "user", "content": user_input})
        
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
                messages=history[-10:]  # Send only the last 10 exchanges
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
        else:  # OpenAI
            # Use streaming API for OpenAI
            print("\nGPT-4: ", end="", flush=True)
            full_response = ""
            
            stream = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "system", "content": system_message}] + history[-10:],
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
        history.append({"role": "assistant", "content": full_response})


def clear_file_cache() -> None:
    """Clear the file list cache."""
    global _file_cache
    _file_cache.clear()


def get_refined_search_terms(prompt: str, matches: List[Dict[str, Any]], 
                            max_terms: int = 10, extensions: Optional[List[str]] = None, 
                            context_lines: int = 3, provider: str = "anthropic") -> List[str]:
    """
    Generate refined search terms based on initial matches.
    
    Args:
        prompt: Original search prompt
        matches: List of match dictionaries from initial search
        max_terms: Maximum number of search terms to generate
        extensions: List of file extensions to focus on
        context_lines: Number of lines of context around matches (default: 3)
        provider: AI provider to use ("anthropic" or "openai")
        
    Returns:
        List of refined search terms
    """
    client = get_ai_client(provider)
    
    # Prepare context from current matches
    context_sections = []
    for i, m in enumerate(matches[:10]):  # Use first 10 matches for context
        # Split context into lines and get the middle
        context_lines_list = m['context'].split('\n')
        middle_line_index = len(context_lines_list) // 2
        
        # Add file and line info
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

Keep your response VERY brief - just list the search terms, one per line.
Respond with ONLY the search terms, with no additional text, explanations, or numbering.
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
                "content": f"Original search prompt: '{prompt}'\n\nCurrent matches:\n{combined_contexts}\n\nGenerate up to {max_terms} refined search terms based on these matches.{extensions_info}"
            }]
        )
        raw_text = response.content[1].text.strip()
    else:  # OpenAI
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Original search prompt: '{prompt}'\n\nCurrent matches:\n{combined_contexts}\n\nGenerate up to {max_terms} refined search terms based on these matches.{extensions_info}"}
            ],
            temperature=1,
            max_completion_tokens=4096
        )
        raw_text = response.choices[0].message.content.strip()
    
    return parse_ai_response(raw_text, max_terms)


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
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic", help="AI provider to use")
    args = parser.parse_args()
    
    # Get search terms
    search_terms = get_search_terms_from_prompt(args.prompt, args.terms, args.extensions, args.provider)
    
    # Search code
    matches = search_code(
        directory=args.directory,
        search_terms=search_terms,
        extensions=args.extensions,
        case_sensitive=not args.insensitive,
        color_output=not args.no_color,
        context_lines=args.context,
        ignore_comments=not args.include_comments,
        max_workers=args.workers
    )
    
    # Chat about results if enabled
    if not args.no_chat and matches:
        chat_about_matches(matches, args.prompt, args.provider)
