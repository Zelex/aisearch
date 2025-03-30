import os
import re
import argparse
import anthropic
import sys
import concurrent.futures
from tqdm import tqdm

def get_search_terms_from_prompt(prompt, max_terms=10, extensions=None):
    client = anthropic.Anthropic()
    
    # Map file extensions to languages
    language_map = {
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
    
    # Convert extensions to languages if recognized
    languages = []
    if extensions:
        for ext in extensions:
            ext = ext if ext.startswith('.') else f'.{ext}'
            if ext in language_map:
                languages.append(language_map[ext])
    
    # Improved system prompt for effective code search
    system_message = """You are a code search expert. Generate search terms that will effectively find relevant code patterns.
Focus on practical patterns that would appear in actual code.
Generate proper regex patterns that can be used with Python's re module, not just literal strings.
Include regex syntax for more powerful searches (e.g., \b for word boundaries, .* for any characters, etc.).
Keep your response VERY brief - just list the search terms, one per line.
Respond with ONLY the search terms, with no additional text, explanations, or numbering.
"""
    
    # Format language/extension info for the prompt
    extensions_info = ""
    if extensions:
        ext_list = ', '.join(extensions)
        if languages:
            lang_list = ', '.join(languages)
            extensions_info = f"\nLook specifically in {ext_list} files ({lang_list}). Tailor search terms to these languages using their specific syntax patterns."
        else:
            extensions_info = f"\nLook specifically in {ext_list} files. Tailor search terms to these file types."
    
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
    
    # Clean and filter the terms
    raw_text = response.content[1].text.strip()
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


def sanitize_regex_pattern(pattern):
    """
    Sanitize a potentially problematic regex pattern.
    This handles common regex issues that might cause 'unterminated character set' errors.
    """
    # If the pattern has unbalanced square brackets, escape them all
    if pattern.count('[') != pattern.count(']'):
        pattern = pattern.replace('[', '\\[').replace(']', '\\]')
    
    # If the pattern has unbalanced curly braces, escape them all
    if pattern.count('{') != pattern.count('}'):
        pattern = pattern.replace('{', '\\{').replace('}', '\\}')
    
    # If the pattern has unbalanced parentheses, escape them all
    if pattern.count('(') != pattern.count(')'):
        pattern = pattern.replace('(', '\\(').replace(')', '\\)')
    
    # Escape other problematic characters if they're not already escaped
    problematic_chars = ['*', '+', '?', '|', '^', '$', '.']
    for char in problematic_chars:
        # Only escape if not already escaped
        pattern = re.sub(r'(?<!\\)' + re.escape(char), f'\\{char}', pattern)
    
    return pattern


def highlight_match(line, term, pattern, color_output, case_sensitive, use_regex):
    if not color_output:
        return line

    color_start = '\033[91m'
    color_end = '\033[0m'

    def replace(match):
        return f"{color_start}{match.group(0)}{color_end}"

    if use_regex:
        return pattern.sub(replace, line)
    else:
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.sub(f'({re.escape(term)})', f'{color_start}\\1{color_end}', line, flags=flags)


def is_comment(line, file_ext):
    """Determine if a line is a comment based on file extension."""
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


def search_file(path, file_ext, search_terms, case_sensitive, use_regex, color_output, context_lines, ignore_comments, sanitized_patterns):
    """Search a single file for all terms and return matches."""
    file_matches = []
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for term in search_terms:
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if use_regex:
                try:
                    pattern = re.compile(term, flags)
                except re.error:
                    # If pattern is already sanitized, use it
                    if term in sanitized_patterns:
                        sanitized_term = sanitized_patterns[term]
                    else:
                        sanitized_term = sanitize_regex_pattern(term)
                        sanitized_patterns[term] = sanitized_term
                    
                    try:
                        pattern = re.compile(sanitized_term, flags)
                    except re.error:
                        continue
            else:
                pattern = None

            for i, line in enumerate(lines):
                line_to_check = line.rstrip()
                
                # Skip comments if enabled
                if ignore_comments and is_comment(line_to_check, file_ext):
                    continue
                    
                if use_regex:
                    match = pattern.search(line_to_check)
                else:
                    # Regular string containment
                    match = (term in line_to_check if case_sensitive else term.lower() in line_to_check.lower())
                
                if match:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "".join(lines[start:end])
                    
                    highlighted = highlight_match(line_to_check, term, pattern if use_regex else None, 
                                                color_output, case_sensitive, use_regex)
                    
                    file_matches.append({
                        "file": path,
                        "line": i + 1,
                        "term": term,
                        "context": context.strip(),
                        "highlighted": highlighted
                    })

    except Exception as e:
        # Just return empty list if file can't be read
        pass
        
    return file_matches


def search_code(directory, search_terms, extensions=None, case_sensitive=False,
                color_output=True, context_lines=3, ignore_comments=True, max_workers=None):
    if extensions:
        extensions = set(ext.lower() for ext in extensions)

    # Track which patterns have already been sanitized to avoid duplicate warnings
    sanitized_patterns = {}
    all_matches = []
    total_files = 0
    processed_files = 0

    # If no custom max_workers specified, determine an appropriate value
    if max_workers is None:
        max_workers = os.cpu_count() * 2
        max_workers = max(1, max_workers)  # Ensure at least 1 worker
    
    # Create a ThreadPoolExecutor to search files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a dictionary to track all futures for files being processed
        futures = {}
        
        # Create a progress bar that will be updated as we go
        pbar = tqdm(desc="Searching", unit="files")
        
        # Process completed futures while we continue to find more files
        def process_completed_futures(block=False):
            nonlocal processed_files
            
            # Use as_completed with timeout=0 for non-blocking or None for blocking
            timeout = None if block else 0
            
            try:
                # Process any completed futures
                for future in concurrent.futures.as_completed(list(futures.keys()), timeout=timeout):
                    file_path = futures.pop(future)
                    try:
                        file_matches = future.result()
                        if file_matches:
                            all_matches.extend(file_matches)
                            # Display matches immediately
                            for match in file_matches:
                                print(f"{match['file']}:{match['line']}: {match['highlighted']}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}", file=sys.stderr)
                    finally:
                        processed_files += 1
                        pbar.update(1)
            except concurrent.futures.TimeoutError:
                # No futures completed within the timeout
                pass
        
        try:
            # Walk the directory tree and submit tasks as we find files
            for root, _, files in os.walk(directory):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()
                    if extensions and not any(file.lower().endswith(ext) for ext in extensions):
                        continue
                    
                    path = os.path.join(root, file)
                    total_files += 1
                    
                    # Submit the file for processing
                    future = executor.submit(
                        search_file, 
                        path, 
                        file_ext, 
                        search_terms, 
                        case_sensitive, 
                        True,  # Always use regex 
                        color_output, 
                        context_lines, 
                        ignore_comments, 
                        sanitized_patterns
                    )
                    futures[future] = path
                    
                    # Periodically process completed futures while we continue searching
                    process_completed_futures(block=False)
                    
                    # Update total in progress bar
                    pbar.total = total_files
                    pbar.refresh()
            
            # Process all remaining futures
            process_completed_futures(block=True)
            
        finally:
            pbar.close()
    
    # Report warnings for sanitized patterns
    for term, sanitized_term in sanitized_patterns.items():
        print(f"Warning: Fixed problematic regex pattern: '{term}' → '{sanitized_term}'")
    
    print(f"\nProcessed {processed_files} files, found {len(all_matches)} matches")
    return all_matches


def chat_about_matches(matches, original_prompt):
    client = anthropic.Anthropic()
    
    # Prepare a more structured context for Claude
    context_sections = []
    for i, m in enumerate(matches[:20]):  # Limit to first 20 matches to avoid token limits
        context_sections.append(f"{m['file']}:{m['line']} (Match #{i+1})\nMatched term: {m['term']}\n{m['context']}")
    
    combined_contexts = "\n\n---\n\n".join(context_sections)
    
    system_message = """You are an expert code analyst helping to interpret search results from a codebase.
Focus on explaining:
1. How the found code works
2. Potential security implications or bugs if relevant 
3. Connections between different matches
4. Clear, factual analysis based only on the provided code

When referring to matches, ALWAYS use the file path and line number (e.g., 'In file.cpp:123') rather than match numbers.
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
        
        # Use streaming API
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
        
        print()  # Add a newline after the streamed response
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search code and chat with AI about results.")
    parser.add_argument("directory", help="Directory to search in")
    parser.add_argument("--prompt", help="Natural language prompt to generate search terms")
    parser.add_argument("-e", "--extensions", nargs='+', help="File extensions to include (e.g. .py .js)")
    parser.add_argument("-i", "--insensitive", action="store_true", help="Case-insensitive search")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--no-chat", action="store_true", help="Skip chat mode")
    parser.add_argument("--include-comments", action="store_true", help="Include comments in search results")
    parser.add_argument("--terms", type=int, default=10, help="Number of search terms to generate (default: 10)")
    parser.add_argument("--context", type=int, default=6, help="Lines of context before/after match")
    parser.add_argument("--workers", type=int, help="Number of parallel workers for search (default: 2x CPU cores)")

    args = parser.parse_args()

    if not args.prompt:
        print("You must provide a --prompt to suggest search terms.")
        sys.exit(1)

    print("\nQuerying Claude for search terms...")
    search_terms = get_search_terms_from_prompt(args.prompt, max_terms=args.terms, extensions=args.extensions)
    print("Suggested terms:")
    for t in search_terms:
        print(f"- {t}")

    print("\nSearching code using regex patterns...")
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

    if matches and not args.no_chat:
        chat_about_matches(matches, args.prompt)
