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


def fast_walk(directory, skip_dirs=None):
    """Faster alternative to os.walk using scandir with Windows-specific optimizations"""
    if skip_dirs is None:
        skip_dirs = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', 'build', 'dist'}
    
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
                            
                        # Use stat attributes directly when possible on Windows
                        if hasattr(entry, 'is_dir') and hasattr(entry, 'is_file'):
                            if entry.is_dir(follow_symlinks=False):
                                # Store full paths directly
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
        except PermissionError:
            continue
        except OSError:
            # Handle Windows-specific path errors
            continue
        except Exception as e:
            # Skip any directories we can't access
            continue


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
                color_output=True, context_lines=3, ignore_comments=True, max_workers=None, stop_requested=None):
    """
    Search code in directory for specified terms.
    
    Parameters:
    -----------
    stop_requested : callable, optional
        A callable that returns True if the search should be stopped.
        This allows external code to interrupt the search process.
    """
    if extensions:
        # Convert extensions to a set of lowercase strings for O(1) lookup
        extensions = set(ext.lower() for ext in extensions)
        # Pre-process extensions to ensure they start with a dot
        extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}

    # Track which patterns have already been sanitized to avoid duplicate warnings
    sanitized_patterns = {}
    all_matches = []
    total_files = 0
    processed_files = 0

    # Skip common non-source directories
    skip_dirs = {'.git', 'node_modules', 'venv', '.venv', '__pycache__', 'build', 'dist', 'obj', 'bin'}

    # If no custom max_workers specified, determine an appropriate value
    if max_workers is None:
        max_workers = os.cpu_count() * 2
        max_workers = max(1, max_workers)  # Ensure at least 1 worker
    
    # Windows-specific: Reduce thread count for better performance on Windows
    if sys.platform == 'win32' and max_workers > 8:
        max_workers = 8  # Windows handles fewer threads more efficiently
    
    # Create a ThreadPoolExecutor to search files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a dictionary to track all futures for files being processed
        futures = {}
        
        # Create a progress bar that will be updated as we go
        pbar = tqdm(desc="Searching", unit="files")
        
        # Process completed futures while we continue to find more files
        def process_completed_futures(block=False):
            nonlocal processed_files
            
            # Check if search should be stopped
            if stop_requested and stop_requested():
                # Cancel all pending futures
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
            
            return False
        
        # Batch files by directory for more efficient processing
        current_batch = []
        current_dir = None
        batch_size = 100  # Adjust based on testing
        
        try:
            # Walk the directory tree using fast_walk and submit tasks as we find files
            for root, dirs, files in fast_walk(directory, skip_dirs):
                # Check if search should be stopped
                if stop_requested and stop_requested():
                    break
                
                # Process files in batches by directory to improve locality
                file_batch = []
                
                for file in files:
                    # Check if search should be stopped
                    if stop_requested and stop_requested():
                        break
                        
                    # Get extension - more efficient than splitext for extension checking
                    file_lower = file.lower()
                    dot_index = file_lower.rfind('.')
                    
                    if dot_index != -1:
                        file_ext = file_lower[dot_index:]
                        # If extensions specified, filter by extension
                        if extensions and file_ext not in extensions:
                            continue
                    elif extensions:
                        # Skip files without extensions if extensions specified
                        continue
                    else:
                        file_ext = ''
                    
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
                    # Process more frequently on Windows to avoid memory buildup
                    if len(futures) > batch_size or (sys.platform == 'win32' and len(futures) > 50):
                        if process_completed_futures(block=False):
                            break  # Stop requested
                    
                    # Update total in progress bar
                    pbar.total = total_files
                    pbar.refresh()
                
                # Process completed futures after each directory to maintain responsiveness
                if process_completed_futures(block=False):
                    break  # Stop requested
            
            # Process all remaining futures if not stopped
            if not (stop_requested and stop_requested()):
                process_completed_futures(block=True)
            
        finally:
            pbar.close()
    
    # Report warnings for sanitized patterns
    for term, sanitized_term in sanitized_patterns.items():
        print(f"Warning: Fixed problematic regex pattern: '{term}' → '{sanitized_term}'")
    
    if stop_requested and stop_requested():
        print(f"\nSearch stopped. Processed {processed_files} files, found {len(all_matches)} matches")
    else:
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
