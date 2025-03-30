# AISearch: AI-Powered Code Search Tool

AISearch is a powerful command-line utility that leverages Claude 3.7 Sonnet to generate intelligent search patterns for codebases. It transforms natural language descriptions into effective regex search patterns, searches your code, and allows you to chat with AI about the results.

## Features

- Generate regex search patterns from natural language descriptions using Claude 3.7
- Search across codebases with multi-threaded processing
- Use powerful regex patterns for accurate matches
- Filter by file extensions
- Highlight matches in color
- Interactive chat to analyze search results with Claude
- Configurable context lines, case sensitivity, and more

## Prerequisites

- Python 3.7+
- Anthropic API key

## Installation

1. Clone this repository or download the script
2. Install dependencies:
```bash
pip install anthropic tqdm
```
3. Set your Anthropic API key:
```bash
# On macOS/Linux
export ANTHROPIC_API_KEY=your_api_key_here

# On Windows
set ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Basic usage:

```bash
python aisearch.py /path/to/codebase --prompt "your search query"
```

### Examples

Search for authentication logic in a Python project:
```bash
python aisearch.py ./my_project --prompt "user authentication implementation" --extensions .py
```

Find security vulnerabilities:
```bash
python aisearch.py ./webapp --prompt "potential SQL injection vulnerabilities"
```

Look for memory leaks in C++ code with extended context:
```bash
python aisearch.py ./src --prompt "memory allocation without proper cleanup" --extensions .cpp .h --context 10
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `directory` | Directory to search in |
| `--prompt` | Natural language prompt to generate search terms |
| `-e, --extensions` | File extensions to include (e.g., .py .js) |
| `-i, --insensitive` | Case-insensitive search |
| `--no-color` | Disable colored output |
| `--no-chat` | Skip chat mode |
| `--include-comments` | Include comments in search results |
| `--terms` | Number of search terms to generate (default: 10) |
| `--context` | Lines of context before/after match (default: 6) |
| `--workers` | Number of parallel workers (default: 2x CPU cores) |

## How It Works

1. You provide a natural language prompt about what you're looking for
2. Claude generates relevant regex search patterns
3. The tool searches your codebase with those patterns in parallel
4. Matches are displayed with context
5. You can chat with Claude about the results for deeper analysis

## Tips for Effective Searches

- Be specific in your prompts for better search patterns
- Mention specific programming languages or patterns in your prompt
- Add file extensions to focus on specific languages
- Increase context lines when you need more surrounding code
- Ask specific questions during the chat phase

## GUI Version

A GUI version of this tool is also available. Run `python aisearch_gui.py` to use the graphical interface. 