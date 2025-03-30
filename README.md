# AI Search

AI Search is a command-line utility that uses AI to generate intelligent search patterns for codebases. It leverages Claude from Anthropic to generate contextually relevant regex search patterns from natural language prompts, then searches your codebase with those patterns.

## Features

- Generate regex search patterns from natural language prompts using Claude 3.7
- Search across codebases with parallel processing
- Always uses regex patterns for powerful pattern matching
- Filter by file extensions
- Highlight matches in color
- Interactive chat to analyze search results with Claude
- Configurable context lines, case sensitivity, and more

## Prerequisites

- Python 3.7+
- Anthropic API key

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install anthropic tqdm
```
3. Set your Anthropic API key:
```
export ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Basic usage:

```
python aisearch.py /path/to/codebase --prompt "your search query"
```

### Examples

Search for authentication logic in a Python project:
```
python aisearch.py ./my_project --prompt "user authentication implementation" --extensions .py
```

Find security vulnerabilities:
```
python aisearch.py ./webapp --prompt "potential SQL injection vulnerabilities"
```

Look for memory leaks in C++ code with extended context:
```
python aisearch.py ./src --prompt "memory allocation without proper cleanup" --extensions .cpp .h --context 10
```

Search with word boundaries to find exact variable names:
```
python aisearch.py ./src --prompt "config variables" --word-boundaries
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
| `--word-boundaries` | Match whole words only |

## Workflow

1. You provide a natural language prompt about what you're looking for
2. Claude generates relevant regex search patterns
3. The tool searches your codebase with those patterns
4. Matches are displayed with context
5. You can chat with Claude about the results for deeper analysis

## Tips

- Be specific in your prompts for better search patterns
- Add file extensions to focus on specific languages
- Increase context lines when you need more surrounding code
- Use word boundaries to avoid partial matches 