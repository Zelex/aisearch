# AISearch: AI-Powered Code Search Tool

AISearch is a powerful tool that leverages Claude 3.7 Sonnet or o4-mini to generate intelligent search patterns for codebases. It transforms natural language descriptions into effective regex search patterns, searches your code, and allows you to chat with AI about the results.

**[Download Latest Release](https://github.com/Zelex/aisearch/releases/latest)**

![AISearch Screenshot](https://i.imgur.com/hAwBneM.png)
![AISearch Chat Screenshot](https://i.imgur.com/lXH9R5B.png)

## Features

- **Natural language search**: Describe what you're looking for in plain English
- **AI-powered pattern generation**: Claude 3.7 or o4-mini creates optimal regex search patterns
- **Multi-threaded search**: Fast parallel processing to scan codebases
- **Interactive analysis**: Chat with AI about the search results
- **Modern UI**: Choose between a sleek GUI or efficient CLI
- **Code context**: View relevant lines before and after matches
- **Language filtering**: Focus on specific file types
- **Smart caching**: Efficient file list caching for faster repeated searches
- **Comment filtering**: Option to exclude code comments from search
- **Clickable results**: Open matched files directly in your editor
- **Search refinement**: AI-powered search term refinement based on initial results
- **Multiple AI providers**: Choose between Anthropic's Claude or OpenAI's o4-mini

## Supported Languages

AISearch supports a wide range of programming languages including:
- Python (.py)
- JavaScript (.js)
- TypeScript (.ts)
- React JSX (.jsx)
- React TSX (.tsx)
- Java (.java)
- C/C++ (.c, .cpp)
- C# (.cs)
- Go (.go)
- Ruby (.rb)
- PHP (.php)
- Swift (.swift)
- Kotlin (.kt)
- Rust (.rs)

## Quick Start

### Easy Installation

Run the installer script to set up all dependencies:

```bash
python install.py
```

### Running AISearch

Use the launcher script to run either the GUI or CLI version:

```bash
# Launch GUI (default if available)
./run_aisearch.py

# Explicitly launch GUI
./run_aisearch.py --gui

# Launch CLI with arguments
./run_aisearch.py --cli /path/to/code --prompt "your search query"
```

## Versions

### GUI Version

The graphical interface provides an intuitive way to:
- Configure search parameters (case sensitivity, context lines, etc.)
- View search results with syntax highlighting
- Chat with AI about findings
- Save and manage search sessions
- Click on results to open files in your editor
- Refine searches based on initial results
- Filter by file extensions
- Exclude comments from search
- Choose between Claude and o4-mini

To launch the GUI directly:
```bash
python aisearch_gui.py
```

### CLI Version

The command-line interface is perfect for:
- Integration with scripts and automation
- Quick searches from the terminal
- Remote server usage
- Power users who prefer terminal workflows

To use the CLI directly:
```bash
python aisearch.py /path/to/codebase --prompt "your search query"
```

For detailed CLI documentation, see [CLI Documentation](aisearch_readme.md).

## Requirements

- Python 3.7+
- Anthropic API key (for Claude 3.7 Sonnet) or OpenAI API key (for o4-mini)
- Dependencies:
  - anthropic
  - openai
  - tqdm
  - pyside6 (for GUI version)

## License

MIT 

## Caching Features

The tool includes two levels of caching for improved performance:

### File List Cache
- Caches the list of files in a directory to avoid repeated filesystem traversal
- Automatically cleared when directory or file extensions change
- Use `--clear-cache` to manually clear

### File Content Cache
- Caches file contents in memory based on file path and modification time
- Automatically invalidates when files are modified
- Configurable cache size (default: 1000 files)
- Uses FIFO eviction when cache is full

### Cache Management Options

```bash
# Set maximum number of files to cache (default: 1000)
python aisearch.py /path/to/code --cache-size 500 --prompt "find authentication code"

# Clear all caches before searching
python aisearch.py /path/to/code --clear-cache --prompt "find database connections"

# Show cache statistics after search
python aisearch.py /path/to/code --cache-stats --prompt "find error handling"
```

## Usage Examples

```bash
# Basic search with caching
python aisearch.py /path/to/project --prompt "find SQL injection vulnerabilities"

# Search specific file types with custom cache size
python aisearch.py /path/to/project --prompt "authentication logic" -e .py .js --cache-size 2000

# Clear cache and show statistics
python aisearch.py /path/to/project --prompt "error handling" --clear-cache --cache-stats

# Disable chat mode and show cache performance
python aisearch.py /path/to/project --prompt "API endpoints" --no-chat --cache-stats
```

## Performance Benefits

File content caching provides significant performance improvements for:
- **Repeated searches** in the same codebase
- **Iterative refinement** of search patterns
- **Large codebases** where file I/O is a bottleneck
- **Network-mounted** or slow storage systems

Cache hit rates of 50-90% are common when performing multiple searches on the same codebase.

## Environment Variables

Set one of these based on your AI provider:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
``` 