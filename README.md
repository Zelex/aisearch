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