# AISearch: AI-Powered Code Search Tool

AISearch is a powerful tool that leverages Claude 3.7 Sonnet to generate intelligent search patterns for codebases. It transforms natural language descriptions into effective regex search patterns, searches your code, and allows you to chat with AI about the results.

![AISearch Screenshot](https://i.imgur.com/pLD4o8T.png)

## Features

- **Natural language search**: Describe what you're looking for in plain English
- **AI-powered pattern generation**: Claude 3.7 creates optimal regex search patterns
- **Multi-threaded search**: Fast parallel processing to scan codebases
- **Interactive analysis**: Chat with Claude about the search results
- **Modern UI**: Choose between a sleek GUI or efficient CLI
- **Code context**: View relevant lines before and after matches
- **Language filtering**: Focus on specific file types

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
- Configure search parameters
- View search results
- Chat with Claude about findings
- Save and manage search sessions

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
- Anthropic API key
- Dependencies:
  - anthropic
  - tqdm
  - pyside6 (for GUI version)

## License

MIT 