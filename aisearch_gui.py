import sys
import os
import threading
import atexit
import gc
import time
import re
import subprocess
import platform
import markdown  # Add markdown library
try:
    import pygments # For code syntax highlighting
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, 
                             QCheckBox, QSpinBox, QGroupBox, QSplitter, QComboBox,
                             QListWidget, QProgressBar, QMessageBox, QDialog, QMenu, QTabWidget, QListWidgetItem,
                             QGridLayout, QTextBrowser)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QSettings, QTimer, QMimeData
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon, QTextCharFormat, QSyntaxHighlighter, QClipboard, QAction

import aisearch

class ResultsBuffer:
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()
    
    def add(self, text):
        with self.lock:
            self.buffer.append(text)
    
    def get_and_clear(self):
        with self.lock:
            result = "".join(self.buffer)
            self.buffer = []
            return result

class StreamRedirector:
    def __init__(self, results_buffer):
        self.results_buffer = results_buffer
        self.line_buffer = ""
        
    def write(self, text):
        self.line_buffer += text
        if '\n' in self.line_buffer or len(self.line_buffer) > 80:
            self.results_buffer.add(self.line_buffer)
            self.line_buffer = ""
            
    def flush(self):
        if self.line_buffer:
            self.results_buffer.add(self.line_buffer)
            self.line_buffer = ""

class SearchThread(threading.Thread):
    def __init__(self, parent, directory, prompt, extensions, case_sensitive, 
                color_output, context_lines, ignore_comments, max_terms, max_workers, provider,
                single_line=False):
        super().__init__()
        self.parent = parent
        self.directory = directory
        self.prompt = prompt
        self.extensions = extensions
        self.case_sensitive = case_sensitive
        self.color_output = False  # Always disable color output for GUI
        self.context_lines = context_lines
        self.ignore_comments = ignore_comments
        self.max_terms = max_terms
        self.max_workers = max_workers
        self.search_terms = []
        self.anti_patterns = []
        self.matches = []
        self.running = True
        self.stop_requested = False
        self.provider = provider
        self.single_line = single_line
        
    def run(self):
        try:
            # Get search terms
            if self.stop_requested:
                self.running = False
                return
                
            self.parent.signal_update_status.emit(f"Querying {self.provider.title()} for search terms...")
            
            # Check if this is a refined search (we have previous matches)
            # This will be true if the parent has matches - for refined searches
            if hasattr(self.parent, 'matches') and self.parent.matches and len(self.parent.matches) > 0:
                self.search_terms, self.anti_patterns = aisearch.get_refined_search_terms(
                    self.prompt,
                    self.parent.matches,
                    max_terms=self.max_terms,
                    extensions=self.extensions,
                    context_lines=self.context_lines,
                    provider=self.provider,
                    multiline=not self.single_line
                )
                self.parent.signal_update_status.emit("Generated refined search terms based on current results...")
            else:
                self.search_terms, self.anti_patterns = aisearch.get_search_terms_from_prompt(
                    self.prompt,
                    max_terms=self.max_terms,
                    extensions=self.extensions,
                    provider=self.provider,
                    multiline=not self.single_line
                )
            
            # Ensure search_terms is a list of strings
            if not isinstance(self.search_terms, list):
                self.search_terms = list(self.search_terms) if self.search_terms else []
            self.search_terms = [str(term) for term in self.search_terms]
            
            # Ensure anti_patterns is a list of strings
            if not isinstance(self.anti_patterns, list):
                self.anti_patterns = list(self.anti_patterns) if self.anti_patterns else []
            self.anti_patterns = [str(pattern) for pattern in self.anti_patterns]
            
            # Display terms
            terms_text = "Suggested terms:\n" + "\n".join([f"- {t}" for t in self.search_terms])
            if self.anti_patterns:
                terms_text += "\n\nAnti-patterns (exclusions):\n" + "\n".join([f"- {t}" for t in self.anti_patterns])
            self.parent.signal_update_terms.emit(terms_text)
            
            if self.stop_requested:
                self.running = False
                return
                
            # Search code
            self.parent.signal_update_status.emit("Searching code using regex patterns...")
            
            # Process in batches
            all_matches = []
            
            try:
                all_matches = aisearch.search_code(
                    directory=self.directory,
                    search_terms=self.search_terms,
                    extensions=self.extensions,
                    case_sensitive=self.case_sensitive,
                    color_output=self.color_output,
                    context_lines=self.context_lines,
                    ignore_comments=self.ignore_comments,
                    max_workers=self.max_workers,
                    stop_requested=lambda: self.stop_requested,  # Pass a callable to check if stop was requested
                    multiline=not self.single_line,
                    anti_regex=self.anti_patterns
                )
                
                self.matches = all_matches
            except Exception as e:
                self.parent.signal_error.emit(f"Search error: {str(e)}")
                self.matches = all_matches
            
            self.parent.signal_search_complete.emit(len(self.matches))
        except Exception as e:
            self.parent.signal_error.emit(str(e))
        finally:
            self.running = False
    
    def stop(self):
        self.stop_requested = True

class ChatThread(threading.Thread):
    def __init__(self, parent, matches, prompt, question, chat_history=None):
        super().__init__()
        self.parent = parent
        self.matches = matches
        self.prompt = prompt
        self.question = question
        self.provider = parent.provider_combo.currentText()
        self.chat_history = chat_history or []
        
    def run(self):
        try:
            self.parent.signal_update_status.emit(f"Waiting for {self.provider.title()}'s response...")
            
            # Use the provider-agnostic client
            client = aisearch.get_ai_client(self.provider)
            
            # Prepare context
            context_sections = []
            for i, m in enumerate(self.matches[:20]):
                context_sections.append(f"{m['file']}:{m['line']} (Match #{i+1})\nMatched term: {m['term']}\n{m['context']}")
            
            combined_contexts = "\n\n---\n\n".join(context_sections)
            
            system_message = """You are an expert code analyst helping to interpret search results from a codebase.
            Focus on explaining:
            1. How the found code works
            2. Potential security implications or bugs if relevant 
            3. Connections between different matches
            4. Clear, factual analysis based only on the provided code

            IMPORTANT CONSTRAINTS:
            - DO NOT provide code fixes or solutions
            - DO NOT generate new code
            - DO NOT suggest code changes
            - Only analyze and explain the existing code
            - If you identify issues, explain them but do not propose fixes

            When referring to matches, ALWAYS use the file path and line number (e.g., 'In file.cpp:123') rather than match numbers.
            Keep your responses concise and to the point.
            
            IMPORTANT: Always directly address the user's questions. If they ask a specific question about the code or results,
            make sure to answer that question directly rather than continuing your previous analysis."""
            
            # Store the initial search results as a fixed context that will always be included
            search_context = f"These are code search results for: '{self.prompt}'\n\n{combined_contexts}"
            
            # Keep track of user-assistant exchanges separately from the fixed context
            exchanges = self.chat_history or []
            
            # Add the current question if provided
            if self.question:
                exchanges.append({"role": "user", "content": self.question})
            
            # Construct messages: first the context message, then recent exchanges
            messages = [{"role": "user", "content": search_context}]
            messages.extend(exchanges[-12:] if len(exchanges) > 12 else exchanges)
            
            if self.provider == "anthropic":
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4096,
                    temperature=1,
                    system=system_message,
                    messages=messages
                )
                response_text = response.content[0].text
            elif self.provider == "azure":
                # Get deployment name from parent's settings
                deployment_name = getattr(self.parent, 'azure_deployment', 'gpt-4')
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[{"role": "system", "content": system_message}] + messages,
                    temperature=1,
                    max_tokens=4096
                )
                response_text = response.choices[0].message.content
            else:  # OpenAI
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": system_message}] + messages,
                    temperature=1,
                    max_tokens=4096
                )
                response_text = response.choices[0].message.content
            
            # Save question and response to history
            if self.question:
                new_history = self.chat_history + [
                    {"role": "user", "content": self.question},
                    {"role": "assistant", "content": response_text}
                ]
            else:
                new_history = self.chat_history + [
                    {"role": "assistant", "content": response_text}
                ]
            
            # Use tuple to send both response and history
            self.parent.signal_chat_response.emit((response_text, new_history))
            
        except Exception as e:
            self.parent.signal_error.emit(str(e))

class CodeSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code snippets that adjusts based on file extension"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_language = "none"
        self.highlighting_rules = []
        self.setup_highlighting_rules()
    
    def compile_regex(self, pattern_str):
        """Compile regex pattern using re"""
        return re.compile(pattern_str)
        
    def set_language(self, file_path):
        """Set the language based on file extension"""
        if not file_path:
            return
            
        # Extract extension
        _, ext = os.path.splitext(file_path)
        if ext:
            ext = ext[1:].lower()  # Remove the dot and convert to lowercase
            
            # Map common extensions to language types
            ext_to_lang = {
                # Python
                "py": "Python", "pyw": "Python", "pyx": "Python", "pxd": "Python", "pxi": "Python",
                # JavaScript
                "js": "JavaScript", "jsx": "JavaScript", "ts": "TypeScript", "tsx": "TypeScript",
                # HTML/XML
                "html": "HTML", "htm": "HTML", "xhtml": "HTML", "xml": "XML",
                # CSS
                "css": "CSS", "scss": "CSS", "sass": "CSS",
                # C/C++
                "c": "C", "cpp": "C++", "cxx": "C++", "h": "C", "hpp": "C++",
                # Java
                "java": "Java",
                # JSON
                "json": "JSON",
                # Go
                "go": "Go",
                # Rust
                "rs": "Rust",
                # Ruby
                "rb": "Ruby",
                # PHP
                "php": "PHP",
                # Swift
                "swift": "Swift",
                # C#
                "cs": "C#",
                # Shell
                "sh": "Shell", "bash": "Shell", "zsh": "Shell",
            }
            
            if ext in ext_to_lang:
                if self.current_language != ext_to_lang[ext]:
                    self.current_language = ext_to_lang[ext]
                    self.setup_highlighting_rules()
                    return True
        
        return False
    
    def setup_highlighting_rules(self):
        """Set up highlighting rules based on current language"""
        self.highlighting_rules = []
        
        # Comment format (common for most languages)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))  # Green
        
        # String format (common for most languages)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))  # Orange/brown
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))  # Blue
        keyword_format.setFontWeight(QFont.Bold)
        
        # Function call format
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))  # Yellow
        
        # Number format
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))  # Light green
        
        # Basic number regex (works for most languages)
        rule = {"pattern": self.compile_regex(r"\b[0-9]+(\.[0-9]+)?\b"), "format": number_format}
        self.highlighting_rules.append(rule)
        
        # Function calls (works for most languages)
        rule = {"pattern": self.compile_regex(r"\b[A-Za-z0-9_]+(?=\s*\()"), "format": function_format}
        self.highlighting_rules.append(rule)
        
        # Language-specific rules
        if self.current_language == "Python":
            # Python keywords
            keywords = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", 
                       "else", "except", "False", "finally", "for", "from", "global", "if", "import", 
                       "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", 
                       "return", "True", "try", "while", "with", "yield"]
            
            keyword_pattern = r"\b(" + "|".join(keywords) + r")\b"
            rule = {"pattern": self.compile_regex(keyword_pattern), "format": keyword_format}
            self.highlighting_rules.append(rule)
            
            # Python docstrings
            rule = {"pattern": self.compile_regex(r'""".*?"""'), "format": comment_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"'''.*?'''"), "format": comment_format}
            self.highlighting_rules.append(rule)
            
            # Python strings
            rule = {"pattern": self.compile_regex(r'"[^"\\]*(\\.[^"\\]*)*"'), "format": string_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"'[^'\\]*(\\.[^'\\]*)*'"), "format": string_format}
            self.highlighting_rules.append(rule)
            
            # Python comments
            rule = {"pattern": self.compile_regex(r"#[^\n]*"), "format": comment_format}
            self.highlighting_rules.append(rule)
            
        elif self.current_language in ["JavaScript", "TypeScript", "React JSX", "React TSX"]:
            # JS/TS keywords
            keywords = ["break", "case", "catch", "class", "const", "continue", "debugger", 
                       "default", "delete", "do", "else", "export", "extends", "false", 
                       "finally", "for", "function", "if", "import", "in", "instanceof", 
                       "new", "null", "return", "super", "switch", "this", "throw", 
                       "true", "try", "typeof", "var", "void", "while", "with", "yield", 
                       "let", "static", "enum", "await", "async", "implements", "interface", 
                       "package", "private", "protected", "public", "as", "readonly"]
            
            keyword_pattern = r"\b(" + "|".join(keywords) + r")\b"
            rule = {"pattern": self.compile_regex(keyword_pattern), "format": keyword_format}
            self.highlighting_rules.append(rule)
            
            # JS/TS strings
            rule = {"pattern": self.compile_regex(r'"[^"\\]*(\\.[^"\\]*)*"'), "format": string_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"'[^'\\]*(\\.[^'\\]*)*'"), "format": string_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"`[^`\\]*(\\.[^`\\]*)*`"), "format": string_format}
            self.highlighting_rules.append(rule)
            
            # JS/TS comments
            rule = {"pattern": self.compile_regex(r"//[^\n]*"), "format": comment_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"/\*.*?\*/"), "format": comment_format}
            self.highlighting_rules.append(rule)
            
        elif self.current_language in ["Java", "C", "C++", "C#"]:
            # Common C-like languages keywords
            keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", 
                       "class", "const", "continue", "default", "do", "double", "else", "enum", 
                       "extends", "final", "finally", "float", "for", "goto", "if", "implements", 
                       "import", "instanceof", "int", "interface", "long", "native", "new", "package", 
                       "private", "protected", "public", "return", "short", "static", "strictfp", 
                       "super", "switch", "synchronized", "this", "throw", "throws", "transient", 
                       "try", "void", "volatile", "while", "true", "false", "null"]
            
            keyword_pattern = r"\b(" + "|".join(keywords) + r")\b"
            rule = {"pattern": self.compile_regex(keyword_pattern), "format": keyword_format}
            self.highlighting_rules.append(rule)
            
            # C-like strings
            rule = {"pattern": self.compile_regex(r'"[^"\\]*(\\.[^"\\]*)*"'), "format": string_format}
            self.highlighting_rules.append(rule)
            
            # C-like comments
            rule = {"pattern": self.compile_regex(r"//[^\n]*"), "format": comment_format}
            self.highlighting_rules.append(rule)
            rule = {"pattern": self.compile_regex(r"/\*.*?\*/"), "format": comment_format}
            self.highlighting_rules.append(rule)
        
        # Add more languages as needed
        # The basic patterns (numbers, functions) are already added for all languages

    def highlightBlock(self, text):
        """Apply syntax highlighting to text block"""
        # Detect language from the file extension in the text
        # Look for file paths in the text block (typical format in search results)
        file_match = re.search(r'([^:]+\.[a-zA-Z0-9]+):', text)
        if file_match:
            file_path = file_match.group(1)
            self.set_language(file_path)
            
        # Apply all rules
        for rule in self.highlighting_rules:
            matches = rule["pattern"].finditer(text)
            for match in matches:
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, rule["format"])

class ClickableTextEdit(QTextEdit):
    """Custom QTextEdit that can detect clicks on file references"""
    def __init__(self, parent=None, use_syntax_highlighter=True):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.setCursorWidth(1)
        # Use a simpler pattern to match line numbers and delegate more complex path validation to a method
        self.file_match_pattern = re.compile(r'([^:]+):(\d+)')
        # Remove text selection flags to prevent automatic selection
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse)
        self.setStatusTip("Click on file:line references to open in your default editor")
        
        # Add syntax highlighter only if requested
        if use_syntax_highlighter:
            self.highlighter = CodeSyntaxHighlighter(self.document())
        else:
            self.highlighter = None
        
        # Set a stylesheet for better markdown rendering
        self.document().setDefaultStyleSheet("""
            code { background-color: #2d2d2d; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
            pre { background-color: #2d2d2d; padding: 10px; border-radius: 5px; overflow-x: auto; }
            blockquote { border-left: 3px solid #555; margin-left: 0; padding-left: 10px; color: #aaa; }
            h1, h2, h3, h4 { color: #ddd; }
            a { color: #3a92ea; text-decoration: none; }
            table { border-collapse: collapse; }
            th, td { border: 1px solid #555; padding: 6px; }
            th { background-color: #2d2d2d; }
            .file-reference { color: #3a92ea; text-decoration: underline; cursor: pointer; }
        """)
        
        # Store original markdown content for copying
        self.markdown_content = ""

    
    @staticmethod
    def is_valid_file_path(path):
        """Validate if a path looks like a legitimate file path across platforms"""
        # Windows extended path (\\?\) validation - must have valid structure
        if path.startswith('\\\\?\\'):
            # Check for drive letter (e.g., C:)
            if len(path) > 4 and path[4].isalpha() and path[5] == ':':
                pass  # Valid Windows drive path
            # Check for UNC path
            elif len(path) > 5 and path[4:8] == 'UNC\\':
                pass  # Valid UNC path
            else:
                return False
        # Either must start with a path separator or have a drive letter
        elif not (path.startswith('/') or path.startswith('\\') or
                 path.startswith('./') or path.startswith('../') or
                 re.match(r'[A-Za-z]:', path)):
            return False
        # Path must have at least one path separator
        if '/' not in path and '\\' not in path:
            return False
        return True
    
    def find_main_window(self):
        """Find the main window by traversing up the parent hierarchy"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AISearchGUI):
                return parent
            parent = parent.parent()
        return None
        
    def mouseMoveEvent(self, event):
        """Handle mouse movement to highlight file references"""
        cursor = self.cursorForPosition(event.pos())
        
        # Get the text cursor at the hover position and the line it's on
        hover_cursor = QTextCursor(cursor)
        click_pos = hover_cursor.position()
        
        # Get the whole line
        line_cursor = QTextCursor(hover_cursor)
        line_cursor.select(QTextCursor.LineUnderCursor)
        line_text = line_cursor.selectedText()
        
        # Get start position of the line
        line_cursor.movePosition(QTextCursor.StartOfLine)
        line_start_pos = line_cursor.position()
        
        # Calculate relative position within the line
        relative_pos = click_pos - line_start_pos
        
        # Look for file:line patterns in the line
        for match in self.file_match_pattern.finditer(line_text):
            # Check if the hover is within the file:line pattern
            match_start = match.start()
            match_end = match.end()
            
            if match_start <= relative_pos <= match_end and self.is_valid_file_path(match.group(1)):
                # We're hovering over a file reference
                self.viewport().setCursor(Qt.PointingHandCursor)
                return
        
        # We're not hovering over a file reference
        self.viewport().setCursor(Qt.IBeamCursor)
        super().mouseMoveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press to detect file references"""
        if event.button() == Qt.LeftButton:
            # Get the text cursor at the click position and the line it's on
            click_cursor = self.cursorForPosition(event.pos())
            
            # Store the absolute cursor position
            click_pos = click_cursor.position()
            
            # Get the whole line
            line_cursor = QTextCursor(click_cursor)
            line_cursor.select(QTextCursor.LineUnderCursor)
            line_text = line_cursor.selectedText()
            
            # Skip empty lines
            if not line_text.strip():
                super().mousePressEvent(event)
                return
                
            # Get start position of the line
            line_cursor.movePosition(QTextCursor.StartOfLine)
            line_start_pos = line_cursor.position()
            
            # Calculate relative position within the line
            relative_pos = click_pos - line_start_pos
            
            # First check for HTML links
            if self.textCursor().charFormat().isAnchor():
                anchor_href = self.textCursor().charFormat().anchorHref()
                if anchor_href:
                    # Process as file reference if it matches our pattern
                    match = self.file_match_pattern.search(anchor_href)
                    if match and self.is_valid_file_path(match.group(1)):
                        file_path = match.group(1)
                        line_number = match.group(2)
                        main_window = self.find_main_window()
                        if main_window:
                            main_window.open_file_in_editor(file_path, line_number)
                            event.accept()
                            return
            
            # Flag to track if we clicked on a file reference
            clicked_file_ref = False
            
            # Look for file:line patterns in the line
            for match in self.file_match_pattern.finditer(line_text):
                # Check if the click is within the file:line pattern
                match_start = match.start()
                match_end = match.end()
                
                if match_start <= relative_pos <= match_end:
                    # Click is within the file:line pattern
                    file_path = match.group(1)
                    line_number = match.group(2)
                    
                    if self.is_valid_file_path(file_path):
                        # Find the main window
                        main_window = self.find_main_window()
                        if main_window is None:
                            return
                        
                        # If it's a relative path, make it absolute
                        if not os.path.isabs(file_path):
                            # Use the current search directory as the base
                            base_dir = main_window.dir_input.text()
                            # Normalize path separators
                            file_path = file_path.replace('\\', '/')
                            base_dir = base_dir.replace('\\', '/')
                            # Remove any leading slashes from file_path
                            file_path = file_path.lstrip('/')
                            # Join paths
                            file_path = os.path.join(base_dir, file_path)
                            # Normalize the final path
                            file_path = os.path.normpath(file_path)
                        else:
                            # For absolute paths, just normalize the separators
                            file_path = os.path.normpath(file_path)
                        
                        if os.path.exists(file_path):
                            main_window.open_file_in_editor(file_path, line_number)
                            clicked_file_ref = True
                            event.accept()
                            return
            
            # If we didn't click on a file reference, check if the line has a file reference
            # and show context popup
            if not clicked_file_ref:
                found_match = None
                file_line_found = None
                
                # Find any file reference in the line
                for match in self.file_match_pattern.finditer(line_text):
                    file_path = match.group(1)
                    line_number = match.group(2)
                    
                    if self.is_valid_file_path(file_path):
                        file_line_found = (file_path, line_number)
                        # Found a valid file reference in this line, now look for its context
                        main_window = self.find_main_window()
                        if main_window and hasattr(main_window, 'matches') and main_window.matches:
                            # Try to find the file in matches
                            found_match = self.find_matching_context(file_path, line_number, main_window.matches)
                            if found_match:
                                break
                
                if found_match:
                    # Display context popup
                    self.show_context_popup(event.globalPos(), found_match)
                    event.accept()
                    return
                elif file_line_found:
                    # We found a file reference but couldn't find its context,
                    # try to load the file directly
                    file_path, line_number = file_line_found
                    
                    # Normalize path
                    main_window = self.find_main_window()
                    if main_window:
                        if not os.path.isabs(file_path):
                            # Use the current search directory as the base
                            base_dir = main_window.dir_input.text()
                            file_path = file_path.replace('\\', '/')
                            base_dir = base_dir.replace('\\', '/')
                            file_path = file_path.lstrip('/')
                            file_path = os.path.join(base_dir, file_path)
                            file_path = os.path.normpath(file_path)
                        else:
                            file_path = os.path.normpath(file_path)
                            
                        if os.path.exists(file_path):
                            # Try to read a few lines around the target line
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    all_lines = f.readlines()
                                
                                line_idx = int(line_number) - 1
                                if 0 <= line_idx < len(all_lines):
                                    # Create a synthetic match
                                    context_lines = 3  # Get 3 lines of context
                                    start_line = max(0, line_idx - context_lines)
                                    end_line = min(len(all_lines), line_idx + context_lines + 1)
                                    
                                    context = ''.join(all_lines[start_line:end_line])
                                    
                                    match_info = {
                                        'file': file_path,
                                        'line': line_number,
                                        'term': "Direct file access",
                                        'context': context
                                    }
                                    
                                    self.show_context_popup(event.globalPos(), match_info)
                                    event.accept()
                                    return
                            except Exception as e:
                                print(f"Error reading file directly: {e}")
        
        # If we get here, either it's not a left click or no file pattern was found
        # Let the parent handle the event for normal text selection
        super().mousePressEvent(event)
    
    def find_matching_context(self, file_path, line_number, matches):
        """Find matching context for a file path and line number"""
        # Normalize paths for comparison
        norm_file_path = os.path.normpath(file_path)
        norm_file_path_lower = norm_file_path.lower() 
        basename = os.path.basename(norm_file_path)
        basename_lower = basename.lower()
        
        # Convert line number to string for comparison
        line_str = str(line_number)
        
        # Try different matching strategies
        for match in matches:
            match_file = os.path.normpath(match['file'])
            match_file_lower = match_file.lower()
            match_basename = os.path.basename(match_file)
            match_basename_lower = match_basename.lower()
            
            # Try different matching strategies with decreasing specificity
            if (match_file == norm_file_path or 
                match_file_lower == norm_file_path_lower or
                # Try relaxed matching using path endings
                match_file.endswith(norm_file_path) or
                match_file_lower.endswith(norm_file_path_lower) or
                # Try basename matching
                match_basename == basename or
                match_basename_lower == basename_lower):
                
                # Check line number
                if str(match['line']) == line_str:
                    return match
        
        # No match found
        return None

    def show_context_popup(self, position, match_info):
        """Show a popup with context information from the match"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextBrowser
        
        # Create a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Code Context")
        dialog.setFixedSize(600, 400)  # Make it a bit larger
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add file info
        file_label = QLabel(f"<b>{match_info['file']}:{match_info['line']}</b>")
        if 'term' in match_info:
            file_label.setText(f"{file_label.text()} (matched: {match_info['term']})")
        file_label.setTextFormat(Qt.RichText)
        file_label.setWordWrap(True)
        layout.addWidget(file_label)
        
        # Add context
        context_browser = QTextBrowser()
        context_browser.setFont(QFont("Menlo, Monaco, Courier New", 10))
        context_browser.setLineWrapMode(QTextBrowser.NoWrap)  # Prevent line wrapping
        
        # Apply syntax highlighting to the context
        if match_info.get('context'):
            context_text = match_info['context']
            
            # Apply some basic syntax highlighting with HTML
            if PYGMENTS_AVAILABLE:
                from pygments import highlight
                from pygments.lexers import guess_lexer_for_filename, TextLexer
                from pygments.formatters import HtmlFormatter
                
                try:
                    # Try to get a lexer based on the file extension
                    try:
                        lexer = guess_lexer_for_filename(match_info['file'], context_text)
                    except:
                        # Fall back to a basic lexer
                        lexer = TextLexer()
                        
                    formatter = HtmlFormatter(style='monokai', linenos=False)
                    highlighted_html = highlight(context_text, lexer, formatter)
                    
                    # Add pygments CSS
                    css = formatter.get_style_defs('.highlight')
                    context_browser.document().setDefaultStyleSheet(css)
                    
                    context_browser.setHtml(highlighted_html)
                except Exception as e:
                    print(f"Syntax highlighting error: {e}")
                    # Fall back to plain text if highlighting fails
                    context_browser.setPlainText(context_text)
            else:
                context_browser.setPlainText(context_text)
        
        layout.addWidget(context_browser, 1)  # Give more space to the context
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        # Open file button
        open_btn = QPushButton("Open in Editor")
        open_btn.clicked.connect(lambda: self.open_file_from_popup(match_info['file'], match_info['line'], dialog))
        button_layout.addWidget(open_btn)
        
        # Add spacing
        button_layout.addStretch(1)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        # Position dialog relative to the main window rather than cursor
        # to ensure it's always visible
        main_window = self.find_main_window()
        if main_window:
            geom = main_window.geometry()
            dialog_x = geom.x() + (geom.width() - dialog.width()) / 2
            dialog_y = geom.y() + (geom.height() - dialog.height()) / 2
            dialog.move(int(dialog_x), int(dialog_y))
        else:
            # Fall back to positioning near cursor
            dialog.move(position.x(), position.y())
            
        # Show dialog
        dialog.show()
    
    def open_file_from_popup(self, file_path, line_number, dialog=None):
        """Open a file at the specified line from a popup dialog"""
        main_window = self.find_main_window()
        if main_window:
            # Normalize path if needed
            if not os.path.isabs(file_path):
                base_dir = main_window.dir_input.text()
                file_path = os.path.normpath(os.path.join(base_dir, file_path))
            
            if os.path.exists(file_path):
                main_window.open_file_in_editor(file_path, line_number)
                
                # Close the dialog if provided
                if dialog:
                    dialog.close()
    
    def contextMenuEvent(self, event):
        """Custom context menu with markdown copy option"""
        menu = QMenu(self)
        
        # Add standard actions
        menu.addAction(self.action(QTextEdit.StandardAction.Cut))
        menu.addAction(self.action(QTextEdit.StandardAction.Copy))
        menu.addAction(self.action(QTextEdit.StandardAction.Paste))
        menu.addSeparator()
        
        # Add custom actions for markdown
        if hasattr(self, 'markdown_content') and self.markdown_content:
            copy_markdown_action = QAction("Copy as Markdown", self)
            copy_markdown_action.triggered.connect(self.copy_markdown_to_clipboard)
            menu.addAction(copy_markdown_action)
            
        select_all_action = self.action(QTextEdit.StandardAction.SelectAll)
        menu.addAction(select_all_action)
        
        menu.exec(event.globalPos())
        
    def copy_markdown_to_clipboard(self):
        """Copy the original markdown content to clipboard"""
        if hasattr(self, 'markdown_content') and self.markdown_content:
            clipboard = QApplication.clipboard()
            mime_data = QMimeData()
            mime_data.setText(self.markdown_content)
            clipboard.setMimeData(mime_data)
            
            # Find main window to show status message
            main_window = self.find_main_window()
            if main_window:
                main_window.statusBar().showMessage("Markdown copied to clipboard", 3000)

        self.anchorClicked.connect(self.handle_link_clicked)
        
        # Set a stylesheet for better markdown rendering
        self.document().setDefaultStyleSheet("""
            code { background-color: #2d2d2d; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
            pre { background-color: #2d2d2d; padding: 10px; border-radius: 5px; overflow-x: auto; }
            blockquote { border-left: 3px solid #555; margin-left: 0; padding-left: 10px; color: #aaa; }
            h1, h2, h3, h4 { color: #ddd; }
            a { color: #3a92ea; text-decoration: none; }
            table { border-collapse: collapse; }
            th, td { border: 1px solid #555; padding: 6px; }
            th { background-color: #2d2d2d; }
            .file-reference { color: #3a92ea; text-decoration: underline; cursor: pointer; }
        """)
    
    @staticmethod
    def is_valid_file_path(path):
        """Validate if a path looks like a legitimate file path across platforms"""
        # Windows extended path (\\?\) validation - must have valid structure
        if path.startswith('\\\\?\\'):
            # Check for drive letter (e.g., C:)
            if len(path) > 4 and path[4].isalpha() and path[5] == ':':
                pass  # Valid Windows drive path
            # Check for UNC path
            elif len(path) > 5 and path[4:8] == 'UNC\\':
                pass  # Valid UNC path
            else:
                return False
        # Either must start with a path separator or have a drive letter
        elif not (path.startswith('/') or path.startswith('\\') or
                 path.startswith('./') or path.startswith('../') or
                 re.match(r'[A-Za-z]:', path)):
            return False
        # Path must have at least one path separator
        if '/' not in path and '\\' not in path:
            return False
        return True
    
    def find_main_window(self):
        """Find the main window by traversing up the parent hierarchy"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AISearchGUI):
                return parent
            parent = parent.parent()
        return None
    
    def handle_link_clicked(self, url):
        """Handle clicked links, especially file references"""
        url_str = url.toString()
        match = self.file_match_pattern.search(url_str)
        
        if match:
            file_path = match.group(1)
            line_number = match.group(2)
            
            if self.is_valid_file_path(file_path):
                # Find the main window
                main_window = self.find_main_window()
                if main_window is None:
                    return
                
                # If it's a relative path, make it absolute
                if not os.path.isabs(file_path):
                    # Use the current search directory as the base
                    base_dir = main_window.dir_input.text()
                    # Normalize path separators
                    file_path = file_path.replace('\\', '/')
                    base_dir = base_dir.replace('\\', '/')
                    # Remove any leading slashes from file_path
                    file_path = file_path.lstrip('/')
                    # Join paths
                    file_path = os.path.join(base_dir, file_path)
                    # Normalize the final path
                    file_path = os.path.normpath(file_path)
                else:
                    # For absolute paths, just normalize the separators
                    file_path = os.path.normpath(file_path)
                
                if os.path.exists(file_path):
                    main_window.open_file_in_editor(file_path, line_number)
    
    def contextMenuEvent(self, event):
        """Custom context menu with markdown copy option"""
        menu = QMenu(self)
        
        # Add standard actions
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy)
        menu.addAction(copy_action)
        
        # Add custom actions for markdown
        if hasattr(self, 'markdown_content') and self.markdown_content:
            copy_markdown_action = QAction("Copy as Markdown", self)
            copy_markdown_action.triggered.connect(self.copy_markdown_to_clipboard)
            menu.addAction(copy_markdown_action)
        
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.selectAll)
        menu.addAction(select_all_action)
        
        menu.exec(event.globalPos())
    
    def copy_markdown_to_clipboard(self):
        """Copy the original markdown content to clipboard"""
        if hasattr(self, 'markdown_content') and self.markdown_content:
            clipboard = QApplication.clipboard()
            mime_data = QMimeData()
            mime_data.setText(self.markdown_content)
            clipboard.setMimeData(mime_data)
            
            # Find main window to show status message
            main_window = self.find_main_window()
            if main_window:
                main_window.statusBar().showMessage("Markdown copied to clipboard", 3000)

class FileClickableBrowser(QTextBrowser):
    """A QTextBrowser that can process clicks on file:line references"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)  # Don't open links automatically
        self.setOpenExternalLinks(False)
        self.file_match_pattern = re.compile(r'([^:]+):(\d+)')
        self.markdown_content = ""  # For storing original markdown
        
        # Connect anchor clicked signal
        self.anchorClicked.connect(self.handle_link_clicked)
        
        # Set a stylesheet for better markdown rendering
        self.document().setDefaultStyleSheet("""
            code { background-color: #2d2d2d; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
            pre { background-color: #2d2d2d; padding: 10px; border-radius: 5px; overflow-x: auto; }
            blockquote { border-left: 3px solid #555; margin-left: 0; padding-left: 10px; color: #aaa; }
            h1, h2, h3, h4 { color: #ddd; }
            a { color: #3a92ea; text-decoration: none; }
            table { border-collapse: collapse; }
            th, td { border: 1px solid #555; padding: 6px; }
            th { background-color: #2d2d2d; }
            .file-reference { color: #3a92ea; text-decoration: underline; cursor: pointer; }
        """)
    
    @staticmethod
    def is_valid_file_path(path):
        """Validate if a path looks like a legitimate file path across platforms"""
        # Windows extended path (\\?\) validation - must have valid structure
        if path.startswith('\\\\?\\'):
            # Check for drive letter (e.g., C:)
            if len(path) > 4 and path[4].isalpha() and path[5] == ':':
                pass  # Valid Windows drive path
            # Check for UNC path
            elif len(path) > 5 and path[4:8] == 'UNC\\':
                pass  # Valid UNC path
            else:
                return False
        # Either must start with a path separator or have a drive letter
        elif not (path.startswith('/') or path.startswith('\\') or
                 path.startswith('./') or path.startswith('../') or
                 re.match(r'[A-Za-z]:', path)):
            return False
        # Path must have at least one path separator
        if '/' not in path and '\\' not in path:
            return False
        return True
    
    def find_main_window(self):
        """Find the main window by traversing up the parent hierarchy"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, AISearchGUI):
                return parent
            parent = parent.parent()
        return None
    
    def handle_link_clicked(self, url):
        """Handle clicked links, especially file references"""
        url_str = url.toString()
        match = self.file_match_pattern.search(url_str)
        
        if match:
            file_path = match.group(1)
            line_number = match.group(2)
            
            if self.is_valid_file_path(file_path):
                # Find the main window
                main_window = self.find_main_window()
                if main_window is None:
                    return
                
                # If it's a relative path, make it absolute
                if not os.path.isabs(file_path):
                    # Use the current search directory as the base
                    base_dir = main_window.dir_input.text()
                    # Normalize path separators
                    file_path = file_path.replace('\\', '/')
                    base_dir = base_dir.replace('\\', '/')
                    # Remove any leading slashes from file_path
                    file_path = file_path.lstrip('/')
                    # Join paths
                    file_path = os.path.join(base_dir, file_path)
                    # Normalize the final path
                    file_path = os.path.normpath(file_path)
                else:
                    # For absolute paths, just normalize the separators
                    file_path = os.path.normpath(file_path)
                
                if os.path.exists(file_path):
                    main_window.open_file_in_editor(file_path, line_number)
    
    def contextMenuEvent(self, event):
        """Custom context menu with markdown copy option"""
        menu = QMenu(self)
        
        # Add standard actions
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy)
        menu.addAction(copy_action)
        
        # Add custom actions for markdown
        if hasattr(self, 'markdown_content') and self.markdown_content:
            copy_markdown_action = QAction("Copy as Markdown", self)
            copy_markdown_action.triggered.connect(self.copy_markdown_to_clipboard)
            menu.addAction(copy_markdown_action)
        
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.selectAll)
        menu.addAction(select_all_action)
        
        menu.exec(event.globalPos())
    
    def copy_markdown_to_clipboard(self):
        """Copy the original markdown content to clipboard"""
        if hasattr(self, 'markdown_content') and self.markdown_content:
            clipboard = QApplication.clipboard()
            mime_data = QMimeData()
            mime_data.setText(self.markdown_content)
            clipboard.setMimeData(mime_data)
            
            # Find main window to show status message
            main_window = self.find_main_window()
            if main_window:
                main_window.statusBar().showMessage("Markdown copied to clipboard", 3000)

class AISearchGUI(QMainWindow):
    signal_update_status = Signal(str)
    signal_update_terms = Signal(str)
    signal_search_complete = Signal(int)
    signal_error = Signal(str)
    signal_chat_response = Signal(object)  # Changed to object to support tuple
    
    def __init__(self):
        super().__init__()
        self.matches = []
        self.search_thread = None
        self.chat_thread = None
        self.chat_history = []  # Add chat history
        self.settings = QSettings("AICodeSearch", "AISearchGUI")
        self.results_buffer = ResultsBuffer()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_results_buffer)
        self.setupUI()
        self.loadSettings()
        
        # Register cleanup handlers
        atexit.register(self.cleanup_resources)
        
    def setupUI(self):
        self.setWindowTitle("AI Code Search")
        self.setMinimumSize(1200, 800)  # Larger default size for more content
        
        # Create central widget with main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ----- Left Sidebar -----
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Directory selection
        dir_group = QGroupBox("Project")
        dir_group.setObjectName("sidebar-group")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_header = QHBoxLayout()
        dir_header.addWidget(QLabel("Directory:"))
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_directory)
        dir_header.addWidget(browse_btn)
        dir_layout.addLayout(dir_header)
        
        self.dir_input = QLineEdit()
        dir_layout.addWidget(self.dir_input)
        
        # Extensions input
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("File Extensions:"))
        dir_layout.addLayout(ext_layout)
        
        self.ext_input = QLineEdit()
        self.ext_input.setPlaceholderText(".py .js .ts (leave empty for all)")
        placeholder_palette = self.ext_input.palette()
        placeholder_palette.setColor(placeholder_palette.ColorRole.PlaceholderText, QColor("#cccccc"))
        self.ext_input.setPalette(placeholder_palette)
        dir_layout.addWidget(self.ext_input)
        
        # File cache action
        clear_cache_btn = QPushButton("Clear File Cache")
        clear_cache_btn.clicked.connect(self.clear_file_cache)
        dir_layout.addWidget(clear_cache_btn)
        
        sidebar_layout.addWidget(dir_group)
        
        # Search options
        search_group = QGroupBox("Search Options")
        search_group.setObjectName("sidebar-group")
        search_layout = QVBoxLayout(search_group)
        
        # Case sensitivity and comments
        options_layout = QVBoxLayout()
        self.case_sensitive = QCheckBox("Case Sensitive")
        self.case_sensitive.setChecked(True)
        options_layout.addWidget(self.case_sensitive)
        
        self.include_comments = QCheckBox("Include Comments")
        self.include_comments.setChecked(True)
        options_layout.addWidget(self.include_comments)
        
        # Add Single Line Mode checkbox
        self.single_line_mode = QCheckBox("Single Line Mode")
        self.single_line_mode.setChecked(False)
        self.single_line_mode.setToolTip("When checked, searches line-by-line instead of supporting multi-line patterns")
        options_layout.addWidget(self.single_line_mode)
        
        search_layout.addLayout(options_layout)
        
        # Terms and workers
        options_grid = QGridLayout()
        options_grid.addWidget(QLabel("Max Terms:"), 0, 0)
        self.max_terms = QSpinBox()
        self.max_terms.setRange(1, 100)
        self.max_terms.setValue(8)
        options_grid.addWidget(self.max_terms, 0, 1)
        
        options_grid.addWidget(QLabel("Workers:"), 1, 0)
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 64)
        self.max_workers.setValue(4)
        options_grid.addWidget(self.max_workers, 1, 1)
        
        options_grid.addWidget(QLabel("Context Lines:"), 2, 0)
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 20)
        self.context_lines.setValue(2)
        options_grid.addWidget(self.context_lines, 2, 1)
        
        search_layout.addLayout(options_grid)
        sidebar_layout.addWidget(search_group)
        
        # AI provider
        ai_group = QGroupBox("AI Provider")
        ai_group.setObjectName("sidebar-group")
        ai_layout = QVBoxLayout(ai_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["anthropic", "openai", "azure"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        ai_layout.addWidget(self.provider_combo)
        
        api_buttons = QHBoxLayout()
        anthropic_key_btn = QPushButton("Anthropic Key")
        anthropic_key_btn.clicked.connect(lambda: self.show_api_key_dialog("anthropic"))
        api_buttons.addWidget(anthropic_key_btn)
        
        openai_key_btn = QPushButton("OpenAI Key")
        openai_key_btn.clicked.connect(lambda: self.show_api_key_dialog("openai"))
        api_buttons.addWidget(openai_key_btn)
        
        azure_key_btn = QPushButton("Azure Key")
        azure_key_btn.clicked.connect(lambda: self.show_api_key_dialog("azure"))
        api_buttons.addWidget(azure_key_btn)
        
        ai_layout.addLayout(api_buttons)
        sidebar_layout.addWidget(ai_group)
        
        # Actions section at bottom of sidebar
        action_group = QGroupBox("Actions")
        action_group.setObjectName("sidebar-group")
        action_layout = QVBoxLayout(action_group)
        
        # Prompt input
        action_layout.addWidget(QLabel("Search Prompt:"))
        self.prompt_input = QLineEdit()
        action_layout.addWidget(self.prompt_input)
        
        # Search buttons
        search_buttons = QHBoxLayout()
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.start_search)
        search_buttons.addWidget(self.search_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_search)
        self.stop_button.setEnabled(False)
        search_buttons.addWidget(self.stop_button)
        
        self.refine_button = QPushButton("Refine")
        self.refine_button.clicked.connect(self.refine_search)
        self.refine_button.setEnabled(False)
        search_buttons.addWidget(self.refine_button)
        
        action_layout.addLayout(search_buttons)
        
        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self.clear_results)
        action_layout.addWidget(clear_button)
        
        sidebar_layout.addWidget(action_group)
        
        # Add stretcher to push everything up
        sidebar_layout.addStretch(1)
        
        main_layout.addWidget(sidebar_widget)
        
        # ----- Main Content Area with Tabs -----
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # ----- Search Results Tab -----
        results_tab = QWidget()
        results_layout = QHBoxLayout(results_tab)
        results_layout.setContentsMargins(10, 10, 10, 10)
        
        # File tree view (left side of results)
        file_tree_widget = QWidget()
        file_tree_widget.setFixedWidth(250)
        file_tree_layout = QVBoxLayout(file_tree_widget)
        file_tree_layout.setContentsMargins(0, 0, 0, 0)
        
        file_tree_header = QHBoxLayout()
        file_tree_header.addWidget(QLabel("Matched Files"))
        file_tree_layout.addLayout(file_tree_header)
        
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(False)  # Disable alternating row colors
        self.file_list.itemClicked.connect(self.on_file_selected)
        file_tree_layout.addWidget(self.file_list)
        
        results_layout.addWidget(file_tree_widget)
        
        # Results panel (right side)
        results_panel = QWidget()
        results_panel_layout = QVBoxLayout(results_panel)
        results_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results toolbar
        results_toolbar = QHBoxLayout()
        results_toolbar.addWidget(QLabel("Results:"))
        
        self.match_count_label = QLabel("0 matches")
        results_toolbar.addWidget(self.match_count_label)
        
        results_toolbar.addStretch(1)
        
        # Filter input for results
        results_toolbar.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter results...")
        self.filter_input.setMaximumWidth(200)
        self.filter_input.textChanged.connect(self.filter_results)
        results_toolbar.addWidget(self.filter_input)
        
        results_panel_layout.addLayout(results_toolbar)
        
        # Results text area with clickable references
        click_hint = QLabel("💡 Click on any file:line reference to open in your default editor")
        click_hint.setStyleSheet("color: #999999; font-style: italic;")
        results_panel_layout.addWidget(click_hint)
        
        self.results_text = ClickableTextEdit(results_panel)
        self.results_text.setFont(QFont("Menlo, Monaco, Courier New", 10))
        results_panel_layout.addWidget(self.results_text)
        
        results_layout.addWidget(results_panel)
        
        self.tabs.addTab(results_tab, "Search Results")
        
        # ----- Chat Tab -----
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        
        chat_header = QHBoxLayout()
        chat_header.addWidget(QLabel("Chat with AI about search results:"))
        
        self.chat_button = QPushButton("Ask AI")
        self.chat_button.clicked.connect(self.start_chat)
        self.chat_button.setEnabled(False)
        chat_header.addWidget(self.chat_button)
        chat_layout.addLayout(chat_header)
        
        chat_splitter = QSplitter(Qt.Vertical)
        
        # Chat input
        chat_input_widget = QWidget()
        chat_input_layout = QVBoxLayout(chat_input_widget)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your question about the search results...")
        self.chat_input.setMaximumHeight(80)
        placeholder_palette = self.chat_input.palette()
        placeholder_palette.setColor(placeholder_palette.ColorRole.PlaceholderText, QColor("#cccccc"))
        self.chat_input.setPalette(placeholder_palette)
        chat_input_layout.addWidget(self.chat_input)
        
        chat_splitter.addWidget(chat_input_widget)
        
        # Chat output
        self.chat_output = FileClickableBrowser(chat_tab)
        self.chat_output.setFont(QFont("Menlo, Monaco, Courier New", 10))
        chat_splitter.addWidget(self.chat_output)
        
        # Set initial sizes
        chat_splitter.setSizes([100, 500])
        chat_layout.addWidget(chat_splitter)
        
        self.tabs.addTab(chat_tab, "AI Chat")
        
        content_layout.addWidget(self.tabs)
        
        main_layout.addWidget(content_widget)
        
        # Status bar with progress
        self.statusBar().showMessage("Ready")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Connect signals
        self.signal_update_status.connect(self.update_status)
        self.signal_update_terms.connect(self.update_terms)
        self.signal_search_complete.connect(self.search_complete)
        self.signal_error.connect(self.show_error)
        self.signal_chat_response.connect(self.update_chat)
        
        self.setCentralWidget(central_widget)
        
        # Apply custom styling
        self.apply_custom_styles()
    
    def apply_custom_styles(self):
        """Apply additional custom styles to the UI components"""
        # Sidebar styling
        self.setStyleSheet(self.styleSheet() + """
            #sidebar {
                background-color: #2d2d2d;
                border-right: 1px solid #444;
            }
            
            #sidebar-group {
                background-color: #333;
                border: 1px solid #444;
                border-radius: 4px;
            }
            
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #252525;
            }
            
            QTabBar::tab {
                background-color: #353535;
                color: #ccc;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: #252525;
                color: white;
                border-bottom: none;
            }
            
            QListWidget {
                background-color: #252525;
                border: 1px solid #444;
                border-radius: 4px;
            }
            
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
            
            QListWidget::item:selected {
                background-color: #2a82da;
                color: white;
            }
            
            /* Removed alternating row styles */
        """)

    def on_file_selected(self, item):
        """Handle file selection in the file list"""
        if not hasattr(self, 'matches') or not self.matches:
            return
            
        file_path = item.data(Qt.UserRole)
        if not file_path:
            return
            
        # Show only matches for this file
        cursor = self.results_text.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.results_text.setTextCursor(cursor)
        
        # Clear current content and show only matches for selected file
        self.results_text.clear()
        file_matches = [m for m in self.matches if m['file'] == file_path]
        
        for i, match in enumerate(file_matches):
            self.results_text.append(f"{match['file']}:{match['line']} (Match #{i+1})\n")
            self.results_text.append(f"Matched term: {match['term']}\n")
            self.results_text.append(match['context'])
            self.results_text.append("\n---\n")
    
    def filter_results(self, filter_text):
        """Filter results based on filter text"""
        if not filter_text:
            # If no filter, restore full results or selected file results
            selected_items = self.file_list.selectedItems()
            if selected_items:
                self.on_file_selected(selected_items[0])
            else:
                self.display_all_results()
            return
            
        # Clear current content
        self.results_text.clear()
        
        if not hasattr(self, 'matches') or not self.matches:
            return
            
        # Filter matches that contain the filter text
        filtered_matches = [m for m in self.matches 
                            if filter_text.lower() in m['file'].lower() 
                            or filter_text.lower() in m['context'].lower()
                            or filter_text.lower() in m['term'].lower()]
        
        for i, match in enumerate(filtered_matches):
            self.results_text.append(f"{match['file']}:{match['line']} (Match #{i+1})\n")
            self.results_text.append(f"Matched term: {match['term']}\n")
            self.results_text.append(match['context'])
            self.results_text.append("\n---\n")
        
        self.match_count_label.setText(f"{len(filtered_matches)} filtered matches")
    
    def display_all_results(self):
        """Display all results in the results text area"""
        if not hasattr(self, 'matches') or not self.matches:
            return
            
        self.results_text.clear()
        
        for i, match in enumerate(self.matches):
            self.results_text.append(f"{match['file']}:{match['line']} (Match #{i+1})\n")
            self.results_text.append(f"Matched term: {match['term']}\n")
            self.results_text.append(match['context'])
            self.results_text.append("\n---\n")
    
    def search_complete(self, match_count):
        """Handle search completion with the new UI"""
        # Reset stdout
        sys.stdout = self.original_stdout
        
        # Process any remaining buffered output
        buffered_text = self.results_buffer.get_and_clear()
        if buffered_text:
            cursor = self.results_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(buffered_text)
            self.results_text.setTextCursor(cursor)
            self.results_text.ensureCursorVisible()
        
        # Stop the update timer
        self.update_timer.stop()
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Get matches from search thread
        if hasattr(self, 'search_thread') and hasattr(self.search_thread, 'matches'):
            self.matches = self.search_thread.matches
        else:
            self.matches = []
            
        # Update file list with unique files
        self.update_file_list()
        
        if match_count > 0:
            self.chat_button.setEnabled(True)
            # Remove chat_action reference
            self.refine_button.setEnabled(True)
            self.statusBar().showMessage(f"Search complete. Found {match_count} matches.")
            self.match_count_label.setText(f"{match_count} matches")
        else:
            self.chat_button.setEnabled(False)
            # Remove chat_action reference
            self.refine_button.setEnabled(False)
            self.statusBar().showMessage("Search complete. No matches found.")
            self.match_count_label.setText("0 matches")
            
        # Switch to results tab
        self.tabs.setCurrentIndex(0)
            
        # Clear reference to search thread
        self.search_thread = None
        gc.collect()
    
    def update_file_list(self):
        """Update the file list in the tree view with matched files"""
        self.file_list.clear()
        
        if not hasattr(self, 'matches') or not self.matches:
            return
            
        # Get unique files and count matches per file
        file_counts = {}
        for match in self.matches:
            file_path = match['file']
            if file_path in file_counts:
                file_counts[file_path] += 1
            else:
                file_counts[file_path] = 1
        
        # Add items to list
        for file_path, count in file_counts.items():
            item = QListWidgetItem(f"{os.path.basename(file_path)} ({count})")
            item.setData(Qt.UserRole, file_path)
            self.file_list.addItem(item)
        
        # Display all results initially
        self.display_all_results()
    
    def start_search(self):
        """Start search with updated UI elements"""
        success = self._prepare_search(is_refine=False)
        if success:
            self.search_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.statusBar().showMessage("Searching...")
            
            # Clear file list
            self.file_list.clear()
    
    def stop_search(self):
        """Stop the currently running search with updated UI"""
        if self.search_thread and self.search_thread.is_alive():
            # Request the search thread to stop
            self.search_thread.stop()
            
            # Update UI
            self.results_buffer.add("\n\n[Search stopped by user]\n")
            self.stop_button.setEnabled(False)
            self.search_button.setEnabled(True)
            self.statusBar().showMessage("Search stopped")

    def _prepare_search(self, is_refine=False):
        """Common setup for both search and refine search operations"""
        directory = self.dir_input.text().strip()
        prompt = self.prompt_input.text().strip()
        provider = self.provider_combo.currentText()
        
        if not directory:
            self.show_error("Please select a directory")
            return False
        
        if not prompt:
            self.show_error("Please enter a search prompt")
            return False
        
        # Check for API key
        api_key = getattr(self, f"{provider}_key", '')
        if not api_key:
            self.show_error(f"Please set your {provider.title()} API key first")
            self.show_api_key_dialog(provider)
            return False
        
        # Get extensions
        extensions_text = self.ext_input.text().strip()
        extensions = extensions_text.split() if extensions_text else None
        
        # Clear previous results - more aggressive memory cleanup
        if not is_refine:
            self.results_text.clear()
            self.chat_history = []  # Clear chat history for new searches
        self.chat_output.clear()
        self.chat_input.clear()
        self.chat_button.setEnabled(False)
        
        # Reset the results buffer
        self.results_buffer = ResultsBuffer()
        
        # Force garbage collection
        previous_matches = None
        if is_refine:
            previous_matches = self.matches
        else:
            self.matches = []
        gc.collect()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Redirect stdout to results buffer
        self.original_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.results_buffer)
        
        # Start search thread
        self.search_thread = SearchThread(
            self,
            directory,
            prompt,
            extensions,
            self.case_sensitive.isChecked(),
            False,  # color_output (we handle this differently in GUI)
            self.context_lines.value(),
            not self.include_comments.isChecked(),
            self.max_terms.value(),
            self.max_workers.value(),
            provider,
            self.single_line_mode.isChecked()
        )
        self.search_thread.start()
        
        # Start the update timer to periodically process the buffer
        self.update_timer.start(250)  # Update UI every 250ms
        
        return True

    def refine_search(self):
        """Refine the search based on current results"""
        if not self.matches:
            self.show_error("No search results to refine from")
            return
            
        self._prepare_search(is_refine=True)
    
    def process_results_buffer(self):
        """Process accumulated results from the buffer and update UI"""
        if not hasattr(self, 'search_thread') or self.search_thread is None:
            self.update_timer.stop()
            return
            
        # Get buffered content
        buffered_text = self.results_buffer.get_and_clear()
        if not buffered_text:
            return
            
        # Check if text widget is getting too large
        if self.results_text.document().characterCount() > 1000000:  # ~1MB text limit
            # Truncate the text to prevent memory issues
            cursor = self.results_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 200000)  # Remove ~200K chars
            cursor.removeSelectedText()
            self.results_text.insertPlainText("[...truncated to prevent memory issues...]\n\n")
        
        # Append new text
        cursor = self.results_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(buffered_text)
        self.results_text.setTextCursor(cursor)
        self.results_text.ensureCursorVisible()
        
        # Check if search is finished
        if hasattr(self, 'search_thread') and self.search_thread and not self.search_thread.running:
            self.update_timer.stop()
        
    @Slot(str)
    def update_status(self, message):
        self.statusBar().showMessage(message)
        
    @Slot(str)
    def update_terms(self, terms_text):
        # Add to buffer instead of directly to UI
        self.results_buffer.add(terms_text + "\n\n")
    
    @Slot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.chat_button.setEnabled(True if self.matches else False)
    
    def show_api_key_dialog(self, provider):
        """Show dialog to input API key"""
        current_key = getattr(self, f"{provider}_key", '')
        title = f"Set {provider.title()} API Key"
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel(f"Enter your {provider.title()} API key below. The key will be stored locally.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # API Key input
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("API Key:"))
        key_input = QLineEdit()
        key_input.setEchoMode(QLineEdit.Password)  # Hide the key
        key_input.setText(current_key)
        key_layout.addWidget(key_input)
        layout.addLayout(key_layout)
        
        # Azure-specific fields
        if provider == "azure":
            # Endpoint input
            endpoint_layout = QHBoxLayout()
            endpoint_layout.addWidget(QLabel("Endpoint:"))
            endpoint_input = QLineEdit()
            endpoint_input.setPlaceholderText("https://your-resource-name.openai.azure.com")
            endpoint_input.setText(getattr(self, "azure_endpoint", ""))
            endpoint_layout.addWidget(endpoint_input)
            layout.addLayout(endpoint_layout)
            
            # Deployment name input
            deployment_layout = QHBoxLayout()
            deployment_layout.addWidget(QLabel("Deployment:"))
            deployment_input = QLineEdit()
            deployment_input.setPlaceholderText("gpt-4")
            deployment_input.setText(getattr(self, "azure_deployment", "gpt-4"))
            deployment_layout.addWidget(deployment_input)
            layout.addLayout(deployment_layout)
            
            # Add help text for Azure configuration
            help_text = QLabel(
                "For Azure OpenAI, you need:\n"
                "1. API Key from Azure Portal\n"
                "2. Endpoint URL from your Azure OpenAI resource\n"
                "3. Deployment name of your model"
            )
            help_text.setWordWrap(True)
            layout.addWidget(help_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect buttons
        if provider == "azure":
            save_btn.clicked.connect(lambda: self.save_azure_config(
                key_input.text(), 
                endpoint_input.text(),
                deployment_input.text(),
                dialog
            ))
        else:
            save_btn.clicked.connect(lambda: self.save_api_key(provider, key_input.text(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def save_azure_config(self, key, endpoint, deployment, dialog):
        """Save Azure OpenAI configuration"""
        if not key:
            QMessageBox.warning(dialog, "Error", "API Key cannot be empty")
            return
            
        if not endpoint:
            QMessageBox.warning(dialog, "Error", "Endpoint URL cannot be empty")
            return
            
        if not deployment:
            deployment = "gpt-4"  # Default deployment name
            
        # Save to instance variables
        self.azure_key = key
        self.azure_endpoint = endpoint
        self.azure_deployment = deployment
        
        # Set environment variables
        os.environ["AZURE_OPENAI_API_KEY"] = key
        os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment
        
        self.saveSettings()
        dialog.accept()
        self.statusBar().showMessage("Azure OpenAI configuration saved")
    
    def save_api_key(self, provider, key, dialog):
        """Save API key and close dialog"""
        if key:
            # Save to instance variable
            setattr(self, f"{provider}_key", key)
            # Set environment variable for immediate use
            os.environ[f"{provider.upper()}_API_KEY"] = key
            self.saveSettings()
            dialog.accept()
            self.statusBar().showMessage(f"{provider.title()} API Key saved")
        else:
            QMessageBox.warning(dialog, "Error", "API Key cannot be empty")
    
    def on_provider_changed(self, provider):
        """Handle provider selection change"""
        self.settings.setValue("provider", provider)
        self.saveSettings()
    
    def open_file_in_editor(self, file_path, line_number):
        """Open a file at a specific line in the default editor"""
        try:
            system = platform.system()
            line_num = int(line_number)
            
            editors_by_platform = {
                "Darwin": [  # macOS
                    # Check if the 'code' command is available in PATH
                    (lambda: subprocess.run(["which", "code"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
                     lambda: subprocess.run(["code", "--goto", f"{file_path}:{line_num}"])),
                    # Check common VS Code installation locations
                    (lambda: any(os.path.exists(p) for p in [
                        "/Applications/Visual Studio Code.app", 
                        "/Applications/VSCode.app",
                        "/Applications/VS Code.app",
                        os.path.expanduser("~/Applications/Visual Studio Code.app"),
                        os.path.expanduser("~/Applications/VSCode.app"),
                        os.path.expanduser("~/Applications/VS Code.app")
                    ]), lambda: subprocess.run(["open", "-a", "Visual Studio Code", "--args", "-g", f"{file_path}:{line_num}"])),
                    # Check if VSCodium exists (open-source VS Code)
                    (lambda: any(os.path.exists(p) for p in [
                        "/Applications/VSCodium.app",
                        os.path.expanduser("~/Applications/VSCodium.app")
                    ]), lambda: subprocess.run(["open", "-a", "VSCodium", "--args", "-g", f"{file_path}:{line_num}"])),
                    (lambda: os.path.exists("/Applications/Sublime Text.app"), 
                     lambda: subprocess.run(["open", "-a", "Sublime Text", "--args", f"{file_path}:{line_num}"])),
                    (lambda: True,  # Fallback to TextEdit
                     lambda: subprocess.run(["open", "-a", "TextEdit", file_path]))
                ],
                "Windows": [
                    # Try VSCode in various possible locations
                    (lambda: any(os.path.exists(p) for p in [
                        os.path.expandvars("%LOCALAPPDATA%\\Programs\\Microsoft VS Code\\Code.exe"),
                        os.path.expandvars("%PROGRAMFILES%\\Microsoft VS Code\\Code.exe"),
                        os.path.expandvars("%USERPROFILE%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe")
                    ]), lambda: subprocess.run([next(p for p in [
                        os.path.expandvars("%LOCALAPPDATA%\\Programs\\Microsoft VS Code\\Code.exe"),
                        os.path.expandvars("%PROGRAMFILES%\\Microsoft VS Code\\Code.exe"),
                        os.path.expandvars("%USERPROFILE%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe")
                    ] if os.path.exists(p)), "-g", f"{file_path}:{line_num}"])),
                    
                    # Try Notepad++
                    (lambda: any(os.path.exists(p) for p in [
                        "C:\\Program Files\\Notepad++\\notepad++.exe",
                        "C:\\Program Files (x86)\\Notepad++\\notepad++.exe",
                        os.path.expandvars("%PROGRAMFILES%\\Notepad++\\notepad++.exe"),
                        os.path.expandvars("%PROGRAMFILES(X86)%\\Notepad++\\notepad++.exe")
                    ]), lambda: subprocess.run([next(p for p in [
                        "C:\\Program Files\\Notepad++\\notepad++.exe",
                        "C:\\Program Files (x86)\\Notepad++\\notepad++.exe",
                        os.path.expandvars("%PROGRAMFILES%\\Notepad++\\notepad++.exe"),
                        os.path.expandvars("%PROGRAMFILES(X86)%\\Notepad++\\notepad++.exe")
                    ] if os.path.exists(p)), "-n" + line_number, file_path])),
                    
                    # Fallback to default application
                    (lambda: True, lambda: subprocess.run(["cmd", "/c", "start", "", "/b", file_path], 
                                                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) 
                                   if subprocess.run == subprocess.run else os.startfile(file_path))
                ],
                "Linux": [
                    (lambda: subprocess.run(["which", "code"], stdout=subprocess.DEVNULL).returncode == 0,
                     lambda: subprocess.run(["code", "-g", f"{file_path}:{line_num}"])),
                    (lambda: subprocess.run(["which", "gedit"], stdout=subprocess.DEVNULL).returncode == 0,
                     lambda: subprocess.run(["gedit", f"{file_path}+{line_num}"])),
                    (lambda: subprocess.run(["which", "vim"], stdout=subprocess.DEVNULL).returncode == 0,
                     lambda: subprocess.run(["x-terminal-emulator", "-e", f"vim +{line_num} {file_path}"])),
                    (lambda: True, lambda: subprocess.run(["xdg-open", file_path]))
                ]
            }
            
            if system in editors_by_platform:
                for condition, action in editors_by_platform[system]:
                    if condition():
                        action()
                        break
                self.statusBar().showMessage(f"Opened {file_path}:{line_number}")
            else:
                self.statusBar().showMessage(f"Unsupported platform: {system}")
                
        except Exception as e:
            self.show_error(f"Error opening file: {str(e)}")

    def clear_file_cache(self):
        """Clear the file cache to force directory re-scan on next search"""
        aisearch.clear_file_cache()
        self.statusBar().showMessage("File cache cleared")

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)

    def clear_results(self):
        """Clear all search results and chat history"""
        self.results_text.clear()
        self.chat_output.clear()
        self.chat_input.clear()
        self.file_list.clear()
        self.matches = []
        self.chat_history = []  # Clear chat history
        self.chat_button.setEnabled(False)
        self.chat_action.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.match_count_label.setText("0 matches")
        self.statusBar().showMessage("Results cleared")

    def start_chat(self):
        """Start a chat with AI about search results"""
        if not self.matches:
            self.show_error("No search results to discuss")
            return
        
        question = self.chat_input.toPlainText().strip()
        prompt = self.prompt_input.text().strip()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.chat_button.setEnabled(False)
        
        # Switch to chat tab
        self.tabs.setCurrentIndex(1)
        
        # Start chat thread with history
        self.chat_thread = ChatThread(self, self.matches, prompt, question, self.chat_history)
        self.chat_thread.start()
    
    @Slot(object)
    def update_chat(self, response_data):
        """Update chat output with AI response"""
        # Unpack response and history
        if isinstance(response_data, tuple):
            response, self.chat_history = response_data
        else:
            # For backward compatibility
            response = response_data
        
        # Store original markdown for copy functionality
        self.chat_output.markdown_content = response
        
        # Clear previous content and show only the latest exchange
        self.chat_output.clear()
        
        # Add user question to the display if it exists
        if self.chat_input.toPlainText().strip():
            # Format with user question and AI response
            content = f"**You:** {self.chat_input.toPlainText().strip()}\n\n**AI:**\n\n{response}"
            self.chat_output.setMarkdown(content)
            # Clear input after sending
            self.chat_input.clear()
        else:
            # Initial analysis (no user question)
            self.chat_output.setMarkdown(response)
            
        # Convert markdown to HTML
        try:
            # Set up extensions based on available libraries
            extensions = ['fenced_code', 'tables']
            extension_configs = {}
            
            # Add syntax highlighting if Pygments is available
            if PYGMENTS_AVAILABLE:
                extensions.append('codehilite')
                formatter = HtmlFormatter(style='monokai')
                extension_configs['codehilite'] = {
                    'css_class': 'highlight',
                    'guess_lang': True,
                    'linenums': False
                }
                
                # Add Pygments CSS to document stylesheet
                pygments_css = formatter.get_style_defs('.highlight')
                # Combine existing stylesheet with pygments styles
                current_css = self.chat_output.document().defaultStyleSheet() 
                self.chat_output.document().setDefaultStyleSheet(current_css + pygments_css)
            
            html_content = markdown.markdown(
                self.chat_output.toMarkdown(), 
                extensions=extensions,
                extension_configs=extension_configs
            )
           
            # Make file references clickable by converting them to actual anchor tags
            html_content = re.sub(
                r'([^:<>\s]+?\.[a-zA-Z0-9]+):(\d+)', 
                r'<a href="\1:\2" class="file-reference">\1:\2</a>', 
                html_content
            )
            
            # Set the HTML content
            self.chat_output.setHtml(html_content)
        
        # Hide progress and enable chat button
        finally:
            self.progress_bar.setVisible(False)
            self.chat_button.setEnabled(True)

    def loadSettings(self):
        """Load saved settings"""
        # Directory
        directory = self.settings.value("directory", "")
        if directory and os.path.exists(directory):
            self.dir_input.setText(directory)
            
        # Prompt
        prompt = self.settings.value("prompt", "")
        self.prompt_input.setText(prompt)
        
        # Extensions
        extensions = self.settings.value("extensions", "")
        self.ext_input.setText(extensions)
        
        # API Keys (stored encrypted)
        self.anthropic_key = self.settings.value("anthropic_api_key", "")
        self.openai_key = self.settings.value("openai_api_key", "")
        self.azure_key = self.settings.value("azure_api_key", "")
        self.azure_endpoint = self.settings.value("azure_endpoint", "")
        self.azure_deployment = self.settings.value("azure_deployment", "gpt-4")
        
        # Set environment variables if API keys are available
        if self.anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_key
        if self.openai_key:
            os.environ["OPENAI_API_KEY"] = self.openai_key
        if self.azure_key and self.azure_endpoint:
            os.environ["AZURE_OPENAI_API_KEY"] = self.azure_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = self.azure_endpoint
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = self.azure_deployment
        
        # Provider
        provider = self.settings.value("provider", "anthropic")
        self.provider_combo.setCurrentText(provider)
        
        # Checkboxes
        self.case_sensitive.setChecked(self.settings.value("case_sensitive", True, type=bool))
        self.include_comments.setChecked(self.settings.value("include_comments", True, type=bool))
        self.single_line_mode.setChecked(self.settings.value("single_line_mode", False, type=bool))
        
        # SpinBoxes
        self.max_terms.setValue(self.settings.value("max_terms", 8, type=int))
        self.max_workers.setValue(self.settings.value("max_workers", 4, type=int))
        self.context_lines.setValue(self.settings.value("context_lines", 2, type=int))
    
    def saveSettings(self):
        """Save current settings"""
        # Directory
        self.settings.setValue("directory", self.dir_input.text())
        
        # Prompt
        self.settings.setValue("prompt", self.prompt_input.text())
        
        # Extensions
        self.settings.setValue("extensions", self.ext_input.text())
        
        # API Keys
        if hasattr(self, 'anthropic_key'):
            self.settings.setValue("anthropic_api_key", self.anthropic_key)
        if hasattr(self, 'openai_key'):
            self.settings.setValue("openai_api_key", self.openai_key)
        if hasattr(self, 'azure_key'):
            self.settings.setValue("azure_api_key", self.azure_key)
            self.settings.setValue("azure_endpoint", self.azure_endpoint)
            self.settings.setValue("azure_deployment", self.azure_deployment)
        
        # Provider
        self.settings.setValue("provider", self.provider_combo.currentText())
        
        # Checkboxes
        self.settings.setValue("case_sensitive", self.case_sensitive.isChecked())
        self.settings.setValue("include_comments", self.include_comments.isChecked())
        self.settings.setValue("single_line_mode", self.single_line_mode.isChecked())
        
        # SpinBoxes
        self.settings.setValue("max_terms", self.max_terms.value())
        self.settings.setValue("max_workers", self.max_workers.value())
        self.settings.setValue("context_lines", self.context_lines.value())
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.saveSettings()
        
        # Clean up threads and processes before closing
        self.cleanup_resources()
        
        super().closeEvent(event)
    
    def cleanup_resources(self):
        """Clean up resources to prevent leaks"""
        # Stop the update timer
        if hasattr(self, 'update_timer') and self.update_timer.isActive():
            self.update_timer.stop()
        
        # Clear references to threads
        if hasattr(self, 'search_thread'):
            self.search_thread = None
        
        if hasattr(self, 'chat_thread'):
            self.chat_thread = None
        
        # Force garbage collection
        gc.collect()
        
        # Explicitly clean multiprocessing resources
        try:
            import multiprocessing as mp
            if hasattr(mp, 'active_children'):
                for child in mp.active_children():
                    try:
                        child.terminate()
                        child.join(0.1)
                    except:
                        pass
        except:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look and feel
    
    # Test the file pattern matching
    #ClickableTextEdit.test_file_pattern()
    
    # Set application icon if available
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aisearch_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Set dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, Qt.white)
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, Qt.white)
    palette.setColor(palette.ColorRole.ToolTipText, Qt.white)
    palette.setColor(palette.ColorRole.Text, Qt.white)
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, Qt.white)
    palette.setColor(palette.ColorRole.BrightText, Qt.red)
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Apply stylesheet
    app.setStyleSheet("""
        QMainWindow, QDialog {
            background-color: #353535;
        }
        QGroupBox {
            border: 1px solid #5c5c5c;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #2a82da;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #3a92ea;
        }
        QPushButton:pressed {
            background-color: #1a72ca;
        }
        QPushButton:disabled {
            background-color: #666666;
        }
        QLineEdit, QTextEdit, QSpinBox {
            background-color: #252525;
            border: 1px solid #5c5c5c;
            border-radius: 4px;
            padding: 4px;
            color: white;
        }
        QLineEdit::placeholder, QTextEdit::placeholder {
            color: #cccccc;
        }
        QLabel {
            color: white;
        }
        QCheckBox {
            color: white;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QProgressBar {
            border: 1px solid #5c5c5c;
            border-radius: 4px;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            width: 20px;
        }
        QToolBar {
            background-color: #252525;
            border-bottom: 1px solid #5c5c5c;
            spacing: 3px;
        }
        QToolBar QToolButton {
            background-color: transparent;
            color: white;
            border-radius: 4px;
            padding: 5px;
        }
        QToolBar QToolButton:hover {
            background-color: #444444;
        }
        QToolBar QToolButton:pressed {
            background-color: #555555;
        }
        QSpinBox {
            color: white;
        }
        QSplitter::handle {
            background-color: #5c5c5c;
        }
    """)
    
    window = AISearchGUI()
    window.show()
    
    # Add cleanup to Python's exit handling
    def cleanup():
        window.cleanup_resources()
        # Extra cleanup for multiprocessing
        import multiprocessing as mp
        if hasattr(mp, '_resource_tracker') and hasattr(mp._resource_tracker, '_stop'):
            try:
                mp._resource_tracker._stop()
            except:
                pass
    
    atexit.register(cleanup)
    
    sys.exit(app.exec()) 