import sys
import os
import threading
import atexit
import gc
import time
import re
import subprocess
import platform
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, 
                             QCheckBox, QSpinBox, QGroupBox, QSplitter, QComboBox,
                             QListWidget, QProgressBar, QMessageBox, QDialog)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QSettings, QTimer
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon, QTextCharFormat

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
                color_output, context_lines, ignore_comments, max_terms, max_workers, provider):
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
        self.matches = []
        self.running = True
        self.stop_requested = False
        self.provider = provider
        
    def run(self):
        try:
            # Get search terms
            if self.stop_requested:
                self.running = False
                return
                
            self.parent.signal_update_status.emit(f"Querying {self.provider.title()} for search terms...")
            
            # Check if this is a refined search (we have previous matches)
            if hasattr(self.parent, 'matches') and self.parent.matches:
                self.search_terms = aisearch.get_refined_search_terms(
                    self.prompt,
                    self.parent.matches,
                    max_terms=self.max_terms,
                    extensions=self.extensions,
                    context_lines=self.context_lines,
                    provider=self.provider
                )
                self.parent.signal_update_status.emit("Generated refined search terms based on current results...")
            else:
                self.search_terms = aisearch.get_search_terms_from_prompt(
                    self.prompt,
                    max_terms=self.max_terms,
                    extensions=self.extensions,
                    provider=self.provider
                )
            
            # Display terms
            terms_text = "Suggested terms:\n" + "\n".join([f"- {t}" for t in self.search_terms])
            self.parent.signal_update_terms.emit(terms_text)
            
            if self.stop_requested:
                self.running = False
                return
                
            # Search code
            self.parent.signal_update_status.emit("Searching code using regex patterns...")
            
            # Process in batches
            all_matches = []
            MAX_RESULTS = 100
            
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
                    stop_requested=lambda: self.stop_requested  # Pass a callable to check if stop was requested
                )
                
                # Truncate matches if too many
                if len(all_matches) > MAX_RESULTS:
                    self.matches = all_matches[:MAX_RESULTS]
                    self.parent.signal_update_status.emit(f"Limited to {MAX_RESULTS} matches to prevent memory issues")
                else:
                    self.matches = all_matches
            except Exception as e:
                self.parent.signal_error.emit(f"Search error: {str(e)}")
                self.matches = all_matches[:50] if len(all_matches) > 50 else all_matches
            
            self.parent.signal_search_complete.emit(len(self.matches))
        except Exception as e:
            self.parent.signal_error.emit(str(e))
        finally:
            self.running = False
    
    def stop(self):
        self.stop_requested = True

class ChatThread(threading.Thread):
    def __init__(self, parent, matches, prompt, question):
        super().__init__()
        self.parent = parent
        self.matches = matches
        self.prompt = prompt
        self.question = question
        self.provider = parent.provider_combo.currentText()
        
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

            When referring to matches, ALWAYS use the file path and line number (e.g., 'In file.cpp:123') rather than match numbers.
            Keep your responses concise and to the point."""
            
            messages = [
                {"role": "user", "content": f"These are code search results for: '{self.prompt}'\n\n{combined_contexts}"}
            ]
            
            if self.question:
                messages.append({"role": "user", "content": self.question})
            else:
                messages.append({"role": "user", "content": "Please analyze these findings."})
            
            if self.provider == "anthropic":
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4096,
                    temperature=1,
                    system=system_message,
                    messages=messages
                )
                response_text = response.content[0].text
            else:  # OpenAI
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": system_message}] + messages,
                    temperature=1,
                    max_tokens=4096
                )
                response_text = response.choices[0].message.content
            
            self.parent.signal_chat_response.emit(response_text)
            
        except Exception as e:
            self.parent.signal_error.emit(str(e))

class ClickableTextEdit(QTextEdit):
    """Custom QTextEdit that can detect clicks on file references"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.setCursorWidth(1)
        # Use a simpler pattern to match line numbers and delegate more complex path validation to a method
        self.file_match_pattern = re.compile(r'([^:]+):(\d+)')
        # Remove text selection flags to prevent automatic selection
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setStatusTip("Click on file:line references to open in your default editor")
    
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
    
    @staticmethod
    def test_file_pattern():
        """Test the file pattern matching with various path formats"""
        test_cases = [
            # Windows paths
            ("C:\\path\\to\\file.py:123", True),
            ("D:/path/to/file.py:123", True),
            ("C:\\Program Files\\App\\file.txt:456", True),
            
            # Windows \\?\ prefixed paths
            ("\\\\?\\C:\\path\\to\\file.py:123", True),
            ("\\\\?\\D:/path/to/file.py:123", True),
            ("\\\\?\\C:\\Program Files\\App\\file.txt:456", True),
            ("\\\\?\\UNC\\server\\share\\file.py:123", True),
            
            # Unix/Mac paths
            ("/path/to/file.py:123", True),
            ("/home/user/project/file.js:789", True),
            ("/usr/local/bin/script.sh:42", True),
            
            # Relative paths
            ("./file.py:123", True),
            ("../file.py:123", True),
            ("../../project/file.py:123", True),
            
            # Edge cases
            ("file.py:123", False),  # No path separator
            ("C:file.py:123", False),  # Missing path separator after drive
            ("/file.py", False),  # No line number
            ("file.py", False),  # No line number
            ("C:\\path\\file", False),  # No line number
            ("/path/file", False),  # No line number
            ("\\\\?\\file.py:123", False),  # \\?\ without drive letter
            ("\\\\?\\C:file.py:123", False),  # \\?\ with missing path separator
        ]
        
        print("\nTesting file pattern matching:")
        print("-" * 50)
        pattern = re.compile(r'([^:]+):(\d+)')
        
        for test_path, expected in test_cases:
            match = pattern.search(test_path)
            if match:
                path = match.group(1)
                result = ClickableTextEdit.is_valid_file_path(path)
            else:
                result = False
            
            status = "✓" if result == expected else "✗"
            print(f"{status} {test_path:<40} Expected: {expected}, Got: {result}")
    
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
        cursor.select(QTextCursor.LineUnderCursor)
        line_text = cursor.selectedText()
        
        # Check if cursor is on file:line pattern
        match = self.file_match_pattern.search(line_text)
        if match and self.is_valid_file_path(match.group(1)):
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.IBeamCursor)
            
        super().mouseMoveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press to detect file references"""
        if event.button() == Qt.LeftButton:
            # Get the line under the cursor
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QTextCursor.LineUnderCursor)
            line_text = cursor.selectedText()
            
            # Look for file:line pattern
            match = self.file_match_pattern.search(line_text)
            if match and self.is_valid_file_path(match.group(1)):
                file_path = match.group(1)
                line_number = match.group(2)
                
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
                # Accept the event to prevent text selection
                event.accept()
                return
        
        # If we get here, either it's not a left click or no file pattern was found
        # Let the parent handle the event for normal text selection
        super().mousePressEvent(event)

class AISearchGUI(QMainWindow):
    signal_update_status = Signal(str)
    signal_update_terms = Signal(str)
    signal_search_complete = Signal(int)
    signal_error = Signal(str)
    signal_chat_response = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.matches = []
        self.search_thread = None
        self.chat_thread = None
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
        self.setMinimumSize(1000, 700)
        
        # Create toolbar
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))
        
        # Add toolbar actions
        search_action = self.toolbar.addAction("Search")
        search_action.triggered.connect(self.start_search)
        search_action.setToolTip("Start a new search")
        
        # Add stop search action
        self.stop_action = self.toolbar.addAction("Stop")
        self.stop_action.triggered.connect(self.stop_search)
        self.stop_action.setToolTip("Stop the current search")
        self.stop_action.setEnabled(False)
        
        self.toolbar.addSeparator()
        
        clear_action = self.toolbar.addAction("Clear")
        clear_action.triggered.connect(self.clear_results)
        clear_action.setToolTip("Clear all results")
        
        self.toolbar.addSeparator()
        
        # Add clear cache action
        clear_cache_action = self.toolbar.addAction("Clear Cache")
        clear_cache_action.triggered.connect(self.clear_file_cache)
        clear_cache_action.setToolTip("Clear file list cache to force directory re-scan")
        
        self.toolbar.addSeparator()
        
        self.chat_action = self.toolbar.addAction("Chat")
        self.chat_action.triggered.connect(self.start_chat)
        self.chat_action.setToolTip("Chat about results")
        self.chat_action.setEnabled(False)
        
        self.toolbar.addSeparator()
        
        # Add API key actions
        anthropic_key_action = self.toolbar.addAction("Anthropic Key")
        anthropic_key_action.triggered.connect(lambda: self.show_api_key_dialog("anthropic"))
        anthropic_key_action.setToolTip("Set your Anthropic API key")
        
        openai_key_action = self.toolbar.addAction("OpenAI Key")
        openai_key_action.triggered.connect(lambda: self.show_api_key_dialog("openai"))
        openai_key_action.setToolTip("Set your OpenAI API key")
        
        self.toolbar.addSeparator()
        
        # Add provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("AI Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["anthropic", "openai"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Search configuration section
        config_group = QGroupBox("Search Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directory:"))
        self.dir_input = QLineEdit()
        dir_layout.addWidget(self.dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        config_layout.addLayout(dir_layout)
        
        # Prompt input
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QLineEdit()
        prompt_layout.addWidget(self.prompt_input)
        config_layout.addLayout(prompt_layout)
        
        # Extensions input
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("File Extensions:"))
        self.ext_input = QLineEdit()
        self.ext_input.setPlaceholderText(".py .js .ts (leave empty for all files)")
        # Make placeholder text more visible
        placeholder_palette = self.ext_input.palette()
        placeholder_palette.setColor(placeholder_palette.ColorRole.PlaceholderText, QColor("#cccccc"))
        self.ext_input.setPalette(placeholder_palette)
        ext_layout.addWidget(self.ext_input)
        config_layout.addLayout(ext_layout)
        
        # Options layout
        options_layout = QHBoxLayout()
        
        # Left options (checkboxes)
        left_options = QVBoxLayout()
        left_options.setSpacing(5)  # Reduce spacing between checkboxes
        self.case_sensitive = QCheckBox("Case Sensitive")
        self.case_sensitive.setChecked(True)
        left_options.addWidget(self.case_sensitive)
        self.include_comments = QCheckBox("Include Comments")
        self.include_comments.setChecked(True)  # Include comments by default
        left_options.addWidget(self.include_comments)
        options_layout.addLayout(left_options)
        
        # Add some spacing between option groups
        options_layout.addSpacing(20)
        
        # Middle options (terms and workers)
        middle_options = QVBoxLayout()
        middle_options.setSpacing(5)  # Reduce spacing between rows
        
        # Terms row
        terms_layout = QHBoxLayout()
        terms_layout.setSpacing(5)  # Reduce spacing between label and spinbox
        terms_layout.addWidget(QLabel("Max Terms:"))
        self.max_terms = QSpinBox()
        self.max_terms.setRange(1, 100)
        self.max_terms.setValue(8)
        self.max_terms.setMinimumWidth(60)  # Make spinbox wider
        terms_layout.addWidget(self.max_terms)
        middle_options.addLayout(terms_layout)
        
        # Workers row
        workers_layout = QHBoxLayout()
        workers_layout.setSpacing(5)  # Reduce spacing between label and spinbox
        workers_layout.addWidget(QLabel("Workers:"))
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 64)
        self.max_workers.setValue(4)
        self.max_workers.setMinimumWidth(60)  # Make spinbox wider
        workers_layout.addWidget(self.max_workers)
        middle_options.addLayout(workers_layout)
        
        options_layout.addLayout(middle_options)
        
        # Add some spacing between option groups
        options_layout.addSpacing(20)
        
        # Right options (context lines and buttons)
        right_options = QVBoxLayout()
        right_options.setSpacing(5)  # Reduce spacing between rows
        
        # Context lines row
        context_layout = QHBoxLayout()
        context_layout.setSpacing(5)  # Reduce spacing between label and spinbox
        context_layout.addWidget(QLabel("Context Lines:"))
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 20)
        self.context_lines.setValue(2)
        self.context_lines.setMinimumWidth(60)  # Make spinbox wider
        context_layout.addWidget(self.context_lines)
        right_options.addLayout(context_layout)
        
        # Search buttons row
        search_buttons_layout = QHBoxLayout()
        search_buttons_layout.setSpacing(5)  # Reduce spacing between buttons
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.start_search)
        self.search_button.setMinimumWidth(80)  # Make button wider
        search_buttons_layout.addWidget(self.search_button)
        
        self.refine_button = QPushButton("Refine Search")
        self.refine_button.clicked.connect(self.refine_search)
        self.refine_button.setToolTip("Refine search based on current results")
        self.refine_button.setEnabled(False)
        self.refine_button.setMinimumWidth(80)  # Make button wider
        search_buttons_layout.addWidget(self.refine_button)
        
        right_options.addLayout(search_buttons_layout)
        options_layout.addLayout(right_options)
        
        config_layout.addLayout(options_layout)
        config_layout.addLayout(provider_layout)
        main_layout.addWidget(config_group)
        
        # Splitter for results and chat
        splitter = QSplitter(Qt.Vertical)
        
        # Results section
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a label to indicate clickable references
        click_hint = QLabel("💡 Click on any file:line reference to open in your default editor")
        click_hint.setStyleSheet("color: #999999; font-style: italic;")
        results_layout.addWidget(click_hint)
        
        # Use our custom ClickableTextEdit instead of QTextEdit
        self.results_text = ClickableTextEdit(results_widget)
        # Use a better cross-platform monospace font
        self.results_text.setFont(QFont("Menlo, Monaco, Courier New", 10))
        results_layout.addWidget(self.results_text)
        
        splitter.addWidget(results_widget)
        
        # Chat section
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        chat_header = QHBoxLayout()
        chat_header.addWidget(QLabel("Chat with AI about results:"))
        self.chat_button = QPushButton("Ask AI")
        self.chat_button.clicked.connect(self.start_chat)
        self.chat_button.setEnabled(False)
        chat_header.addWidget(self.chat_button)
        chat_layout.addLayout(chat_header)
        
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your question about the search results...")
        self.chat_input.setMaximumHeight(60)
        # Make placeholder text more visible
        placeholder_palette = self.chat_input.palette()
        placeholder_palette.setColor(placeholder_palette.ColorRole.PlaceholderText, QColor("#cccccc"))
        self.chat_input.setPalette(placeholder_palette)
        chat_layout.addWidget(self.chat_input)
        
        # Use our custom ClickableTextEdit for chat output as well
        self.chat_output = ClickableTextEdit(chat_widget)
        chat_layout.addWidget(self.chat_output)
        
        splitter.addWidget(chat_widget)
        
        # Set initial sizes
        splitter.setSizes([400, 300])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Progress bar
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
        
        self.setCentralWidget(main_widget)
    
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
        
        # Set environment variables if API keys are available
        if self.anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_key
        if self.openai_key:
            os.environ["OPENAI_API_KEY"] = self.openai_key
        
        # Provider
        provider = self.settings.value("provider", "anthropic")
        self.provider_combo.setCurrentText(provider)
        
        # Checkboxes
        self.case_sensitive.setChecked(self.settings.value("case_sensitive", True, type=bool))
        self.include_comments.setChecked(self.settings.value("include_comments", True, type=bool))
        
        # SpinBoxes
        self.max_terms.setValue(self.settings.value("max_terms", 8, type=int))
        self.max_workers.setValue(self.settings.value("max_workers", 4, type=int))
        self.context_lines.setValue(self.settings.value("context_lines", 2, type=int))
    
    def saveSettings(self):
        """Save current settings"""
        self.settings.setValue("directory", self.dir_input.text())
        self.settings.setValue("prompt", self.prompt_input.text())
        self.settings.setValue("extensions", self.ext_input.text())
        self.settings.setValue("case_sensitive", self.case_sensitive.isChecked())
        self.settings.setValue("include_comments", self.include_comments.isChecked())
        self.settings.setValue("max_terms", self.max_terms.value())
        self.settings.setValue("max_workers", self.max_workers.value())
        self.settings.setValue("context_lines", self.context_lines.value())
        self.settings.setValue("provider", self.provider_combo.currentText())
        
        # Save API keys if present
        if hasattr(self, 'anthropic_key') and self.anthropic_key:
            self.settings.setValue("anthropic_api_key", self.anthropic_key)
        if hasattr(self, 'openai_key') and self.openai_key:
            self.settings.setValue("openai_api_key", self.openai_key)
    
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
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
    
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
        self.chat_output.clear()
        self.chat_input.clear()
        self.chat_button.setEnabled(False)
        
        # Reset the results buffer
        self.results_buffer = ResultsBuffer()
        
        # Force garbage collection
        if not is_refine:
            self.matches = []
        gc.collect()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.stop_action.setEnabled(True)
        
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
            provider
        )
        self.search_thread.start()
        
        # Start the update timer to periodically process the buffer
        self.update_timer.start(250)  # Update UI every 250ms
        
        return True

    def start_search(self):
        self._prepare_search(is_refine=False)
        
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
    
    @Slot(int)
    def search_complete(self, match_count):
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
        self.stop_action.setEnabled(False)
        
        # Ensure we don't process too many matches
        if hasattr(self, 'search_thread') and hasattr(self.search_thread, 'matches'):
            if len(self.search_thread.matches) > 100:
                self.matches = self.search_thread.matches[:100]
                self.results_text.insertPlainText(f"\n\nLimiting displayed matches to 100 (out of {len(self.search_thread.matches)}) to prevent memory issues.\n")
            else:
                self.matches = self.search_thread.matches
        else:
            self.matches = []
            
        if match_count > 0:
            self.chat_button.setEnabled(True)
            self.chat_action.setEnabled(True)
            self.refine_button.setEnabled(True)  # Enable refine button when we have results
            self.statusBar().showMessage(f"Search complete. Found {match_count} matches (displaying up to 100).")
        else:
            self.chat_button.setEnabled(False)
            self.chat_action.setEnabled(False)
            self.refine_button.setEnabled(False)  # Disable refine button when no results
            self.statusBar().showMessage("Search complete. No matches found.")
            
        # Clear reference to search thread
        self.search_thread = None
        gc.collect()
    
    def clear_results(self):
        """Clear all search results and chat history"""
        self.results_text.clear()
        self.chat_output.clear()
        self.chat_input.clear()
        self.matches = []
        self.chat_button.setEnabled(False)
        self.chat_action.setEnabled(False)
        self.refine_button.setEnabled(False)  # Disable refine button when clearing
        self.statusBar().showMessage("Results cleared")
    
    def start_chat(self):
        if not self.matches:
            self.show_error("No search results to discuss")
            return
        
        question = self.chat_input.toPlainText().strip()
        prompt = self.prompt_input.text().strip()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.chat_button.setEnabled(False)
        
        # Start chat thread
        self.chat_thread = ChatThread(self, self.matches, prompt, question)
        self.chat_thread.start()
    
    @Slot(str)
    def update_chat(self, response):
        self.chat_output.setPlainText(response)
        self.progress_bar.setVisible(False)
        self.chat_button.setEnabled(True)
        self.chat_action.setEnabled(True)
        self.statusBar().showMessage("Chat response received")
    
    @Slot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        self.chat_button.setEnabled(True if self.matches else False)
    
    def show_api_key_dialog(self, provider):
        """Show dialog to input API key"""
        current_key = getattr(self, f"{provider}_key", '')
        title = "Set Anthropic API Key" if provider == "anthropic" else "Set OpenAI API Key"
        
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
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect buttons
        save_btn.clicked.connect(lambda: self.save_api_key(provider, key_input.text(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
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
                    (lambda: os.path.exists("/Applications/Visual Studio Code.app"), 
                     lambda: subprocess.run(["open", "-a", "Visual Studio Code", "--args", "-g", f"{file_path}:{line_num}"])),
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

    def stop_search(self):
        """Stop the currently running search"""
        if self.search_thread and self.search_thread.is_alive():
            # Request the search thread to stop
            self.search_thread.stop()
            
            # Update UI
            self.results_buffer.add("\n\n[Search stopped by user]\n")
            self.stop_action.setEnabled(False)
            self.search_button.setEnabled(True)
            self.statusBar().showMessage("Search stopped")
            
            # No need to reset stdout here as that will be done in search_complete
            # when the thread finishes

    def clear_file_cache(self):
        """Clear the file cache to force directory re-scan on next search"""
        aisearch.clear_file_cache()
        self.statusBar().showMessage("File cache cleared")

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