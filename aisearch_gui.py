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
                color_output, context_lines, ignore_comments, max_terms, max_workers):
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
        
    def run(self):
        try:
            # Get search terms
            if self.stop_requested:
                self.running = False
                return
                
            self.parent.signal_update_status.emit("Querying Claude for search terms...")
            self.search_terms = aisearch.get_search_terms_from_prompt(
                self.prompt, 
                max_terms=self.max_terms, 
                extensions=self.extensions)
            
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
            MAX_RESULTS = 100  # Hard limit on results
            
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
        
    def run(self):
        try:
            self.parent.signal_update_status.emit("Waiting for Claude's response...")
            
            # Use a simplified version since we can't easily use streaming in the GUI
            client = aisearch.anthropic.Anthropic()
            
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
            
            response = client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=4096,
                temperature=1,
                system=system_message,
                messages=messages
            )
            
            response_text = response.content[0].text
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
        # Enhanced pattern to match more file reference formats
        self.file_match_pattern = re.compile(r'([\/\w\.-]+\.[a-zA-Z0-9]+):(\d+)')
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setStatusTip("Click on file:line references to open in your default editor")
    
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
        if self.file_match_pattern.search(line_text):
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.IBeamCursor)
            
        super().mouseMoveEvent(event)
        
    def mousePressEvent(self, event):
        """Handle mouse press to detect file references"""
        if event.button() == Qt.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QTextCursor.LineUnderCursor)
            line_text = cursor.selectedText()
            
            # Look for file:line pattern
            match = self.file_match_pattern.search(line_text)
            if match:
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
                    file_path = os.path.join(base_dir, file_path)
                
                if os.path.exists(file_path):
                    main_window.open_file_in_editor(file_path, line_number)
                    return
        
        # Call the parent implementation for normal text selection
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
        
        self.chat_action = self.toolbar.addAction("Chat")
        self.chat_action.triggered.connect(self.start_chat)
        self.chat_action.setToolTip("Chat about results")
        self.chat_action.setEnabled(False)
        
        self.toolbar.addSeparator()
        
        api_action = self.toolbar.addAction("API Key")
        api_action.triggered.connect(self.show_api_key_dialog)
        api_action.setToolTip("Set your Anthropic API key")
        
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
        
        # Left options
        left_options = QVBoxLayout()
        self.case_sensitive = QCheckBox("Case Sensitive")
        self.case_sensitive.setChecked(True)
        left_options.addWidget(self.case_sensitive)
        self.include_comments = QCheckBox("Include Comments")
        self.include_comments.setChecked(True)  # Include comments by default
        left_options.addWidget(self.include_comments)
        options_layout.addLayout(left_options)
        
        # Middle options
        middle_options = QVBoxLayout()
        terms_layout = QHBoxLayout()
        terms_layout.addWidget(QLabel("Max Terms:"))
        self.max_terms = QSpinBox()
        self.max_terms.setRange(1, 100)  # Increased max
        self.max_terms.setValue(8)
        terms_layout.addWidget(self.max_terms)
        middle_options.addLayout(terms_layout)
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Workers:"))
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 64)  # Increased max workers
        self.max_workers.setValue(4)
        workers_layout.addWidget(self.max_workers)
        middle_options.addLayout(workers_layout)
        options_layout.addLayout(middle_options)
        
        # Right options
        right_options = QVBoxLayout()
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Context Lines:"))
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 20)  # Increased max context lines
        self.context_lines.setValue(2)
        context_layout.addWidget(self.context_lines)
        right_options.addLayout(context_layout)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.start_search)
        right_options.addWidget(self.search_button)
        options_layout.addLayout(right_options)
        
        config_layout.addLayout(options_layout)
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
        chat_header.addWidget(QLabel("Chat with Claude about results:"))
        self.chat_button = QPushButton("Ask Claude")
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
        
        # API Key (stored encrypted)
        self.api_key = self.settings.value("api_key", "")
        # Set environment variable if API key is available
        if self.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        
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
        
        # Save API key if present
        if hasattr(self, 'api_key') and self.api_key:
            self.settings.setValue("api_key", self.api_key)
    
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
        
        # Stop any running threads
        if hasattr(self, 'search_thread') and self.search_thread is not None:
            # The thread might be accessing multiprocessing resources
            self.search_thread = None
        
        if hasattr(self, 'chat_thread') and self.chat_thread is not None:
            self.chat_thread = None
        
        # Force garbage collection
        gc.collect()
        
        # Explicitly clean multiprocessing resources
        try:
            import multiprocessing as mp
            if hasattr(mp, 'active_children'):
                for child in mp.active_children():
                    child.terminate()
                    child.join(0.1)
        except:
            pass
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
    
    def start_search(self):
        # Validate inputs
        directory = self.dir_input.text().strip()
        prompt = self.prompt_input.text().strip()
        
        if not directory:
            self.show_error("Please select a directory")
            return
        
        if not prompt:
            self.show_error("Please enter a search prompt")
            return
        
        # Check for API key
        if not hasattr(self, 'api_key') or not self.api_key:
            self.show_error("Please set your Anthropic API key first")
            self.show_api_key_dialog()
            return
        
        # Get extensions
        extensions_text = self.ext_input.text().strip()
        extensions = extensions_text.split() if extensions_text else None
        
        # Clear previous results - more aggressive memory cleanup
        self.results_text.clear()
        self.chat_output.clear()
        self.chat_input.clear()
        self.chat_button.setEnabled(False)
        
        # Reset the results buffer
        self.results_buffer = ResultsBuffer()
        
        # Force garbage collection
        self.matches = []
        import gc
        gc.collect()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
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
            self.max_workers.value()
        )
        self.search_thread.start()
        
        # Start the update timer to periodically process the buffer
        self.update_timer.start(250)  # Update UI every 250ms
        
    def process_results_buffer(self):
        """Process accumulated results from the buffer and update UI"""
        if not hasattr(self, 'search_thread') or self.search_thread is None:
            self.update_timer.stop()
            return
            
        # Get buffered content
        buffered_text = self.results_buffer.get_and_clear()
        if buffered_text:
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
        
        # Check if search is still running
        if hasattr(self, 'search_thread') and self.search_thread and not self.search_thread.running:
            # Final update to make sure we catch everything
            buffered_text = self.results_buffer.get_and_clear()
            if buffered_text:
                cursor = self.results_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertText(buffered_text)
                self.results_text.setTextCursor(cursor)
                self.results_text.ensureCursorVisible()
            
            # Stop the timer
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
            self.statusBar().showMessage(f"Search complete. Found {match_count} matches (displaying up to 100).")
        else:
            self.chat_button.setEnabled(False)
            self.chat_action.setEnabled(False)
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
    
    def show_api_key_dialog(self):
        """Show dialog to input API key"""
        current_key = getattr(self, 'api_key', '')
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Anthropic API Key")
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel("Enter your Anthropic API key below. The key will be stored locally.")
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
        save_btn.clicked.connect(lambda: self.save_api_key(key_input.text(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def save_api_key(self, key, dialog):
        """Save API key and close dialog"""
        if key:
            self.api_key = key
            # Set environment variable for immediate use
            os.environ["ANTHROPIC_API_KEY"] = key
            self.saveSettings()
            dialog.accept()
            self.statusBar().showMessage("API Key saved")
        else:
            QMessageBox.warning(dialog, "Error", "API Key cannot be empty")

    def open_file_in_editor(self, file_path, line_number):
        """Open a file at a specific line in the default editor"""
        try:
            system = platform.system()
            line_num = int(line_number)
            
            if system == "Darwin":  # macOS
                # Try to use Visual Studio Code if available
                if os.path.exists("/Applications/Visual Studio Code.app"):
                    subprocess.run(["open", "-a", "Visual Studio Code", "--args", "-g", f"{file_path}:{line_num}"])
                # Try to use Sublime Text if available
                elif os.path.exists("/Applications/Sublime Text.app"):
                    subprocess.run(["open", "-a", "Sublime Text", "--args", f"{file_path}:{line_num}"])
                else:
                    # Fallback to TextEdit (doesn't support line numbers)
                    subprocess.run(["open", "-a", "TextEdit", file_path])
                
                self.statusBar().showMessage(f"Opened {file_path}:{line_number}")
                
            elif system == "Windows":
                # Try to use VSCode if installed
                vscode_path = os.path.expandvars("%LOCALAPPDATA%\\Programs\\Microsoft VS Code\\Code.exe")
                if os.path.exists(vscode_path):
                    subprocess.run([vscode_path, "-g", f"{file_path}:{line_num}"])
                # Try Notepad++ if installed
                elif os.path.exists("C:\\Program Files\\Notepad++\\notepad++.exe"):
                    subprocess.run(["C:\\Program Files\\Notepad++\\notepad++.exe", "-n" + line_number, file_path])
                else:
                    # Fallback to default application
                    os.startfile(file_path)
                
                self.statusBar().showMessage(f"Opened {file_path}:{line_number}")
                
            elif system == "Linux":
                # Try common editors with line number support
                # Try VSCode
                if subprocess.run(["which", "code"], stdout=subprocess.DEVNULL).returncode == 0:
                    subprocess.run(["code", "-g", f"{file_path}:{line_num}"])
                # Try gedit
                elif subprocess.run(["which", "gedit"], stdout=subprocess.DEVNULL).returncode == 0:
                    subprocess.run(["gedit", f"{file_path}+{line_num}"])
                # Try vim in terminal
                elif subprocess.run(["which", "vim"], stdout=subprocess.DEVNULL).returncode == 0:
                    subprocess.run(["x-terminal-emulator", "-e", f"vim +{line_num} {file_path}"])
                else:
                    # Fallback to xdg-open
                    subprocess.run(["xdg-open", file_path])
                
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look and feel
    
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