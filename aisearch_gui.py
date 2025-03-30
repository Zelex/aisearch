import sys
import os
import threading
import atexit
import gc
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, 
                             QCheckBox, QSpinBox, QGroupBox, QSplitter, QComboBox,
                             QListWidget, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QSettings
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon, QTextCharFormat

import aisearch

class StreamRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        self.max_buffer_size = 1000000  # ~1MB text limit
        
    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer or len(self.buffer) > 80:
            # Check if text widget is getting too large
            if self.text_widget.document().characterCount() > self.max_buffer_size:
                # Truncate the text to prevent memory issues
                cursor = self.text_widget.textCursor()
                cursor.movePosition(QTextCursor.Start)
                cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 200000)  # Remove ~200K chars
                cursor.removeSelectedText()
                self.text_widget.insertPlainText("[...truncated to prevent memory issues...]\n\n")
            
            # Use insertPlainText instead of append to avoid extra newlines
            cursor = self.text_widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(self.buffer)
            self.text_widget.setTextCursor(cursor)
            self.text_widget.ensureCursorVisible()
            self.buffer = ""
            
    def flush(self):
        if self.buffer:
            cursor = self.text_widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(self.buffer)
            self.text_widget.setTextCursor(cursor)
            self.text_widget.ensureCursorVisible()
            self.buffer = ""

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
        self.context_lines = min(context_lines, 3)  # Limit context lines to 3 max
        self.ignore_comments = ignore_comments
        self.max_terms = min(max_terms, 10)  # Limit max terms to 10
        self.max_workers = max_workers
        self.search_terms = []
        self.matches = []
        
    def run(self):
        try:
            # Get search terms
            self.parent.signal_update_status.emit("Querying Claude for search terms...")
            self.search_terms = aisearch.get_search_terms_from_prompt(
                self.prompt, 
                max_terms=self.max_terms, 
                extensions=self.extensions)
            
            # Limit search terms to prevent memory issues
            if len(self.search_terms) > 8:
                self.search_terms = self.search_terms[:8]
                
            # Display terms
            terms_text = "Suggested terms:\n" + "\n".join([f"- {t}" for t in self.search_terms])
            self.parent.signal_update_terms.emit(terms_text)
            
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
                    max_workers=self.max_workers
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
        
        self.toolbar.addSeparator()
        
        clear_action = self.toolbar.addAction("Clear")
        clear_action.triggered.connect(self.clear_results)
        clear_action.setToolTip("Clear all results")
        
        self.toolbar.addSeparator()
        
        self.chat_action = self.toolbar.addAction("Chat")
        self.chat_action.triggered.connect(self.start_chat)
        self.chat_action.setToolTip("Chat about results")
        self.chat_action.setEnabled(False)
        
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
        self.max_terms.setRange(1, 10)  # Limit to 10 max
        self.max_terms.setValue(8)
        terms_layout.addWidget(self.max_terms)
        middle_options.addLayout(terms_layout)
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Workers:"))
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 16)  # Reduced max workers
        self.max_workers.setValue(4)  # Reduced default
        workers_layout.addWidget(self.max_workers)
        middle_options.addLayout(workers_layout)
        options_layout.addLayout(middle_options)
        
        # Right options
        right_options = QVBoxLayout()
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Context Lines:"))
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 3)  # Limit to 3 context lines
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
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
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
        
        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
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
        
        # Checkboxes
        self.case_sensitive.setChecked(self.settings.value("case_sensitive", True, type=bool))
        self.include_comments.setChecked(self.settings.value("include_comments", True, type=bool))
        
        # SpinBoxes
        self.max_terms.setValue(min(self.settings.value("max_terms", 8, type=int), 10))
        self.max_workers.setValue(min(self.settings.value("max_workers", 4, type=int), 16))
        self.context_lines.setValue(min(self.settings.value("context_lines", 2, type=int), 3))
    
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
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.saveSettings()
        
        # Clean up threads and processes before closing
        self.cleanup_resources()
        
        super().closeEvent(event)
    
    def cleanup_resources(self):
        """Clean up resources to prevent leaks"""
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
        
        # Get extensions
        extensions_text = self.ext_input.text().strip()
        extensions = extensions_text.split() if extensions_text else None
        
        # Clear previous results - more aggressive memory cleanup
        self.results_text.clear()
        self.chat_output.clear()
        self.chat_input.clear()
        self.chat_button.setEnabled(False)
        
        # Force garbage collection
        self.matches = []
        import gc
        gc.collect()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
        
        # Redirect stdout to results_text
        self.original_stdout = sys.stdout
        sys.stdout = StreamRedirector(self.results_text)
        
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
        
    @Slot(str)
    def update_status(self, message):
        self.statusBar().showMessage(message)
        
    @Slot(str)
    def update_terms(self, terms_text):
        # Clear first if getting large to prevent memory issues
        if self.results_text.document().characterCount() > 500000:
            self.results_text.clear()
        self.results_text.insertPlainText(terms_text + "\n\n")
    
    @Slot(int)
    def search_complete(self, match_count):
        # Reset stdout
        sys.stdout = self.original_stdout
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        
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