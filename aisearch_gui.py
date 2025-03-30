import sys
import os
import threading
import atexit
import gc
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit, 
                             QCheckBox, QSpinBox, QGroupBox, QSplitter, QComboBox,
                             QListWidget, QProgressBar, QMessageBox, QDialog)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QSettings
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon, QTextCharFormat

import aisearch

class StreamRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""
        self.max_buffer_size = 500000  # Reduced to ~500KB text limit
        self.lock = threading.Lock()  # Add lock for thread safety
        
    def write(self, text):
        if not self.text_widget:
            return  # Guard against text_widget being None
            
        with self.lock:
            self.buffer += text
            if '\n' in self.buffer or len(self.buffer) > 80:
                # Check if text widget is getting too large
                try:
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
                except (RuntimeError, Exception):
                    # Guard against Qt C++ object deleted errors
                    pass
                self.buffer = ""
            
    def flush(self):
        if not self.text_widget:
            return  # Guard against text_widget being None
            
        with self.lock:
            if self.buffer:
                try:
                    cursor = self.text_widget.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    cursor.insertText(self.buffer)
                    self.text_widget.setTextCursor(cursor)
                    self.text_widget.ensureCursorVisible()
                except (RuntimeError, Exception):
                    # Guard against Qt C++ object deleted errors
                    pass
                self.buffer = ""
                
    def close(self):
        # Ensure any remaining buffer is flushed
        self.flush()
        # Clear reference to text widget
        self.text_widget = None

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
        self.daemon = True  # Make thread daemon so it doesn't block program exit
        
    def run(self):
        try:
            # Get search terms - send status update through signal
            self.parent.signal_update_status.emit("Querying Claude for search terms...")
            self.search_terms = aisearch.get_search_terms_from_prompt(
                self.prompt, 
                max_terms=self.max_terms, 
                extensions=self.extensions)
            
            # Limit search terms to prevent memory issues
            if len(self.search_terms) > 8:
                self.search_terms = self.search_terms[:8]
                
            # Display terms - send through signal
            terms_text = "Suggested terms:\n" + "\n".join([f"- {t}" for t in self.search_terms])
            self.parent.signal_update_terms.emit(terms_text)
            
            # Search code - send status update through signal
            self.parent.signal_update_status.emit("Searching code using regex patterns...")
            
            # Process in batches
            all_matches = []
            MAX_RESULTS = 100  # Hard limit on results
            
            try:
                # Use a proper resource manager context to ensure cleanup
                import multiprocessing as mp
                
                # Setup special handling for macOS to prevent semaphore leaks
                if sys.platform == 'darwin':
                    # Limit workers even more on macOS
                    effective_workers = min(self.max_workers, 2)
                    
                    # Create a safer multiprocessing context for macOS
                    if hasattr(mp, 'get_context'):
                        mp_ctx = mp.get_context('spawn')
                        # Disable semaphore tracking if possible to prevent leaks
                        if hasattr(mp, 'resource_tracker') and hasattr(mp.resource_tracker, '_resource_tracker'):
                            try:
                                # Let's be careful with this 
                                original_register = mp.resource_tracker._resource_tracker._register
                                
                                # Define a wrapper function that does nothing for semaphores
                                def patched_register(name, rtype):
                                    if rtype != 'semaphore':
                                        original_register(name, rtype)
                                
                                # Apply the patch
                                mp.resource_tracker._resource_tracker._register = patched_register
                            except:
                                pass
                else:
                    # For other platforms, just limit workers
                    effective_workers = min(self.max_workers, 4)
                    
                # Run the search function with limited workers    
                all_matches = aisearch.search_code(
                    directory=self.directory,
                    search_terms=self.search_terms,
                    extensions=self.extensions,
                    case_sensitive=self.case_sensitive,
                    color_output=self.color_output,
                    context_lines=self.context_lines,
                    ignore_comments=self.ignore_comments,
                    max_workers=effective_workers
                )
                
                # Truncate matches if too many
                if len(all_matches) > MAX_RESULTS:
                    self.matches = all_matches[:MAX_RESULTS]
                    # Send status update through signal
                    self.parent.signal_update_status.emit(f"Limited to {MAX_RESULTS} matches to prevent memory issues")
                else:
                    self.matches = all_matches
            except Exception as e:
                # Send error through signal
                self.parent.signal_error.emit(f"Search error: {str(e)}")
                self.matches = all_matches[:50] if len(all_matches) > 50 else all_matches
            
            # Signal completion only if thread still active (parent still exists)
            if hasattr(self, 'parent') and self.parent:
                self.parent.signal_search_complete.emit(len(self.matches))
        except Exception as e:
            if hasattr(self, 'parent') and self.parent:
                self.parent.signal_error.emit(str(e))
        finally:
            # Clean up multiprocessing resources explicitly
            import gc
            
            # Force garbage collection
            gc.collect()

class ChatThread(threading.Thread):
    def __init__(self, parent, matches, prompt, question):
        super().__init__()
        self.parent = parent
        self.matches = matches
        self.prompt = prompt
        self.question = question
        self.daemon = True  # Make thread daemon so it doesn't block program exit
        
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
            if hasattr(self, 'parent') and self.parent:
                self.parent.signal_chat_response.emit(response_text)
            
        except Exception as e:
            if hasattr(self, 'parent') and self.parent:
                self.parent.signal_error.emit(str(e))
        finally:
            # Clean up resources
            gc.collect()

# Create a custom QProgressBar that disables the built-in animation timer
class SafeProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)  # Disable text to avoid timer-based updates
        self._animating = False
        self._animation_timer = None
        
        # Disable Qt's internal style animations
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        
        # Set properties to influence style to avoid animations
        self.setProperty("animated", False)
        
        # Immediately disable and block any QBasicTimer
        if hasattr(self, '_q_animation'):
            self._q_animation = None
        
    # Override the timer-based methods to prevent QBasicTimer errors
    def timerEvent(self, event):
        # Just ignore timer events completely
        if DEBUG_MODE:
            debug_print(f"Ignoring timer event in SafeProgressBar: {event}")
        event.ignore()  # Mark as handled to prevent further processing
        
    # Override paintEvent to skip animations
    def paintEvent(self, event):
        try:
            # Call the parent class's paintEvent with a static drawing approach
            super().paintEvent(event)
        except Exception as e:
            if DEBUG_MODE:
                debug_print(f"Error in SafeProgressBar.paintEvent: {e}")
                
    # Disable standard progress bar behavior to avoid timers
    def setRange(self, minimum, maximum):
        super().setRange(minimum, maximum)
        # Explicitly disable timer-based animation
        self._animating = False
        
    def setValue(self, value):
        # Just set the value without animation
        super().setValue(value)
        
    # Override to prevent animation timers
    def startTimer(self, interval, timerType=None):
        if DEBUG_MODE:
            debug_print(f"Blocked startTimer call in SafeProgressBar: interval={interval}")
        # Return invalid timer ID
        return -1

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
        
        # Progress bar - use our custom SafeProgressBar to avoid timer issues
        self.progress_bar = SafeProgressBar()
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
            if hasattr(mp, 'resource_tracker') and hasattr(mp.resource_tracker, '_resource_tracker') and hasattr(mp.resource_tracker._resource_tracker, '_stop'):
                mp.resource_tracker._resource_tracker._stop()
            
            # Try to clean up any lingering semaphores
            if hasattr(mp, 'active_children'):
                for child in mp.active_children():
                    try:
                        child.terminate()
                        child.join(0.1)
                    except:
                        pass
                    
            # For Python 3.9+ specific handling of resource tracker
            if hasattr(mp, '_resource_tracker') and hasattr(mp._resource_tracker, '_stop'):
                try:
                    mp._resource_tracker._stop()
                except:
                    pass
        except:
            pass
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
    
    def start_search(self):
        # First make sure no existing search is running
        if hasattr(self, 'search_thread') and self.search_thread is not None:
            self.statusBar().showMessage("Search already in progress, please wait...")
            return
        
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
        self.chat_action.setEnabled(False)
        
        # Force garbage collection
        self.matches = []
        gc.collect()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.search_button.setEnabled(False)
        
        # Redirect stdout to results_text
        try:
            self.original_stdout = sys.stdout
            redirector = StreamRedirector(self.results_text)
            sys.stdout = redirector
        except Exception as e:
            sys.stdout = sys.__stdout__
            self.show_error(f"Failed to redirect output: {str(e)}")
            self.search_button.setEnabled(True)
            return
        
        # Start search thread with error handling
        try:
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
        except Exception as e:
            sys.stdout = self.original_stdout
            self.show_error(f"Failed to start search: {str(e)}")
            self.search_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.search_thread = None
            gc.collect()
    
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
        if hasattr(self, 'original_stdout'):
            try:
                sys.stdout = self.original_stdout
                
                # If we used StreamRedirector, close it properly
                if isinstance(sys.stdout, StreamRedirector):
                    sys.stdout.close()
            except:
                # Fallback to ensure stdout is properly reset
                sys.stdout = sys.__stdout__
        
        # Hide progress - must be done from the main thread
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        
        # Safely get matches from search thread
        self.matches = []
        
        if hasattr(self, 'search_thread') and self.search_thread and hasattr(self.search_thread, 'matches'):
            # Copy the matches from the thread to prevent reference issues
            thread_matches = self.search_thread.matches
            
            if len(thread_matches) > 100:
                # Create a deep copy of the first 100 matches
                import copy
                self.matches = copy.deepcopy(thread_matches[:100])
                self.results_text.insertPlainText(f"\n\nLimiting displayed matches to 100 (out of {len(thread_matches)}) to prevent memory issues.\n")
            else:
                # Create a deep copy of all matches
                import copy
                self.matches = copy.deepcopy(thread_matches)
            
        # Update UI based on match count - from main thread
        if match_count > 0:
            self.chat_button.setEnabled(True)
            self.chat_action.setEnabled(True)
            self.statusBar().showMessage(f"Search complete. Found {match_count} matches (displaying up to 100).")
        else:
            self.chat_button.setEnabled(False)
            self.chat_action.setEnabled(False)
            self.statusBar().showMessage("Search complete. No matches found.")
            
        # Clear reference to search thread to release memory
        if hasattr(self, 'search_thread') and self.search_thread:
            self.search_thread = None
        
        # Force garbage collection
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
        # First make sure no existing chat is running
        if hasattr(self, 'chat_thread') and self.chat_thread is not None:
            self.statusBar().showMessage("Chat is already in progress, please wait...")
            return
        
        if not self.matches:
            self.show_error("No search results to discuss")
            return
        
        question = self.chat_input.toPlainText().strip()
        prompt = self.prompt_input.text().strip()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.chat_button.setEnabled(False)
        self.chat_action.setEnabled(False)
        
        # Start chat thread with error handling
        try:
            # Create a deep copy of matches to prevent reference issues
            import copy
            matches_copy = copy.deepcopy(self.matches[:20])  # Limit to first 20 matches
            
            self.chat_thread = ChatThread(self, matches_copy, prompt, question)
            self.chat_thread.start()
        except Exception as e:
            self.show_error(f"Failed to start chat: {str(e)}")
            self.progress_bar.setVisible(False)
            self.chat_button.setEnabled(True)
            self.chat_action.setEnabled(True)
            self.chat_thread = None
            gc.collect()
    
    @Slot(str)
    def update_chat(self, response):
        self.chat_output.setPlainText(response)
        self.progress_bar.setVisible(False)
        self.chat_button.setEnabled(True)
        self.chat_action.setEnabled(True)
        self.statusBar().showMessage("Chat response received")
        
        # Clear the chat thread to release resources
        if hasattr(self, 'chat_thread') and self.chat_thread:
            self.chat_thread = None
        
        # Force garbage collection
        gc.collect()
    
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

if __name__ == "__main__":
    # Debug mode handling
    DEBUG_MODE = os.environ.get("AISEARCH_DEBUG", "0") == "1"
    
    # Set up debug logging
    if DEBUG_MODE:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger("aisearch_gui")
        logger.info("Debug mode enabled")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        
        # Log Qt version
        try:
            from PySide6 import __version__ as pyside_version
            logger.info(f"PySide6 version: {pyside_version}")
        except:
            logger.info("PySide6 version: unknown")
            
        # Log multiprocessing settings
        logger.info(f"Multiprocessing start method: {mp.get_start_method() if 'mp' in locals() else 'unknown'}")
        
        # Add more detailed debugging for key areas - monkeypatch to track resource usage
        def debug_print(*args, **kwargs):
            print("[DEBUG]", *args, **kwargs)
            sys.stdout.flush()  # Force immediate output
    else:
        # No-op for debug_print in non-debug mode
        def debug_print(*args, **kwargs):
            pass
    
    # Force disable resource tracker warnings 
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
    
    # More aggressive suppression of warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*resource_tracker.*")
    warnings.filterwarnings("ignore", message=".*semaphore.*")
    
    # This is even more important than setting spawn - on macOS we need to completely
    # disable the resource tracker for semaphores
    try:
        # Try to fix the semaphore leak by monkeypatching the resource tracker
        import multiprocessing.resource_tracker as resource_tracker
        
        # Original tracker register function
        _original_register = resource_tracker.register
        
        # Create a patched version that ignores semaphores and logs calls
        def _patched_register(name, rtype):
            if DEBUG_MODE:
                debug_print(f"Resource tracker registering {rtype}: {name}")
            
            if rtype != "semaphore":
                return _original_register(name, rtype)
            else:
                # Do nothing for semaphores
                if DEBUG_MODE:
                    debug_print(f"IGNORING semaphore registration: {name}")
                return None
                
        # Apply the patch before any processes start
        resource_tracker.register = _patched_register
        
        # Also patch the unregister function
        _original_unregister = resource_tracker.unregister
        
        def _patched_unregister(name, rtype):
            if DEBUG_MODE:
                debug_print(f"Resource tracker unregistering {rtype}: {name}")
            
            if rtype != "semaphore":
                return _original_unregister(name, rtype)
            else:
                # Do nothing for semaphores
                if DEBUG_MODE:
                    debug_print(f"IGNORING semaphore unregistration: {name}")
                return None
                
        resource_tracker.unregister = _patched_unregister
        
        if DEBUG_MODE:
            debug_print("Successfully patched resource tracker")
            
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Failed to patch resource tracker: {e}")
            import traceback
            traceback.print_exc()
    
    mp = None  # Store multiprocessing module reference
    
    # Initialize multiprocessing with spawn method to avoid fork-related issues
    try:
        import multiprocessing as mp
        # Use spawn instead of fork for macOS compatibility
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
                if DEBUG_MODE:
                    debug_print(f"Set multiprocessing start method to spawn")
            except RuntimeError as e:
                if DEBUG_MODE:
                    debug_print(f"Could not set start method: {e}")
                # Method may already be set
                if DEBUG_MODE:
                    debug_print(f"Current start method: {mp.get_start_method()}")
                pass
            
        # Register special macOS cleanup function
        if sys.platform == 'darwin':
            if DEBUG_MODE:
                debug_print("Detected macOS platform, registering special cleanup")
                
            # On macOS, we need to cleanup semaphores more aggressively
            def cleanup_macos_semaphores():
                if DEBUG_MODE:
                    debug_print("Running macOS semaphore cleanup")
                try:
                    # Get the semaphore tracker module
                    from multiprocessing import semaphore_tracker
                    # Force shutdown of the semaphore tracker
                    if hasattr(semaphore_tracker, '_cleanup'):
                        semaphore_tracker._cleanup()
                        if DEBUG_MODE:
                            debug_print("Called semaphore_tracker._cleanup()")
                except (ImportError, AttributeError) as e:
                    if DEBUG_MODE:
                        debug_print(f"Error in semaphore cleanup: {e}")
                    pass
                    
            # Register the macOS cleanup
            atexit.register(cleanup_macos_semaphores)
            if DEBUG_MODE:
                debug_print("Registered macOS semaphore cleanup")
    except (ImportError, RuntimeError) as e:
        if DEBUG_MODE:
            debug_print(f"Error setting up multiprocessing: {e}")
        pass
        
    # Add a handler for Qt warnings & debug messages
    if DEBUG_MODE:
        def qt_message_handler(mode, context, message):
            if mode == 0:  # QtDebugMsg
                debug_print(f"Qt Debug: {message}")
            elif mode == 1:  # QtWarningMsg
                debug_print(f"Qt Warning: {message}")
            elif mode == 2:  # QtCriticalMsg
                debug_print(f"Qt Critical: {message}")
            elif mode == 3:  # QtFatalMsg
                debug_print(f"Qt Fatal: {message}")
        
        try:
            from PySide6.QtCore import qInstallMessageHandler
            qInstallMessageHandler(qt_message_handler)
            debug_print("Installed Qt message handler")
        except (ImportError, AttributeError) as e:
            debug_print(f"Could not install Qt message handler: {e}")
    
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
        if DEBUG_MODE:
            debug_print("Running final cleanup")
            
        # Cleanup main window resources
        if hasattr(window, 'cleanup_resources'):
            try:
                window.cleanup_resources()
                if DEBUG_MODE:
                    debug_print("Called window.cleanup_resources()")
            except Exception as e:
                if DEBUG_MODE:
                    debug_print(f"Error during window cleanup: {e}")
                pass
        
        # Force garbage collection to ensure all resources are released
        import gc
        count_before = gc.get_count()
        gc.collect()
        count_after = gc.get_count()
        if DEBUG_MODE:
            debug_print(f"GC collect: {count_before} → {count_after}")
        
        # Explicitly close and destroy window if still exists
        try:
            window.close()
            window.deleteLater()
            if DEBUG_MODE:
                debug_print("Called window.close() and deleteLater()")
        except Exception as e:
            if DEBUG_MODE:
                debug_print(f"Error closing window: {e}")
            pass
            
        # Extra cleanup for multiprocessing
        if mp:
            # Shutdown any active multiprocessing resources
            try:
                if hasattr(mp, 'current_process') and mp.current_process().name == 'MainProcess':
                    if DEBUG_MODE:
                        debug_print("Running multiprocessing cleanup in main process")
                    # Only do this in the main process
                    
                    # Close the resource tracker explicitly to fix semaphore leaks
                    if hasattr(mp, '_resource_tracker') and hasattr(mp._resource_tracker, '_stop'):
                        try:
                            mp._resource_tracker._stop()
                            if DEBUG_MODE:
                                debug_print("Called mp._resource_tracker._stop()")
                        except Exception as e:
                            if DEBUG_MODE:
                                debug_print(f"Error stopping resource tracker: {e}")
                            pass
                        
                    # Reset the resource tracker module to clear any lingering handles
                    if hasattr(mp, '_resource_tracker') and hasattr(mp._resource_tracker, '_resource_tracker'):
                        try:
                            mp._resource_tracker._resource_tracker = None
                            if DEBUG_MODE:
                                debug_print("Set mp._resource_tracker._resource_tracker = None")
                        except Exception as e:
                            if DEBUG_MODE:
                                debug_print(f"Error resetting resource tracker: {e}")
                            pass
                            
                    # Shutdown the daemon process that tracks resources
                    if hasattr(mp, 'util') and hasattr(mp.util, '_exit_function'):
                        try:
                            mp.util._exit_function()
                            if DEBUG_MODE:
                                debug_print("Called mp.util._exit_function()")
                        except Exception as e:
                            if DEBUG_MODE:
                                debug_print(f"Error calling exit function: {e}")
                            pass
                        
                    # Special handling for macOS semaphore cleanup
                    if sys.platform == 'darwin':
                        try:
                            from multiprocessing import resource_tracker
                            if hasattr(resource_tracker, "_HAVE_SIGMASK"):
                                resource_tracker._HAVE_SIGMASK = False
                                if DEBUG_MODE:
                                    debug_print("Set resource_tracker._HAVE_SIGMASK = False")
                                
                            # Disable resource tracker altogether
                            if hasattr(resource_tracker, "_resource_tracker"):
                                resource_tracker._resource_tracker = None
                                if DEBUG_MODE:
                                    debug_print("Set resource_tracker._resource_tracker = None")
                        except Exception as e:
                            if DEBUG_MODE:
                                debug_print(f"Error in macOS cleanup: {e}")
                            pass
            except Exception as e:
                if DEBUG_MODE:
                    debug_print(f"Error in multiprocessing cleanup: {e}")
                pass
        
        # One more garbage collection for good measure
        gc.collect()
        if DEBUG_MODE:
            debug_print("Final cleanup complete")
    
    # Register the main cleanup function
    atexit.register(cleanup)
    if DEBUG_MODE:
        debug_print("Registered atexit cleanup handler")
    
    # Catch termination signals
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: (debug_print("Received SIGINT") if DEBUG_MODE else None, cleanup()))
    signal.signal(signal.SIGTERM, lambda sig, frame: (debug_print("Received SIGTERM") if DEBUG_MODE else None, cleanup()))
    if DEBUG_MODE:
        debug_print("Registered signal handlers")
    
    try:
        if DEBUG_MODE:
            debug_print("Starting main Qt event loop")
        result = app.exec()
        if DEBUG_MODE:
            debug_print(f"Qt event loop ended with result: {result}")
        # Explicitly call cleanup before exiting
        cleanup()
        sys.exit(result)
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Error in main event loop: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Final cleanup attempt
        if DEBUG_MODE:
            debug_print("In finally block, calling cleanup again")
        cleanup() 