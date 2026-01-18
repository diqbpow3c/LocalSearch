import time,logging,os
from PySide6.QtCore import (QThread, Signal)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class IndexRebuilderThread(QThread):
    """
    A QThread for rebuilding the search index in the background.
    Emits signals to update the UI status.
    """

    rebuild_started = Signal()
    rebuild_finished = Signal(bool)  # True for success, False for failure

    def __init__(self, searcher_instance):
        super().__init__()
        self.searcher = searcher_instance

    def run(self):
        """
        The main execution method for the thread.
        Calls the rebuild_index method of the Searcher instance.
        """
        self.rebuild_started.emit()
        success = self.searcher.rebuild_index()
        if success:
            logger.info(
                "Index rebuild successful, now saving index components in background thread..."
            )
            save_success = self.searcher.save_index_components()  # Call save here
            if not save_success:
                logger.info("Warning: Index components failed to save after rebuild.")
                success = False
        self.rebuild_finished.emit(success)

    def stop(self):
        self.terminate()

class ChangeTracker(FileSystemEventHandler):
    def __init__(self, watch_files, watch_folders):
        # Convert to absolute paths for reliable comparison
        self.watch_files = {os.path.abspath(f) for f in watch_files}
        self.watch_folders = {os.path.abspath(f) for f in watch_folders}
        self.has_changes = False

    def on_modified(self, event):
        if event.is_directory:
            return

        current_path = os.path.abspath(event.src_path)

        # Check 1: Is it one of our specific files?
        if current_path in self.watch_files:
            self.has_changes = True
            return

        # Check 2: Is it inside one of our watched folders?
        for folder in self.watch_folders:
            if current_path.startswith(folder):
                self.has_changes = True
                break

class FileChangeMonitorThread(QThread):
    """
    A QThread for continuously monitoring indexed files and folders for changes.
    Emits a signal if changes are detected.
    """

    changes_detected = Signal()
    # Signal to update the status bar with monitoring status
    monitoring_status_update = Signal(str)

    def __init__(self, searcher_instance, interval=60):
        super().__init__()
        self.searcher_instance = searcher_instance
        self.interval = interval  # Check interval in seconds
        self._running = True
        self.handler = ChangeTracker(self.searcher_instance.file_paths, self.searcher_instance.folder_paths)
        self.observer = Observer()
        # Determine unique parent directories to watch
        # Watchdog needs to watch the folder containing a file to see file changes
        self.directories_to_watch = set(self.handler.watch_folders)
        for f in self.handler.watch_files:
            self.directories_to_watch.add(os.path.dirname(f))
        for directory in self.directories_to_watch:
            if os.path.exists(directory):
                # recursive=True for the folders, False for parent dirs of files
                # But for simplicity, True is safer if folders are nested
                self.observer.schedule(self.handler, directory, recursive=True)
        self.observer.start()
        self.first_run_since_app_start = True

    def run(self):
        """
        Continuously monitors files for changes.
        """

        self.monitoring_status_update.emit("Monitoring file changes...")
        time.sleep(20) # wait for some time before the first automatic check
        while self._running:
            logger.info(f"Monitoring check")
            if self.first_run_since_app_start:
                self.first_run_since_app_start = False
                # check for file changes during the period that the app was not running
                current_mtimes = self.searcher_instance.get_current_mtimes()
                last_saved_mtimes = self.searcher_instance.get_saved_mtimes()
                if current_mtimes!=last_saved_mtimes:
                    logger.info("Changes detected for the period that the app was closed. Starting re-index...")
                    self.handler.has_changes = False
                    self.changes_detected.emit()

            elif self.handler.has_changes:
                logger.info("Changes detected in 30s window. Starting re-index...")
                self.handler.has_changes = False
                self.changes_detected.emit()

            time.sleep(self.interval)

    def stop(self):
        """Stops the monitoring thread."""
        self._running = False
        self.observer.stop()
        self.observer.join()
        self.terminate()
