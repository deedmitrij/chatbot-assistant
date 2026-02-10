import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FAQUpdateHandler(FileSystemEventHandler):

    def __init__(self, file_path, sync_callback):
        self.file_path = os.path.abspath(file_path)
        self.sync_callback = sync_callback
        self.last_triggered = 0  # Time of last synchronization start
        self.debounce_interval = 2 # Interval in seconds

    def on_modified(self, event):
        if event.is_directory:
            return

        current_event_path = os.path.abspath(event.src_path)

        if current_event_path == self.file_path:
            current_time = time.time()

            # If less than 2 seconds have passed since the last triggering, ignore it
            if current_time - self.last_triggered < self.debounce_interval:
                return

            print(f"📝 Detect changes in {self.file_path}. Triggering sync...")

            # Update the timestamp BEFORE sleeping and calling the function
            self.last_triggered = current_time

            # Short pause so that the file has time to be completely written to the disk
            time.sleep(1)
            self.sync_callback()


def start_faq_watcher(file_path, sync_callback):
    event_handler = FAQUpdateHandler(file_path, sync_callback)
    observer = Observer()
    watch_dir = os.path.dirname(os.path.abspath(file_path))
    observer.schedule(event_handler, path=watch_dir, recursive=False)
    observer.start()
    return observer
