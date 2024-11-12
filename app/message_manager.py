from datetime import datetime
from typing import List, Optional
import queue
import threading

class MessageManager:
    def __init__(self, max_messages: int = 100):
        self._messages: List[str] = []
        self._max_messages = max_messages
        self._message_queue = queue.Queue()
        self._lock = threading.Lock()

    def add_message(self, message: str, message_type: str = "INFO") -> None:
        """Add a new message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{message_type}] {message}"
        
        with self._lock:
            self._messages.append(formatted_message)
            if len(self._messages) > self._max_messages:
                self._messages.pop(0)

    def add_success(self, message: str) -> None:
        """Add a success message."""
        self.add_message(message, "SUCCESS")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.add_message(message, "WARNING")

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.add_message(message, "ERROR")

    def get_messages(self) -> str:
        """Get all messages as a single string."""
        with self._lock:
            return "\n".join(self._messages)

    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self._messages.clear()
