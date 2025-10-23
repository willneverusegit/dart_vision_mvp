"""
Hotkey Handler - Command pattern-based keyboard input handling.

Provides clean, maintainable hotkey management with:
- Categorical organization
- Easy binding/rebinding
- Help text generation
- Logging integration
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class HotkeyCategory(Enum):
    """Categories for organizing hotkeys."""
    NAVIGATION = auto()      # Arrow keys, jkli movement
    OVERLAY = auto()         # Overlay modes, rotation, scale
    GAME = auto()            # Game controls
    DEBUG = auto()           # Debug toggles
    CALIBRATION = auto()     # Calibration & Hough
    MOTION_TUNING = auto()   # Live motion parameter tuning
    PRESETS = auto()         # Detector presets
    HEATMAP = auto()         # Heatmap toggles
    SYSTEM = auto()          # Quit, screenshot, help


@dataclass
class HotkeyAction:
    """
    Represents a single hotkey action.

    Attributes:
        key: Key code or character (e.g., 'q', VK_LEFT)
        callback: Function to execute when key is pressed
        description: Human-readable description for help text
        category: Category this hotkey belongs to
        enabled: Whether this hotkey is currently active
    """
    key: Any
    callback: Callable[[], None]
    description: str
    category: HotkeyCategory
    enabled: bool = True


class HotkeyHandler:
    """
    Centralized hotkey management using Command pattern.

    Features:
    - Register hotkeys with callbacks
    - Organize by category
    - Enable/disable hotkeys dynamically
    - Auto-generate help text
    - Logging support
    """

    def __init__(self):
        """Initialize hotkey handler."""
        self.actions: Dict[Any, HotkeyAction] = {}
        self.logger = logging.getLogger("HotkeyHandler")

    def register(
        self,
        key: Any,
        callback: Callable[[], None],
        description: str,
        category: HotkeyCategory = HotkeyCategory.SYSTEM,
        enabled: bool = True
    ):
        """
        Register a hotkey action.

        Args:
            key: Key code or character (e.g., ord('q'), VK_LEFT)
            callback: Function to call when key is pressed
            description: Description for help text
            category: Category for organization
            enabled: Whether hotkey is active
        """
        action = HotkeyAction(key, callback, description, category, enabled)
        self.actions[key] = action
        self.logger.debug(f"Registered hotkey: {self._key_name(key)} -> {description}")

    def register_multiple(self, bindings: List[Tuple[Any, Callable, str, HotkeyCategory]]):
        """
        Register multiple hotkeys at once.

        Args:
            bindings: List of (key, callback, description, category) tuples
        """
        for key, callback, description, category in bindings:
            self.register(key, callback, description, category)

    def handle(self, key: Any) -> bool:
        """
        Handle a key press.

        Args:
            key: Key code from cv2.waitKeyEx or ord()

        Returns:
            True if key was handled, False otherwise
        """
        if key == -1:
            return False

        action = self.actions.get(key)
        if action is None:
            return False

        if not action.enabled:
            self.logger.debug(f"Hotkey {self._key_name(key)} is disabled")
            return False

        try:
            self.logger.debug(f"Executing: {action.description}")
            action.callback()
            return True
        except Exception as e:
            self.logger.error(f"Error executing hotkey {self._key_name(key)}: {e}", exc_info=True)
            return False

    def enable(self, key: Any):
        """Enable a specific hotkey."""
        if key in self.actions:
            self.actions[key].enabled = True

    def disable(self, key: Any):
        """Disable a specific hotkey."""
        if key in self.actions:
            self.actions[key].enabled = False

    def enable_category(self, category: HotkeyCategory):
        """Enable all hotkeys in a category."""
        for action in self.actions.values():
            if action.category == category:
                action.enabled = True

    def disable_category(self, category: HotkeyCategory):
        """Disable all hotkeys in a category."""
        for action in self.actions.values():
            if action.category == category:
                action.enabled = False

    def get_help_text(self) -> Dict[HotkeyCategory, List[Tuple[str, str]]]:
        """
        Generate help text organized by category.

        Returns:
            Dict mapping category to list of (key_name, description) tuples
        """
        help_text: Dict[HotkeyCategory, List[Tuple[str, str]]] = {}

        for action in self.actions.values():
            if action.category not in help_text:
                help_text[action.category] = []

            key_name = self._key_name(action.key)
            help_text[action.category].append((key_name, action.description))

        # Sort each category's entries
        for category in help_text:
            help_text[category].sort(key=lambda x: x[0])

        return help_text

    def print_help(self):
        """Print formatted help text to console."""
        help_text = self.get_help_text()

        print("\n" + "=" * 60)
        print("DART VISION MVP - HOTKEYS")
        print("=" * 60)

        for category in HotkeyCategory:
            if category not in help_text:
                continue

            print(f"\n{category.name}:")
            print("-" * 40)
            for key_name, description in help_text[category]:
                print(f"  {key_name:15} {description}")

        print("=" * 60 + "\n")

    def _key_name(self, key: Any) -> str:
        """
        Convert key code to readable name.

        Args:
            key: Key code

        Returns:
            Human-readable key name
        """
        # Special keys (extended codes)
        special_keys = {
            0x250000: "LEFT",
            0x260000: "UP",
            0x270000: "RIGHT",
            0x280000: "DOWN",
            2424832: "LEFT (OCV)",
            2490368: "UP (OCV)",
            2555904: "RIGHT (OCV)",
            2621440: "DOWN (OCV)",
        }

        if key in special_keys:
            return special_keys[key]

        # ASCII printable characters
        if 32 <= key <= 126:
            char = chr(key)
            # Show shift combinations
            if char.isupper() or char in '!@#$%^&*()_+{}|:"<>?':
                return f"Shift+{char}"
            return char

        # Fallback: hex representation
        return f"0x{key:X}"

    def get_registered_keys(self) -> List[Any]:
        """Get list of all registered key codes."""
        return list(self.actions.keys())

    def unregister(self, key: Any):
        """
        Unregister a hotkey.

        Args:
            key: Key code to unregister
        """
        if key in self.actions:
            del self.actions[key]
            self.logger.debug(f"Unregistered hotkey: {self._key_name(key)}")

    def clear(self):
        """Clear all registered hotkeys."""
        self.actions.clear()
        self.logger.info("Cleared all hotkeys")

    def get_category_keys(self, category: HotkeyCategory) -> List[Any]:
        """
        Get all keys in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of key codes in that category
        """
        return [key for key, action in self.actions.items() if action.category == category]
