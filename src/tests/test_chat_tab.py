import unittest
from unittest.mock import MagicMock
import tkinter as tk
from src.gui.chat_tab import ChatTab
from src.llm.model_manager import ModelManager

class TestChatTab(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.model_manager = ModelManager()
        self.chat_tab = ChatTab(self.root, self.model_manager)

    def tearDown(self):
        self.root.destroy()

    def test_initialization(self):
        self.assertIsInstance(self.chat_tab, ChatTab)
        self.assertEqual(self.chat_tab.provider_var.get(), "Select Provider")

    def test_update_model_menu(self):
        self.chat_tab.update_model_menu("openai")
        self.assertEqual(self.chat_tab.model_var.get(), "廉価モデル")

    def test_send_prompt_with_no_selection(self):
        self.chat_tab.send_prompt()
        response = self.chat_tab.response_text.get("1.0", tk.END).strip()
        self.assertIn("Error: Please select a provider, model, and enter a prompt.", response)

    def test_send_prompt(self):
        self.model_manager.generate = MagicMock(return_value="Test response")
        self.chat_tab.provider_var.set("openai")
        self.chat_tab.update_model_menu("openai")
        self.chat_tab.model_var.set("廉価モデル")
        self.chat_tab.prompt_entry.insert(0, "Test prompt")
        self.chat_tab.send_prompt()
        response = self.chat_tab.response_text.get("1.0", tk.END).strip()
        self.assertIn("> Test prompt", response)
        self.assertIn("Test response", response)

if __name__ == '__main__':
    unittest.main()
