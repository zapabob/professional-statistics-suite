import unittest
from unittest.mock import patch, MagicMock
from src.llm.model_manager import ModelManager

class TestModelManager(unittest.TestCase):

    def setUp(self):
        # 環境変数をモック
        self.patcher_openai = patch.dict(os.environ, {'OPENAI_API_KEY': 'test_openai_key'})
        self.patcher_claude = patch.dict(os.environ, {'CLAUDE_API_KEY': 'test_claude_key'})
        self.patcher_gemini = patch.dict(os.environ, {'GEMINI_API_KEY': 'test_gemini_key'})
        self.patcher_openai.start()
        self.patcher_claude.start()
        self.patcher_gemini.start()

        # ollama.list()のモック
        self.mock_ollama_list = patch('ollama.list').start()
        self.mock_ollama_list.return_value = {'models': [
            {'name': 'llama2:latest'},
            {'name': 'mistral:latest'}
        ]}

        # openai.ChatCompletion.createのモック
        self.mock_openai_chat_create = patch('openai.ChatCompletion.create').start()
        self.mock_openai_chat_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI response"))]
        )

        # anthropic.Anthropic().messages.createのモック
        self.mock_claude_messages_create = patch('anthropic.Anthropic').start().return_value.messages.create
        self.mock_claude_messages_create.return_value = MagicMock(content="Claude response")

        # google.generativeai.GenerativeModel().generate_contentのモック
        self.mock_gemini_generate_content = patch('google.generativeai.GenerativeModel').start().return_value.generate_content
        self.mock_gemini_generate_content.return_value = MagicMock(text="Gemini response")

        self.model_manager = ModelManager()

    def tearDown(self):
        patch.stopall()

    def test_get_ollama_models_success(self):
        models = self.model_manager.get_ollama_models()
        self.assertEqual(models, ['llama2:latest', 'mistral:latest'])
        self.mock_ollama_list.assert_called_once()

    def test_get_ollama_models_failure(self):
        self.mock_ollama_list.side_effect = Exception("Ollama not running")
        models = self.model_manager.get_ollama_models()
        self.assertEqual(models, [])

    def test_select_model_ollama(self):
        self.model_manager.select_model("ollama", "llama2:latest")
        self.assertEqual(self.model_manager.selected_provider, "ollama")
        self.assertEqual(self.model_manager.selected_model, "llama2:latest")

    def test_select_model_openai(self):
        self.model_manager.select_model("openai", "廉価モデル")
        self.assertEqual(self.model_manager.selected_provider, "openai")
        self.assertEqual(self.model_manager.selected_model, "gpt-3.5-turbo")

    def test_generate_ollama(self):
        self.model_manager.select_model("ollama", "llama2:latest")
        with patch('ollama.chat') as mock_ollama_chat:
            mock_ollama_chat.return_value = {'message': {'content': 'Ollama chat response'}}
            response = self.model_manager.generate("Hello")
            self.assertEqual(response, "Ollama chat response")
            mock_ollama_chat.assert_called_once_with(model='llama2:latest', messages=[{'role': 'user', 'content': 'Hello'}])

    def test_generate_openai(self):
        self.model_manager.select_model("openai", "廉価モデル")
        response = self.model_manager.generate("Hello")
        self.assertEqual(response, "OpenAI response")
        self.mock_openai_chat_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )

    def test_generate_claude(self):
        self.model_manager.select_model("claude", "廉価モデル")
        response = self.model_manager.generate("Hello")
        self.assertEqual(response, "Claude response")
        self.mock_claude_messages_create.assert_called_once_with(
            model="claude-2.1",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )

    def test_generate_gemini(self):
        self.model_manager.select_model("gemini", "廉価モデル")
        response = self.model_manager.generate("Hello")
        self.assertEqual(response, "Gemini response")
        self.mock_gemini_generate_content.assert_called_once_with("Hello")

    def test_generate_no_model_selected(self):
        with self.assertRaises(ValueError) as cm:
            self.model_manager.generate("Hello")
        self.assertEqual(str(cm.exception), "No model selected.")

if __name__ == '__main__':
    unittest.main()
