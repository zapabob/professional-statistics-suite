
import os
from dotenv import load_dotenv
import ollama
import openai
import anthropic
import google.generativeai as genai

load_dotenv()

class ModelManager:
    def __init__(self):
        self.models = {
            "ollama": self.get_ollama_models(),
            "openai": {
                "廉価モデル": "gpt-3.5-turbo",
                "高機能モデル": "gpt-4"
            },
            "claude": {
                "廉価モデル": "claude-2.1",
                "高機能モデル": "claude-3-opus-20240229"
            },
            "gemini": {
                "廉価モデル": "gemini-1.0-pro",
                "高機能モデル": "gemini-1.5-pro-latest"
            }
        }
        self.selected_model = None
        self.selected_provider = None

        # APIキーの設定
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def get_ollama_models(self):
        try:
            return [model['name'] for model in ollama.list()['models']]
        except Exception as e:
            print(f"Ollama is not running or accessible: {e}")
            return []

    def select_model(self, provider, model_name):
        self.selected_provider = provider
        if provider == "ollama":
            self.selected_model = model_name
        else:
            self.selected_model = self.models[provider][model_name]

    def generate(self, prompt):
        if not self.selected_model:
            raise ValueError("No model selected.")

        if self.selected_provider == "ollama":
            response = ollama.chat(model=self.selected_model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        elif self.selected_provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.selected_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif self.selected_provider == "claude":
            message = self.claude_client.messages.create(
                model=self.selected_model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content
        elif self.selected_provider == "gemini":
            model = genai.GenerativeModel(self.selected_model)
            response = model.generate_content(prompt)
            return response.text
