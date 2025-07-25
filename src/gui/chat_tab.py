import tkinter as tk
from tkinter import ttk, scrolledtext
from src.llm.model_manager import ModelManager

class ChatTab(ttk.Frame):
    def __init__(self, parent, model_manager: ModelManager):
        super().__init__(parent)
        self.model_manager = model_manager

        self.create_widgets()

    def create_widgets(self):
        # Provider selection
        provider_frame = ttk.LabelFrame(self, text="Provider")
        provider_frame.pack(fill=tk.X, padx=5, pady=5)

        self.provider_var = tk.StringVar()
        self.provider_menu = ttk.OptionMenu(provider_frame, self.provider_var, "Select Provider", *self.model_manager.models.keys(), command=self.update_model_menu)
        self.provider_menu.pack(side=tk.LEFT, padx=5, pady=5)

        # Model selection
        self.model_var = tk.StringVar()
        self.model_menu = ttk.OptionMenu(provider_frame, self.model_var, "Select Model")
        self.model_menu.pack(side=tk.LEFT, padx=5, pady=5)

        # Prompt input
        prompt_frame = ttk.LabelFrame(self, text="Prompt")
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)

        self.prompt_entry = ttk.Entry(prompt_frame)
        self.prompt_entry.pack(fill=tk.X, padx=5, pady=5)

        # Send button
        send_button = ttk.Button(prompt_frame, text="Send", command=self.send_prompt)
        send_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Response display
        response_frame = ttk.LabelFrame(self, text="Response")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.response_text = scrolledtext.ScrolledText(response_frame, state='disabled')
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_model_menu(self, selected_provider):
        models = self.model_manager.models.get(selected_provider, {})
        self.model_menu['menu'].delete(0, 'end')
        for model_name in models:
            self.model_menu['menu'].add_command(label=model_name, command=lambda value=model_name: self.model_var.set(value))
        if models:
            self.model_var.set(list(models.keys())[0])

    def send_prompt(self):
        provider = self.provider_var.get()
        model = self.model_var.get()
        prompt = self.prompt_entry.get()

        if not provider or not model or not prompt:
            self.response_text.configure(state='normal')
            self.response_text.insert(tk.END, "Error: Please select a provider, model, and enter a prompt.\n")
            self.response_text.configure(state='disabled')
            return

        self.model_manager.select_model(provider, model)
        response = self.model_manager.generate(prompt)

        self.response_text.configure(state='normal')
        self.response_text.insert(tk.END, f"> {prompt}\n")
        self.response_text.insert(tk.END, f"{response}\n\n")
        self.response_text.configure(state='disabled')
        self.prompt_entry.delete(0, tk.END)
