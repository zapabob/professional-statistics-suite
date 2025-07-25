#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GGUF Model Selector GUI Component
GGUFモデル選択GUIコンポーネント

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
import json

class GGUFModelSelector:
    """GGUFモデル選択GUIコンポーネント"""
    
    def __init__(self, parent: tk.Widget, on_model_selected: Optional[Callable] = None):
        self.parent = parent
        self.on_model_selected = on_model_selected
        self.selected_model_path = None
        self.available_models = []
        
        # 設定ファイルパス
        self.config_file = Path("gguf_model_config.json")
        self.last_directory = self._load_last_directory()
        
        self._create_widgets()
        self._load_saved_models()
    
    def _create_widgets(self):
        """ウィジェット作成"""
        # メインフレーム
        self.frame = ttk.LabelFrame(self.parent, text="GGUFモデル選択", padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # モデルディレクトリ選択
        dir_frame = ttk.Frame(self.frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dir_frame, text="モデルディレクトリ:").pack(side=tk.LEFT)
        
        self.dir_var = tk.StringVar(value=self.last_directory or "")
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=50)
        self.dir_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        self.browse_btn = ttk.Button(dir_frame, text="参照", command=self._browse_directory)
        self.browse_btn.pack(side=tk.RIGHT)
        
        # スキャンボタン
        scan_frame = ttk.Frame(self.frame)
        scan_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.scan_btn = ttk.Button(scan_frame, text="GGUFファイルをスキャン", command=self._scan_gguf_files)
        self.scan_btn.pack(side=tk.LEFT)
        
        self.refresh_btn = ttk.Button(scan_frame, text="更新", command=self._refresh_models)
        self.refresh_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # モデルリスト
        list_frame = ttk.Frame(self.frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(list_frame, text="利用可能なGGUFモデル:").pack(anchor=tk.W)
        
        # リストボックスとスクロールバー
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.model_listbox = tk.Listbox(list_container, height=8)
        self.model_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.model_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=scrollbar.set)
        
        # 選択ボタン
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.select_btn = ttk.Button(btn_frame, text="選択", command=self._select_model)
        self.select_btn.pack(side=tk.LEFT)
        
        self.clear_btn = ttk.Button(btn_frame, text="クリア", command=self._clear_selection)
        self.clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # 選択されたモデル表示
        self.selected_label = ttk.Label(btn_frame, text="選択なし", foreground="gray")
        self.selected_label.pack(side=tk.RIGHT)
        
        # ダブルクリックで選択
        self.model_listbox.bind('<Double-Button-1>', self._on_double_click)
    
    def _browse_directory(self):
        """ディレクトリ選択ダイアログ"""
        directory = filedialog.askdirectory(
            title="GGUFモデルディレクトリを選択",
            initialdir=self.last_directory or os.getcwd()
        )
        
        if directory:
            self.dir_var.set(directory)
            self.last_directory = directory
            self._save_last_directory()
            self._scan_gguf_files()
    
    def _scan_gguf_files(self):
        """GGUFファイルをスキャン"""
        directory = self.dir_var.get().strip()
        
        if not directory:
            messagebox.showwarning("警告", "ディレクトリを選択してください")
            return
        
        if not os.path.exists(directory):
            messagebox.showerror("エラー", "指定されたディレクトリが存在しません")
            return
        
        try:
            self.available_models = []
            path = Path(directory)
            
            # .ggufファイルを再帰的に検索
            gguf_files = list(path.rglob('*.gguf'))
            
            for file_path in gguf_files:
                relative_path = file_path.relative_to(path)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                
                model_info = {
                    'path': str(file_path),
                    'name': str(relative_path),
                    'size_mb': round(file_size, 2)
                }
                self.available_models.append(model_info)
            
            self._update_model_list()
            
            if self.available_models:
                messagebox.showinfo("成功", f"{len(self.available_models)}個のGGUFファイルを発見しました")
            else:
                messagebox.showwarning("警告", "GGUFファイルが見つかりませんでした")
                
        except Exception as e:
            messagebox.showerror("エラー", f"スキャン中にエラーが発生しました: {e}")
    
    def _update_model_list(self):
        """モデルリストを更新"""
        self.model_listbox.delete(0, tk.END)
        
        for model_info in self.available_models:
            display_text = f"{model_info['name']} ({model_info['size_mb']} MB)"
            self.model_listbox.insert(tk.END, display_text)
    
    def _refresh_models(self):
        """モデルリストを更新"""
        self._scan_gguf_files()
    
    def _select_model(self):
        """選択されたモデルを取得"""
        selection = self.model_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("警告", "モデルを選択してください")
            return
        
        selected_index = selection[0]
        selected_model = self.available_models[selected_index]
        
        self.selected_model_path = selected_model['path']
        self.selected_label.config(
            text=f"選択: {selected_model['name']}", 
            foreground="green"
        )
        
        # コールバック関数を呼び出し
        if self.on_model_selected:
            self.on_model_selected(selected_model)
        
        # 設定を保存
        self._save_selected_model(selected_model)
    
    def _on_double_click(self, event):
        """ダブルクリックで選択"""
        self._select_model()
    
    def _clear_selection(self):
        """選択をクリア"""
        self.selected_model_path = None
        self.selected_label.config(text="選択なし", foreground="gray")
        self.model_listbox.selection_clear(0, tk.END)
    
    def _load_last_directory(self) -> str:
        """最後に使用したディレクトリを読み込み"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('last_directory', '')
        except Exception:
            pass
        return ""
    
    def _save_last_directory(self):
        """最後に使用したディレクトリを保存"""
        try:
            config = {
                'last_directory': self.last_directory,
                'selected_models': self._get_saved_models()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def _save_selected_model(self, model_info: Dict):
        """選択されたモデルを保存"""
        try:
            saved_models = self._get_saved_models()
            saved_models.append(model_info)
            
            # 重複を除去
            unique_models = []
            seen_paths = set()
            for model in saved_models:
                if model['path'] not in seen_paths:
                    unique_models.append(model)
                    seen_paths.add(model['path'])
            
            # 最新の5個のみ保持
            if len(unique_models) > 5:
                unique_models = unique_models[-5:]
            
            config = {
                'last_directory': self.last_directory,
                'selected_models': unique_models
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"モデル保存エラー: {e}")
    
    def _get_saved_models(self) -> List[Dict]:
        """保存されたモデル一覧を取得"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('selected_models', [])
        except Exception:
            pass
        return []
    
    def _load_saved_models(self):
        """保存されたモデルを読み込み"""
        saved_models = self._get_saved_models()
        if saved_models:
            self.available_models = saved_models
            self._update_model_list()
    
    def get_selected_model_path(self) -> Optional[str]:
        """選択されたモデルパスを取得"""
        return self.selected_model_path
    
    def get_available_models(self) -> List[Dict]:
        """利用可能なモデル一覧を取得"""
        return self.available_models.copy()
    
    def set_directory(self, directory: str):
        """ディレクトリを設定"""
        self.dir_var.set(directory)
        self.last_directory = directory
        self._save_last_directory()
    
    def refresh(self):
        """コンポーネントを更新"""
        self._load_saved_models()

def create_gguf_selector_dialog(parent: tk.Widget, title: str = "GGUFモデル選択") -> Optional[str]:
    """GGUFモデル選択ダイアログを作成"""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry("600x500")
    dialog.resizable(True, True)
    
    # ダイアログをモーダルにする
    dialog.transient(parent)
    dialog.grab_set()
    
    selected_path = None
    
    def on_model_selected(model_info: Dict):
        nonlocal selected_path
        selected_path = model_info['path']
        dialog.destroy()
    
    selector = GGUFModelSelector(dialog, on_model_selected)
    
    # ダイアログが閉じられるまで待機
    dialog.wait_window()
    
    return selected_path

if __name__ == "__main__":
    # テスト用
    root = tk.Tk()
    root.title("GGUF Model Selector Test")
    root.geometry("700x600")
    
    selector = GGUFModelSelector(root)
    
    def test_selection():
        selected = selector.get_selected_model_path()
        if selected:
            messagebox.showinfo("選択結果", f"選択されたモデル: {selected}")
        else:
            messagebox.showwarning("警告", "モデルが選択されていません")
    
    test_btn = ttk.Button(root, text="選択テスト", command=test_selection)
    test_btn.pack(pady=10)
    
    root.mainloop() 