#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI応答性最適化システム
Professional Statistics Suite - GUI Responsiveness Optimizer

Author: Professional Statistics Suite Team
Email: r.minegishi1987@gmail.com
License: MIT
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import gc
import psutil
import weakref
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class ResponsivenessMetrics:
    """応答性メトリクス"""
    button_click_time_ms: float
    ui_update_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    event_queue_size: int
    timestamp: datetime

class GUIResponsivenessOptimizer:
    """GUI応答性最適化システム"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.metrics_history: List[ResponsivenessMetrics] = []
        self.async_queue = queue.Queue()
        self.ui_update_queue = queue.Queue()
        self.background_tasks: Dict[str, threading.Thread] = {}
        self.weak_refs: List[weakref.ref] = []
        
        # 応答性設定
        self.max_response_time_ms = 100  # 100ms以下を目標
        self.max_memory_mb = 1000  # 1GB以下
        self.max_cpu_percent = 80  # 80%以下
        self.ui_update_batch_size = 10  # バッチ更新サイズ
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 初期化
        self._setup_event_loop_optimization()
        self._setup_memory_monitoring()
        self._setup_async_processor()
        self._setup_ui_update_optimizer()
        
        self.logger.info("🚀 GUI応答性最適化システム初期化完了")
    
    def _setup_event_loop_optimization(self):
        """イベントループ最適化"""
        # メインループの最適化
        self.root.after(1, self._process_async_queue)
        self.root.after(10, self._process_ui_updates)
        self.root.after(100, self._cleanup_resources)
        
        # イベントバインディング最適化
        self.root.bind('<Configure>', self._on_window_resize)
        self.root.bind('<FocusIn>', self._on_focus_in)
        self.root.bind('<FocusOut>', self._on_focus_out)
    
    def _setup_memory_monitoring(self):
        """メモリ監視システム"""
        self.memory_monitor_active = True
        self.memory_monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, 
            daemon=True
        )
        self.memory_monitor_thread.start()
    
    def _setup_async_processor(self):
        """非同期処理システム"""
        self.async_processor_active = True
        self.async_processor_thread = threading.Thread(
            target=self._async_processor_worker,
            daemon=True
        )
        self.async_processor_thread.start()
    
    def _setup_ui_update_optimizer(self):
        """UI更新最適化システム"""
        self.ui_updates_pending = []
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.016  # 60FPS (16.67ms)
    
    def execute_async(self, task: Callable, task_name: str = "unknown", 
                     callback: Optional[Callable] = None, 
                     error_callback: Optional[Callable] = None):
        """非同期タスク実行"""
        task_id = f"{task_name}_{int(time.time() * 1000)}"
        
        def async_wrapper():
            try:
                start_time = time.time()
                result = task()
                execution_time = (time.time() - start_time) * 1000
                
                # メトリクス記録
                self._record_metrics(execution_time, task_name)
                
                # コールバック実行
                if callback:
                    self.async_queue.put((callback, result))
                
                self.logger.info(f"✅ 非同期タスク完了: {task_name} ({execution_time:.1f}ms)")
                
            except Exception as e:
                self.logger.error(f"❌ 非同期タスクエラー: {task_name} - {e}")
                if error_callback:
                    self.async_queue.put((error_callback, str(e)))
        
        # バックグラウンドスレッドで実行
        thread = threading.Thread(target=async_wrapper, daemon=True)
        self.background_tasks[task_id] = thread
        thread.start()
        
        return task_id
    
    def schedule_ui_update(self, update_func: Callable, priority: int = 0):
        """UI更新スケジューリング"""
        self.ui_update_queue.put((priority, time.time(), update_func))
    
    def optimize_button_response(self, button: tk.Widget, command: Callable):
        """ボタン応答性最適化"""
        def optimized_command():
            start_time = time.time()
            
            # ボタンを一時的に無効化
            original_state = button.cget("state")
            button.configure(state="disabled")
            
            try:
                # 非同期実行
                self.execute_async(
                    task=command,
                    task_name="button_click",
                    callback=lambda result: self._on_button_complete(button, original_state, result),
                    error_callback=lambda error: self._on_button_error(button, original_state, error)
                )
                
                response_time = (time.time() - start_time) * 1000
                self.logger.info(f"⚡ ボタン応答時間: {response_time:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"❌ ボタン実行エラー: {e}")
                button.configure(state=original_state)
        
        return optimized_command
    
    def _on_button_complete(self, button: tk.Widget, original_state: str, result: Any):
        """ボタン完了処理"""
        button.configure(state=original_state)
        self.logger.info("✅ ボタン処理完了")
    
    def _on_button_error(self, button: tk.Widget, original_state: str, error: str):
        """ボタンエラー処理"""
        button.configure(state=original_state)
        self.logger.error(f"❌ ボタン処理エラー: {error}")
    
    def _process_async_queue(self):
        """非同期キュー処理"""
        try:
            while not self.async_queue.empty():
                callback, result = self.async_queue.get_nowait()
                callback(result)
        except queue.Empty:
            pass
        
        # 次の処理をスケジュール
        self.root.after(1, self._process_async_queue)
    
    def _process_ui_updates(self):
        """UI更新処理"""
        try:
            updates = []
            while not self.ui_update_queue.empty() and len(updates) < self.ui_update_batch_size:
                priority, timestamp, update_func = self.ui_update_queue.get_nowait()
                updates.append((priority, timestamp, update_func))
            
            # 優先度順にソート
            updates.sort(key=lambda x: x[0])
            
            # バッチ更新実行
            for priority, timestamp, update_func in updates:
                try:
                    update_func()
                except Exception as e:
                    self.logger.error(f"❌ UI更新エラー: {e}")
                    
        except queue.Empty:
            pass
        
        # 次の処理をスケジュール
        self.root.after(10, self._process_ui_updates)
    
    def _cleanup_resources(self):
        """リソースクリーンアップ"""
        # 完了したタスクの削除
        completed_tasks = []
        for task_id, thread in self.background_tasks.items():
            if not thread.is_alive():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del self.background_tasks[task_id]
        
        # ガベージコレクション
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
        
        # 弱参照のクリーンアップ
        self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
        
        # 次のクリーンアップをスケジュール
        self.root.after(100, self._cleanup_resources)
    
    def _monitor_memory_usage(self):
        """メモリ使用量監視"""
        while self.memory_monitor_active:
            try:
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_usage = psutil.cpu_percent()
                
                if memory_usage > self.max_memory_mb:
                    self.logger.warning(f"⚠️ メモリ使用量警告: {memory_usage:.1f}MB")
                    self._optimize_memory_usage()
                
                if cpu_usage > self.max_cpu_percent:
                    self.logger.warning(f"⚠️ CPU使用率警告: {cpu_usage:.1f}%")
                
                time.sleep(1)  # 1秒間隔で監視
                
            except Exception as e:
                self.logger.error(f"❌ メモリ監視エラー: {e}")
                time.sleep(5)
    
    def _async_processor_worker(self):
        """非同期処理ワーカー"""
        while self.async_processor_active:
            try:
                # 非同期タスクの処理
                time.sleep(0.001)  # 1ms間隔
            except Exception as e:
                self.logger.error(f"❌ 非同期処理エラー: {e}")
                time.sleep(1)
    
    def _optimize_memory_usage(self):
        """メモリ使用量最適化"""
        # ガベージコレクション実行
        gc.collect()
        
        # 古いメトリクス履歴の削除
        if len(self.metrics_history) > 500:
            self.metrics_history = self.metrics_history[-250:]
        
        self.logger.info("🧹 メモリ最適化実行")
    
    def _record_metrics(self, execution_time: float, task_name: str):
        """メトリクス記録"""
        metrics = ResponsivenessMetrics(
            button_click_time_ms=execution_time,
            ui_update_time_ms=0,  # UI更新時間は別途計測
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            event_queue_size=self.async_queue.qsize(),
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
    
    def _on_window_resize(self, event):
        """ウィンドウリサイズイベント"""
        # リサイズ時の最適化
        self.schedule_ui_update(lambda: None, priority=1)
    
    def _on_focus_in(self, event):
        """フォーカスインイベント"""
        # フォーカス復帰時の最適化
        self.schedule_ui_update(lambda: None, priority=2)
    
    def _on_focus_out(self, event):
        """フォーカスアウトイベント"""
        # フォーカス喪失時の最適化
        pass
    
    def get_responsiveness_report(self) -> Dict[str, Any]:
        """応答性レポート取得"""
        if not self.metrics_history:
            return {"error": "メトリクスデータがありません"}
        
        recent_metrics = self.metrics_history[-100:]  # 最新100件
        
        avg_response_time = sum(m.button_click_time_ms for m in recent_metrics) / len(recent_metrics)
        max_response_time = max(m.button_click_time_ms for m in recent_metrics)
        avg_memory_usage = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "average_response_time_ms": avg_response_time,
            "max_response_time_ms": max_response_time,
            "average_memory_usage_mb": avg_memory_usage,
            "average_cpu_usage_percent": avg_cpu_usage,
            "total_tasks_processed": len(self.metrics_history),
            "active_background_tasks": len(self.background_tasks),
            "queue_size": self.async_queue.qsize(),
            "optimization_status": "active"
        }
    
    def shutdown(self):
        """システムシャットダウン"""
        self.memory_monitor_active = False
        self.async_processor_active = False
        
        # バックグラウンドタスクの終了待機
        for thread in self.background_tasks.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.logger.info("🛑 GUI応答性最適化システム終了")

class ResponsivenessTestSuite:
    """応答性テストスイート"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.optimizer = GUIResponsivenessOptimizer(root)
        self.test_results = []
        
    def run_button_response_test(self, button: tk.Widget, test_name: str) -> Dict[str, Any]:
        """ボタン応答性テスト"""
        start_time = time.time()
        
        # ボタンクリック実行
        button.invoke()
        
        response_time = (time.time() - start_time) * 1000
        
        result = {
            "test_name": test_name,
            "response_time_ms": response_time,
            "timestamp": datetime.now(),
            "success": response_time < 100  # 100ms以下を成功とする
        }
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """包括的応答性テスト"""
        # 応答性レポート取得
        responsiveness_report = self.optimizer.get_responsiveness_report()
        
        # テスト結果集計
        successful_tests = sum(1 for result in self.test_results if result["success"])
        total_tests = len(self.test_results)
        
        return {
            "responsiveness_report": responsiveness_report,
            "test_results": self.test_results,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
        } 