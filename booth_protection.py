#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Booth Protection System
商用版保護システム - リバースエンジニアリング対策
"""

import os
import sys
import hashlib
import time
import base64
import ctypes
import threading
import subprocess
from typing import Optional, Dict, Any, List
import json
import uuid
import socket
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class AntiDebugProtection:
    """アンチデバッグ保護システム"""
    
    def __init__(self):
        self.debug_detected = False
        self.protection_active = True
        self.check_interval = 5  # 5秒間隔でチェック
        
    def detect_debugger(self) -> bool:
        """デバッガー検出"""
        try:
            # プロセス名チェック（一般的なデバッガー）
            debugger_processes = [
                'x64dbg.exe', 'x32dbg.exe', 'windbg.exe', 'ida.exe', 'ida64.exe',
                'ollydbg.exe', 'immunitydebugger.exe', 'cheatengine.exe',
                'processhacker.exe', 'procmon.exe', 'wireshark.exe',
                'fiddler.exe', 'httpdebugger.exe', 'python.exe -m pdb'
            ]
            
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    proc_name = proc.info['name'].lower()
                    cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                    
                    for debugger in debugger_processes:
                        if debugger.lower() in proc_name or debugger.lower() in cmdline:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Windows固有のデバッガー検出
            if platform.system() == "Windows":
                # IsDebuggerPresent API
                try:
                    kernel32 = ctypes.windll.kernel32
                    if kernel32.IsDebuggerPresent():
                        return True
                except:
                    pass
                
                # CheckRemoteDebuggerPresent API
                try:
                    current_process = kernel32.GetCurrentProcess()
                    debug_flag = ctypes.c_bool()
                    kernel32.CheckRemoteDebuggerPresent(current_process, ctypes.byref(debug_flag))
                    if debug_flag.value:
                        return True
                except:
                    pass
            
            # Python固有のデバッグ検出
            if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
                return True
                
            # 実行時間異常検出（ステップ実行検出）
            start_time = time.perf_counter()
            time.sleep(0.1)
            end_time = time.perf_counter()
            if end_time - start_time > 0.5:  # 通常の5倍以上時間がかかった場合
                return True
                
            return False
            
        except Exception:
            return False
    
    def start_protection(self):
        """保護システム開始"""
        def protection_thread():
            while self.protection_active:
                if self.detect_debugger():
                    self.debug_detected = True
                    self._handle_debug_detection()
                    break
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=protection_thread, daemon=True)
        thread.start()
    
    def _handle_debug_detection(self):
        """デバッグ検出時の処理"""
        # データ破壊
        try:
            # 重要なメモリ領域をクリア
            import gc
            gc.collect()
            
            # プロセス終了
            os._exit(1)
        except:
            sys.exit(1)

class LicenseManager:
    """ライセンス管理システム"""
    
    def __init__(self, server_url: str = "https://your-license-server.com"):
        self.server_url = server_url
        self.license_file = Path("license.enc")
        self.machine_id = self._generate_machine_id()
        self.crypto_key = self._derive_key()
        
    def _generate_machine_id(self) -> str:
        """マシン固有ID生成"""
        try:
            # CPUアーキテクチャ
            cpu_info = platform.processor()
            
            # MAC アドレス
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0,2*6,2)][::-1])
            
            # マザーボード情報（Windows）
            motherboard = ""
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ['wmic', 'baseboard', 'get', 'serialnumber'],
                        capture_output=True, text=True, timeout=5
                    )
                    motherboard = result.stdout.strip()
                except:
                    pass
            
            # ハードドライブ情報
            hdd_serial = ""
            try:
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ['wmic', 'diskdrive', 'get', 'serialnumber'],
                        capture_output=True, text=True, timeout=5
                    )
                    hdd_serial = result.stdout.strip()
            except:
                pass
            
            # 組み合わせてハッシュ化
            machine_data = f"{cpu_info}{mac}{motherboard}{hdd_serial}{platform.system()}"
            return hashlib.sha256(machine_data.encode()).hexdigest()[:16]
            
        except Exception:
            # フォールバック
            return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:16]
    
    def _derive_key(self) -> bytes:
        """暗号化キー導出"""
        password = f"booth_license_{self.machine_id}".encode()
        salt = b'statisticssuite2025'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def verify_license(self) -> Dict[str, Any]:
        """ライセンス検証"""
        try:
            # ローカルライセンスファイル確認
            if self.license_file.exists():
                local_result = self._verify_local_license()
                if local_result["valid"]:
                    # オンライン検証（可能な場合）
                    online_result = self._verify_online_license()
                    if online_result["success"]:
                        return online_result
                    return local_result
            
            # 新規ライセンス取得
            return self._request_new_license()
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"ライセンス検証エラー: {str(e)}",
                "trial_available": True
            }
    
    def _verify_local_license(self) -> Dict[str, Any]:
        """ローカルライセンス検証"""
        try:
            with open(self.license_file, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(self.crypto_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            license_data = json.loads(decrypted_data.decode())
            
            # 有効期限チェック
            expiry_date = datetime.fromisoformat(license_data["expiry"])
            if datetime.now() > expiry_date:
                return {"valid": False, "error": "ライセンスが期限切れです"}
            
            # マシンIDチェック
            if license_data["machine_id"] != self.machine_id:
                return {"valid": False, "error": "このマシンでは使用できません"}
            
            return {
                "valid": True,
                "license_type": license_data["type"],
                "expiry": license_data["expiry"],
                "features": license_data.get("features", [])
            }
            
        except Exception as e:
            return {"valid": False, "error": f"ライセンスファイル読み込みエラー: {str(e)}"}
    
    def _verify_online_license(self) -> Dict[str, Any]:
        """オンラインライセンス検証"""
        try:
            import requests
            
            response = requests.post(
                f"{self.server_url}/api/verify",
                json={
                    "machine_id": self.machine_id,
                    "version": "1.0.0"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["valid"]:
                    # ライセンス情報を更新
                    self._save_license(result["license_data"])
                return result
            else:
                return {"success": False, "error": "サーバー応答エラー"}
                
        except Exception as e:
            return {"success": False, "error": f"オンライン検証失敗: {str(e)}"}
    
    def _request_new_license(self) -> Dict[str, Any]:
        """新規ライセンス要求"""
        return {
            "valid": False,
            "error": "ライセンスが見つかりません",
            "activation_url": f"{self.server_url}/activate?machine_id={self.machine_id}",
            "trial_available": True
        }
    
    def _save_license(self, license_data: Dict[str, Any]):
        """ライセンス保存"""
        try:
            fernet = Fernet(self.crypto_key)
            encrypted_data = fernet.encrypt(json.dumps(license_data).encode())
            
            with open(self.license_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception:
            pass

class CodeObfuscator:
    """コード難読化システム"""
    
    @staticmethod
    def obfuscate_string(text: str) -> str:
        """文字列難読化"""
        encoded = base64.b64encode(text.encode()).decode()
        return f"__import__('base64').b64decode('{encoded}').decode()"
    
    @staticmethod
    def obfuscate_numbers(value: int) -> str:
        """数値難読化"""
        # 複雑な計算式で数値を隠蔽
        a, b = divmod(value, 17)
        c = a + 42
        return f"({c} - 42) * 17 + {b}"
    
    @staticmethod
    def create_dummy_functions():
        """ダミー関数生成"""
        dummy_code = """
def __dummy_func_1():
    import random
    return random.randint(1000, 9999)

def __dummy_func_2():
    import hashlib
    return hashlib.md5(str(__dummy_func_1()).encode()).hexdigest()

def __dummy_func_3():
    for i in range(100):
        pass
    return True
"""
        exec(dummy_code)

class IntegrityChecker:
    """整合性チェッカー"""
    
    def __init__(self, expected_hashes: Dict[str, str]):
        self.expected_hashes = expected_hashes
    
    def verify_files(self) -> bool:
        """ファイル整合性検証"""
        try:
            for file_path, expected_hash in self.expected_hashes.items():
                if not Path(file_path).exists():
                    return False
                
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                actual_hash = hashlib.sha256(content).hexdigest()
                if actual_hash != expected_hash:
                    return False
            
            return True
        except Exception:
            return False
    
    def check_runtime_integrity(self) -> bool:
        """実行時整合性チェック"""
        try:
            # スタックトレース検証
            import inspect
            frame = inspect.currentframe()
            if frame is None:
                return False
            
            # 予期しない呼び出し元チェック
            caller = frame.f_back
            if caller and hasattr(caller, 'f_code'):
                filename = caller.f_code.co_filename
                if 'pdb' in filename or 'debugger' in filename:
                    return False
            
            return True
        except Exception:
            return False

class BoothProtectionSystem:
    """Booth版保護システム統合"""
    
    def __init__(self):
        self.anti_debug = AntiDebugProtection()
        self.license_manager = LicenseManager()
        self.obfuscator = CodeObfuscator()
        self.integrity_checker = IntegrityChecker({})
        self.protection_level = "MAXIMUM"
        
    def initialize_protection(self) -> Dict[str, Any]:
        """保護システム初期化"""
        try:
            # アンチデバッグ開始
            if self.protection_level in ["HIGH", "MAXIMUM"]:
                self.anti_debug.start_protection()
            
            # ライセンス検証
            license_result = self.license_manager.verify_license()
            
            # 整合性チェック
            if not self.integrity_checker.check_runtime_integrity():
                return {
                    "success": False,
                    "error": "プログラムの整合性に問題があります"
                }
            
            # ダミー関数生成（リバースエンジニアリング妨害）
            self.obfuscator.create_dummy_functions()
            
            return {
                "success": license_result.get("valid", False),
                "license_info": license_result,
                "protection_active": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"保護システム初期化失敗: {str(e)}"
            }
    
    def check_runtime_protection(self) -> bool:
        """実行時保護チェック"""
        if self.anti_debug.debug_detected:
            return False
        
        return self.integrity_checker.check_runtime_integrity()
    
    def create_protected_wrapper(self, func):
        """保護ラッパー作成"""
        def wrapper(*args, **kwargs):
            if not self.check_runtime_protection():
                os._exit(1)
            return func(*args, **kwargs)
        return wrapper

# グローバル保護システム
booth_protection = BoothProtectionSystem()

def require_license(license_type: Optional[str] = None):
    """ライセンス必須デコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            license_result = booth_protection.license_manager.verify_license()
            if not license_result.get("valid", False):
                raise Exception("有効なライセンスが必要です")
            
            if license_type and license_result.get("license_type") != license_type:
                raise Exception(f"{license_type}ライセンスが必要です")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def anti_debug(func):
    """アンチデバッグデコレータ"""
    def wrapper(*args, **kwargs):
        if booth_protection.anti_debug.detect_debugger():
            os._exit(1)
        return func(*args, **kwargs)
    return wrapper

# 使用例
if __name__ == "__main__":
    # 保護システム初期化
    protection_result = booth_protection.initialize_protection()
    
    if protection_result["success"]:
        print("保護システム: アクティブ")
        print(f"ライセンス: {protection_result['license_info']}")
    else:
        print(f"保護システムエラー: {protection_result['error']}")
        sys.exit(1) 