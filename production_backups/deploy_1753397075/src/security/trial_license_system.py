#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trial License System
14日間試用期間とライセンス管理システム
"""

import os
import sys
import json
import time
import hashlib
import base64
import uuid
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class TrialLicenseManager:
    """14日間試用期間管理システム"""
    
    def __init__(self, app_name: str = "Statistics Suite"):
        self.app_name = app_name
        self.trial_days = 14
        self.machine_id = self._generate_machine_id()
        self.crypto_key = self._derive_key()
        
        # 複数の隠蔽された設定ファイル
        self.trial_files = [
            Path(os.path.expanduser("~")) / ".cache" / f".{app_name.lower().replace(' ', '_')}_trial.dat",
            Path(os.environ.get('TEMP', '/tmp')) / f".{self.machine_id[:8]}.tmp",
            Path(os.environ.get('APPDATA', os.path.expanduser("~"))) / "Microsoft" / "Windows" / f".{self.machine_id[8:16]}.sys",
            Path(".") / ".system" / f".{hashlib.md5(self.machine_id.encode()).hexdigest()[:12]}.bin"
        ]
        
        # レジストリ情報（Windows）
        self.registry_keys = [
            f"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\{self._obfuscate_string('TrialData')}",
            f"HKLM\\SOFTWARE\\{self._obfuscate_string('SystemCache')}\\{app_name.replace(' ', '')}"
        ]
        
    def _generate_machine_id(self) -> str:
        """マシン固有ID生成（より詳細）"""
        try:
            # CPUアーキテクチャ
            cpu_info = platform.processor()
            
            # MAC アドレス
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0,2*6,2)][::-1])
            
            # システム情報
            system_info = {
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': platform.python_version()
            }
            
            # ハードウェア情報（Windows）
            hw_info = ""
            if platform.system() == "Windows":
                try:
                    import subprocess
                    # マザーボード
                    result = subprocess.run(['wmic', 'baseboard', 'get', 'serialnumber'], 
                                          capture_output=True, text=True, timeout=5)
                    hw_info += result.stdout.strip()
                    
                    # CPU
                    result = subprocess.run(['wmic', 'cpu', 'get', 'processorid'], 
                                          capture_output=True, text=True, timeout=5)
                    hw_info += result.stdout.strip()
                    
                    # BIOS
                    result = subprocess.run(['wmic', 'bios', 'get', 'serialnumber'], 
                                          capture_output=True, text=True, timeout=5)
                    hw_info += result.stdout.strip()
                except:
                    pass
            
            # 組み合わせてハッシュ化
            machine_data = f"{cpu_info}{mac}{json.dumps(system_info, sort_keys=True)}{hw_info}"
            return hashlib.sha256(machine_data.encode()).hexdigest()
            
        except Exception:
            # フォールバック
            return hashlib.sha256(f"{uuid.getnode()}{platform.platform()}".encode()).hexdigest()
    
    def _derive_key(self) -> bytes:
        """暗号化キー導出"""
        password = f"trial_system_{self.machine_id}_{self.app_name}".encode()
        salt = b'professional_statistics_suite_2025_trial_system'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=150000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    def _obfuscate_string(self, text: str) -> str:
        """文字列難読化"""
        return base64.b64encode(text.encode()).decode().replace('=', '')
    
    def _create_trial_data(self) -> Dict[str, Any]:
        """試用期間データ作成"""
        now = datetime.now()
        trial_data = {
            'machine_id': self.machine_id,
            'app_name': self.app_name,
            'start_date': now.isoformat(),
            'end_date': (now + timedelta(days=self.trial_days)).isoformat(),
            'install_time': time.time(),
            'usage_count': 0,
            'checksum': hashlib.sha256(f"{self.machine_id}{now.isoformat()}".encode()).hexdigest(),
            'version': '1.0.0',
            'features_used': [],
            'last_check': now.isoformat()
        }
        return trial_data
    
    def _save_trial_data(self, trial_data: Dict[str, Any]):
        """試用期間データ保存（複数箇所）"""
        try:
            fernet = Fernet(self.crypto_key)
            encrypted_data = fernet.encrypt(json.dumps(trial_data).encode())
            
            # 複数のファイルに保存
            for trial_file in self.trial_files:
                try:
                    trial_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(trial_file, 'wb') as f:
                        f.write(encrypted_data)
                except:
                    continue
            
            # Windowsレジストリにも保存
            if platform.system() == "Windows":
                self._save_to_registry(trial_data)
                
        except Exception:
            pass
    
    def _save_to_registry(self, trial_data: Dict[str, Any]):
        """Windowsレジストリに保存"""
        try:
            import winreg
            
            # HKEY_CURRENT_USER に保存
            key_path = f"Software\\Microsoft\\Windows\\CurrentVersion\\{self._obfuscate_string('SystemCache')}"
            
            try:
                key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
                encrypted_data = base64.b64encode(json.dumps(trial_data).encode()).decode()
                winreg.SetValueEx(key, self._obfuscate_string('AppData'), 0, winreg.REG_SZ, encrypted_data)
                winreg.CloseKey(key)
            except:
                pass
                
        except ImportError:
            pass
    
    def _load_trial_data(self) -> Optional[Dict[str, Any]]:
        """試用期間データ読み込み"""
        trial_data = None
        
        # ファイルから読み込み
        for trial_file in self.trial_files:
            if trial_file.exists():
                try:
                    with open(trial_file, 'rb') as f:
                        encrypted_data = f.read()
                    
                    fernet = Fernet(self.crypto_key)
                    decrypted_data = fernet.decrypt(encrypted_data)
                    trial_data = json.loads(decrypted_data.decode())
                    break
                except:
                    continue
        
        # レジストリから読み込み（Windows）
        if trial_data is None and platform.system() == "Windows":
            trial_data = self._load_from_registry()
        
        return trial_data
    
    def _load_from_registry(self) -> Optional[Dict[str, Any]]:
        """Windowsレジストリから読み込み"""
        try:
            import winreg
            
            key_path = f"Software\\Microsoft\\Windows\\CurrentVersion\\{self._obfuscate_string('SystemCache')}"
            
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
                encrypted_data, _ = winreg.QueryValueEx(key, self._obfuscate_string('AppData'))
                winreg.CloseKey(key)
                
                decoded_data = base64.b64decode(encrypted_data).decode()
                return json.loads(decoded_data)
            except:
                return None
                
        except ImportError:
            return None
    
    def check_trial_status(self) -> Dict[str, Any]:
        """試用期間状態チェック"""
        try:
            # 既存の試用期間データ読み込み
            trial_data = self._load_trial_data()
            
            if trial_data is None:
                # 新規インストール
                trial_data = self._create_trial_data()
                self._save_trial_data(trial_data)
                
                return {
                    'status': 'trial_started',
                    'valid': True,
                    'days_remaining': self.trial_days,
                    'end_date': trial_data['end_date'],
                    'message': f'{self.trial_days}日間の試用期間が開始されました'
                }
            
            # データ整合性チェック
            if trial_data.get('machine_id') != self.machine_id:
                return {
                    'status': 'invalid_machine',
                    'valid': False,
                    'message': 'このマシンでは使用できません'
                }
            
            # 試用期間チェック
            end_date = datetime.fromisoformat(trial_data['end_date'])
            now = datetime.now()
            
            if now > end_date:
                return {
                    'status': 'trial_expired',
                    'valid': False,
                    'message': '試用期間が終了しました。ライセンスを購入してください',
                    'expired_days': (now - end_date).days
                }
            
            # 残り日数計算
            days_remaining = (end_date - now).days
            
            # 使用回数更新
            trial_data['usage_count'] += 1
            trial_data['last_check'] = now.isoformat()
            self._save_trial_data(trial_data)
            
            return {
                'status': 'trial_active',
                'valid': True,
                'days_remaining': days_remaining,
                'end_date': trial_data['end_date'],
                'usage_count': trial_data['usage_count'],
                'message': f'試用期間: あと{days_remaining}日'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'valid': False,
                'message': f'試用期間チェックエラー: {str(e)}'
            }
    
    def verify_license_key(self, license_key: str) -> Dict[str, Any]:
        """ライセンスキー検証"""
        try:
            # ライセンスキー形式: XXXX-XXXX-XXXX-XXXX
            if not license_key or len(license_key.replace('-', '')) != 16:
                return {'valid': False, 'message': 'ライセンスキーの形式が正しくありません'}
            
            # 基本的な検証（実際にはサーバーで検証）
            clean_key = license_key.replace('-', '').upper()
            
            # チェックサム検証
            checksum = sum(ord(c) for c in clean_key[:12]) % 10000
            provided_checksum = int(clean_key[12:16])
            
            if checksum != provided_checksum:
                return {'valid': False, 'message': 'ライセンスキーが無効です'}
            
            # ライセンスデータ作成
            license_data = {
                'license_key': license_key,
                'machine_id': self.machine_id,
                'app_name': self.app_name,
                'activation_date': datetime.now().isoformat(),
                'license_type': 'full',
                'valid_until': (datetime.now() + timedelta(days=365)).isoformat(),
                'features': ['all']
            }
            
            # ライセンスファイル保存
            self._save_license_data(license_data)
            
            # 試用期間データ削除
            self._clear_trial_data()
            
            return {
                'valid': True,
                'message': 'ライセンスが正常に認証されました',
                'license_data': license_data
            }
            
        except Exception as e:
            return {'valid': False, 'message': f'ライセンス検証エラー: {str(e)}'}
    
    def _save_license_data(self, license_data: Dict[str, Any]):
        """ライセンスデータ保存"""
        try:
            license_file = Path("license.dat")
            fernet = Fernet(self.crypto_key)
            encrypted_data = fernet.encrypt(json.dumps(license_data).encode())
            
            with open(license_file, 'wb') as f:
                f.write(encrypted_data)
                
        except Exception:
            pass
    
    def _clear_trial_data(self):
        """試用期間データ削除"""
        try:
            # ファイル削除
            for trial_file in self.trial_files:
                if trial_file.exists():
                    try:
                        trial_file.unlink()
                    except:
                        pass
            
            # レジストリ削除（Windows）
            if platform.system() == "Windows":
                try:
                    import winreg
                    key_path = f"Software\\Microsoft\\Windows\\CurrentVersion\\{self._obfuscate_string('SystemCache')}"
                    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
                except:
                    pass
                    
        except Exception:
            pass
    
    def check_license(self) -> Dict[str, Any]:
        """ライセンス状態チェック"""
        try:
            license_file = Path("license.dat")
            if license_file.exists():
                with open(license_file, 'rb') as f:
                    encrypted_data = f.read()
                
                fernet = Fernet(self.crypto_key)
                decrypted_data = fernet.decrypt(encrypted_data)
                license_data = json.loads(decrypted_data.decode())
                
                # ライセンス有効期限チェック
                valid_until = datetime.fromisoformat(license_data['valid_until'])
                if datetime.now() > valid_until:
                    return {
                        'status': 'license_expired',
                        'valid': False,
                        'message': 'ライセンスが期限切れです'
                    }
                
                # マシンIDチェック
                if license_data.get('machine_id') != self.machine_id:
                    return {
                        'status': 'invalid_machine',
                        'valid': False,
                        'message': 'このマシンでは使用できません'
                    }
                
                return {
                    'status': 'licensed',
                    'valid': True,
                    'license_data': license_data,
                    'message': 'ライセンス認証済み'
                }
            
            # ライセンスがない場合は試用期間チェック
            return self.check_trial_status()
            
        except Exception as e:
            return {
                'status': 'error',
                'valid': False,
                'message': f'ライセンスチェックエラー: {str(e)}'
            }

def require_trial_or_license(app_name: str = "Statistics Suite"):
    """試用期間またはライセンス必須デコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            trial_manager = TrialLicenseManager(app_name)
            license_status = trial_manager.check_license()
            
            if not license_status['valid']:
                print(f"❌ {license_status['message']}")
                if license_status['status'] == 'trial_expired':
                    print("💰 ライセンスを購入してください: https://your-license-store.com")
                sys.exit(1)
            
            if license_status['status'] == 'trial_active':
                print(f"⏰ {license_status['message']}")
            elif license_status['status'] == 'licensed':
                print("✅ ライセンス認証済み")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 使用例
if __name__ == "__main__":
    trial_manager = TrialLicenseManager("Professional Statistics Suite")
    
    # ライセンス状態チェック
    status = trial_manager.check_license()
    print(f"Status: {status}")
    
    # ライセンスキーテスト
    test_key = "ABCD-EFGH-IJKL-1234"
    license_result = trial_manager.verify_license_key(test_key)
    print(f"License verification: {license_result}") 