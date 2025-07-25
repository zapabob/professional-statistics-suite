# -*- coding: utf-8 -*-
"""
包括的監査・コンプライアンスシステム
Comprehensive Audit and Compliance System

Author: Kiro AI Assistant
Email: r.minegishi1987@gmail.com
License: MIT
"""

import json
import os
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

class AuditLevel(Enum):
    """監査レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceType(Enum):
    """コンプライアンスタイプ"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    CUSTOM = "custom"

@dataclass
class AuditEvent:
    """監査イベント"""
    event_id: str
    timestamp: datetime
    user_id: str
    session_id: str
    event_type: str
    event_description: str
    data_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    severity: AuditLevel = AuditLevel.MEDIUM
    metadata: Dict[str, Any] = None

@dataclass
class ComplianceViolation:
    """コンプライアンス違反"""
    violation_id: str
    timestamp: datetime
    compliance_type: ComplianceType
    violation_type: str
    description: str
    severity: AuditLevel
    affected_data: Optional[str] = None
    remediation_required: bool = True
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class DataAccessLog:
    """データアクセスログ"""
    access_id: str
    timestamp: datetime
    user_id: str
    data_type: str
    access_type: str  # read, write, delete, export
    data_fingerprint: str
    justification: Optional[str] = None
    approved_by: Optional[str] = None

class AuditTrailManager:
    """包括的監査トレイル管理システム"""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_database()
        self.setup_logging()
    
    def setup_database(self):
        """データベースの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 監査イベントテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_description TEXT NOT NULL,
                        data_fingerprint TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        severity TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # コンプライアンス違反テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_violations (
                        violation_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        compliance_type TEXT NOT NULL,
                        violation_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        affected_data TEXT,
                        remediation_required BOOLEAN DEFAULT 1,
                        resolved BOOLEAN DEFAULT 0,
                        resolution_notes TEXT
                    )
                """)
                
                # データアクセスログテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_access_logs (
                        access_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        access_type TEXT NOT NULL,
                        data_fingerprint TEXT NOT NULL,
                        justification TEXT,
                        approved_by TEXT
                    )
                """)
                
                # インデックスの作成
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON compliance_violations(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_timestamp ON data_access_logs(timestamp)")
                
        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")
            raise
    
    def setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "audit_trail.log"),
                logging.StreamHandler()
            ]
        )
    
    def log_analysis_operation(self, operation: Dict[str, Any], user: Dict[str, Any]) -> str:
        """分析操作のログ記録"""
        try:
            event_id = self._generate_event_id()
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                user_id=user.get('user_id', 'unknown'),
                session_id=user.get('session_id', 'unknown'),
                event_type='analysis_operation',
                event_description=f"統計分析実行: {operation.get('method', 'unknown')}",
                data_fingerprint=operation.get('data_fingerprint'),
                severity=self._determine_severity(operation),
                metadata=operation
            )
            
            self._store_audit_event(event)
            self.logger.info(f"分析操作ログ記録: {event_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"分析操作ログ記録エラー: {e}")
            raise
    
    def log_data_access(self, access: Dict[str, Any], user: Dict[str, Any]) -> str:
        """データアクセスのログ記録"""
        try:
            access_id = self._generate_event_id()
            access_log = DataAccessLog(
                access_id=access_id,
                timestamp=datetime.now(),
                user_id=user.get('user_id', 'unknown'),
                data_type=access.get('data_type', 'unknown'),
                access_type=access.get('access_type', 'read'),
                data_fingerprint=access.get('data_fingerprint', ''),
                justification=access.get('justification'),
                approved_by=access.get('approved_by')
            )
            
            self._store_data_access_log(access_log)
            self.logger.info(f"データアクセスログ記録: {access_id}")
            return access_id
            
        except Exception as e:
            self.logger.error(f"データアクセスログ記録エラー: {e}")
            raise
    
    def log_ai_interaction(self, interaction: Dict[str, Any], user: Dict[str, Any]) -> str:
        """AI相互作用のログ記録"""
        try:
            event_id = self._generate_event_id()
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                user_id=user.get('user_id', 'unknown'),
                session_id=user.get('session_id', 'unknown'),
                event_type='ai_interaction',
                event_description=f"AI相互作用: {interaction.get('query', 'unknown')}",
                data_fingerprint=interaction.get('data_fingerprint'),
                severity=AuditLevel.MEDIUM,
                metadata=interaction
            )
            
            self._store_audit_event(event)
            self.logger.info(f"AI相互作用ログ記録: {event_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"AI相互作用ログ記録エラー: {e}")
            raise
    
    def generate_audit_report(self, time_range: Tuple[datetime, datetime], 
                            user: Optional[str] = None) -> Dict[str, Any]:
        """監査レポートの生成"""
        try:
            start_time, end_time = time_range
            
            with sqlite3.connect(self.db_path) as conn:
                # 監査イベントの取得
                query = """
                    SELECT * FROM audit_events 
                    WHERE timestamp BETWEEN ? AND ?
                """
                params = [start_time.isoformat(), end_time.isoformat()]
                
                if user:
                    query += " AND user_id = ?"
                    params.append(user)
                
                query += " ORDER BY timestamp DESC"
                
                events = pd.read_sql_query(query, conn, params=params)
                
                # 統計情報の計算
                report = {
                    'time_range': {
                        'start': start_time.isoformat(),
                        'end': end_time.isoformat()
                    },
                    'total_events': len(events),
                    'events_by_type': events['event_type'].value_counts().to_dict(),
                    'events_by_severity': events['severity'].value_counts().to_dict(),
                    'unique_users': events['user_id'].nunique(),
                    'events': events.to_dict('records')
                }
                
                self.logger.info(f"監査レポート生成: {len(events)}件のイベント")
                return report
                
        except Exception as e:
            self.logger.error(f"監査レポート生成エラー: {e}")
            raise
    
    def check_compliance_violations(self) -> List[ComplianceViolation]:
        """コンプライアンス違反のチェック"""
        try:
            violations = []
            
            # GDPR違反チェック
            gdpr_violations = self._check_gdpr_compliance()
            violations.extend(gdpr_violations)
            
            # HIPAA違反チェック
            hipaa_violations = self._check_hipaa_compliance()
            violations.extend(hipaa_violations)
            
            # カスタムコンプライアンスチェック
            custom_violations = self._check_custom_compliance()
            violations.extend(custom_violations)
            
            # 違反のデータベース保存
            for violation in violations:
                self._store_compliance_violation(violation)
            
            self.logger.info(f"コンプライアンス違反チェック完了: {len(violations)}件の違反")
            return violations
            
        except Exception as e:
            self.logger.error(f"コンプライアンス違反チェックエラー: {e}")
            raise
    
    def _check_gdpr_compliance(self) -> List[ComplianceViolation]:
        """GDPRコンプライアンスチェック"""
        violations = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 個人データの処理チェック
                personal_data_events = pd.read_sql_query("""
                    SELECT * FROM audit_events 
                    WHERE event_description LIKE '%personal%' 
                    OR event_description LIKE '%PII%'
                    OR event_description LIKE '%sensitive%'
                """, conn)
                
                for _, event in personal_data_events.iterrows():
                    violation = ComplianceViolation(
                        violation_id=self._generate_event_id(),
                        timestamp=datetime.now(),
                        compliance_type=ComplianceType.GDPR,
                        violation_type="personal_data_processing",
                        description=f"個人データ処理: {event['event_description']}",
                        severity=AuditLevel.HIGH,
                        affected_data=event.get('data_fingerprint')
                    )
                    violations.append(violation)
                    
        except Exception as e:
            self.logger.error(f"GDPRコンプライアンスチェックエラー: {e}")
        
        return violations
    
    def _check_hipaa_compliance(self) -> List[ComplianceViolation]:
        """HIPAAコンプライアンスチェック"""
        violations = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 医療データの処理チェック
                medical_data_events = pd.read_sql_query("""
                    SELECT * FROM audit_events 
                    WHERE event_description LIKE '%medical%' 
                    OR event_description LIKE '%health%'
                    OR event_description LIKE '%patient%'
                """, conn)
                
                for _, event in medical_data_events.iterrows():
                    violation = ComplianceViolation(
                        violation_id=self._generate_event_id(),
                        timestamp=datetime.now(),
                        compliance_type=ComplianceType.HIPAA,
                        violation_type="medical_data_processing",
                        description=f"医療データ処理: {event['event_description']}",
                        severity=AuditLevel.CRITICAL,
                        affected_data=event.get('data_fingerprint')
                    )
                    violations.append(violation)
                    
        except Exception as e:
            self.logger.error(f"HIPAAコンプライアンスチェックエラー: {e}")
        
        return violations
    
    def _check_custom_compliance(self) -> List[ComplianceViolation]:
        """カスタムコンプライアンスチェック"""
        violations = []
        
        try:
            # カスタムルールの実装
            # 例: データエクスポートの制限チェック
            with sqlite3.connect(self.db_path) as conn:
                export_events = pd.read_sql_query("""
                    SELECT * FROM data_access_logs 
                    WHERE access_type = 'export'
                """, conn)
                
                for _, event in export_events.iterrows():
                    # 承認なしのエクスポートチェック
                    if not event.get('approved_by'):
                        violation = ComplianceViolation(
                            violation_id=self._generate_event_id(),
                            timestamp=datetime.now(),
                            compliance_type=ComplianceType.CUSTOM,
                            violation_type="unauthorized_export",
                            description=f"承認なしのデータエクスポート: {event['data_type']}",
                            severity=AuditLevel.HIGH,
                            affected_data=event.get('data_fingerprint')
                        )
                        violations.append(violation)
                        
        except Exception as e:
            self.logger.error(f"カスタムコンプライアンスチェックエラー: {e}")
        
        return violations
    
    def _generate_event_id(self) -> str:
        """イベントIDの生成"""
        timestamp = datetime.now().isoformat()
        random_component = os.urandom(8).hex()
        return f"event_{timestamp}_{random_component}"
    
    def _determine_severity(self, operation: Dict[str, Any]) -> AuditLevel:
        """操作の重要度を判定"""
        method = operation.get('method', '').lower()
        
        if any(keyword in method for keyword in ['delete', 'remove', 'drop']):
            return AuditLevel.HIGH
        elif any(keyword in method for keyword in ['export', 'download', 'save']):
            return AuditLevel.MEDIUM
        else:
            return AuditLevel.LOW
    
    def _store_audit_event(self, event: AuditEvent):
        """監査イベントの保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events 
                    (event_id, timestamp, user_id, session_id, event_type, 
                     event_description, data_fingerprint, ip_address, user_agent, 
                     severity, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.session_id,
                    event.event_type,
                    event.event_description,
                    event.data_fingerprint,
                    event.ip_address,
                    event.user_agent,
                    event.severity.value,
                    json.dumps(event.metadata) if event.metadata else None
                ))
                
        except Exception as e:
            self.logger.error(f"監査イベント保存エラー: {e}")
            raise
    
    def _store_compliance_violation(self, violation: ComplianceViolation):
        """コンプライアンス違反の保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO compliance_violations 
                    (violation_id, timestamp, compliance_type, violation_type,
                     description, severity, affected_data, remediation_required,
                     resolved, resolution_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.violation_id,
                    violation.timestamp.isoformat(),
                    violation.compliance_type.value,
                    violation.violation_type,
                    violation.description,
                    violation.severity.value,
                    violation.affected_data,
                    violation.remediation_required,
                    violation.resolved,
                    violation.resolution_notes
                ))
                
        except Exception as e:
            self.logger.error(f"コンプライアンス違反保存エラー: {e}")
            raise
    
    def _store_data_access_log(self, access_log: DataAccessLog):
        """データアクセスログの保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO data_access_logs 
                    (access_id, timestamp, user_id, data_type, access_type,
                     data_fingerprint, justification, approved_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    access_log.access_id,
                    access_log.timestamp.isoformat(),
                    access_log.user_id,
                    access_log.data_type,
                    access_log.access_type,
                    access_log.data_fingerprint,
                    access_log.justification,
                    access_log.approved_by
                ))
                
        except Exception as e:
            self.logger.error(f"データアクセスログ保存エラー: {e}")
            raise

class ComplianceChecker:
    """コンプライアンスチェッカー"""
    
    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(__name__)
    
    def check_data_privacy_compliance(self, data: pd.DataFrame, 
                                    user_context: Dict[str, Any]) -> Dict[str, Any]:
        """データプライバシーコンプライアンスチェック"""
        try:
            compliance_report = {
                'gdpr_compliant': True,
                'hipaa_compliant': True,
                'violations': [],
                'recommendations': []
            }
            
            # 個人識別情報の検出
            pii_columns = self._detect_pii_columns(data)
            if pii_columns:
                compliance_report['gdpr_compliant'] = False
                compliance_report['violations'].append({
                    'type': 'pii_detected',
                    'columns': pii_columns,
                    'severity': 'high'
                })
                compliance_report['recommendations'].append(
                    "個人識別情報を含む列を匿名化または削除してください"
                )
            
            # 医療データの検出
            medical_columns = self._detect_medical_data(data)
            if medical_columns:
                compliance_report['hipaa_compliant'] = False
                compliance_report['violations'].append({
                    'type': 'medical_data_detected',
                    'columns': medical_columns,
                    'severity': 'critical'
                })
                compliance_report['recommendations'].append(
                    "医療データの処理には特別な許可が必要です"
                )
            
            # データエクスポート制限のチェック
            if user_context.get('access_level') == 'restricted':
                compliance_report['recommendations'].append(
                    "制限されたアクセスレベルではデータエクスポートが禁止されています"
                )
            
            return compliance_report
            
        except Exception as e:
            self.logger.error(f"データプライバシーコンプライアンスチェックエラー: {e}")
            raise
    
    def _detect_pii_columns(self, data: pd.DataFrame) -> List[str]:
        """個人識別情報を含む列の検出"""
        pii_columns = []
        
        # 一般的なPIIパターン
        pii_patterns = [
            'email', 'phone', 'address', 'ssn', 'passport', 'id',
            'name', 'birth', 'age', 'gender', 'zip', 'postal'
        ]
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in pii_patterns):
                pii_columns.append(col)
        
        return pii_columns
    
    def _detect_medical_data(self, data: pd.DataFrame) -> List[str]:
        """医療データを含む列の検出"""
        medical_columns = []
        
        # 医療データパターン
        medical_patterns = [
            'diagnosis', 'treatment', 'medication', 'symptom',
            'patient', 'medical', 'health', 'clinical', 'therapy'
        ]
        
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in medical_patterns):
                medical_columns.append(col)
        
        return medical_columns

class DataPrivacyManager:
    """データプライバシー管理システム"""
    
    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(__name__)
    
    def classify_data_sensitivity(self, data: pd.DataFrame) -> Dict[str, str]:
        """データ感度の分類"""
        try:
            sensitivity_levels = {}
            
            for col in data.columns:
                sensitivity = self._determine_column_sensitivity(data[col])
                sensitivity_levels[col] = sensitivity
            
            return sensitivity_levels
            
        except Exception as e:
            self.logger.error(f"データ感度分類エラー: {e}")
            raise
    
    def _determine_column_sensitivity(self, column: pd.Series) -> str:
        """列の感度レベルを判定"""
        # 数値データの統計的感度チェック
        if column.dtype in ['int64', 'float64']:
            # 極端な値の存在チェック
            if column.max() > column.mean() + 3 * column.std():
                return 'high'
            elif column.max() > column.mean() + 2 * column.std():
                return 'medium'
            else:
                return 'low'
        
        # カテゴリデータの感度チェック
        elif column.dtype == 'object':
            unique_ratio = column.nunique() / len(column)
            if unique_ratio > 0.8:  # 高カーディナリティ
                return 'high'
            elif unique_ratio > 0.5:
                return 'medium'
            else:
                return 'low'
        
        return 'low'
    
    def anonymize_data(self, data: pd.DataFrame, 
                      columns_to_anonymize: List[str]) -> pd.DataFrame:
        """データの匿名化"""
        try:
            anonymized_data = data.copy()
            
            for col in columns_to_anonymize:
                if col in anonymized_data.columns:
                    # ハッシュ化による匿名化
                    anonymized_data[col] = anonymized_data[col].astype(str).apply(
                        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
                    )
            
            return anonymized_data
            
        except Exception as e:
            self.logger.error(f"データ匿名化エラー: {e}")
            raise
    
    def enforce_privacy_controls(self, data: pd.DataFrame, 
                               user_context: Dict[str, Any]) -> pd.DataFrame:
        """プライバシー制御の適用"""
        try:
            controlled_data = data.copy()
            
            # ユーザーレベルの制御
            if user_context.get('access_level') == 'restricted':
                # 制限されたアクセスでは高感度データを除外
                sensitivity_levels = self.classify_data_sensitivity(data)
                high_sensitivity_cols = [
                    col for col, level in sensitivity_levels.items() 
                    if level == 'high'
                ]
                controlled_data = controlled_data.drop(columns=high_sensitivity_cols)
            
            return controlled_data
            
        except Exception as e:
            self.logger.error(f"プライバシー制御適用エラー: {e}")
            raise

def main():
    """メイン関数（テスト用）"""
    # 監査マネージャーの初期化
    audit_manager = AuditTrailManager()
    
    # テストデータの作成
    test_data = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'email': ['test1@example.com', 'test2@example.com', 'test3@example.com'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    # テストユーザーコンテキスト
    test_user = {
        'user_id': 'test_user',
        'session_id': 'test_session',
        'access_level': 'standard'
    }
    
    # 監査イベントの記録
    operation = {
        'method': 'descriptive_statistics',
        'data_fingerprint': hashlib.sha256(str(test_data).encode()).hexdigest()
    }
    
    event_id = audit_manager.log_analysis_operation(operation, test_user)
    print(f"監査イベント記録: {event_id}")
    
    # コンプライアンスチェッカーのテスト
    compliance_checker = ComplianceChecker(audit_manager)
    compliance_report = compliance_checker.check_data_privacy_compliance(test_data, test_user)
    print(f"コンプライアンスレポート: {compliance_report}")
    
    # データプライバシーマネージャーのテスト
    privacy_manager = DataPrivacyManager(audit_manager)
    sensitivity_levels = privacy_manager.classify_data_sensitivity(test_data)
    print(f"データ感度レベル: {sensitivity_levels}")
    
    # 監査レポートの生成
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    audit_report = audit_manager.generate_audit_report((start_time, end_time))
    print(f"監査レポート生成: {audit_report['total_events']}件のイベント")

if __name__ == "__main__":
    main() 