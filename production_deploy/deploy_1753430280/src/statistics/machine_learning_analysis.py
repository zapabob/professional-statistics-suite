#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Machine Learning and Deep Learning Analysis Module
機械学習・深層学習分析モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MachineLearningAnalyzer:
    """機械学習・深層学習分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
    def classification_analysis(self, data: pd.DataFrame, target_col: str, 
                              test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        分類分析
        
        Args:
            data: データフレーム
            target_col: 目的変数名
            test_size: テストデータの割合
            random_state: 乱数シード
        """
        try:
            # データ準備
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['classification'] = encoders
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['classification'] = scaler
            
            # モデル定義
            models = {
                'Logistic Regression': LogisticRegression(random_state=random_state),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'SVM': SVC(probability=True, random_state=random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Gradient Boosting': GradientBoostingClassifier(random_state=random_state)
            }
            
            results = {}
            
            for name, model in models.items():
                # モデル訓練
                model.fit(X_train_scaled, y_train)
                self.models[f'classification_{name}'] = model
                
                # 予測
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None
                
                # 評価指標
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # ROC-AUC（2クラス分類の場合）
                roc_auc = None
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # 混同行列
                cm = confusion_matrix(y_test, y_pred)
                
                # クロスバリデーション
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                # 特徴量重要度（Random Forestの場合）
                if name == 'Random Forest':
                    feature_importance = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[f'classification_{name}'] = importance_df
                    results[name]['feature_importance'] = importance_df.to_dict('records')
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def regression_analysis(self, data: pd.DataFrame, target_col: str, 
                          test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        回帰分析
        
        Args:
            data: データフレーム
            target_col: 目的変数名
            test_size: テストデータの割合
            random_state: 乱数シード
        """
        try:
            # データ準備
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['regression'] = encoders
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=test_size, random_state=random_state
            )
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['regression'] = scaler
            
            # モデル定義
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'SVR': SVR(kernel='rbf'),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
            
            results = {}
            
            for name, model in models.items():
                # モデル訓練
                model.fit(X_train_scaled, y_train)
                self.models[f'regression_{name}'] = model
                
                # 予測
                y_pred = model.predict(X_test_scaled)
                
                # 評価指標
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # クロスバリデーション
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred.tolist(),
                    'actual': y_test.tolist()
                }
                
                # 特徴量重要度（Random Forestの場合）
                if name == 'Random Forest':
                    feature_importance = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[f'regression_{name}'] = importance_df
                    results[name]['feature_importance'] = importance_df.to_dict('records')
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clustering_analysis(self, data: pd.DataFrame, n_clusters: int = 3, 
                          method: str = 'kmeans') -> Dict[str, Any]:
        """
        クラスタリング分析
        
        Args:
            data: データフレーム
            n_clusters: クラスタ数
            method: クラスタリング手法
        """
        try:
            # データ準備
            X = data.copy()
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['clustering'] = encoders
            
            # スケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            self.scalers['clustering'] = scaler
            
            # クラスタリング手法選択
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                return {"success": False, "error": "無効なクラスタリング手法"}
            
            # クラスタリング実行
            clusters = model.fit_predict(X_scaled)
            self.models[f'clustering_{method}'] = model
            
            # 結果分析
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = clusters
            
            # クラスタごとの統計
            cluster_stats = data_with_clusters.groupby('cluster').agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            # クラスタサイズ
            cluster_sizes = data_with_clusters['cluster'].value_counts().to_dict()
            
            # クラスタ間の距離（K-meansの場合）
            cluster_centers = None
            if method == 'kmeans':
                cluster_centers = model.cluster_centers_
                # スケールを元に戻す
                cluster_centers_original = scaler.inverse_transform(cluster_centers)
                cluster_centers_df = pd.DataFrame(
                    cluster_centers_original,
                    columns=X.columns,
                    index=[f'Cluster_{i}' for i in range(n_clusters)]
                )
            
            results = {
                'method': method,
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'cluster_statistics': cluster_stats.to_dict(),
                'cluster_centers': cluster_centers_df.to_dict() if cluster_centers_df is not None else None,
                'silhouette_score': self._calculate_silhouette_score(X_scaled, clusters) if method != 'dbscan' else None
            }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def dimensionality_reduction(self, data: pd.DataFrame, n_components: int = 2, 
                               method: str = 'pca') -> Dict[str, Any]:
        """
        次元削減分析
        
        Args:
            data: データフレーム
            n_components: 削減後の次元数
            method: 次元削減手法
        """
        try:
            # データ準備
            X = data.copy()
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['dimensionality'] = encoders
            
            # スケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            self.scalers['dimensionality'] = scaler
            
            # 次元削減手法選択
            if method == 'pca':
                model = PCA(n_components=n_components, random_state=42)
            elif method == 'factor':
                model = FactorAnalysis(n_components=n_components, random_state=42)
            elif method == 'tsne':
                model = TSNE(n_components=n_components, random_state=42)
            else:
                return {"success": False, "error": "無効な次元削減手法"}
            
            # 次元削減実行
            X_reduced = model.fit_transform(X_scaled)
            self.models[f'dimensionality_{method}'] = model
            
            # 結果分析
            if method == 'pca':
                explained_variance_ratio = model.explained_variance_ratio_
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
                
                results = {
                    'method': method,
                    'n_components': n_components,
                    'explained_variance_ratio': explained_variance_ratio.tolist(),
                    'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
                    'total_explained_variance': cumulative_variance_ratio[-1],
                    'reduced_data': X_reduced.tolist()
                }
            else:
                results = {
                    'method': method,
                    'n_components': n_components,
                    'reduced_data': X_reduced.tolist()
                }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def feature_selection(self, data: pd.DataFrame, target_col: str, 
                         n_features: int = 10, method: str = 'f_classif') -> Dict[str, Any]:
        """
        特徴量選択
        
        Args:
            data: データフレーム
            target_col: 目的変数名
            n_features: 選択する特徴量数
            method: 選択手法
        """
        try:
            # データ準備
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['feature_selection'] = encoders
            
            # 特徴量選択手法
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=n_features)
            elif method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            else:
                return {"success": False, "error": "無効な特徴量選択手法"}
            
            # 特徴量選択実行
            X_selected = selector.fit_transform(X_encoded, y)
            self.models['feature_selection'] = selector
            
            # 選択された特徴量の情報
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = selector.scores_[selector.get_support()]
            
            # 特徴量重要度ランキング
            feature_ranking = pd.DataFrame({
                'feature': X.columns,
                'score': selector.scores_,
                'p_value': selector.pvalues_
            }).sort_values('score', ascending=False)
            
            results = {
                'method': method,
                'n_selected_features': len(selected_features),
                'selected_features': selected_features,
                'feature_scores': feature_scores.tolist(),
                'feature_ranking': feature_ranking.to_dict('records'),
                'selected_data_shape': X_selected.shape
            }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def hyperparameter_tuning(self, data: pd.DataFrame, target_col: str, 
                            model_type: str = 'classification', 
                            test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        ハイパーパラメータチューニング
        
        Args:
            data: データフレーム
            target_col: 目的変数名
            model_type: モデルタイプ
            test_size: テストデータの割合
            random_state: 乱数シード
        """
        try:
            # データ準備
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # カテゴリ変数のエンコーディング
            X_encoded, encoders = self._encode_categorical_features(X)
            self.encoders['hyperparameter_tuning'] = encoders
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=test_size, random_state=random_state,
                stratify=y if model_type == 'classification' else None
            )
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['hyperparameter_tuning'] = scaler
            
            # モデルとパラメータグリッド定義
            if model_type == 'classification':
                model = RandomForestClassifier(random_state=random_state)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(random_state=random_state)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                scoring = 'r2'
            
            # グリッドサーチ
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring, n_jobs=-1, random_state=random_state
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # 最適モデル
            best_model = grid_search.best_estimator_
            self.models['hyperparameter_tuning_best'] = best_model
            
            # 予測
            y_pred = best_model.predict(X_test_scaled)
            
            # 評価
            if model_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                results = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_accuracy': accuracy,
                    'cv_results': grid_search.cv_results_
                }
            else:
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                results = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_r2': r2,
                    'cv_results': grid_search.cv_results_
                }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """カテゴリ変数のエンコーディング"""
        X_encoded = X.copy()
        encoders = {}
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        
        return X_encoded, encoders
    
    def _calculate_silhouette_score(self, X: np.ndarray, clusters: np.ndarray) -> float:
        """シルエットスコア計算"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, clusters)
        except:
            return None 