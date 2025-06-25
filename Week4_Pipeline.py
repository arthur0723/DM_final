import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from mapie.regression import MapieQuantileRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
from datetime import datetime
import time
import joblib

class AdvancedHousePricePredictor:
    """
    進階房價預測區間模型
    """
    
    def __init__(self, target_coverage: float = 0.9, random_state: int = 42):
        self.target_coverage = target_coverage
        self.alpha = 1 - target_coverage
        self.random_state = random_state
        
        # 模型組件
        self.feature_processor = None
        self.base_models = {}
        self.cqr_models = {}
        self.ensemble_weights = {}
        
        # 特徵相關
        self.selected_features = []
        self.feature_importance = {}
        
    def advanced_feature_engineering(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        進階特徵工程，不使用目標變數
        """
        df_processed = df.copy()
        
        print("開始進階特徵工程...")
        
        # 1. 時間特徵工程
        if 'sale_date' in df_processed.columns:
            df_processed['sale_date'] = pd.to_datetime(df_processed['sale_date'])
            df_processed['sale_year'] = df_processed['sale_date'].dt.year
            df_processed['sale_month'] = df_processed['sale_date'].dt.month
            df_processed['sale_quarter'] = df_processed['sale_date'].dt.quarter
            df_processed['sale_day_of_year'] = df_processed['sale_date'].dt.dayofyear
            
            # 季節性特徵
            df_processed['is_spring'] = (df_processed['sale_month'].isin([3, 4, 5])).astype(int)
            df_processed['is_summer'] = (df_processed['sale_month'].isin([6, 7, 8])).astype(int)
            df_processed['is_fall'] = (df_processed['sale_month'].isin([9, 10, 11])).astype(int)
            df_processed['is_winter'] = (df_processed['sale_month'].isin([12, 1, 2])).astype(int)
        
        # 2. 數據特徵工程
        if 'year_built' in df_processed.columns:
            current_year = datetime.now().year
            df_processed['house_age'] = current_year - df_processed['year_built']
            df_processed['house_age_squared'] = df_processed['house_age'] ** 2
            
            # 房屋年代分類
            df_processed['age_category'] = pd.cut(
                df_processed['house_age'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['new', 'modern', 'middle', 'old', 'historic']
            ).astype(str)
        
        # 3. 空間面積特徵
        area_features = ['sqft', 'sqft_lot', 'sqft_basement', 'sqft_garage']
        available_area_features = [col for col in area_features if col in df_processed.columns]
        
        if len(available_area_features) >= 2:
            # 總建築面積
            if 'sqft' in df_processed.columns and 'sqft_basement' in df_processed.columns:
                df_processed['total_building_sqft'] = df_processed['sqft'] + df_processed['sqft_basement'].fillna(0)
            
            # 土地利用率
            if 'sqft' in df_processed.columns and 'sqft_lot' in df_processed.columns:
                df_processed['land_usage_ratio'] = df_processed['sqft'] / (df_processed['sqft_lot'] + 1)
                df_processed['lot_size_category'] = pd.qcut(
                    df_processed['sqft_lot'],
                    q=5,
                    labels=['tiny', 'small', 'medium', 'large', 'huge']
                ).astype(str)
        
        # 4. 房間特徵工程
        room_features = ['beds', 'bath_full', 'bath_3qtr', 'bath_half']
        available_room_features = [col for col in room_features if col in df_processed.columns]
        
        if len(available_room_features) > 0:
            df_processed['total_rooms'] = df_processed[available_room_features].sum(axis=1)
            
            # 房間密度
            if 'sqft' in df_processed.columns:
                df_processed['room_density'] = df_processed['total_rooms'] / (df_processed['sqft'] + 1)
            
            # 浴室品質分數
            if all(col in df_processed.columns for col in ['bath_full', 'bath_3qtr', 'bath_half']):
                df_processed['bathroom_score'] = (
                    df_processed['bath_full'] * 1.0 +
                    df_processed['bath_3qtr'] * 0.75 +
                    df_processed['bath_half'] * 0.5
                )
        
        # 5. 品質與價值特徵
        value_features = ['imp_val', 'land_val']
        if all(col in df_processed.columns for col in value_features):
            df_processed['total_assessed_value'] = df_processed['imp_val'] + df_processed['land_val']
            df_processed['improvement_ratio'] = df_processed['imp_val'] / (df_processed['total_assessed_value'] + 1)
            
            # 價值等級
            df_processed['value_tier'] = pd.qcut(
                df_processed['total_assessed_value'],
                q=4,
                labels=['budget', 'mid', 'premium', 'luxury']
            ).astype(str)
        
        # 6. 景觀特徵聚合
        view_features = [col for col in df_processed.columns if col.startswith('view_')]
        if len(view_features) > 0:
            df_processed['total_view_score'] = df_processed[view_features].sum(axis=1)
            df_processed['has_premium_view'] = (df_processed['total_view_score'] > 0).astype(int)
            
            # 景觀類型統計
            df_processed['num_view_types'] = (df_processed[view_features] > 0).sum(axis=1)
        
        # 7. 地理位置特徵
        if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
            # 位置聚類特徵（使用簡單的分桶方法）
            df_processed['lat_bucket'] = pd.cut(df_processed['latitude'], bins=20, labels=False)
            df_processed['lon_bucket'] = pd.cut(df_processed['longitude'], bins=20, labels=False)
            df_processed['location_cluster'] = df_processed['lat_bucket'].astype(str) + '_' + df_processed['lon_bucket'].astype(str)
        
        # 8. 互動特徵
        if 'grade' in df_processed.columns and 'sqft' in df_processed.columns:
            df_processed['grade_sqft_interaction'] = df_processed['grade'] * df_processed['sqft']
        
        if 'beds' in df_processed.columns and 'bath_full' in df_processed.columns:
            df_processed['bed_bath_ratio'] = df_processed['beds'] / (df_processed['bath_full'] + 1)
        
        print(f"特徵工程完成，新增特徵數量: {len(df_processed.columns) - len(df.columns)}")
        return df_processed
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        準備訓練和測試特徵
        """
        print("開始特徵準備...")
        
        # 特徵工程
        train_processed = self.advanced_feature_engineering(train_df, is_train=True)
        if test_df is not None:
            test_processed = self.advanced_feature_engineering(test_df, is_train=False)
        else:
            test_processed = None
        
        # 處理類別特徵
        categorical_features = train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_features:
            if col in ['id', 'sale_date']:
                continue
                
            # 高基數特徵使用頻率編碼
            unique_count = train_processed[col].nunique()
            
            if unique_count > 50:
                freq_map = train_processed[col].value_counts(normalize=True).to_dict()
                train_processed[col + '_freq'] = train_processed[col].map(freq_map).fillna(0)
                
                if test_processed is not None:
                    test_processed[col + '_freq'] = test_processed[col].map(freq_map).fillna(0)
                
                train_processed.drop(col, axis=1, inplace=True)
                if test_processed is not None:
                    test_processed.drop(col, axis=1, inplace=True)
            
            else:
                # 低基數特徵使用 Label Encoding
                le = LabelEncoder()
                train_processed[col] = le.fit_transform(train_processed[col].astype(str))
                
                if test_processed is not None:
                    test_values = test_processed[col].astype(str)
                    mask = test_values.isin(le.classes_)
                    test_processed[col] = 0
                    test_processed.loc[mask, col] = le.transform(test_values[mask])
        
        # 處理缺失值
        for col in train_processed.columns:
            if train_processed[col].isnull().sum() > 0:
                if train_processed[col].dtype in ['float64', 'int64']:
                    train_processed[col].fillna(train_processed[col].median(), inplace=True)
                else:
                    train_processed[col].fillna(train_processed[col].mode()[0] if len(train_processed[col].mode()) > 0 else 'Unknown', inplace=True)
        
        if test_processed is not None:
            for col in test_processed.columns:
                if test_processed[col].isnull().sum() > 0:
                    if test_processed[col].dtype in ['float64', 'int64']:
                        test_processed[col].fillna(train_processed[col].median(), inplace=True)
                    else:
                        test_processed[col].fillna(train_processed[col].mode()[0] if len(train_processed[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # 特徵選擇
        if test_processed is not None:
            exclude_cols = ['id', 'sale_price', 'sale_date']
            train_features = [col for col in train_processed.columns if col not in exclude_cols]
            test_features = [col for col in test_processed.columns if col not in ['id', 'sale_date']]
            common_features = list(set(train_features) & set(test_features))
            self.selected_features = common_features
        else:
            exclude_cols = ['id', 'sale_price', 'sale_date']
            self.selected_features = [col for col in train_processed.columns if col not in exclude_cols]
        
        print(f"最終選擇特徵數量: {len(self.selected_features)}")
        return train_processed, test_processed
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        訓練基礎分位數回歸模型
        """
        print("訓練基礎分位數回歸模型...")
        
        # 對數變換目標變數
        y_log = np.log1p(y_train)
        
        # 模型配置
        models_config = {
            'lgb_lower': lgb.LGBMRegressor(
                objective='quantile',
                alpha=self.alpha/2,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=self.random_state,
                verbose=-1
            ),
            'lgb_upper': lgb.LGBMRegressor(
                objective='quantile',
                alpha=1-self.alpha/2,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=self.random_state,
                verbose=-1
            ),
            'hgb_lower': HistGradientBoostingRegressor(
                loss='quantile',
                quantile=self.alpha/2,
                max_iter=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=self.random_state
            ),
            'hgb_upper': HistGradientBoostingRegressor(
                loss='quantile',
                quantile=1-self.alpha/2,
                max_iter=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=self.random_state
            )
        }
        
        # 訓練模型
        for name, model in models_config.items():
            print(f"訓練 {name}...")
            try:
                model.fit(X_train, y_log)
                self.base_models[name] = model
            except Exception as e:
                print(f"訓練 {name} 失敗: {e}")
        
        return self.base_models
    
    def train_cqr_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        訓練 Conformalized Quantile Regression 模型
        """
        print("訓練 Conformalized Quantile Regression 模型...")
        
        # 對數變換
        y_log = np.log1p(y_train)
        
        # 準備多個基礎模型的CQR
        base_estimators = {
            'lgb': lgb.LGBMRegressor(
                objective='quantile',
                alpha=0.5,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=self.random_state,
                verbose=-1
            ),
            'hgb': HistGradientBoostingRegressor(
                loss='quantile',
                quantile=0.5,
                max_iter=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=self.random_state
            )
        }
        
        # 檢查 mapie 版本
        import mapie
        print(f"MAPIE 版本: {mapie.__version__}")
        
        # 為每個基礎估計器創建CQR模型
        for name, estimator in base_estimators.items():
            print(f"訓練 CQR-{name}...")
            
            # 創建 MapieQuantileRegressor
            try:
                cqr_model = MapieQuantileRegressor(
                    estimator=estimator,
                    cv="split",
                    alpha=self.alpha
                )
                
                # 訓練CQR模型
                cqr_model.fit(X_train, y_log)
                self.cqr_models[f'cqr_{name}'] = cqr_model
            except Exception as e:
                print(f"CQR-{name} 訓練失敗: {e}")
        
        return self.cqr_models
    
    def predict_intervals(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測區間，結合多種方法
        """
        print("開始預測區間...")
        
        all_lower_preds = []
        all_upper_preds = []
        
        # 1. 基礎分位數回歸預測
        for model_type in ['lgb', 'hgb']:
            if f'{model_type}_lower' in self.base_models:
                lower_model = self.base_models[f'{model_type}_lower']
                upper_model = self.base_models[f'{model_type}_upper']
                
                lower_pred_log = lower_model.predict(X_test)
                upper_pred_log = upper_model.predict(X_test)
                
                # 轉換回原始空間
                lower_pred = np.expm1(lower_pred_log)
                upper_pred = np.expm1(upper_pred_log)
                
                print(f"{model_type} 基礎模型預測形狀: 下界 {lower_pred.shape}, 上界 {upper_pred.shape}")
                
                all_lower_preds.append(lower_pred)
                all_upper_preds.append(upper_pred)
        
        # 2. CQR預測
        for name, cqr_model in self.cqr_models.items():
            try:
                y_pred, y_pis = cqr_model.predict(X_test, alpha=self.alpha)
                
                # 確保 y_pis 是正確形狀
                if y_pis.ndim == 3:
                    y_pis = y_pis.squeeze(axis=2)  # 移除多餘維度
                
                # 轉換回原始空間
                lower_pred = np.expm1(y_pis[:, 0])
                upper_pred = np.expm1(y_pis[:, 1])
                
                print(f"CQR-{name} 預測形狀: 下界 {lower_pred.shape}, 上界 {upper_pred.shape}")
                
                all_lower_preds.append(lower_pred)
                all_upper_preds.append(upper_pred)
            except Exception as e:
                print(f"CQR-{name} 預測失敗: {e}")
        
        # 3. 檢查預測數組形狀
        if len(all_lower_preds) > 0:
            try:
                # 確保所有預測數組形狀一致
                shapes = [pred.shape for pred in all_lower_preds]
                if len(set(shapes)) > 1:
                    print(f"警告：預測數組形狀不一致: {shapes}")
                    # 取最小長度
                    min_len = min([len(pred) for pred in all_lower_preds])
                    all_lower_preds = [pred[:min_len] for pred in all_lower_preds]
                    all_upper_preds = [pred[:min_len] for pred in all_upper_preds]
                
                # 轉為數組並計算中位數
                all_lower_preds = np.array(all_lower_preds)
                all_upper_preds = np.array(all_upper_preds)
                
                final_lower = np.median(all_lower_preds, axis=0)
                final_upper = np.median(all_upper_preds, axis=0)
            except Exception as e:
                print(f"集成預測失敗: {e}")
                # 後備方案：使用第一個成功預測
                final_lower = all_lower_preds[0]
                final_upper = all_upper_preds[0]
        else:
            print("警告：沒有成功的預測模型，使用基本方法")
            from sklearn.linear_model import LinearRegression
            backup_model = LinearRegression()
            y_log = np.log1p(self.y_train_backup) if hasattr(self, 'y_train_backup') else np.zeros(len(X_test))
            backup_model.fit(self.X_train_backup if hasattr(self, 'X_train_backup') else X_test, y_log)
            
            pred_log = backup_model.predict(X_test)
            pred = np.expm1(pred_log)
            
            # 使用預測值的±30%作為區間
            final_lower = pred * 0.7
            final_upper = pred * 1.3
        
        # 確保 lower <= upper
        final_lower = np.minimum(final_lower, final_upper)
        
        return final_lower, final_upper
    
    def evaluate_model(self, y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, float]:
        """
        評估模型性能
        """
        # Winkler Score
        winkler_scores = []
        for y, l, u in zip(y_true, y_lower, y_upper):
            if y < l:
                score = (u - l) + (2/self.alpha) * (l - y)
            elif l <= y <= u:
                score = (u - l)
            else:  # y > u
                score = (u - l) + (2/self.alpha) * (y - u)
            winkler_scores.append(score)
        
        winkler_score = np.mean(winkler_scores)
        
        # 覆蓋率
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        
        # 平均區間寬度
        avg_width = np.mean(y_upper - y_lower)
        
        # 區間寬度變異係數
        width_cv = np.std(y_upper - y_lower) / avg_width if avg_width > 0 else 0
        
        return {
            'winkler_score': winkler_score,
            'coverage_rate': coverage,
            'avg_width': avg_width,
            'width_cv': width_cv,
            'coverage_deviation': abs(coverage - self.target_coverage)
        }
    
    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        """
        完整的訓練流程
        """
        print("開始完整訓練流程...")
        
        # 1. 特徵準備
        start_time = time.time()
        train_processed, test_processed = self.prepare_features(train_df, test_df)
        print(f"特徵準備完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 2. 準備訓練數據
        X_train = train_processed[self.selected_features]
        y_train = train_processed['sale_price']
        
        # 儲存用於後備方案
        self.X_train_backup = X_train.copy()
        self.y_train_backup = y_train.copy()
        
        # 3. 數據標準化
        start_time = time.time()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        print(f"數據標準化完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 4. 訓練基礎模型
        start_time = time.time()
        try:
            self.train_base_models(X_train_scaled, y_train)
        except Exception as e:
            print(f"基礎模型訓練出現錯誤：{e}")
        print(f"基礎模型訓練完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 5. 訓練CQR模型
        start_time = time.time()
        try:
            self.train_cqr_models(X_train_scaled, y_train)
        except Exception as e:
            print(f"CQR模型訓練出現錯誤：{e}")
        print(f"CQR模型訓練完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 6. 驗證模型
        start_time = time.time()
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=self.random_state
        )
        
        try:
            self.train_base_models(X_train_split, y_train_split)
            self.train_cqr_models(X_train_split, y_train_split)
            
            val_lower, val_upper = self.predict_intervals(X_val_split)
            val_metrics = self.evaluate_model(y_val_split.values, val_lower, val_upper)
            
            print(f"驗證結果:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            print(f"驗證過程出現錯誤：{e}")
            val_metrics = {'error': str(e)}
        print(f"驗證完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 7. 在完整數據上重新訓練
        start_time = time.time()
        try:
            self.train_base_models(X_train_scaled, y_train)
            self.train_cqr_models(X_train_scaled, y_train)
        except Exception as e:
            print(f"最終訓練出現錯誤：{e}")
        print(f"最終訓練完成，耗時: {time.time() - start_time:.2f} 秒")
        
        # 8. 如果有測試集，進行預測
        if test_processed is not None:
            start_time = time.time()
            X_test = test_processed[self.selected_features]
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            test_lower, test_upper = self.predict_intervals(X_test_scaled)
            
            # 創建提交文件
            submission = pd.DataFrame({
                'id': test_processed['id'],
                'pi_lower': test_lower,
                'pi_upper': test_upper
            })
            
            print(f"測試預測完成，耗時: {time.time() - start_time:.2f} 秒")
            return submission, val_metrics
        
        return None, val_metrics

def main():
    """主函數"""
    print("載入數據...")
    start_time = time.time()
    train_df = pd.read_csv('dataset.csv')
    test_df = pd.read_csv('test.csv')
    print(f"數據載入完成，耗時: {time.time() - start_time:.2f} 秒")
    
    # 創建進階預測器
    predictor = AdvancedHousePricePredictor(target_coverage=0.9, random_state=42)
    
    # 訓練模型並預測
    start_time = time.time()
    submission, val_metrics = predictor.fit(train_df, test_df)
    print(f"模型訓練與預測完成，耗時: {time.time() - start_time:.2f} 秒")
    
    if submission is not None:
        # 保存提交文件
        submission.to_csv('advanced_submission.csv', index=False)
        print("進階提交文件已保存為 advanced_submission.csv")
        
        # 顯示提交文件預覽
        print("\n提交文件預覽:")
        print(submission.head(10))
        
        # 預測區間統計
        print(f"\n預測區間統計:")
        print(f"平均下界: {submission['pi_lower'].mean():.2f}")
        print(f"平均上界: {submission['pi_upper'].mean():.2f}")
        print(f"平均間寬度: {(submission['pi_upper'] - submission['pi_lower']).mean():.2f}")
    
    return submission, val_metrics

if __name__ == "__main__":
    submission, metrics = main()