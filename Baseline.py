import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def winkler_score(y_true, y_lower, y_upper, alpha=0.1):
    """
    計算 Winkler Interval Score
    """
    scores = []
    for y, l, u in zip(y_true, y_lower, y_upper):
        if y < l:
            score = (u - l) + (2/alpha) * (l - y)
        elif l <= y <= u:
            score = (u - l)
        else:  # y > u
            score = (u - l) + (2/alpha) * (y - u)
        scores.append(score)
    return np.mean(scores)

def coverage_rate(y_true, y_lower, y_upper):
    """計算覆蓋率"""
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).sum()
    return covered / len(y_true)

class HousePricePreprocessor:
    def __init__(self):
        self.numerical_features = []
        self.categorical_features = []
        self.high_cardinality_features = []
        self.low_cardinality_features = []
        self.label_encoders = {}
        self.target_encoders = {}
        self.scaler = StandardScaler()
        
    def identify_features(self, df):
        """識別不同類型的特徵"""
        # 排除 ID 和目標變數
        exclude_cols = ['id', 'sale_price']
        
        # 數值特徵
        self.numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numerical_features = [col for col in self.numerical_features if col not in exclude_cols]
        
        # 類別特徵
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # 高基數和低基數類別特徵
        for col in self.categorical_features:
            unique_count = df[col].nunique()
            if unique_count > 50:
                self.high_cardinality_features.append(col)
            else:
                self.low_cardinality_features.append(col)
        
        print(f"數值特徵 ({len(self.numerical_features)}): {self.numerical_features[:10]}...")
        print(f"低基數類別特徵 ({len(self.low_cardinality_features)}): {self.low_cardinality_features}")
        print(f"高基數類別特徵 ({len(self.high_cardinality_features)}): {self.high_cardinality_features}")
    
    def handle_missing_values(self, df):
        """處理缺失值"""
        df_processed = df.copy()
        
        # 數值特徵：使用中位數填補
        for col in self.numerical_features:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"填補 {col} 缺失值，使用中位數: {median_val}")
        
        # 類別特徵：使用 'Unknown' 填補
        for col in self.categorical_features:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna('Unknown', inplace=True)
                print(f"填補 {col} 缺失值，使用 'Unknown'")
        
        return df_processed
    
    def create_new_features(self, df, is_train=True):
        """創建新特徵"""
        df_processed = df.copy()
        
        # 房屋年齡
        if 'year_built' in df_processed.columns:
            current_year = 2024  # 假設當前年份
            df_processed['house_age'] = current_year - df_processed['year_built']
            df_processed['house_age'] = df_processed['house_age'].clip(lower=0)
        
        # 總房間數
        room_cols = ['beds', 'bath_full', 'bath_3qtr', 'bath_half']
        available_room_cols = [col for col in room_cols if col in df_processed.columns]
        if len(available_room_cols) > 0:
            df_processed['total_rooms'] = df_processed[available_room_cols].sum(axis=1)
        
        # 價格每平方英尺 (僅訓練資料)
        if is_train and 'sale_price' in df_processed.columns and 'sqft' in df_processed.columns:
            df_processed['price_per_sqft'] = df_processed['sale_price'] / (df_processed['sqft'] + 1)
        
        # 土地使用率
        if 'sqft' in df_processed.columns and 'sqft_lot' in df_processed.columns:
            df_processed['land_usage_ratio'] = df_processed['sqft'] / (df_processed['sqft_lot'] + 1)
        
        # 改善價值比例
        if 'imp_val' in df_processed.columns and 'land_val' in df_processed.columns:
            total_val = df_processed['imp_val'] + df_processed['land_val']
            df_processed['imp_ratio'] = df_processed['imp_val'] / (total_val + 1)
        
        # 景觀特徵總和
        view_cols = [col for col in df_processed.columns if col.startswith('view_')]
        if len(view_cols) > 0:
            df_processed['total_view_score'] = df_processed[view_cols].sum(axis=1)
        
        # 更新數值特徵列表 (只在訓練時更新)
        if is_train:
            new_features = ['house_age', 'total_rooms', 'price_per_sqft', 'land_usage_ratio', 
                           'imp_ratio', 'total_view_score']
            for feature in new_features:
                if feature in df_processed.columns and feature not in self.numerical_features:
                    self.numerical_features.append(feature)
        
        return df_processed
    
    def handle_outliers(self, df, method='clip'):
        """處理異常值"""
        df_processed = df.copy()
        
        for col in self.numerical_features:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.01)
                Q3 = df_processed[col].quantile(0.99)
                
                if method == 'clip':
                    df_processed[col] = df_processed[col].clip(lower=Q1, upper=Q3)
                elif method == 'remove':
                    # 標記異常值，但不在這裡移除
                    pass
        
        return df_processed
    
    def encode_categorical_features(self, df_train, df_test=None):
        """編碼類別特徵"""
        df_train_processed = df_train.copy()
        df_test_processed = df_test.copy() if df_test is not None else None
        
        # 低基數特徵：Label Encoding
        for col in self.low_cardinality_features:
            if col in df_train_processed.columns:
                le = LabelEncoder()
                df_train_processed[col] = le.fit_transform(df_train_processed[col].astype(str))
                self.label_encoders[col] = le
                
                if df_test_processed is not None:
                    # 處理測試集中的未見值
                    test_values = df_test_processed[col].astype(str)
                    mask = test_values.isin(le.classes_)
                    df_test_processed[col] = 0  # 預設值
                    df_test_processed.loc[mask, col] = le.transform(test_values[mask])
        
        # 高基數特徵：Target Encoding (僅在有目標變數時)
        if 'sale_price' in df_train_processed.columns:
            for col in self.high_cardinality_features:
                if col in df_train_processed.columns:
                    # 計算每個類別的目標變數平均值
                    target_mean = df_train_processed.groupby(col)['sale_price'].mean()
                    global_mean = df_train_processed['sale_price'].mean()
                    
                    # 使用平滑技術
                    counts = df_train_processed.groupby(col).size()
                    smooth = 1 / (1 + np.exp(-(counts - 5) / 1))
                    target_encoding = global_mean * (1 - smooth) + target_mean * smooth
                    
                    self.target_encoders[col] = target_encoding
                    df_train_processed[col + '_encoded'] = df_train_processed[col].map(target_encoding).fillna(global_mean)
                    
                    if df_test_processed is not None:
                        df_test_processed[col + '_encoded'] = df_test_processed[col].map(target_encoding).fillna(global_mean)
                    
                    # 移除原始欄位
                    df_train_processed.drop(col, axis=1, inplace=True)
                    if df_test_processed is not None:
                        df_test_processed.drop(col, axis=1, inplace=True)
        else:
            # 測試時使用頻率編碼
            for col in self.high_cardinality_features:
                if col in df_train_processed.columns:
                    freq_encoding = df_train_processed[col].value_counts(normalize=True)
                    df_train_processed[col + '_freq'] = df_train_processed[col].map(freq_encoding)
                    
                    if df_test_processed is not None:
                        df_test_processed[col + '_freq'] = df_test_processed[col].map(freq_encoding).fillna(0)
                    
                    df_train_processed.drop(col, axis=1, inplace=True)
                    if df_test_processed is not None:
                        df_test_processed.drop(col, axis=1, inplace=True)
        
        return df_train_processed, df_test_processed
    
    def fit_transform(self, df_train, df_test=None):
        """完整的預處理流程"""
        print("開始預處理...")
        
        # 1. 識別特徵類型
        self.identify_features(df_train)
        
        # 2. 處理缺失值
        df_train = self.handle_missing_values(df_train)
        if df_test is not None:
            df_test = self.handle_missing_values(df_test)
        
        # 3. 創建新特徵
        df_train = self.create_new_features(df_train, is_train=True)
        if df_test is not None:
            df_test = self.create_new_features(df_test, is_train=False)
        
        # 4. 處理異常值
        df_train = self.handle_outliers(df_train)
        
        # 5. 編碼類別特徵
        df_train, df_test = self.encode_categorical_features(df_train, df_test)
        
        # 6. 選擇特徵用於建模 - 確保訓練和測試集有相同的特徵
        exclude_cols = ['id', 'sale_price']
        train_feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        if df_test is not None:
            test_feature_cols = [col for col in df_test.columns if col not in ['id']]
            # 取交集，確保特徵一致
            feature_cols = list(set(train_feature_cols) & set(test_feature_cols))
        else:
            feature_cols = train_feature_cols
        
        print(f"最終使用的特徵數量: {len(feature_cols)}")
        
        X_train = df_train[feature_cols]
        y_train = df_train['sale_price'] if 'sale_price' in df_train.columns else None
        
        # 7. 標準化數值特徵
        if y_train is not None:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if df_test is not None:
            X_test = df_test[feature_cols]
            X_test_scaled = self.scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        else:
            X_test = None
        
        print(f"預處理完成！最終特徵數量: {X_train.shape[1]}")
        return X_train, y_train, X_test

class BaselineModel:
    def __init__(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.lgb_lower = None
        self.lgb_upper = None
        
    def train_random_forest_baseline(self, X_train, y_train):
        """訓練 Random Forest Baseline"""
        print("訓練 Random Forest Baseline...")
        
        # 對目標變數進行對數變換
        y_log = np.log1p(y_train)
        
        # 訓練模型
        self.rf_model.fit(X_train, y_log)
        
        # 預測並計算預測區間
        y_pred_log = self.rf_model.predict(X_train)
        y_pred = np.expm1(y_pred_log)
        
        # 計算殘差的標準差
        residuals = y_train - y_pred
        residual_std = np.std(residuals)
        
        # 90% 信賴區間 (假設常數方差)
        lower_bound = y_pred - 1.645 * residual_std
        upper_bound = y_pred + 1.645 * residual_std
        
        # 確保邊界為正值
        lower_bound = np.maximum(lower_bound, 0)
        
        # 評估
        winkler = winkler_score(y_train, lower_bound, upper_bound)
        coverage = coverage_rate(y_train, lower_bound, upper_bound)
        
        print(f"Random Forest Baseline 結果:")
        print(f"Winkler Score: {winkler:.2f}")
        print(f"Coverage Rate: {coverage:.3f}")
        print(f"Average Interval Width: {np.mean(upper_bound - lower_bound):.2f}")
        
        return {
            'winkler_score': winkler,
            'coverage_rate': coverage,
            'avg_width': np.mean(upper_bound - lower_bound),
            'residual_std': residual_std
        }
    
    def train_lgb_quantile(self, X_train, y_train):
        """訓練 LightGBM Quantile Regression"""
        print("訓練 LightGBM Quantile Regression...")
        
        # 對目標變數進行對數變換
        y_log = np.log1p(y_train)
        
        # 5% 分位數模型
        self.lgb_lower = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.05,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbose=-1
        )
        
        # 95% 分位數模型  
        self.lgb_upper = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.95,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbose=-1
        )
        
        # 訓練兩個模型
        self.lgb_lower.fit(X_train, y_log)
        self.lgb_upper.fit(X_train, y_log)
        
        # 預測
        y_lower_log = self.lgb_lower.predict(X_train)
        y_upper_log = self.lgb_upper.predict(X_train)
        
        y_lower = np.expm1(y_lower_log)
        y_upper = np.expm1(y_upper_log)
        
        # 確保 lower <= upper
        y_lower = np.minimum(y_lower, y_upper)
        
        # 評估
        winkler = winkler_score(y_train, y_lower, y_upper)
        coverage = coverage_rate(y_train, y_lower, y_upper)
        
        print(f"LightGBM Quantile Regression 結果:")
        print(f"Winkler Score: {winkler:.2f}")
        print(f"Coverage Rate: {coverage:.3f}")
        print(f"Average Interval Width: {np.mean(y_upper - y_lower):.2f}")
        
        return {
            'winkler_score': winkler,
            'coverage_rate': coverage,
            'avg_width': np.mean(y_upper - y_lower)
        }
    
    def predict_intervals(self, X_test):
        """預測測試集的區間"""
        if self.lgb_lower is None or self.lgb_upper is None:
            raise ValueError("請先訓練 LightGBM 模型")
        
        # 預測對數空間的分位數
        y_lower_log = self.lgb_lower.predict(X_test)
        y_upper_log = self.lgb_upper.predict(X_test)
        
        # 轉換回原始空間
        y_lower = np.expm1(y_lower_log)
        y_upper = np.expm1(y_upper_log)
        
        # 確保 lower <= upper
        y_lower = np.minimum(y_lower, y_upper)
        
        return y_lower, y_upper

def main():
    """主函數"""
    print("載入資料...")
    train_df = pd.read_csv('dataset.csv')
    test_df = pd.read_csv('test.csv')
    
    # 初始化預處理器
    preprocessor = HousePricePreprocessor()
    
    # 預處理
    X_train, y_train, X_test = preprocessor.fit_transform(train_df, test_df)
    
    # 分割訓練和驗證集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 初始化模型
    model = BaselineModel()
    
    # 訓練 Random Forest Baseline
    rf_results = model.train_random_forest_baseline(X_train_split, y_train_split)
    
    # 訓練 LightGBM Quantile Regression
    lgb_results = model.train_lgb_quantile(X_train_split, y_train_split)
    
    # 在驗證集上評估 LightGBM
    print("\n在驗證集上評估 LightGBM...")
    val_lower, val_upper = model.predict_intervals(X_val_split)
    val_winkler = winkler_score(y_val_split, val_lower, val_upper)
    val_coverage = coverage_rate(y_val_split, val_lower, val_upper)
    
    print(f"驗證集結果:")
    print(f"Winkler Score: {val_winkler:.2f}")
    print(f"Coverage Rate: {val_coverage:.3f}")
    print(f"Average Interval Width: {np.mean(val_upper - val_lower):.2f}")
    
    # 在完整訓練集上重新訓練最佳模型
    print("\n在完整訓練集上重新訓練...")
    model.train_lgb_quantile(X_train, y_train)
    
    # 預測測試集
    print("預測測試集...")
    test_lower, test_upper = model.predict_intervals(X_test)
    
    # 創建提交檔案
    submission = pd.DataFrame({
        'id': test_df['id'],
        'pi_lower': test_lower,
        'pi_upper': test_upper
    })
    
    submission.to_csv('submission.csv', index=False)
    print("提交檔案已保存為 submission.csv")
    
    # 顯示提交檔案預覽
    print("\n提交檔案預覽:")
    print(submission.head(10))
    
    return submission

if __name__ == "__main__":
    submission = main()