import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# 設定繪圖風格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_basic_info():
    """載入資料並顯示基本資訊"""
    print("=" * 60)
    print("1. 載入資料並顯示基本資訊")
    print("=" * 60)
    
    # 載入資料
    train_df = pd.read_csv('dataset.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_submission.shape}")
    
    print("\nTraining data columns:")
    print(train_df.columns.tolist())
    
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print("\nBasic statistics:")
    print(train_df.describe())
    
    print("\nData types:")
    print(train_df.dtypes.value_counts())
    
    return train_df, test_df, sample_submission

def missing_value_analysis(df):
    """缺失值分析"""
    print("\n" + "=" * 60)
    print("2. 缺失值分析")
    print("=" * 60)
    
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    print(f"總共有 {len(missing_table)} 個特徵有缺失值")
    print("\nTop 20 features with missing values:")
    print(missing_table.head(20))
    
    # 視覺化缺失值
    plt.figure(figsize=(12, 8))
    if len(missing_table) > 0:
        top_missing = missing_table.head(20)
        plt.barh(range(len(top_missing)), top_missing['Missing Percentage'])
        plt.yticks(range(len(top_missing)), top_missing.index)
        plt.xlabel('Missing Percentage (%)')
        plt.title('Top 20 Features with Missing Values')
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return missing_table

def target_variable_analysis(df):
    """目標變數分析"""
    print("\n" + "=" * 60)
    print("3. 目標變數 (SalePrice) 分析")
    print("=" * 60)
    
    target_col = 'sale_price'  # 假設目標變數名稱
    if target_col not in df.columns:
        # 嘗試找到可能的目標變數
        possible_targets = [col for col in df.columns if 'price' in col.lower() or 'sale' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            print(f"找到可能的目標變數: {target_col}")
        else:
            print("無法找到明確的目標變數，請檢查資料")
            return None
    
    target = df[target_col]
    
    print(f"Target variable: {target_col}")
    print(f"Mean: ${target.mean():,.2f}")
    print(f"Median: ${target.median():,.2f}")
    print(f"Std: ${target.std():,.2f}")
    print(f"Min: ${target.min():,.2f}")
    print(f"Max: ${target.max():,.2f}")
    print(f"Skewness: {skew(target):.4f}")
    print(f"Kurtosis: {kurtosis(target):.4f}")
    
    # 視覺化目標變數分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始分布
    axes[0,0].hist(target, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title(f'{target_col} Distribution')
    axes[0,0].set_xlabel('Price')
    axes[0,0].set_ylabel('Frequency')
    
    # 對數變換分布
    log_target = np.log1p(target)
    axes[0,1].hist(log_target, bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_title(f'Log({target_col}) Distribution')
    axes[0,1].set_xlabel('Log(Price)')
    axes[0,1].set_ylabel('Frequency')
    
    # Q-Q plot
    stats.probplot(target, dist="norm", plot=axes[1,0])
    axes[1,0].set_title(f'{target_col} Q-Q Plot')
    
    # Box plot
    axes[1,1].boxplot(target)
    axes[1,1].set_title(f'{target_col} Box Plot')
    axes[1,1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return target_col

def numerical_features_analysis(df, target_col):
    """數值特徵分析"""
    print("\n" + "=" * 60)
    print("4. 數值特徵分析")
    print("=" * 60)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_features:
        numerical_features.remove(target_col)
    
    print(f"總共有 {len(numerical_features)} 個數值特徵")
    
    # 計算與目標變數的相關性
    correlations = []
    for feature in numerical_features:
        if df[feature].notna().sum() > 0:  # 確保有非缺失值
            corr = df[feature].corr(df[target_col])
            correlations.append((feature, corr))
    
    correlations.sort(key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0, reverse=True)
    
    print("\nTop 15 features most correlated with target:")
    for feature, corr in correlations[:15]:
        print(f"{feature}: {corr:.4f}")
    
    # 相關性熱力圖 (top 20 features)
    top_features = [item[0] for item in correlations[:20] if not pd.isna(item[1])]
    if len(top_features) > 0:
        corr_matrix = df[top_features + [target_col]].corr()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Top Numerical Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return correlations

def categorical_features_analysis(df):
    """類別特徵分析"""
    print("\n" + "=" * 60)
    print("5. 類別特徵分析")
    print("=" * 60)
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    print(f"總共有 {len(categorical_features)} 個類別特徵")
    
    # 分析每個類別特徵的唯一值數量
    cat_info = []
    for feature in categorical_features:
        unique_count = df[feature].nunique()
        missing_count = df[feature].isnull().sum()
        cat_info.append((feature, unique_count, missing_count))
    
    cat_info.sort(key=lambda x: x[1], reverse=True)
    
    print("\nCategorical features info:")
    print("Feature | Unique Values | Missing Values")
    print("-" * 50)
    for feature, unique, missing in cat_info:
        print(f"{feature:<20} | {unique:<12} | {missing}")
    
    # 高基數和低基數特徵
    high_cardinality = [item[0] for item in cat_info if item[1] > 20]
    low_cardinality = [item[0] for item in cat_info if item[1] <= 20]
    
    print(f"\nHigh cardinality features (>20 unique values): {len(high_cardinality)}")
    print(high_cardinality[:10])  # 顯示前10個
    
    print(f"\nLow cardinality features (≤20 unique values): {len(low_cardinality)}")
    print(low_cardinality[:10])  # 顯示前10個
    
    return cat_info

def outlier_analysis(df, target_col):
    """異常值分析"""
    print("\n" + "=" * 60)
    print("6. 異常值分析")
    print("=" * 60)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 使用IQR方法檢測異常值
    outlier_info = []
    for feature in numerical_features:
        if df[feature].notna().sum() > 0:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_info.append((feature, outlier_count, outlier_percentage))
    
    outlier_info.sort(key=lambda x: x[2], reverse=True)
    
    print("Top 15 features with most outliers:")
    print("Feature | Outlier Count | Outlier Percentage")
    print("-" * 55)
    for feature, count, percentage in outlier_info[:15]:
        print(f"{feature:<20} | {count:<12} | {percentage:.2f}%")
    
    return outlier_info

def feature_engineering_suggestions(df, target_col, correlations):
    """特徵工程建議"""
    print("\n" + "=" * 60)
    print("7. 特徵工程建議")
    print("=" * 60)
    
    print("基於EDA結果的特徵工程建議：")
    print("\n1. 數值特徵處理：")
    print("   - 對高度偏態的特徵進行對數變換")
    print("   - 對異常值進行winsorization或移除")
    print("   - 標準化/正規化數值特徵")
    
    print("\n2. 類別特徵處理：")
    print("   - 低基數特徵：使用One-hot encoding")
    print("   - 高基數特徵：使用Target encoding或Frequency encoding")
    print("   - 缺失值：考慮創建'Unknown'類別")
    
    print("\n3. 新特徵創建：")
    # 基於常見的房地產特徵
    possible_features = [
        "創建總居住面積 (如果有多個面積特徵)",
        "創建房間總數 (臥室+浴室+其他房間)",
        "創建房屋年齡 (當前年份 - 建造年份)",
        "創建價格每平方英尺",
        "創建地段比率 (建築面積/土地面積)"
    ]
    for suggestion in possible_features:
        print(f"   - {suggestion}")
    
    print("\n4. 交互特徵：")
    print("   - 高相關特徵間的乘積或比率")
    print("   - 地理位置相關特徵的組合")

def generate_summary_report(train_df, missing_table, correlations, cat_info, outlier_info):
    """生成總結報告"""
    print("\n" + "=" * 60)
    print("8. EDA 總結報告")
    print("=" * 60)
    
    print(f"資料集基本資訊：")
    print(f"- 訓練資料樣本數：{len(train_df):,}")
    print(f"- 特徵總數：{len(train_df.columns)}")
    print(f"- 數值特徵數：{len(train_df.select_dtypes(include=[np.number]).columns)}")
    print(f"- 類別特徵數：{len(train_df.select_dtypes(include=['object']).columns)}")
    print(f"- 有缺失值的特徵數：{len(missing_table)}")
    print(f"- 平均缺失值比例：{missing_table['Missing Percentage'].mean():.2f}%")
    
    return {
        'total_samples': len(train_df),
        'total_features': len(train_df.columns),
        'numerical_features': len(train_df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(train_df.select_dtypes(include=['object']).columns),
        'features_with_missing': len(missing_table),
        'avg_missing_percentage': missing_table['Missing Percentage'].mean() if len(missing_table) > 0 else 0
    }

def main():
    """主函數"""
    print("開始執行探索性資料分析...")
    
    # 1. 載入資料
    train_df, test_df, sample_submission = load_and_basic_info()
    
    # 2. 缺失值分析
    missing_table = missing_value_analysis(train_df)
    
    # 3. 目標變數分析
    target_col = target_variable_analysis(train_df)
    
    if target_col is None:
        print("無法進行進一步分析，請檢查目標變數")
        return
    
    # 4. 數值特徵分析
    correlations = numerical_features_analysis(train_df, target_col)
    
    # 5. 類別特徵分析
    cat_info = categorical_features_analysis(train_df)
    
    # 6. 異常值分析
    outlier_info = outlier_analysis(train_df, target_col)
    
    # 7. 特徵工程建議
    feature_engineering_suggestions(train_df, target_col, correlations)
    
    # 8. 生成總結報告
    summary = generate_summary_report(train_df, missing_table, correlations, cat_info, outlier_info)
    
    print("\nEDA 分析完成！")
    print("生成的檔案：")
    print("- missing_values.png")
    print("- target_analysis.png") 
    print("- correlation_matrix.png")
    
    return train_df, test_df, target_col, summary

if __name__ == "__main__":
    train_df, test_df, target_col, summary = main()