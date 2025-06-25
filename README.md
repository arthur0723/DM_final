# House Price Prediction Intervals using Conformalized Quantile Regression

## 專案概述
本專案實現了基於 **Conformalized Quantile Regression (CQR)** 的房價預測區間模型，旨在提供90%信賴區間的房價預測，同時量化預測的不確定性。該方法結合了LightGBM和HistGradientBoosting等先進的機器學習技術，在Kaggle "Prediction Interval Competition II: House Price"競賽數據集上取得了優異的表現。

## 主要特色
- **不確定性量化**: 提供預測區間而非單點預測，更適用於風險敏感的房地產決策
- **分佈無關**: CQR方法無需假設數據分佈，具有理論覆蓋保證
- **先進特徵工程**: 包含時間特徵、空間特徵、房屋特性等多維度特徵
- **模型集成**: 結合多種基礎模型以提升預測性能

## 檔案結構
```
DM_final/
├── Baseline.py              # 基礎模型實現
├── EDA.py                   # 探索性資料分析
├── Week4_Pipeline.py        # 完整的進階建模流程
├── dataset.csv              # 訓練資料集
├── test.csv                 # 測試資料集
├── submission.csv           # 最終預測結果
├── sample_submission.csv    # 提交格式範例
├── Week2.pdf               # 技術報告
├── correlation_matrix.png   # 特徵相關性分析圖
├── missing_values.png       # 缺失值分析圖
└── target_analysis.png      # 目標變數分析圖
```

## 技術方法

### 1. 特徵工程
- **時間特徵**: 銷售年份、月份、季節性指標
- **空間特徵**: 地理聚類、土地使用率、位置特徵
- **房屋特性**: 房屋年齡、房間密度、浴室品質分數
- **交互特徵**: 跨特徵組合以捕捉複雜關係

### 2. 模型架構
- **基礎分位數回歸**: LightGBM與HistGradientBoosting
- **Conformalized Quantile Regression**: 提供分佈無關的覆蓋保證
- **模型集成**: 結合多個預測器的中位數預測

### 3. 評估指標
- **Winkler Score**: 主要評估指標，綜合考慮覆蓋率和區間寬度
- **覆蓋率**: 目標90%覆蓋率（實際範圍88-92%）
- **平均區間寬度**: 預測區間的平均寬度

## 實驗結果

| 方法 | Winkler Score | 覆蓋率 | 平均寬度 |
|------|---------------|--------|----------|
| LightGBM Baseline | 368,245 | 0.874 | 264,890 |
| CQR-LightGBM | 355,180 | 0.885 | 258,420 |
| CQR-Ensemble | **342,960** | **0.891** | **246,366** |

## 使用方法

### 環境需求
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn mapie
```

### 快速開始
```bash
# 探索性資料分析
python EDA.py

# 基礎模型訓練
python Baseline.py

# 完整進階流程
python Week4_Pipeline.py
```

### 程式碼說明

#### Baseline.py
- 實現基礎的分位數回歸模型
- 包含Random Forest baseline和LightGBM quantile regression
- 提供基本的特徵工程和預處理

#### Week4_Pipeline.py
- 完整的進階建模流程
- 實現Conformalized Quantile Regression
- 包含進階特徵工程和模型集成

#### EDA.py
- 全面的探索性資料分析
- 缺失值分析、目標變數分布、特徵相關性等
- 生成視覺化圖表

## 技術亮點

### 1. 預測區間順序修正
解決了傳統分位數回歸中可能出現的上界小於下界的問題：
```python
# 確保 lower <= upper
final_lower = np.minimum(final_lower, final_upper)
```

### 2. 多階段開發流程
- **Stage 1**: 基礎分位數回歸
- **Stage 2**: CQR校準和集成方法
- **Stage 3**: 超參數優化

### 3. 穩健的特徵工程
- 處理缺失值和異常值
- 類別特徵編碼（高基數使用頻率編碼，低基數使用Label Encoding）
- 特徵標準化

## 作者
- **Ya-Hsuan Sun** - 國立成功大學資料科學研究所
- **Hon Kuen Hui** - 國立成功大學醫學資訊研究所

---
*最後更新: 2025年6月*