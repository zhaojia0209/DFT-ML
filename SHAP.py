import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ----------------------------
# 数据读取
# ----------------------------
df = pd.read_csv('binary_alloy_features_extended.csv')

feature_cols = [
    'chi_A','chi_B','delta_chi','r_A','r_B','delta_r','Z_A','Z_B',
    'valence_A','valence_B','delta_valence','V_atom_A','V_atom_B','delta_V',
    'EA_A','EA_B','delta_EA','IE_A','IE_B','delta_IE',
    'chi_prod','chi_avg','delta_chi_sq','delta_r_sq','chi_r_interaction','delta_r_host_sum'
]

X = df[feature_cols].values
y = df[['E_far','E_nn']].values  # 预测 E_far 和 E_nn

X = np.nan_to_num(X)  # 处理NaN值
y = np.nan_to_num(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 特征标准化

# ----------------------------
# 模型定义（线性模型）
# ----------------------------
linear_models = {'Ridge': Ridge(alpha=0.042, max_iter=5000)}  # 重命名变量更准确
targets = ['E_far', 'E_nn']

# ----------------------------
# SHAP 分析 & 保存数据
# ----------------------------
for name, model in linear_models.items():
    print(f"\nSHAP 分析: {name}")
    model.fit(X_scaled, y)  # 训练模型（Ridge支持多输出）
    
    # 线性模型使用LinearExplainer，传入训练数据作为背景
    explainer = shap.LinearExplainer(model, X_scaled)
    # 获取所有目标的SHAP值（shape: [n_targets, n_samples, n_features]）
    shap_values = explainer.shap_values(X_scaled)
    
    for i, target_name in enumerate(targets):
        # 提取当前目标的SHAP值
        target_shap = shap_values[i]
        
        # 保存SHAP值到CSV
        shap_df = pd.DataFrame(target_shap, columns=feature_cols)
        shap_df.to_csv(f"{name}_{target_name}_shap.csv", index=False)
        
        # 绘制SHAP summary热图
        shap.summary_plot(target_shap, X_scaled, feature_names=feature_cols, show=False)
        plt.title(f"{name} - {target_name} SHAP Summary")
        plt.show(block=True)