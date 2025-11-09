# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 17:29:41 2025

@author: zhaojia
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 1. 全局绘图样式配置（学术期刊风格）
# --------------------------
plt.rcParams["font.family"] = "Arial"  # 期刊常用字体
plt.rcParams["font.size"] = 14        # 基础字体大小
plt.rcParams["axes.linewidth"] = 1    # 坐标轴线条宽度
sns.set_style("whitegrid", {"grid.linestyle": ":"})  # 网格线样式

# --------------------------
# 2. 数据准备（双掺杂+单掺杂）
# --------------------------
# 2.1 双掺杂数据（掺杂对、构型、形成能，单位：meV/atom）
double_doped_data = [
    ("CuFe", "far", -33.8244475), ("CuFe", "nn", -35.17926258),
    ("CuMn", "far", -16.37096273), ("CuMn", "nn", -18.39142742),
    ("CuSc", "far", -38.57633258), ("CuSc", "nn", -37.14393195),
    ("CuSi", "far", 8.562030703), ("CuSi", "nn", 7.188026328),
    ("CuTi", "far", -35.79378211), ("CuTi", "nn", -36.62597617),
    ("CuZn", "far", -0.119327422), ("CuZn", "nn", 0.015796328),
    ("CuZr", "far", -37.36882727), ("CuZr", "nn", -36.89529258),
    ("FeSi", "far", -15.75975484), ("FeSi", "nn", -19.87337547),
    ("LiCu", "far", -13.50582992), ("LiCu", "nn", -12.51680711),
    ("LiFe", "far", -38.35271984), ("LiFe", "nn", -37.98913266),
    ("LiMg", "far", -4.946867031), ("LiMg", "nn", -4.833702969),
    ("LiMn", "far", -20.90665656), ("LiMn", "nn", -22.13811281),
    ("LiSc", "far", -41.74233609), ("LiSc", "nn", -44.81139297),
    ("LiSi", "far", 4.565114063), ("LiSi", "nn", 4.124644062),
    ("LiTi", "far", -39.04700219), ("LiTi", "nn", -42.7962225),
    ("LiZn", "far", -5.598310938), ("LiZn", "nn", -5.238254062),
    ("LiZr", "far", -41.10647984), ("LiZr", "nn", -45.46783984),
    ("MgCu", "far", -1.335009453), ("MgCu", "nn", 0.273439922),
    ("MgFe", "far", -26.74137469), ("MgFe", "nn", -23.04068969),
    ("MgMn", "far", -9.866213594), ("MgMn", "nn", -6.781390156),
    ("MgSi", "far", 16.21688016), ("MgSi", "nn", 16.46289859),
    ("MgSc", "far", -29.77125875), ("MgSc", "nn", -32.062885),
    ("MgTi", "far", -27.06879578), ("MgTi", "nn", -28.12738797),
    ("MgZn", "far", 6.296954219), ("MgZn", "nn", 6.653756719),
    ("MgZr", "far", -29.00632188), ("MgZr", "nn", -32.08208844),
    ("MnFe", "far", -38.87544172), ("MnFe", "nn", -46.14431234),
    ("MnSi", "far", 0.669762188), ("MnSi", "nn", -3.52070625),
    ("ScFe", "far", -66.02110531), ("ScFe", "nn", -62.87598312),
    ("ScMn", "far", -49.80637578), ("ScMn", "nn", -48.59883234),
    ("ScSi", "far", -18.64320547), ("ScSi", "nn", -18.19144203),
    ("ScTi", "far", -66.66688766), ("ScTi", "nn", -72.16678797),
    ("ScZr", "far", -68.46199781), ("ScZr", "nn", -77.24332563),
    ("TiFe", "far", -62.94129797), ("TiFe", "nn", -63.48537734),
    ("TiMn", "far", -47.15873875), ("TiMn", "nn", -49.45795125),
    ("TiSi", "far", -16.933785), ("TiSi", "nn", -17.44467125),
    ("ZnFe", "far", -25.67636609), ("ZnFe", "nn", -25.64822672),
    ("ZnMn", "far", -8.74186125), ("ZnMn", "nn", -8.93607375),
    ("ZnSc", "far", -29.96155609), ("ZnSc", "nn", -29.77825516),
    ("ZnSi", "far", 15.77912719), ("ZnSi", "nn", 15.74588844),
    ("ZnTi", "far", -27.3682125), ("ZnTi", "nn", -27.88828906),
    ("ZnZr", "far", -28.70520609), ("ZnZr", "nn", -29.24537797),
    ("ZrFe", "far", -65.91795656), ("ZrFe", "nn", -62.39842844),
    ("ZrMn", "far", -50.43956422), ("ZrMn", "nn", -49.48885047),
    ("ZrSi", "far", -18.12493609), ("ZrSi", "nn", -16.90927734),
    ("ZrTi", "far", -69.64838984), ("ZrTi", "nn", -75.55957266),
]

# 2.2 单掺杂数据（元素、形成能，单位：meV/atom）
single_doped_data = [
    ("Zr", -32.20101484), ("Zn", 2.98574), ("Ti", -28.80724312),
    ("Si", 13.51035406), ("Sc", -30.93087516), ("Mn", -12.44403125),
    ("Mg", 1.559582344), ("Li", -7.203355937), ("Cu", -4.395505547),
    ("Fe", -29.06594484)
]

# 2.3 转换为DataFrame
df_double = pd.DataFrame(
    double_doped_data,
    columns=["dopant_pair", "configuration", "formation_energy"]
)
df_single = pd.DataFrame(
    single_doped_data,
    columns=["element", "formation_energy"]
)

# 2.4 计算协同效应
single_energy_dict = dict(zip(df_single["element"], df_single["formation_energy"]))

def calculate_synergy(row):
    elem1, elem2 = row["dopant_pair"][:2], row["dopant_pair"][2:]
    e1 = single_energy_dict.get(elem1, 0)
    e2 = single_energy_dict.get(elem2, 0)
    return row["formation_energy"] - (e1 + e2)

df_double["synergy"] = df_double.apply(calculate_synergy, axis=1)


# --------------------------
# --------------------------
# 3. 创建两个独立的图像
# --------------------------

# 3.1 第一幅图：双掺杂构型对比
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
sns.barplot(
    x="dopant_pair", y="formation_energy", hue="configuration",
    data=df_double, ax=ax1, palette=["#3498db", "#e74c3c"],
    edgecolor="black", linewidth=0.5
)
ax1.set_xlabel("Dopant Pairs", fontsize=18)
ax1.set_ylabel("Formation Energy (meV/atom)", fontsize=16)
ax1.tick_params(axis="x", rotation=90)
ax1.legend(
    title="Configuration", 
    loc="lower left", 
    frameon=True, 
    framealpha=0.9,
    fontsize=16,
    markerscale=1.2
)
ax1.grid(axis="y", alpha=0.5)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.8)
plt.tight_layout()
plt.savefig(
    "configuration_comparison.png",
    dpi=300, bbox_inches="tight", format="png"
)
plt.show()

# 3.2 第二幅图：协同效应分析
plt.figure(figsize=(12, 6))
ax2 = plt.gca()
sns.barplot(
    x="dopant_pair", y="synergy", hue="configuration",
    data=df_double, ax=ax2, palette=["#3498db", "#e74c3c"],
    edgecolor="black", linewidth=0.5
)
ax2.set_xlabel("Dopant Pairs", fontsize=16)
ax2.set_ylabel("Synergy Energy (meV/atom)", fontsize=16)
ax2.tick_params(axis="x", rotation=90)
ax2.legend(
    title="Configuration", 
    loc="lower left", 
    frameon=True, 
    framealpha=0.9,
    fontsize=16,
    markerscale=1.2
)
ax2.grid(axis="y", alpha=0.5)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.8)
ax2.text(
    0.98, 0.05,
    "Negative: Synergistic Stabilization\nPositive: Antagonistic Destabilization",
    transform=ax2.transAxes, bbox=dict(facecolor="white", alpha=0.8),
    fontsize=10, verticalalignment="bottom", horizontalalignment="right"
)
plt.tight_layout()
plt.savefig(
    "synergy_effect_analysis.png",
    dpi=300, bbox_inches="tight", format="png"
)
plt.show()