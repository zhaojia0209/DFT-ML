# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 16:43:26 2025

@author: zhaojia
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 替换为你的实际数据（元素、形成能、误差）
elements = ["Sc", "Cu", "Mg", "Zn", "Li", "Si", "Mn", "Ti", "Fe", "Zr"]  # 元素列表
formation_energies = [-30.93088, -4.39551, 1.55958, 2.98574, -7.20336, 13.51035, -12.44403, -28.80724, -29.06594,-32.20101]  # 形成能（eV/atom）


# 2. 按形成能排序（从低到高）
sorted_indices = np.argsort(formation_energies)
elements = [elements[i] for i in sorted_indices]
formation_energies = [formation_energies[i] for i in sorted_indices]

# 3. 设置颜色（负→蓝，正→红）
colors = ["#1f77b4" if e < 0 else "#ff6b6b" for e in formation_energies]

# 4. 绘图
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

x_pos = np.arange(len(elements))
bars = plt.bar(x_pos, formation_energies, capsize=5, width=0.6, 
               color=colors, edgecolor="black", linewidth=0.8)

# 5. 添加基线和标签
plt.axhline(y=0, color="gray", linestyle="--", linewidth=1.5)  # 0能量基线
plt.xlabel("Dopant Elements", fontsize=20, labelpad=10)
plt.ylabel("Formation Energy (meV/atom)", fontsize=20, labelpad=10)
#plt.title("Figure 2. Formation Energies of Single-Doped Al Alloys", fontsize=16,loc="bottom", pad=-50,)
plt.xticks(x_pos, elements, rotation=45, ha="right")  # 元素名旋转45度，避免重叠
plt.ylim(min(formation_energies)-10, max(formation_energies)+10)  # 预留上下空间
plt.grid(axis="y", linestyle=":", alpha=0.5)

# 6. 保存高清图片
plt.tight_layout()
plt.savefig("single_doped_formation_energies.png", dpi=300, bbox_inches="tight")
plt.show()
    