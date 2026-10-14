# Codex 实验工具

## 磁珠定焦辅助：自动对齐版 v1.0.0

比较参考珠与当前珠的衍射图样，自动修正选区的 XY 偏差，再计算二维和径向相关值。

- 支持输入几何中心坐标（含小数）以及鼠标选择。
- 显示二维相关值、对齐后径向相关值、dx 和 dy；默认允许各方向 ±8 px。
- 支持删除选中结果、清空列表和 CSV 导出。
- 提供 Python 调用接口、命令行及带已知位置的 Z 扫描选峰功能。

**[中文使用说明](bead-focus/README_中文.md)** · **[下载完整程序包](downloads/bead-focus-auto-alignment-v1.0.0.zip)**

### 启动

下载压缩包并解压，进入 `bead_focus` 文件夹。已安装配套运行环境的电脑可双击 `启动定焦工具.cmd`。

其他电脑先安装 Python 3.10 或更新版本，再在该文件夹运行：

```powershell
python -m pip install -r requirements.txt
python focus_gui.py
```

也可以克隆本仓库，进入 `bead-focus` 文件夹后运行以上两条命令。

### 在 Python 中调用

在 `bead-focus` 文件夹中，或将该文件夹加入 Python 搜索路径：

```python
from bead_focus import Config, FocusReference, load_image

reference = FocusReference.from_image(
    load_image("reference.tif"),
    center=(320.25, 240.5),
    config=Config(roi_size=129, max_shift_px=8),
)
result = reference.compare(load_image("current.tif"), center=(410.3, 285.75))
print(result["ncc"], result["radial_correlation"])
print(result["shift_dx_px"], result["shift_dy_px"])
```

相关值反映图样相似度，不能由单张图直接换算成物理 Z 高度。跨珠比较应保持珠型号、直径和成像条件一致。30 项检查已通过；示例是模拟图，尚未以真实实验图验证纳米精度。

### 验证

在 `bead-focus` 文件夹中运行：

```powershell
python -m unittest discover -s tests -v
```
