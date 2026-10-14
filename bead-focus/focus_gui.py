"""A small local interface for reference-based bead focus comparison."""
from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from bead_focus import Config, FocusError, FocusReference, crop_bead, load_image, scan_focus, write_results_csv, write_scan_svg


class ImagePanel(ttk.Frame):
    def __init__(self, master, title):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text=title, font=("Microsoft YaHei", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(self, width=470, height=340, background="#101827", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=6)
        self.label = ttk.Label(self, text="加载图片后，输入珠中心坐标。")
        self.label.grid(row=2, column=0, sticky="w")
        coordinates = ttk.Frame(self)
        coordinates.grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.center_x = tk.StringVar(value="")
        self.center_y = tk.StringVar(value="")
        for label, variable in (("中心 X", self.center_x), ("中心 Y", self.center_y)):
            ttk.Label(coordinates, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(coordinates, textvariable=variable, width=10)
            entry.pack(side="left", padx=(0, 8))
            entry.bind("<Return>", self.apply_center)
        ttk.Button(coordinates, text="应用坐标", command=self.apply_center).pack(side="left")
        ttk.Label(self, text="当前图片坐标：左上角 (0, 0)，X 向右、Y 向下；支持小数。",
                  foreground="#475569").grid(row=4, column=0, sticky="w", pady=(3, 0))
        self.image = None
        self.center = None
        self.path = ""
        self.roi_size = 129
        self.scale = 1
        self.offset = (0, 0)
        self.photo = None
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.canvas.bind("<Button-1>", self.pick_center)

    def set_image(self, array, path, center=None):
        self.image, self.path, self.center = array, str(path), center
        self.center_x.set("" if center is None else f"{center[0]:.8f}".rstrip("0").rstrip("."))
        self.center_y.set("" if center is None else f"{center[1]:.8f}".rstrip("0").rstrip("."))
        self.redraw()

    def get_center(self):
        if self.image is None:
            raise FocusError("请先加载图片，再输入中心坐标。")
        try:
            center = (float(self.center_x.get()), float(self.center_y.get()))
        except ValueError as exc:
            raise FocusError("中心 X、Y 都需要填写数字，例如 320.25、240.5。") from exc
        crop_bead(self.image, self.roi_size, center)
        self.center = center
        self.redraw()
        return center

    def apply_center(self, event=None):
        try:
            self.get_center()
        except (ValueError, OSError) as exc:
            messagebox.showerror("坐标无效", str(exc))
        return "break"

    def pick_center(self, event):
        if self.image is None:
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        h, w = self.image.shape
        if 0 <= x < w and 0 <= y < h:
            self.center = (x, y)
            self.center_x.set(f"{x:.8f}".rstrip("0").rstrip("."))
            self.center_y.set(f"{y:.8f}".rstrip("0").rstrip("."))
            self.redraw()

    def redraw(self):
        if self.image is None:
            return
        # Display stretching and resizing do NOT modify analysis pixels.
        a = self.image
        low, high = np.percentile(a, [1, 99.8])
        display = np.clip((a-low)/max(high-low, 1e-12)*255, 0, 255).astype(np.uint8)
        ch, cw = max(20, self.canvas.winfo_height()), max(20, self.canvas.winfo_width())
        h, w = a.shape
        scale = min(cw/w, ch/h)
        dw, dh = max(1, int(w*scale)), max(1, int(h*scale))
        self.scale = scale
        self.offset = ((cw-dw)/2, (ch-dh)/2)
        pil = Image.fromarray(display).resize((dw, dh), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        ox, oy = self.offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        if self.center is not None:
            x, y = self.center
            half = self.roi_size/2
            self.canvas.create_rectangle(ox+(x-half)*scale, oy+(y-half)*scale,
                                         ox+(x+half)*scale, oy+(y+half)*scale,
                                         outline="#38bdf8", width=2)
            self.canvas.create_line(ox+x*scale-7, oy+y*scale, ox+x*scale+7, oy+y*scale, fill="#fbbf24")
            self.canvas.create_line(ox+x*scale, oy+y*scale-7, ox+x*scale, oy+y*scale+7, fill="#fbbf24")
            center_label = f"中心 ({x:.4f}, {y:.4f})"
        else:
            center_label = "请输入中心坐标"
        self.label.config(text=f"{Path(self.path).name}  |  {w}×{h}  |  {center_label}")


class FocusApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=14)
        self.pack(fill="both", expand=True)
        self.root = root
        root.title("磁珠定焦辅助")
        root.geometry(f"{min(1080, root.winfo_screenwidth()-60)}x{min(850, root.winfo_screenheight()-80)}")
        root.minsize(900, 700)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        self.rows = []
        self.displayed_result_id = None
        self.last_scan = None
        self.roi = tk.StringVar(value="129")
        self.shift = tk.StringVar(value="8")
        self.inner = tk.StringVar(value="0")
        self.threshold = tk.StringVar(value="")
        self.saturation = tk.StringVar(value="")
        ttk.Label(self, text="自动平移对齐后比较衍射图样", font=("Microsoft YaHei", 17, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text="跨珠比较：请使用相同型号和直径。相关值是图样相似度，不是 Z 高度或对焦误差的纳米数。",
                  foreground="#475569").grid(row=1, column=0, sticky="w", pady=(4, 10))
        tools = ttk.Frame(self)
        tools.grid(row=2, column=0, sticky="ew")
        for label, command in [("选择 reference 图片", self.open_reference),
                               ("选择当前图片", self.open_current),
                               ("比较", self.compare),
                               ("保存 reference", self.save_reference),
                               ("载入 reference", self.load_reference),
                               ("导出结果 CSV", self.export)]:
            ttk.Button(tools, text=label, command=command).pack(side="left", padx=(0, 6))
        settings = ttk.Frame(self)
        settings.grid(row=3, column=0, sticky="ew", pady=10)
        for label, variable, width in [("选区边长(px)", self.roi, 6),
                                        ("允许 XY 偏移(px)", self.shift, 4),
                                        ("排除内圈(px)", self.inner, 4), ("径向阈值(可空)", self.threshold, 6),
                                        ("饱和值(可空)", self.saturation, 6)]:
            ttk.Label(settings, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(settings, textvariable=variable, width=width).pack(side="left", padx=(0, 10))
        self.roi.trace_add("write", lambda *args: self.update_roi())
        images = ttk.Frame(self)
        images.grid(row=4, column=0, sticky="nsew")
        self.reference_panel = ImagePanel(images, "Reference：选择目标焦位的单颗珠")
        self.current_panel = ImagePanel(images, "当前实验组：选择本组单颗珠")
        self.reference_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.current_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.status = tk.StringVar(value="选两张图，分别输入珠中心坐标，再按“比较”。")
        ttk.Label(self, textvariable=self.status, font=("Microsoft YaHei", 10, "bold"), wraplength=990).grid(row=5, column=0, sticky="w", pady=10)
        results = ttk.Frame(self)
        results.grid(row=6, column=0, sticky="ew")
        results.columnconfigure(0, weight=1)
        columns = ("file", "ncc", "radial", "dx", "dy", "state")
        self.table = ttk.Treeview(results, columns=columns, show="headings", height=5, selectmode="extended")
        for key, title, width in zip(columns, ["图片", "二维相关值", "径向相关值", "dx (px)", "dy (px)", "图样判定"],
                                     [240, 140, 140, 100, 100, 170]):
            self.table.heading(key, text=title)
            self.table.column(key, width=width)
        self.table.grid(row=0, column=0, sticky="ew")
        self.table.bind("<Delete>", self.delete_selected)
        deletion = ttk.Frame(results)
        deletion.grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Button(deletion, text="删除选中", command=self.delete_selected).pack(side="left")
        ttk.Button(deletion, text="清空列表", command=self.clear_results).pack(side="left", padx=8)
        ttk.Label(deletion, text="Ctrl / Shift 可多选；Delete 键也可删除。", foreground="#475569").pack(side="left")
        bottom = ttk.Frame(self)
        bottom.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(bottom, text="读取 Z 扫描 CSV，寻找相关峰", command=self.open_scan).pack(side="left")
        ttk.Button(bottom, text="保存扫描曲线 SVG / 结果 JSON", command=self.save_scan).pack(side="left", padx=8)
        self.plot = tk.Canvas(self, height=115, background="white", highlightthickness=1, highlightbackground="#cbd5e1")
        self.plot.grid(row=8, column=0, sticky="ew", pady=(8, 0))

    def config(self):
        config = Config(roi_size=int(self.roi.get()),
                        max_shift_px=int(self.shift.get()),
                        inner_radius_px=float(self.inner.get()),
                        threshold=float(self.threshold.get()) if self.threshold.get().strip() else None,
                        saturation_level=float(self.saturation.get()) if self.saturation.get().strip() else None)
        config.validate()
        return config

    def update_roi(self):
        try:
            size = int(self.roi.get())
        except ValueError:
            return
        for panel in (self.reference_panel, self.current_panel):
            panel.roi_size = size
            panel.redraw()

    def open_image(self, panel):
        path = filedialog.askopenfilename(title="选择单帧原始图片", filetypes=[
            ("图片", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.npy"), ("所有文件", "*.*")])
        if path:
            try:
                panel.set_image(load_image(path), path)
                self.status.set("请输入新图片的珠中心坐标；计算使用原始像素与输入的小数坐标。")
            except (OSError, ValueError) as exc:
                messagebox.showerror("无法加载", str(exc))

    def open_reference(self):
        self.open_image(self.reference_panel)

    def open_current(self):
        self.open_image(self.current_panel)

    def get_reference(self):
        p = self.reference_panel
        if p.image is None:
            raise FocusError("请先加载 reference，并输入参考珠的中心坐标。")
        return FocusReference.from_image(p.image, p.get_center(), self.config(), p.path)

    def compare(self):
        try:
            p = self.current_panel
            if p.image is None:
                raise FocusError("请先加载当前图片，并输入当前珠的中心坐标。")
            result = self.get_reference().compare(p.image, p.get_center())
            result.update(image=p.path, status="ok")
            state = "未设阈值"
            if not result["quality_ok"]:
                state = "图片质量需复核"
            elif result["matches_threshold"] is not None:
                state = "达到图样阈值" if result["matches_threshold"] else "低于图样阈值"
            row_id = self.table.insert("", "end", values=(Path(p.path).name,
                f'{result["ncc"]:.6f}', f'{result["radial_correlation"]:.6f}',
                f'{result["shift_dx_px"]:.3f}', f'{result["shift_dy_px"]:.3f}', state))
            result["_row_id"] = row_id
            self.rows.append(result)
            self.displayed_result_id = row_id
            self.status.set(f'二维相关值 {result["ncc"]:.6f}；对齐后径向相关值 {result["radial_correlation"]:.6f}；'
                            f'dx {result["shift_dx_px"]:+.3f} px，dy {result["shift_dy_px"]:+.3f} px。{state}。')
            self.draw_profiles(result)
            extra = result["warnings"][1:]
            if extra:
                messagebox.showwarning("需要复核", "\n".join(extra))
        except (OSError, ValueError) as exc:
            messagebox.showerror("不能比较", str(exc))

    def delete_selected(self, event=None):
        selected = set(self.table.selection())
        if not selected:
            self.status.set("请先选中要删除的结果行。")
            return "break"
        for row_id in selected:
            self.table.delete(row_id)
        self.rows = [row for row in self.rows if row["_row_id"] not in selected]
        if self.displayed_result_id in selected:
            self.refresh_result_plot()
        self.status.set(f"已删除 {len(selected)} 条结果；列表剩余 {len(self.rows)} 条。")
        return "break"

    def clear_results(self):
        count = len(self.rows)
        self.table.delete(*self.table.get_children())
        self.rows.clear()
        if self.displayed_result_id is not None:
            self.refresh_result_plot()
        self.status.set(f"已清空 {count} 条结果。")

    def refresh_result_plot(self):
        if self.rows:
            self.displayed_result_id = self.rows[-1]["_row_id"]
            self.draw_profiles(self.rows[-1])
        else:
            self.displayed_result_id = None
            self.plot.delete("all")

    def draw_profiles(self, result):
        self.plot.delete("all")
        width = max(self.plot.winfo_width(), 800)
        self.plot.create_text(12, 10, anchor="nw", text="对齐后径向轮廓：蓝色 reference，橙色当前；各自去均值并归一化", fill="#334155")
        profiles = [np.array(result[k]) for k in ("reference_profile", "current_profile")]
        profiles = [(p-p.mean())/max(p.std(), 1e-12) for p in profiles]
        low, high = min(p.min() for p in profiles), max(p.max() for p in profiles)
        for profile, color in zip(profiles, ["#2563eb", "#ea580c"]):
            points = []
            for i, v in enumerate(profile):
                points.extend([25 + i/(len(profile)-1)*(width-50), 100-(v-low)/max(high-low, 1e-12)*70])
            self.plot.create_line(*points, fill=color, width=2)

    def save_reference(self):
        try:
            ref = self.get_reference()
            path = filedialog.asksaveasfilename(title="保存 reference", defaultextension=".npz", filetypes=[("参考文件", "*.npz")])
            if path:
                ref.save(path)
                self.status.set(f"参考已保存：{Path(path).name}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("不能保存", str(exc))

    def load_reference(self):
        path = filedialog.askopenfilename(title="载入 reference", filetypes=[("参考文件", "*.npz")])
        if path:
            try:
                ref = FocusReference.load(path)
                c = ref.config
                for variable, value in [(self.roi, c.roi_size), (self.shift, c.max_shift_px), (self.inner, c.inner_radius_px),
                                         (self.threshold, c.threshold), (self.saturation, c.saturation_level)]:
                    variable.set("" if value is None else str(value))
                self.reference_panel.set_image(ref.roi, path, ref.center)
                self.status.set("参考已载入；输入当前图片的珠中心坐标。" +
                                ("旧参考的阈值已清空，请按本版对齐后的径向指标重新设置。" if ref.legacy_reference else ""))
            except (OSError, ValueError, KeyError) as exc:
                messagebox.showerror("不能载入", str(exc))

    def export(self):
        if not self.rows:
            messagebox.showinfo("没有结果", "先比较至少一张图片。")
            return
        path = filedialog.asksaveasfilename(title="导出比较结果", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                write_results_csv(path, self.rows)
            except OSError as exc:
                messagebox.showerror("不能导出", str(exc))

    def open_scan(self):
        try:
            ref = self.get_reference()
            path = filedialog.askopenfilename(title="选择含已知 Z 位置的扫描 CSV", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            self.last_scan = scan_focus(ref, path)
            r = self.last_scan
            best = r["best_sampled_stage_z_nm"]
            estimate = r["interpolated_stage_z_nm"]
            text = f"最匹配的采样位置：{best:g} nm"
            if estimate is not None:
                text += f"；局部插值：{estimate:.1f} nm（不是精度保证）"
            if r["peak_at_edge"] or r["ambiguous_peak"]:
                text += "；需扩大扫描或复核多个峰"
            self.status.set(text)
            self.displayed_result_id = None
            self.plot.delete("all")
            self.plot.create_text(12, 10, anchor="nw", text="相关值 vs. 输入的扫描台 Z 位置（nm）", fill="#334155")
            curve = r["curve"]
            zz = np.array([p["stage_z_nm"] for p in curve])
            ss = np.array([p["radial_correlation"] for p in curve])
            width = max(self.plot.winfo_width(), 800)
            points = []
            for z, score in zip(zz, ss):
                points.extend([25+(z-zz.min())/(zz.max()-zz.min())*(width-50),
                               100-(score-ss.min())/max(np.ptp(ss), 1e-12)*70])
            self.plot.create_line(*points, fill="#2563eb", width=2)
            if r["peak_at_edge"] or r["ambiguous_peak"]:
                messagebox.showwarning("扫描尚未确定唯一峰", "\n".join(r["warnings"]))
        except (OSError, ValueError) as exc:
            messagebox.showerror("不能读取扫描", str(exc))

    def save_scan(self):
        if self.last_scan is None:
            messagebox.showinfo("没有扫描", "先读取 Z 扫描 CSV。")
            return
        path = filedialog.asksaveasfilename(title="保存扫描结果", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.last_scan, ensure_ascii=False, indent=2), encoding="utf-8")
                write_scan_svg(Path(path).with_suffix(".svg"), self.last_scan)
            except OSError as exc:
                messagebox.showerror("不能保存", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    FocusApp(root)
    root.mainloop()
