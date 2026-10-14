"""Check typed coordinates and result deletion without opening a visible window."""
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import tkinter as tk
    from focus_gui import FocusApp
except ImportError:
    tk = None

from test_bead_focus import bead


@unittest.skipIf(tk is None,"Tkinter unavailable")
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root=tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(str(exc))
        cls.root.withdraw()
        cls.root.update_idletasks()

    @classmethod
    def tearDownClass(cls):
        cls.root.update_idletasks()
        cls.root.destroy()

    def setUp(self):
        self.app=FocusApp(self.root)
        self.app.pack_forget()
        self.app.place(x=0,y=0,width=1080,height=850)
        self.app.reference_panel.set_image(bead(size=257),"reference.tif")
        self.app.current_panel.set_image(bead(dx=.25,dy=-.4,size=257),"current.tif")
        for panel,center in [(self.app.reference_panel,(128,128)),(self.app.current_panel,(128.25,127.6))]:
            panel.center_x.set(str(center[0]))
            panel.center_y.set(str(center[1]))

    def tearDown(self):
        self.app.destroy()
        self.root.update_idletasks()

    def test_typed_centers_work_without_click_or_apply(self):
        self.app.compare()
        self.assertEqual(len(self.app.rows),1)
        self.assertGreater(self.app.rows[0]["radial_correlation"],.999)
        self.assertEqual(self.app.current_panel.center,(128.25,127.6))
        self.assertEqual(tuple(self.app.table["columns"]),("file","ncc","radial","dx","dy","state"))
        self.assertIn("二维",self.app.status.get())
        self.assertGreater(self.app.rows[0]["ncc"],.999)

    def test_automatic_alignment_displays_offsets_without_changing_typed_center(self):
        self.app.current_panel.set_image(bead(dx=4,dy=-3,size=257),"offset.tif")
        self.app.current_panel.center_x.set("128")
        self.app.current_panel.center_y.set("128")
        self.app.compare()
        row=self.app.rows[0]
        self.assertGreater(row["radial_correlation"],.9999)
        values=self.app.table.item(row["_row_id"],"values")
        self.assertEqual(len(values),6)
        self.assertAlmostEqual(float(values[3]),4,places=3)
        self.assertAlmostEqual(float(values[4]),-3,places=3)
        self.assertEqual(self.app.current_panel.center,(128,128))
        self.assertEqual(self.app.current_panel.center_x.get(),"128")

    def test_invalid_numeric_center_does_not_add_a_result(self):
        self.app.current_panel.center_x.set("NaN")
        with patch("focus_gui.messagebox.showerror") as dialog:
            self.app.compare()
        self.assertTrue(dialog.called)
        self.assertEqual(len(self.app.rows),0)
        self.app.current_panel.center_x.set("abc")
        with patch("focus_gui.messagebox.showerror") as dialog:
            self.app.current_panel.apply_center()
        self.assertTrue(dialog.called)

    def test_delete_multiple_rows_and_export_remaining_only(self):
        for name in ("first.tif","second.tif","third.tif"):
            self.app.current_panel.path=name
            self.app.compare()
        ids=self.app.table.get_children()
        self.app.table.selection_set((ids[0],ids[2]))
        self.app.delete_selected()
        self.assertEqual(self.app.table.get_children(),(ids[1],))
        self.assertEqual([r["image"] for r in self.app.rows],["second.tif"])
        self.assertEqual(self.app.displayed_result_id,ids[1])
        scratch=Path(os.environ.get("BEAD_FOCUS_TEST_DIR",str(Path.cwd()/".bead_focus_test_tmp"))).resolve()
        scratch.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=scratch)
        path=Path(temporary.name)
        try:
            with patch("focus_gui.filedialog.asksaveasfilename",return_value=str(path/"results.csv")):
                self.app.export()
            with (path/"results.csv").open(encoding="utf-8-sig") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([r["image"] for r in rows],["second.tif"])
            for field in ("ncc","radial_correlation","shift_dx_px","shift_dy_px"):
                self.assertIn(field,rows[0])
        finally:
            if path.resolve().parent!=scratch:
                raise RuntimeError("Unexpected test path")
            temporary.cleanup()

    def test_clear_results_removes_export_records_and_profile(self):
        self.app.compare()
        self.app.clear_results()
        self.assertEqual(self.app.rows,[])
        self.assertEqual(self.app.table.get_children(),())
        self.assertEqual(self.app.plot.find_all(),())
        self.app.clear_results()

    def test_click_updates_coordinate_fields(self):
        self.root.update_idletasks()
        panel=self.app.current_panel
        x,y=128.25,127.6
        event=type("Event",(),{"x":panel.offset[0]+x*panel.scale,"y":panel.offset[1]+y*panel.scale})()
        panel.pick_center(event)
        self.assertAlmostEqual(float(panel.center_x.get()),x,places=6)
        self.assertAlmostEqual(float(panel.center_y.get()),y,places=6)

    def test_coordinate_controls_and_delete_buttons_fit_small_window(self):
        self.app.place_configure(width=1000,height=700)
        self.root.update_idletasks()
        self.assertLessEqual(self.app.plot.winfo_y()+self.app.plot.winfo_height(),self.app.winfo_height())
        panel=self.app.current_panel
        self.assertGreater(panel.canvas.winfo_height(),20)
        self.assertGreater(panel.label.winfo_height(),0)


if __name__=="__main__":
    unittest.main()
