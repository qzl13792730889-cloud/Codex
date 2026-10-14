"""Numerical checks with synthetic images, not experimental validation."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bead_focus import Config, FocusError, FocusReference, load_image, scan_focus, write_scan_svg


def bead(z=0, dx=0, dy=0, size=129):
    yy, xx = np.indices((size, size), dtype=float)
    r = np.hypot(xx-(size-1)/2-dx, yy-(size-1)/2-dy)
    radius = 23 + z/100
    pattern = (np.exp(-((r-radius)/2.4)**2)
               - 0.5*np.exp(-((r-radius-7)/3.1)**2)
               + 0.3*np.exp(-((r-radius-15)/3.3)**2))
    return 6000 + 2200*pattern


class FocusTests(unittest.TestCase):
    def setUp(self):
        self.scratch_root = Path(os.environ.get("BEAD_FOCUS_TEST_DIR", str(Path.cwd()/".bead_focus_test_tmp"))).resolve()
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.scratch_root)
        self.path = Path(self.tmp.name)
        self.reference = FocusReference.from_image(bead())

    def tearDown(self):
        if self.path.resolve().parent != self.scratch_root:
            raise RuntimeError("Refusing to clean a test directory outside the chosen scratch root.")
        self.tmp.cleanup()

    def save_image(self, path, image):
        Image.fromarray(np.clip(image, 0, 65535).astype(np.uint16)).save(path)

    def manifest(self, values, prefix="scan", image_function=bead):
        manifest = self.path/f"{prefix}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stage_z_nm", "image_path"])
            for index, z in enumerate(values):
                name = f"{prefix}_{index}.tif"
                self.save_image(self.path/name, image_function(z))
                writer.writerow([z, name])
        return manifest

    def test_self_correlation_and_no_fabricated_z(self):
        result = self.reference.compare(bead())
        self.assertAlmostEqual(result["radial_correlation"], 1, places=10)
        self.assertAlmostEqual(result["ncc"], 1, places=10)
        self.assertAlmostEqual(result["shift_dx_px"], 0, places=8)
        self.assertAlmostEqual(result["shift_dy_px"], 0, places=8)
        self.assertNotIn("axial_offset_nm", result)
        self.assertIsNone(result["matches_threshold"])

    def test_brightness_gain_and_offset(self):
        result = self.reference.compare(bead()*2.7+1050)
        self.assertAlmostEqual(result["radial_correlation"], 1, places=10)
        self.assertAlmostEqual(result["ncc"], 1, places=10)

    def test_coordinates_on_distinct_full_images(self):
        ref = FocusReference.from_image(bead(size=257), center=(128,128))
        current = bead(dx=4, dy=-3, size=257)
        correct = ref.compare(current, center=(132,125))
        offset = ref.compare(current, center=(128,128))
        self.assertAlmostEqual(correct["radial_correlation"], 1, places=10)
        self.assertAlmostEqual(offset["radial_correlation"], 1, places=10)
        self.assertAlmostEqual(offset["ncc"], 1, places=10)
        self.assertAlmostEqual(correct["shift_dx_px"], 0, places=8)
        self.assertAlmostEqual(correct["shift_dy_px"], 0, places=8)
        self.assertAlmostEqual(offset["shift_dx_px"], 4, places=8)
        self.assertAlmostEqual(offset["shift_dy_px"], -3, places=8)

    def test_fractional_coordinates_are_honored(self):
        ref = FocusReference.from_image(bead(size=257), center=(128,128), config=Config(max_shift_px=0))
        current = bead(dx=2.4, dy=-1.65, size=257)
        exact = ref.compare(current, center=(130.4,126.35))["radial_correlation"]
        rounded = ref.compare(current, center=(130,126))["radial_correlation"]
        self.assertGreater(exact, 0.999)
        self.assertGreater(exact, rounded)

    def test_subpixel_alignment_corrects_center_error_before_radial_comparison(self):
        image=bead(dx=2.4,dy=-1.65,size=257)
        ref=FocusReference.from_image(bead(size=257),center=(128,128))
        result=ref.compare(image,center=(128,128))
        disabled=FocusReference.from_image(bead(size=257),center=(128,128),config=Config(max_shift_px=0))
        before=disabled.compare(image,center=(128,128))
        self.assertAlmostEqual(result["shift_dx_px"],2.4,delta=.03)
        self.assertAlmostEqual(result["shift_dy_px"],-1.65,delta=.03)
        self.assertGreater(result["radial_correlation"],.999)
        self.assertGreater(result["radial_correlation"],before["radial_correlation"]+.01)
        self.assertGreater(result["ncc"],.999)
        np.testing.assert_allclose(result["aligned_current_center"],(130.4,126.35),atol=.03)

    def test_alignment_limit_blocks_threshold_judgment(self):
        ref=FocusReference.from_image(bead(size=257),config=Config(threshold=.9))
        result=ref.compare(bead(dx=10,size=257))
        self.assertTrue(result["alignment_at_limit"])
        self.assertFalse(result["quality_ok"])
        self.assertIsNone(result["matches_threshold"])
        self.assertAlmostEqual(result["shift_dx_px"],8)
        self.assertTrue(any("边界" in warning for warning in result["warnings"]))

    def test_zero_alignment_range_preserves_unaligned_input(self):
        ref=FocusReference.from_image(bead(),config=Config(max_shift_px=0))
        result=ref.compare(bead(dx=4,dy=-3))
        self.assertEqual((result["shift_dx_px"],result["shift_dy_px"]),(0,0))
        self.assertFalse(result["alignment_at_limit"])
        self.assertLess(result["radial_correlation"],.99)

    def test_2d_score_is_pearson_on_fixed_aligned_pixels(self):
        image=bead(z=120,dx=3,dy=-2)
        result=self.reference.compare(image)
        self.assertAlmostEqual(result["shift_dx_px"],3,delta=.03)
        self.assertAlmostEqual(result["shift_dy_px"],-2,delta=.03)
        margin=self.reference.config.max_shift_px+2
        reference=self.reference.roi[margin:-margin,margin:-margin]
        current=image[margin-2:129-margin-2,margin+3:129-margin+3]
        expected=np.corrcoef(reference.ravel(),current.ravel())[0,1]
        self.assertAlmostEqual(result["ncc"],expected,delta=1e-5)

    def test_shift_configuration_validation(self):
        for maximum in (-1,1.5,100):
            with self.assertRaises(FocusError):
                FocusReference.from_image(bead(),config=Config(max_shift_px=maximum))

    def test_radial_correlation_matches_numpy(self):
        result = self.reference.compare(bead(z=135))
        expected = np.corrcoef(result["reference_profile"],result["current_profile"])[0,1]
        self.assertAlmostEqual(result["radial_correlation"],expected,places=12)

    def test_ring_change_is_not_normalized_away(self):
        near = self.reference.compare(bead(z=50))["radial_correlation"]
        far = self.reference.compare(bead(z=300))["radial_correlation"]
        self.assertLess(near, 0.999)
        self.assertLess(far, near-0.1)

    def test_known_saturation_prevents_threshold_pass(self):
        image = np.minimum(bead(), 7000)
        ref = FocusReference.from_image(image, config=Config(threshold=0.9, saturation_level=7000))
        result = ref.compare(image)
        self.assertGreater(result["radial_correlation"], 0.99)
        self.assertFalse(result["quality_ok"])
        self.assertIsNone(result["matches_threshold"])

    def test_flat_nan_and_out_of_bounds_rejected(self):
        for array in [np.ones((129,129)), np.full((129,129),np.nan)]:
            with self.assertRaises(FocusError):
                self.reference.compare(array)
        with self.assertRaises(FocusError):
            self.reference.compare(bead(),center=(10,10))

    def test_16_bit_tiff_keeps_values(self):
        array = np.arange(129*129,dtype=np.uint16).reshape(129,129)+1000
        path = self.path/"16bit.tif"
        Image.fromarray(array).save(path)
        np.testing.assert_array_equal(load_image(path),array)

    def test_multiframe_requires_explicit_frame(self):
        path = self.path/"stack.tif"
        frames = [Image.fromarray(np.full((129,129),v,dtype=np.uint16)) for v in [1234,2345]]
        frames[0].save(path,save_all=True,append_images=frames[1:])
        with self.assertRaises(FocusError):
            load_image(path)
        self.assertEqual(float(load_image(path,frame=1)[0,0]),2345)

    def test_reference_roundtrip(self):
        path = self.path/"reference.npz"
        ref = FocusReference.from_image(bead(dx=.25,dy=-.4,size=257),center=(128.25,127.6),config=Config(max_shift_px=6,threshold=.95))
        ref.save(path)
        loaded = FocusReference.load(path)
        self.assertEqual(loaded.reference_id,ref.reference_id)
        np.testing.assert_allclose(loaded.center,(64.25,63.6),atol=1e-10)
        self.assertEqual(loaded.config.max_shift_px,6)
        self.assertEqual(loaded.config.threshold,.95)
        self.assertAlmostEqual(loaded.compare(bead())["radial_correlation"],ref.compare(bead())["radial_correlation"])

    def test_legacy_reference_can_be_loaded(self):
        path = self.path/"old.npz"
        metadata = {"version":1,"config":{"roi_size":129,"max_shift_px":8,"metric":"ncc",
                    "inner_radius_px":0,"threshold":.95,"saturation_level":None},"source":"old.tif"}
        np.savez_compressed(path,roi=bead(),metadata=np.array(json.dumps(metadata)))
        loaded = FocusReference.load(path)
        self.assertTrue(loaded.legacy_reference)
        self.assertIsNone(loaded.config.threshold)
        self.assertAlmostEqual(loaded.compare(bead())["radial_correlation"],1,places=10)

    def test_radial_version_reference_keeps_fractional_center(self):
        path=self.path/"radial.npz"
        metadata={"version":2,"config":{"roi_size":129,"inner_radius_px":0,"threshold":.98,
                  "saturation_level":None},"center":[64.25,63.6],"source":"radial.tif"}
        np.savez_compressed(path,roi=bead(dx=.25,dy=-.4),metadata=np.array(json.dumps(metadata)))
        loaded=FocusReference.load(path)
        np.testing.assert_allclose(loaded.center,(64.25,63.6))
        self.assertEqual(loaded.config.max_shift_px,8)
        self.assertIsNone(loaded.config.threshold)
        self.assertGreater(loaded.compare(bead())["radial_correlation"],.999)

    def test_scan_finds_known_center_and_writes_plot(self):
        result = scan_focus(self.reference,self.manifest([-300,-200,-100,0,100,200,300]))
        self.assertEqual(result["best_sampled_stage_z_nm"],0)
        self.assertAlmostEqual(result["interpolated_stage_z_nm"],0,delta=3)
        self.assertFalse(result["peak_at_edge"])
        self.assertIn("ncc",result["frames"][0])
        plot = self.path/"curve.svg"
        write_scan_svg(plot,result)
        self.assertIn("Stage position",plot.read_text())

    def test_scan_edge_and_multiple_peaks_are_flagged(self):
        edge = scan_focus(self.reference,self.manifest([0,100,200]))
        self.assertTrue(edge["peak_at_edge"])
        self.assertIsNone(edge["interpolated_stage_z_nm"])
        tied = scan_focus(self.reference,self.manifest([-100,0,100],prefix="tied",image_function=lambda z: bead()))
        self.assertTrue(tied["ambiguous_peak"])
        self.assertIsNone(tied["interpolated_stage_z_nm"])

    def test_invalid_scan_rows_are_not_silently_dropped(self):
        manifest=self.manifest([-100,0,100])
        with manifest.open("a",encoding="utf-8") as stream:
            stream.write("200,missing.tif\n")
        with self.assertRaises(FocusError):
            scan_focus(self.reference,manifest)

    def test_scan_repeated_z_frames_are_averaged(self):
        result=scan_focus(self.reference,self.manifest([-100,0,0,100]))
        self.assertEqual(len(result["curve"]),3)
        self.assertEqual(len(result["frames"]),4)

    def test_cli_reference_and_csv_with_unicode_paths(self):
        script=Path(__file__).resolve().parents[1]/"bead_focus.py"
        image=self.path/"参考 珠.tif"
        current=self.path/"当前 珠.tif"
        ref=self.path/"reference.npz"
        output=self.path/"results.csv"
        self.save_image(image,bead())
        self.save_image(current,bead())
        env=dict(os.environ,PYTHONIOENCODING="utf-8")
        first=subprocess.run([sys.executable,str(script),"init",str(image),"--output",str(ref)],capture_output=True,encoding="utf-8",env=env)
        self.assertEqual(first.returncode,0,first.stderr)
        second=subprocess.run([sys.executable,str(script),"compare",str(ref),str(current),"--csv",str(output)],capture_output=True,encoding="utf-8",env=env)
        self.assertEqual(second.returncode,0,second.stderr)
        result=json.loads(second.stdout)[0]
        self.assertGreater(result["radial_correlation"],0.999)
        with output.open(encoding="utf-8-sig") as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]["image"],str(current))
        for field in ("ncc","radial_correlation","shift_dx_px","shift_dy_px"):
            self.assertIn(field,rows[0])
        self.assertAlmostEqual(float(rows[0]["ncc"]),1,places=6)


if __name__ == "__main__":
    unittest.main()
