import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import model_runtime as subject


class ModelContracts(unittest.TestCase):
    def test_pair_has_original_six_channel_order_and_normalized_range(self):
        before = Image.new('RGB', (256, 256), (255, 0, 0))
        after = Image.new('RGB', (256, 256), (0, 0, 255))
        pair = subject.prepare_pair(before, after)
        self.assertEqual(pair.shape, (1, 128, 128, 6))
        self.assertEqual(pair.dtype, np.float32)
        np.testing.assert_array_equal(pair[0, 40, 40], [1, 0, 0, 0, 0, 1])

    def test_raw_rgb_is_percentile_stretched_before_model_resize(self):
        pixels = np.tile(np.arange(128, dtype=np.uint8)[None, :, None], (128, 1, 3))
        raw = Image.fromarray(pixels)
        pair = subject.prepare_pair(raw, raw)
        low, high = np.percentile(pixels, [2, 98])
        expected = (np.clip((pixels - low) / (high - low), 0, 1) * 255).astype(np.uint8) / 255
        np.testing.assert_allclose(pair[0, :, :, :3], expected, atol=1e-7)

    def test_constant_rgb_normalization_does_not_produce_nan(self):
        raw = Image.new('RGB', (128, 128), (50, 50, 50))
        pair = subject.prepare_pair(raw, raw)
        self.assertTrue(np.isfinite(pair).all())

    def test_clouds_in_either_image_cannot_be_reported_as_change(self):
        before = np.full((512, 512), 255, dtype=np.uint8)
        after = before.copy()
        before[:256] = 0
        after[:, :256] = 0
        valid = subject.common_clear_mask(Image.fromarray(before), Image.fromarray(after))
        changed, percentage = subject.threshold_prediction(np.ones((128, 128)), valid)
        self.assertEqual(changed.sum(), 64 * 64)
        self.assertEqual(percentage, 100.0)
        self.assertFalse(changed[:64].any())
        self.assertFalse(changed[:, :64].any())

    def test_no_clear_overlap_is_not_reported_as_zero_change(self):
        with self.assertRaisesRegex(subject.ModelError, 'clear'):
            subject.threshold_prediction(np.zeros((128, 128)), np.zeros((128, 128), dtype=bool))

    def test_nonfinite_or_wrong_sized_predictions_are_rejected(self):
        valid = np.ones((128, 128), dtype=bool)
        for output in (np.zeros((128, 128, 1)), np.full((128, 128), np.nan)):
            with self.subTest(shape=output.shape), self.assertRaises(subject.ModelError):
                subject.threshold_prediction(output, valid)

    def test_empty_prediction_produces_transparent_overlay(self):
        overlay = subject.make_overlay(np.zeros((128, 128), dtype=bool), (512, 512))
        self.assertEqual(overlay.mode, 'RGBA')
        self.assertIsNone(overlay.getchannel('A').getbbox())

    def test_only_positive_predictions_get_a_yellow_overlay(self):
        mask = np.zeros((128, 128), dtype=bool)
        mask[32:64, 32:64] = True
        overlay = np.asarray(subject.make_overlay(mask, (512, 512)))
        self.assertTrue((overlay[:128, :, 3] == 0).all())
        self.assertGreater(overlay[160, 160, 3], 0)
        self.assertGreater(overlay[128, 128, 0], 240)


if __name__ == '__main__':
    unittest.main()
