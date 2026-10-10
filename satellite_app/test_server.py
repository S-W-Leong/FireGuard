import importlib
import json
import tempfile
import datetime as dt
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    subject = importlib.import_module('server')
except ModuleNotFoundError:
    subject = None


class PreviewContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'Preview backend has not been implemented')
        self.request = {
            'bbox': [104.85, -3.35, 104.95, -3.25],
            'before_start': '2023-07-01', 'before_end': '2023-07-31',
            'after_start': '2023-10-01', 'after_end': '2023-10-31',
        }

    def test_rejects_invalid_or_unbounded_imagery_requests(self):
        for box in ([104, -3, 103, -2], [0, 0, 180, 90], [104, -3, float('nan'), -2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                subject.validate_request(dict(self.request, bbox=box))

    def test_rejects_overlapping_before_after_windows(self):
        request = dict(self.request, after_start='2023-07-20', after_end='2023-07-31')
        with self.assertRaises(ValueError):
            subject.validate_request(request)

    def test_preserves_valid_historical_dates_and_geography(self):
        result = subject.validate_request(self.request)
        self.assertEqual(result['bbox'], [104.85, -3.35, 104.95, -3.25])
        self.assertEqual(result['before_start'], '2023-07-01')
        self.assertEqual(result['after_end'], '2023-10-31')

    def test_defaults_use_two_recent_nonoverlapping_windows(self):
        request = subject.recent_request(today=dt.date(2026, 10, 11))
        self.assertEqual(request['after_end'], '2026-10-10')
        self.assertEqual(request['after_start'], '2026-09-11')
        self.assertEqual(request['before_end'], '2026-09-10')
        self.assertEqual(request['before_start'], '2026-08-12')
        self.assertEqual(subject.validate_request(request), request)

    def test_recent_defaults_preserve_the_selected_location(self):
        box = [105.14, -3.56, 105.16, -3.54]
        self.assertEqual(subject.recent_request(box, dt.date(2026, 10, 11))['bbox'], box)

    def test_git_lfs_pointer_is_not_reported_as_available_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'unet_model.h5'
            model.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5726504\n')
            result = subject.describe_model(model)
            self.assertFalse(result['available'])
            self.assertEqual(result['reason'], 'weights_missing')
            self.assertEqual(result['expected_bytes'], 5726504)

    @staticmethod
    def quality_day(day, mean, sample_count=1000, no_data=0):
        return {
            'interval': {'from': '2023-07-%02dT00:00:00Z' % day,
                         'to': '2023-07-%02dT00:00:00Z' % (day + 1)},
            'outputs': {'clear': {'bands': {'B0': {'stats': {
                'mean': mean, 'sampleCount': sample_count, 'noDataCount': no_data,
            }}}}},
        }

    def test_local_quality_penalizes_missing_coverage(self):
        result = subject.select_daily_quality({'data': [
            self.quality_day(1, 1.0, no_data=900),
            self.quality_day(2, 0.8),
        ]})
        self.assertEqual(result['interval']['from'], '2023-07-02T00:00:00Z')
        self.assertEqual(result['quality'], {'clear_percent': 80.0, 'coverage_percent': 100.0})

    def test_local_quality_skips_missing_invalid_and_empty_statistics(self):
        result = subject.select_daily_quality({'data': [
            {'interval': {}, 'error': {'message': 'No data'}},
            self.quality_day(1, float('nan')),
            self.quality_day(2, 1.0, no_data=1000),
            self.quality_day(3, 0.75, no_data=200),
        ]})
        self.assertEqual(result['interval']['from'], '2023-07-03T00:00:00Z')
        self.assertEqual(result['quality'], {'clear_percent': 60.0, 'coverage_percent': 80.0})

    def test_no_usable_statistics_reports_no_imagery(self):
        with self.assertRaises(subject.ProviderError):
            subject.select_daily_quality({'data': [self.quality_day(1, None)]})

    def test_old_cached_cutouts_are_not_restored(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(subject, 'CACHE', Path(directory)):
            cached = {'before': {'image_url': '/images/old-before.png'},
                      'after': {'image_url': '/images/old-after.png'}}
            for filename in ('old-before.png', 'old-after.png'):
                (Path(directory) / filename).write_bytes(b'old cutout')
            (Path(directory) / 'latest.json').write_text(json.dumps(cached))
            self.assertIsNone(subject.get_cached())


class CredentialConfiguration(unittest.TestCase):
    def test_loads_app_local_credentials_without_exporting_variables(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(subject.os.environ, {}, clear=True):
            env = Path(directory) / '.env'
            env.write_text('# Local credentials\nSENTINEL_CLIENT_ID="test-client"\nSENTINEL_CLIENT_SECRET=\'test-secret\'\n')
            client = subject.SentinelClient(env_file=env)
            self.assertTrue(client.configured)
            self.assertEqual(client.values['CLIENT_ID'], 'test-client')
            self.assertEqual(client.values['CLIENT_SECRET'], 'test-secret')

    def test_environment_overrides_local_credential_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(subject.os.environ, {
            'SENTINEL_CLIENT_ID': 'environment-client',
            'SENTINEL_CLIENT_SECRET': 'environment-secret',
        }, clear=True):
            env = Path(directory) / '.env'
            env.write_text('SENTINEL_CLIENT_ID=file-client\nSENTINEL_CLIENT_SECRET=file-secret\n')
            client = subject.SentinelClient(env_file=env)
            self.assertEqual(client.values['CLIENT_ID'], 'environment-client')
            self.assertEqual(client.values['CLIENT_SECRET'], 'environment-secret')

    def test_fresh_checkout_reports_missing_credentials_without_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(subject.os.environ, {}, clear=True):
            client = subject.SentinelClient(env_file=Path(directory) / '.env')
            self.assertFalse(client.configured)
            with self.assertRaisesRegex(subject.ProviderError, 'not configured'):
                client.authenticate()


if __name__ == '__main__':
    unittest.main()
