"""Local-only Sentinel imagery preview. Credentials never enter the browser."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from model_runtime import ANALYSIS_VERSION, MODEL_PATH, ModelError, analyze_pair, model_status

ROOT = Path(__file__).resolve().parent
SECRET_FILE = ROOT / '.env'
MODEL_FILE = MODEL_PATH
CACHE = ROOT / 'cache'
CACHE_VERSION = 'daily-aoi-rgb-clear-mask-v3'
API_BASE = 'https://services.sentinel-hub.com'
TOKEN_URL = API_BASE + '/auth/realms/main/protocol/openid-connect/token'


def recent_request(bbox=None, today=None):
    today = today or dt.datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).date()
    end = today - dt.timedelta(days=1)
    after_start = end - dt.timedelta(days=29)
    before_end = after_start - dt.timedelta(days=1)
    return {'bbox': list(bbox or [105.14, -3.56, 105.16, -3.54]),
            'before_start': (before_end - dt.timedelta(days=29)).isoformat(),
            'before_end': before_end.isoformat(),
            'after_start': after_start.isoformat(), 'after_end': end.isoformat()}


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('Provide an image-pair request.')
    box = payload.get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError('A bounding box must contain west, south, east, north.')
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in box):
        raise ValueError('Bounding-box coordinates must be finite numbers.')
    west, south, east, north = box
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('The selected area has invalid coordinates.')
    if east - west > 0.25 or north - south > 0.25:
        raise ValueError('Keep the preview area within 0.25 degrees on each side.')
    keys = ('before_start', 'before_end', 'after_start', 'after_end')
    dates = {}
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            raise ValueError('Choose valid before and after dates.')
        try:
            dates[key] = dt.date.fromisoformat(value)
        except ValueError:
            raise ValueError('Choose valid calendar dates.') from None
    for prefix in ('before', 'after'):
        start, end = dates[prefix + '_start'], dates[prefix + '_end']
        if not 0 <= (end - start).days <= 62:
            raise ValueError('Each imagery window must be ordered and at most 63 days long.')
        if start < dt.date(2015, 6, 23) or end > dt.datetime.now(dt.timezone.utc).date():
            raise ValueError('Choose historical dates within the Sentinel-2 archive.')
    if dates['before_end'] >= dates['after_start']:
        raise ValueError('The before window must end before the after window starts.')
    return dict(bbox=list(box), **{key: dates[key].isoformat() for key in keys})


def describe_model(path=MODEL_FILE):
    return model_status(path)


class ProviderError(Exception):
    pass


def select_daily_quality(result):
    """Rank estimated usable pixels over the full bbox, including data gaps."""
    candidates = []
    for day in result.get('data', []):
        try:
            if day.get('error'):
                continue
            stats = day['outputs']['clear']['bands']['B0']['stats']
            mean, total, missing = (stats[key] for key in ('mean', 'sampleCount', 'noDataCount'))
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in (mean, total, missing)):
                continue
            if not (0 <= mean <= 1 and total > 0 and 0 <= missing < total):
                continue
            interval = day['interval']
            start = dt.datetime.fromisoformat(interval['from'].replace('Z', '+00:00'))
            end = dt.datetime.fromisoformat(interval['to'].replace('Z', '+00:00'))
            if end - start != dt.timedelta(days=1):
                continue
            coverage = (total - missing) / total
            clear = mean * coverage
            candidates.append((clear, coverage, interval['from'], {
                'interval': dict(interval),
                'quality': {'clear_percent': round(100 * clear, 2),
                            'coverage_percent': round(100 * coverage, 2)},
            }))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    if not candidates:
        raise ProviderError('No usable Sentinel-2 observations were found in one of these date windows. Try different dates.')
    # Earliest day breaks ties consistently; no tile-wide cloud metadata is used.
    return min(candidates, key=lambda value: (-value[0], -value[1], value[2]))[3]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class SentinelClient:
    def __init__(self, env_file=SECRET_FILE):
        file_values = {}
        env_file = Path(env_file)
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    key, value = line.split('=', 1)
                    file_values[key.strip()] = value.strip().strip('\"\'')
        self.values = {
            key: os.environ.get('SENTINEL_' + key, file_values.get('SENTINEL_' + key, ''))
            for key in ('CLIENT_ID', 'CLIENT_SECRET')
        }
        self.token = None
        self.expires = 0
        self.status = 'not_tested'
        self.opener = urllib.request.build_opener(NoRedirect())
        self.lock = threading.Lock()

    @property
    def configured(self):
        return bool(self.values.get('CLIENT_ID') and self.values.get('CLIENT_SECRET'))

    def authenticate(self):
        if not self.configured:
            raise ProviderError('Sentinel Hub credentials are not configured on this server.')
        if self.token and time.monotonic() < self.expires:
            return self.token
        body = urllib.parse.urlencode({
            'grant_type': 'client_credentials',
            'client_id': self.values['CLIENT_ID'],
            'client_secret': self.values['CLIENT_SECRET'],
        }).encode()
        request = urllib.request.Request(TOKEN_URL, data=body, headers={
            'Content-Type': 'application/x-www-form-urlencoded',
        })
        try:
            with self.opener.open(request, timeout=30) as response:
                result = json.load(response)
            self.token = result['access_token']
            self.expires = time.monotonic() + max(1, result.get('expires_in', 300) - 30)
            self.status = 'connected'
            return self.token
        except urllib.error.HTTPError as exc:
            self.status = 'error'
            raise ProviderError('Sentinel Hub rejected authentication (HTTP %s).' % exc.code) from None
        except Exception:
            self.status = 'error'
            raise ProviderError('Could not reach Sentinel Hub authentication. Try again shortly.') from None

    def request(self, path, payload, binary=False):
        token = self.authenticate()
        request = urllib.request.Request(API_BASE + path, data=json.dumps(payload).encode(), headers={
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json',
            'Accept': ('image/png' if binary else
                       'application/geo+json' if path.startswith('/api/v1/catalog/') else 'application/json'),
        })
        try:
            with self.opener.open(request, timeout=75) as response:
                result = response.read() if binary else json.load(response)
            self.status = 'connected'
            return result
        except urllib.error.HTTPError as exc:
            self.status = 'error'
            if exc.code == 401:
                self.token = None
            if exc.code in (401, 403):
                raise ProviderError('Sign-in worked, but Sentinel Hub denied imagery access (HTTP %s). Check account access and subscription.' % exc.code) from None
            if exc.code == 429:
                raise ProviderError('Sentinel Hub is rate-limiting requests. Wait briefly and try again.') from None
            raise ProviderError('Sentinel Hub could not complete the imagery request (HTTP %s).' % exc.code) from None
        except Exception:
            self.status = 'error'
            raise ProviderError('The satellite imagery request timed out or could not be completed.') from None

    def scene(self, box, start, end):
        exclusive_end = (dt.date.fromisoformat(end) + dt.timedelta(days=1)).isoformat()
        payload = {
            'input': {
                'bounds': {'bbox': box, 'properties': {'crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}},
                'data': [{'type': 'sentinel-2-l2a', 'dataFilter': {'mosaickingOrder': 'leastCC'}}],
            },
            'aggregation': {
                'timeRange': {'from': start + 'T00:00:00Z', 'to': exclusive_end + 'T00:00:00Z'},
                'aggregationInterval': {'of': 'P1D'},
                'width': 256, 'height': 256,
                'evalscript': '''//VERSION=3
function setup() {
  return {input:["SCL","dataMask"],output:[
    {id:"clear",bands:1,sampleType:"FLOAT32"},{id:"dataMask",bands:1}]};
}
function evaluatePixel(s) {
  return {clear:[![0,1,3,8,9,10,11].includes(s.SCL)?1:0],dataMask:[s.dataMask]};
}''',
            },
        }
        return select_daily_quality(self.request('/api/v1/statistics', payload))

    def image(self, box, scene, clear_mask=False, model_input=False):
        payload = {
            'input': {
                'bounds': {'bbox': box, 'properties': {'crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}},
                'data': [{'type': 'sentinel-2-l2a', 'dataFilter': {
                    'timeRange': scene['interval'],
                    'mosaickingOrder': 'leastCC',
                }}],
            },
            'output': {'width': 512, 'height': 512, 'responses': [{'identifier': 'default', 'format': {'type': 'image/png'}}]},
            'evalscript': '''//VERSION=3
function setup() { return {input:["B02","B03","B04","dataMask"],output:{bands:4}}; }
function evaluatePixel(s) {
  return [Math.min(1,2.5*s.B04),Math.min(1,2.5*s.B03),Math.min(1,2.5*s.B02),s.dataMask];
}''',
        }
        if clear_mask:
            payload['evalscript'] = '''//VERSION=3
function setup() { return {input:["SCL","dataMask"],output:{bands:1,sampleType:"UINT8"}}; }
function evaluatePixel(s) {
  return [s.dataMask && ![0,1,3,8,9,10,11].includes(s.SCL)?255:0];
}'''
        elif model_input:
            payload['evalscript'] = '''//VERSION=3
function setup() { return {input:["B04","B03","B02"],output:{bands:3,sampleType:"AUTO"}}; }
function evaluatePixel(s) { return [s.B04,s.B03,s.B02]; }
'''
        content = self.request('/api/v1/process', payload, binary=True)
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ProviderError('Sentinel Hub returned an unexpected image format.')
        return content


CLIENT = SentinelClient()


def attach_change(result, cache_key):
    result['model'] = describe_model()
    analysis_version = ANALYSIS_VERSION
    filename = cache_key + '-changes.png'
    previous = result.get('change', {})
    if previous.get('analysis_version') == analysis_version and previous.get('available') and (CACHE / filename).is_file():
        previous['validation_status'] = 'unvalidated'
        return
    if not result['model']['available']:
        result['change'] = {'available': False, 'reason': result['model']['reason']}
        return
    def asset(period, key):
        return CACHE / result[period][key].split('/')[-1]
    try:
        change = analyze_pair(asset('before', 'model_image_url'), asset('after', 'model_image_url'),
                              asset('before', 'clear_mask_url'), asset('after', 'clear_mask_url'),
                              CACHE / filename)
        change.update(overlay_url='/images/' + filename, analysis_version=analysis_version)
        result['change'] = change
    except ModelError as exc:
        result['change'] = {'available': False, 'reason': str(exc)}


def get_cached():
    try:
        result = json.loads((CACHE / 'latest.json').read_text())
        if result.get('cache_version') != CACHE_VERSION:
            return None
        for key in ('before', 'after'):
            for asset in ('image_url', 'clear_mask_url', 'model_image_url'):
                filename = result[key][asset].split('/')[-1]
                if not (CACHE / filename).is_file():
                    return None
        if result.get('change', {}).get('available'):
            if result['change'].get('analysis_version') != ANALYSIS_VERSION:
                return None
            if not (CACHE / result['change']['overlay_url'].split('/')[-1]).is_file():
                return None
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def make_preview(payload):
    validated = validate_request(payload)
    cache_key = hashlib.sha256((CACHE_VERSION + json.dumps(validated, sort_keys=True)).encode()).hexdigest()[:24]
    manifest = CACHE / (cache_key + '.json')
    with CLIENT.lock:
        CLIENT.authenticate()
        if manifest.is_file():
            result = json.loads(manifest.read_text())
            if result.get('cache_version') == CACHE_VERSION and all(
                    (CACHE / result[k][asset].split('/')[-1]).is_file()
                    for k in ('before', 'after') for asset in ('image_url', 'clear_mask_url', 'model_image_url')):
                attach_change(result, cache_key)
                manifest.write_text(json.dumps(result, indent=2))
                (CACHE / 'latest.json').write_text(json.dumps(result))
                return result
        CACHE.mkdir(exist_ok=True)
        result = {'cache_version': CACHE_VERSION, 'bbox': validated['bbox'], 'source': 'Sentinel-2 L2A', 'historical': True,
                  'resolution_note': '512 × 512 natural-color display; RGB bands have native 10 m resolution. Each image is a daily mosaic selected for estimated clear coverage inside this area, sampled at 256 × 256. Clouds remain visible; only missing data are transparent. Clear coverage excludes cloud, shadow, snow and defective pixels using Sentinel-2 scene classification. A separate clear mask is supplied for analysis.',
                  'model': describe_model(), 'change': {'available': False, 'reason': 'weights_missing'},
                  'firms': {'configured': False}, 'request': validated}
        for period in ('before', 'after'):
            scene = CLIENT.scene(validated['bbox'], validated[period + '_start'], validated[period + '_end'])
            content = CLIENT.image(validated['bbox'], scene)
            clear_content = CLIENT.image(validated['bbox'], scene, clear_mask=True)
            model_content = CLIENT.image(validated['bbox'], scene, model_input=True)
            filename = cache_key + '-' + period + '.png'
            clear_filename = cache_key + '-' + period + '-clear.png'
            model_filename = cache_key + '-' + period + '-model.png'
            (CACHE / filename).write_bytes(content)
            (CACHE / clear_filename).write_bytes(clear_content)
            (CACHE / model_filename).write_bytes(model_content)
            result[period] = {'image_url': '/images/' + filename, 'clear_mask_url': '/images/' + clear_filename,
                              'model_image_url': '/images/' + model_filename,
                              'acquired_at': scene['interval']['from'][:10], 'interval': scene['interval'],
                              'quality': scene['quality'], 'mosaic': 'daily', 'cloud_cover': None, 'scene_id': None}
        attach_change(result, cache_key)
        manifest.write_text(json.dumps(result, indent=2))
        (CACHE / 'latest.json').write_text(json.dumps(result, indent=2))
        return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Never log request bodies, authentication headers, or query parameters.
        pass

    def reply(self, code, data, content_type='application/json'):
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self):
        return self.headers.get('Host', '').split(':')[0] in ('127.0.0.1', 'localhost')

    def do_GET(self):
        if not self.host_ok():
            return self.reply(403, {'error': 'Local access only.'})
        path = urllib.parse.urlsplit(self.path).path
        if path in ('/', '/index.html'):
            return self.reply(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/status':
            cached = get_cached()
            return self.reply(200, {'sentinel': {'configured': CLIENT.configured, 'status': CLIENT.status},
                                    'model': describe_model(), 'firms': {'configured': False},
                                    'default_request': recent_request(cached['bbox'] if cached else None),
                                    'cached_preview': cached})
        if re.fullmatch(r'/images/[a-f0-9]{24}-(?:(?:before|after)(?:-clear|-model)?|changes)\.png', path):
            image = CACHE / path.split('/')[-1]
            if image.is_file():
                return self.reply(200, image.read_bytes(), 'image/png')
        return self.reply(404, {'error': 'Not found.'})

    def do_POST(self):
        if not self.host_ok():
            return self.reply(403, {'error': 'Local access only.'})
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.reply(403, {'error': 'Cross-origin requests are not allowed.'})
        if self.path != '/api/preview':
            return self.reply(404, {'error': 'Not found.'})
        if self.headers.get_content_type() != 'application/json':
            return self.reply(415, {'error': 'Use a JSON request.'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 4096:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(length))
            result = make_preview(body)
            return self.reply(200, result)
        except (ValueError, UnicodeError) as exc:
            return self.reply(400, {'error': str(exc)})
        except ProviderError as exc:
            return self.reply(502, {'error': str(exc)})
        except Exception:
            return self.reply(500, {'error': 'The preview could not be completed. Try again.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print('FireGuard satellite preview at http://127.0.0.1:%s' % args.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
