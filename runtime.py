"""RunPod transport and resource limits. No CUDA work during module import."""
import base64
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

VERSION = 'rtx-vsr-v1'
HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
MAX_INPUT = 6 * 1024**2
MAX_OUTPUT = 6 * 1024**2
QUALITIES = ('LOW', 'MEDIUM', 'HIGH', 'ULTRA', 'HIGHBITRATE_HIGH', 'HIGHBITRATE_ULTRA')


def safe_text(value):
    s = str(value)
    for name, secret in os.environ.items():
        if any(k in name.upper() for k in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY')) and len(secret) >= 4:
            s = s.replace(secret, '[redacted]')
    s = re.sub(r'(?:hf_|rpa_|ghp_)[A-Za-z0-9_-]+', '[redacted]', s)
    return s[-3000:]


def validate(p):
    if not isinstance(p, dict) or set(p) - {'video_base64', 'scale', 'quality', 'strength', 'diagnose', 'smoke_test'}:
        raise ValueError('Unsupported input; use video_base64, scale, quality and strength')
    for flag in ('diagnose', 'smoke_test'):
        if flag in p and type(p[flag]) is not bool:
            raise ValueError(flag + ' must be true or false')
    if p.get('diagnose') and p.get('smoke_test'):
        raise ValueError('Choose diagnose or smoke_test')
    if (p.get('diagnose') or p.get('smoke_test')) and set(p) - {'diagnose', 'smoke_test'}:
        raise ValueError('Diagnostics cannot contain video or processing settings')
    if p.get('diagnose'):
        return {'diagnose': True}
    if p.get('smoke_test'):
        return {'smoke_test': True}
    encoded = p.get('video_base64')
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4 * ((MAX_INPUT + 2)//3):
        raise ValueError('Supply a video as base64; maximum input is 6 MiB')
    data = base64.b64decode(encoded, validate=True)
    if not data or len(data) > MAX_INPUT:
        raise ValueError('Video exceeds 6 MiB')
    scale = p.get('scale', 2)
    if type(scale) is not int or scale not in (2, 3, 4):
        raise ValueError('scale must be 2, 3 or 4')
    quality = p.get('quality', 'HIGH')
    if quality not in QUALITIES:
        raise ValueError('Unsupported VSR quality')
    strength = p.get('strength', 1.0)
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be between 0 and 1')
    return {'data': data, 'scale': scale, 'quality': quality, 'strength': strength}


def worker_child(folder, args):
    command = [sys.executable, str(HERE / 'vsr_engine.py'), '--directory', str(folder), *args]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
        raise TimeoutError('VSR processing exceeded 10 minutes; worker child was stopped')
    if child.returncode:
        raise RuntimeError('VSR runtime failed: ' + safe_text(output))
    result = json.loads((folder / 'result.json').read_text())
    if not isinstance(result, dict):
        raise RuntimeError('Invalid worker result')
    return result


def execute(p):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='rtx-vsr-') as name:
        folder = Path(name)
        if p.get('diagnose'):
            result = worker_child(folder, ['--diagnose'])
        elif p.get('smoke_test'):
            result = worker_child(folder, ['--smoke-test'])
        else:
            (folder / 'input.mp4').write_bytes(p['data'])
            settings = {k: p[k] for k in ('scale', 'quality', 'strength')}
            (folder / 'settings.json').write_text(json.dumps(settings))
            result = worker_child(folder, [])
            video = folder / 'output.mp4'
            if not video.is_file() or not 1 <= video.stat().st_size <= MAX_OUTPUT:
                raise RuntimeError('Output exceeds the 6 MiB response limit; use a shorter clip or lower scale')
            result.update(video_base64=base64.b64encode(video.read_bytes()).decode(), mime_type='video/mp4')
        result.update(worker_version=VERSION, worker_total_seconds=round(time.monotonic()-started, 2))
        return result
