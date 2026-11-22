"""FFmpeg metadata validation shared by CPU checks and the GPU child."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

MAX_FRAMES = 900


def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-show_streams', '-show_format', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(r.stdout)


def metadata(path, scale):
    p = probe(path)
    videos = [s for s in p['streams'] if s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if len(videos) != 1:
        raise ValueError('Input must have exactly one video stream')
    v = videos[0]
    if v.get('pix_fmt', '').startswith(('yuv420p10', 'yuv422p10', 'yuv444p10')) or v.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('This first version accepts SDR 8-bit video; HDR is not supported')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', '0:1', 'N/A'):
        raise ValueError('Anamorphic video is not supported; export square pixels')
    width, height = int(v['width']), int(v['height'])
    rotation = next((float(s['rotation']) for s in v.get('side_data_list', []) if 'rotation' in s), 0) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError('Unsupported video rotation')
    if rotation in (90, 270):
        width, height = height, width
    duration = float(v.get('duration') or p.get('format', {}).get('duration') or 0)
    try:
        fps = Fraction(v.get('avg_frame_rate', '0/1'))
    except (ValueError, ZeroDivisionError):
        raise ValueError('Invalid video frame rate')
    if not math.isfinite(duration) or not 0 < duration <= 15.1 or not 1 <= fps <= 60:
        raise ValueError('Input must be at most 15 seconds and 1-60 fps')
    if min(width, height) < 64 or max(width, height) > 1920 or width * height > 1920 * 1080:
        raise ValueError('Input dimensions must be 64-1920px, with area <= 1920x1080')
    ow, oh = max(8, round(width*scale/8)*8), max(8, round(height*scale/8)*8)
    if max(ow, oh) > 3840 or ow*oh > 3840*2160:
        raise ValueError('Output exceeds 3840px per edge or 3840x2160 pixels of area')
    # Inspect timestamps: reject VFR rather than silently changing its timing.
    r = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'v:0', '-read_intervals', '%+#1000', '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], capture_output=True, text=True, timeout=45, check=True)
    frames = json.loads(r.stdout).get('frames', [])
    if not 2 <= len(frames) <= MAX_FRAMES:
        raise ValueError('Input must contain 2-900 video frames')
    times = [float(f['best_effort_timestamp_time']) for f in frames]
    if any(not math.isfinite(t) for t in times):
        raise ValueError('Invalid video timestamps')
    if any(abs((b-a)-float(1/fps)) > max(.002, float(1/fps)*.05) for a,b in zip(times,times[1:])):
        raise ValueError('Variable-frame-rate input is unsupported; export constant frame rate')
    return {'input_width': width, 'input_height': height, 'width': ow, 'height': oh,
            'fps_fraction': str(fps), 'fps': float(fps), 'frames': len(frames),
            'duration_seconds': float(len(frames)/fps),
            'has_audio': any(s.get('codec_type') == 'audio' for s in p['streams'])}
