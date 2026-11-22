"""Isolated SDK process; one frame at a time, no H3 or ComfyUI dependency."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from media import metadata, probe


def read_frame(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size-len(chunks))
        if not part:
            if chunks:
                raise RuntimeError('Truncated decoded video frame')
            return None
        chunks.extend(part)
    return bytes(chunks)


def environment():
    import torch
    import nvvfx
    if int((torch.version.cuda or '0').split('.')[0]) < 13:
        raise RuntimeError('Use the CUDA 13 PyTorch image')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select a CUDA 13 compatible GPU host')
    return torch, nvvfx, {'gpu': torch.cuda.get_device_name(0),
        'pytorch': torch.__version__, 'cuda_build': torch.version.cuda,
        'nvidia_vfx': importlib.metadata.version('nvidia-vfx')}


def smoke(torch, sdk):
    # Synthetic gradient only; verifies inference, output ownership and range.
    image = torch.linspace(0, 1, 96, device='cuda').view(1, 1, 96).expand(3, 64, 96).contiguous()
    with sdk.VideoSuperRes(quality=sdk.QualityLevel.HIGH, strength=1.0, device=0,
                          image_encoding=sdk.ImageEncoding.RGB8) as effect:
        effect.output_width, effect.output_height = 192, 128
        effect.load()
        result = torch.from_dlpack(effect.run(image).image).clone()
        torch.cuda.synchronize()
        if tuple(result.shape) != (3, 128, 192) or not torch.isfinite(result).all() or float(result.std()) < .001:
            raise RuntimeError('SDK smoke test produced invalid or constant output')
        return {'ok': True, 'sdk_inference_verified': True,
                'smoke_width': 192, 'smoke_height': 128}


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)


def process_video(folder, settings, info, torch, sdk):
    source, output = folder/'input.mp4', folder/'output.mp4'
    width, height = info['input_width'], info['input_height']
    decode = ['ffmpeg', '-v', 'error', '-nostdin', '-protocol_whitelist', 'file,pipe',
              '-i', str(source), '-map', '0:v:0', '-an', '-sn',
              '-vf', f'scale={width}:{height}:flags=lanczos',
              '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    encode = ['ffmpeg', '-v', 'error', '-nostdin', '-y',
              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{info['width']}x{info['height']}",
              '-framerate', info['fps_fraction'], '-i', 'pipe:0',
              '-protocol_whitelist', 'file,pipe', '-i', str(source),
              '-map', '0:v:0', '-map', '1:a:0?', '-map_metadata', '-1',
              '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '2',
              '-pix_fmt', 'yuv420p', '-frames:v', str(info['frames'])]
    if info['has_audio']:
        encode += ['-c:a', 'aac', '-b:a', '128k', '-af', 'apad']
    encode += ['-t', str(info['duration_seconds']), '-movflags', '+faststart', str(output)]
    with ExitStack() as stack:
        de = stack.enter_context((folder/'decode.log').open('w+b'))
        en = stack.enter_context((folder/'encode.log').open('w+b'))
        decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=de)
        encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=en)
        start = time.monotonic()
        count = 0
        try:
            with torch.cuda.device(0), sdk.VideoSuperRes(quality=sdk.QualityLevel[settings['quality']],
                    strength=settings['strength'], device=0, image_encoding=sdk.ImageEncoding.RGB8) as effect:
                effect.output_width, effect.output_height = info['width'], info['height']
                effect.load()
                while True:
                    raw = read_frame(decoder.stdout, width*height*3)
                    if raw is None:
                        break
                    if count >= info['frames']:
                        raise RuntimeError('Decoded frame count exceeds validated metadata')
                    frame = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(height, width, 3)
                    frame = frame.to(device='cuda', dtype=torch.float32).div_(255).movedim(-1,0).contiguous()
                    result = torch.from_dlpack(effect.run(frame).image).clone()
                    if tuple(result.shape) != (3, info['height'], info['width']) or not torch.isfinite(result).all():
                        raise RuntimeError('Invalid SDK output dimensions or pixel values')
                    encoded_frame = result.clamp(0,1).mul(255).round().to(torch.uint8).movedim(0,-1).contiguous().cpu()
                    encoder.stdin.write(encoded_frame.numpy().tobytes())
                    count += 1
                    if count % 24 == 0:
                        print(f'[VSR] {count}/{info["frames"]} frames', flush=True)
                torch.cuda.synchronize()
            encoder.stdin.close()
            if decoder.wait(timeout=30) or encoder.wait(timeout=60):
                raise RuntimeError('FFmpeg decode/encode failed')
            if count != info['frames']:
                raise RuntimeError('Decoded frame count does not match metadata')
        except Exception as error:
            stop(decoder); stop(encoder)
            de.seek(0); en.seek(0)
            raise RuntimeError(str(error) + '; ' + de.read()[-1000:].decode(errors='replace') + en.read()[-1000:].decode(errors='replace')) from error
        finally:
            decoder.stdout.close()
            if not encoder.stdin.closed:
                encoder.stdin.close()
            stop(decoder); stop(encoder)
    actual = probe(output)
    stream = next(s for s in actual['streams'] if s.get('codec_type')=='video')
    audio = any(s.get('codec_type')=='audio' for s in actual['streams'])
    if (stream['width'],stream['height']) != (info['width'],info['height']) or int(stream['nb_frames']) != count or audio != info['has_audio']:
        raise RuntimeError('Output validation failed: size, frame count or audio')
    return {**info, 'frames': count, 'audio': audio, 'quality': settings['quality'],
            'scale': settings['scale'], 'strength': settings['strength'],
            'sdk_inference_verified': True, 'processing_seconds': round(time.monotonic()-start,2)}


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--directory', required=True)
    parser.add_argument('--diagnose',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    args = parser.parse_args();folder=Path(args.directory)
    # Validate media before loading the GPU SDK.
    settings = None;info = None
    if not args.diagnose and not args.smoke_test:
        settings=json.loads((folder/'settings.json').read_text())
        info=metadata(folder/'input.mp4',settings['scale'])
    torch,sdk,env=environment()
    if args.diagnose:
        result={'ok':True,'sdk_inference_verified':False,'note':'GPU and SDK import checked; run smoke_test to verify inference'}
    elif args.smoke_test:
        result=smoke(torch,sdk)
    else:
        result=process_video(folder,settings,info,torch,sdk)
    result['runtime_environment']=env
    (folder/'result.json').write_text(json.dumps(result))


if __name__=='__main__':
    main()
