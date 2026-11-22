"""CPU-only checks. The SDK in the pipeline test is a nearest-neighbor fake."""
import base64
from contextlib import nullcontext
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import handler
import media
import runtime
from vsr_engine import process_video, read_frame


class Tensor:
    def __init__(self, a): self.a=np.asarray(a)
    @property
    def shape(self): return self.a.shape
    def reshape(self,*shape): return Tensor(self.a.reshape(*shape))
    def to(self,*args,device=None,dtype=None): return Tensor(self.a.astype(dtype or args[0] if args else dtype or self.a.dtype))
    def div_(self,n): self.a/=n;return self
    def movedim(self,a,b): return Tensor(np.moveaxis(self.a,a,b))
    def contiguous(self): return Tensor(np.ascontiguousarray(self.a))
    def clone(self): return Tensor(self.a.copy())
    def clamp(self,a,b): return Tensor(np.clip(self.a,a,b))
    def mul(self,n): return Tensor(self.a*n)
    def round(self): return Tensor(np.round(self.a))
    def cpu(self): return self
    def numpy(self): return self.a


class FakeEffect:
    def __init__(self,**kw): pass
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def load(self): pass
    def run(self,frame):
        a=frame.a
        out=np.repeat(np.repeat(a,self.output_height//a.shape[1],axis=1),self.output_width//a.shape[2],axis=2)
        return SimpleNamespace(image=Tensor(out))


FAKE_TORCH=SimpleNamespace(uint8=np.uint8,float32=np.float32,
    cuda=SimpleNamespace(device=lambda i:nullcontext(),synchronize=lambda:None),
    frombuffer=lambda b,dtype:Tensor(np.frombuffer(b,dtype=dtype)),from_dlpack=lambda t:t,
    isfinite=lambda t:np.isfinite(t.a))
FAKE_SDK=SimpleNamespace(VideoSuperRes=FakeEffect,QualityLevel={'HIGH':'HIGH'},ImageEncoding=SimpleNamespace(RGB8='RGB8'))


class Contracts(unittest.TestCase):
    def test_input_limits_before_gpu(self):
        with patch.object(runtime,'execute') as execute:
            for p in [None,{}, {'video_base64':'!!'}, {'video_base64':'YWJj','scale':True},
                      {'video_base64':'YWJj','scale':1},{'video_base64':'YWJj','quality':'BICUBIC'},
                      {'video_base64':'YWJj','strength':float('nan')}, {'video_url':'https://example.com'},
                      {'diagnose':True,'video_base64':'YWJj'}, {'diagnose':True,'smoke_test':True}]:
                self.assertIn('error',handler.handler({'input':p}))
            execute.assert_not_called()
        self.assertEqual(runtime.validate({'video_base64':'YWJj'})['scale'],2)

    def test_read_exact_and_truncated_frames(self):
        self.assertEqual(read_frame(io.BytesIO(b'abc'),3),b'abc')
        self.assertIsNone(read_frame(io.BytesIO(),3))
        with self.assertRaises(RuntimeError):read_frame(io.BytesIO(b'a'),3)

    def test_secret_mask(self):
        with patch.dict(runtime.os.environ,{'RUNPOD_API_KEY':'secret-key-value'}):
            self.assertNotIn('secret-key-value',runtime.safe_text('secret-key-value'))

    def test_transport_and_cleanup(self):
        seen=[]
        def fake(folder,args):
            seen.append(folder);self.assertEqual((folder/'input.mp4').read_bytes(),b'abc')
            (folder/'output.mp4').write_bytes(b'mock-mp4')
            return {'width':128,'height':128,'audio':True}
        with patch.object(runtime,'worker_child',side_effect=fake):
            r=runtime.execute(runtime.validate({'video_base64':'YWJj'}))
        self.assertEqual(base64.b64decode(r['video_base64']),b'mock-mp4')
        self.assertTrue(all(not p.exists() for p in seen))

    def test_failed_child_cleans_private_files(self):
        seen=[]
        def fail(folder,args):seen.append(folder);raise RuntimeError('simulated SDK failure')
        with patch.object(runtime,'worker_child',side_effect=fail):
            with self.assertRaises(RuntimeError):runtime.execute(runtime.validate({'video_base64':'YWJj'}))
        self.assertTrue(all(not p.exists() for p in seen))

    def test_diagnostic_dispatch_does_not_make_video(self):
        with patch.object(runtime,'worker_child',return_value={'ok':True,'sdk_inference_verified':False}) as child:
            r=runtime.execute(runtime.validate({'diagnose':True}))
        self.assertEqual(child.call_args.args[1],['--diagnose'])
        self.assertFalse(r['sdk_inference_verified']);self.assertNotIn('video_base64',r)

    def test_media_caps_and_vfr_reject(self):
        meta={'streams':[{'codec_type':'video','width':800,'height':800,'avg_frame_rate':'24/1','duration':'16'}]}
        with patch.object(media,'probe',return_value=meta):
            with self.assertRaisesRegex(ValueError,'15 seconds'):media.metadata(Path('x'),2)
        meta['streams'][0]['duration']='1';meta['streams'][0]['width']=1920;meta['streams'][0]['height']=1080
        with patch.object(media,'probe',return_value=meta):
            with self.assertRaisesRegex(ValueError,'Output exceeds'):media.metadata(Path('x'),4)
        meta['streams'][0].update(width=800,height=800)
        frames={'frames':[{'best_effort_timestamp_time':str(t)} for t in (0,.04,.12)]}
        with patch.object(media,'probe',return_value=meta),patch.object(media.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(frames))):
            with self.assertRaisesRegex(ValueError,'Variable'):media.metadata(Path('x'),2)

    def test_real_ffmpeg_stream_pipeline_audio_frames_and_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=64x64:rate=8:duration=0.5',
                '-f','lavfi','-i','sine=frequency=440:duration=0.5','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(folder/'input.mp4')],check=True)
            info=media.metadata(folder/'input.mp4',2)
            r=process_video(folder,{'quality':'HIGH','scale':2,'strength':1},info,FAKE_TORCH,FAKE_SDK)
            self.assertEqual((r['width'],r['height']),(128,128))
            self.assertEqual(r['frames'],4);self.assertEqual(r['fps'],8)
            self.assertTrue(r['audio'])
            self.assertAlmostEqual(r['duration_seconds'],.5)
            self.assertTrue((folder/'output.mp4').is_file())


if __name__=='__main__':unittest.main()
