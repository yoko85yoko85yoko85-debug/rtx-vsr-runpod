"""Create a RunPod request JSON locally; this script never submits a paid job."""
import argparse
import base64
import json
from pathlib import Path
from runtime import validate

p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--output',type=Path,default=Path('request.json'))
p.add_argument('--scale',type=int,default=2);p.add_argument('--quality',default='HIGH');a=p.parse_args()
settings={'video_base64':base64.b64encode(a.video.read_bytes()).decode(),'scale':a.scale,'quality':a.quality}
validate(settings)
a.output.write_text(json.dumps({'input':settings}))
print('Saved request JSON:',a.output)
