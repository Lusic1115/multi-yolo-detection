"""Export inference-only weights and prove their predictions match the checkpoint."""
import argparse,copy,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
(ROOT/'cache/ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import torch
from ultralytics import YOLO

def export(source, output):
    model=YOLO(str(source))
    inference=copy.deepcopy(model.model).cpu().half()
    if hasattr(inference,'criterion'):
        delattr(inference,'criterion')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':inference,'optimizer':None,'epoch':-1,
                'train_args':{'task':'detect','imgsz':416}},output)
    clean=YOLO(str(output))
    sources=[str(p) for p in sorted((ROOT/'data/images/val').glob('*.jpg'))[:8]]
    before=model.predict(sources,device='cpu',imgsz=416,conf=.25,verbose=False)
    after=clean.predict(sources,device='cpu',imgsz=416,conf=.25,verbose=False)
    maxdiff=0.
    for a,b in zip(before,after):
        assert a.boxes.data.shape==b.boxes.data.shape
        assert torch.allclose(a.boxes.data,b.boxes.data,rtol=1e-5,atol=1e-5)
        if a.boxes.data.numel():
            maxdiff=max(maxdiff,float((a.boxes.data-b.boxes.data).abs().max()))
    info={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'export_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'comparison_images':len(sources),'max_absolute_box_difference':maxdiff,
          'operation':'remove training-only criterion and optimizer; preserve model parameters'}
    (ROOT/'export_verification.json').write_text(json.dumps(info,indent=2))
    metrics=ROOT/'final_metrics.json'
    if metrics.exists():
        final=json.loads(metrics.read_text()); final['model_sha256']=info['export_sha256']
        metrics.write_text(json.dumps(final,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--output',type=Path,default=ROOT.parents[1]/'models/drone_yolov8n_v1.pt')
    args=ap.parse_args(); export(args.source,args.output)
