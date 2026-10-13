"""Reproduce the two controlled runs; test is used only after validation selection."""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
(ROOT/'cache/ultralytics').mkdir(parents=True,exist_ok=True)
(ROOT/'cache/matplotlib').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from download_initial import download_initial
import torch
from ultralytics import YOLO
import ultralytics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='0'); ap.add_argument('--epochs',type=int,default=30)
    args=ap.parse_args()
    os.chdir(ROOT)
    if not (ROOT/'data.yaml').exists():
        raise SystemExit('Run prepare.py first')
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
         'ultralytics':ultralytics.__version__, 'cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    (ROOT/'requirements-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu124\n'+subprocess.check_output([sys.executable,'-m','pip','list','--format=freeze','--disable-pip-version-check'],text=True))
    rows=[]
    download_initial()
    initial=json.loads((ROOT/'initial_weights.json').read_text())
    assert hashlib.sha256((ROOT/'yolov8n.pt').read_bytes()).hexdigest()==initial['sha256'], 'Initial weights checksum mismatch'
    for name,augment in [('baseline',False),('augmented',True)]:
        dest=ROOT/'runs'/name
        if dest.exists():
            raise SystemExit(f'{dest} exists. Use a fresh directory to reproduce; results are never overwritten.')
        model=YOLO(str(ROOT/'yolov8n.pt'))
        start=time.perf_counter()
        model.train(data=str(ROOT/'data.yaml'),epochs=args.epochs,imgsz=416,batch=16,device=args.device,
                    workers=0,seed=20261013,deterministic=True,amp=False,optimizer='AdamW',lr0=.001,
                    patience=0,project=str(ROOT/'runs'),name=name,exist_ok=False,plots=True,
                    mosaic=1.0 if augment else 0.0, fliplr=.5 if augment else 0.,
                    hsv_h=.015 if augment else 0.,hsv_s=.7 if augment else 0.,hsv_v=.4 if augment else 0.,
                    translate=.1 if augment else 0.,scale=.5 if augment else 0.,close_mosaic=5,
                    degrees=0.,shear=0.,perspective=0.,flipud=0.,mixup=0.,copy_paste=0.)
        seconds=time.perf_counter()-start
        best=dest/'weights/best.pt'
        model=YOLO(str(best))
        val=model.val(data=str(ROOT/'data.yaml'),split='val',imgsz=416,batch=16,device=args.device,
                      workers=0,conf=.001,iou=.7,plots=True,project=str(ROOT/'evaluations'),name=name+'_val')
        rows.append(dict(run=name,weights=str(best.relative_to(ROOT)),training_seconds=seconds,
                         validation={k:float(v) for k,v in val.results_dict.items()}))
        (ROOT/'comparison.json').write_text(json.dumps(rows,indent=2))
    selected=max(rows,key=lambda r:r['validation']['metrics/mAP50-95(B)'])
    model=YOLO(str(ROOT/selected['weights']))
    test=model.val(data=str(ROOT/'data.yaml'),split='test',imgsz=416,batch=16,device=args.device,
                   workers=0,conf=.001,iou=.7,plots=True,project=str(ROOT/'evaluations'),name='selected_test')
    out=ROOT.parents[1]/'models/drone_yolov8n_v1.pt'; out.parent.mkdir(exist_ok=True)
    from export_compat import export
    export(ROOT/selected['weights'],out)
    samples=sorted((ROOT/'data/images/test').glob('*.jpg'))[:12]
    results=model.predict(source=[str(p) for p in samples],imgsz=416,conf=.25,device=args.device,
                          save=True,save_txt=True,save_conf=True,project=str(ROOT/'examples'),name='test_predictions')
    report=dict(selected_run=selected['run'],selection_metric='validation mAP50-95',
                test={k:float(v) for k,v in test.results_dict.items()},speed_ms=test.speed,
                model_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),names=model.names,
                prediction_threshold=.25,sample_detection_counts=[len(r.boxes) for r in results])
    (ROOT/'final_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
