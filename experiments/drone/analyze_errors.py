"""Fixed-threshold detection error review on the frozen test set; no model selection."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import numpy as np
from ultralytics import YOLO

def main():
    model=YOLO(str(ROOT.parents[1]/'models/drone_yolov8n_v1.pt'))
    files=sorted((ROOT/'data/images/test').glob('*.jpg'))
    results=model.predict([str(p) for p in files],imgsz=416,conf=.25,iou=.7,device=0,stream=True,verbose=False)
    cases=[]
    for p,r in zip(files,results):
        height,width=r.orig_shape
        gt=[]
        for line in (ROOT/'data/labels/test'/(p.stem+'.txt')).read_text().splitlines():
            _,x,y,w,h=map(float,line.split())
            gt.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
        matched=set(); tp=0
        pred=r.boxes.data.cpu().numpy()
        for box in sorted(pred,key=lambda x:-x[4]):
            best=(-1,0.)
            for i,target in enumerate(gt):
                if i in matched: continue
                x1,y1=np.maximum(box[:2],target[:2]); x2,y2=np.minimum(box[2:4],target[2:])
                intersection=max(0,x2-x1)*max(0,y2-y1)
                union=(box[2]-box[0])*(box[3]-box[1])+(target[2]-target[0])*(target[3]-target[1])-intersection
                iou=intersection/union if union else 0
                if iou>best[1]: best=(i,float(iou))
            if best[1]>=.5: matched.add(best[0]); tp+=1
        cases.append({'image':p.name,'tp':tp,'fp':len(pred)-tp,'fn':len(gt)-tp,
                      'ground_truth_boxes':gt,'predictions':pred.tolist()})
    totals={k:sum(c[k] for c in cases) for k in ['tp','fp','fn']}
    summary={'confidence':.25,'matching_iou':.5,'nms_iou':.7,'imgsz':416,**totals,
             'precision':totals['tp']/max(1,totals['tp']+totals['fp']),
             'recall':totals['tp']/max(1,totals['tp']+totals['fn']),
             'matching':'predictions sorted by confidence; greedy unmatched GT IoU>=0.5; single class',
             'cases':sorted(cases,key=lambda c:-(c['fp']+c['fn']))}
    (ROOT/'error_review.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))

if __name__=='__main__': main()
