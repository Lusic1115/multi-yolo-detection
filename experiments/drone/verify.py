"""Verify delivery integrity, split isolation, and an actual boxes inference."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
(ROOT/'cache/ultralytics').mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'cache/ultralytics')
from ultralytics import YOLO

def main():
    records=json.loads((ROOT/'split_manifest.json').read_text())
    repairs={r['original']:r['current'] for r in json.loads((ROOT/'image_repairs.json').read_text())}
    groups={}; pixels={}
    for r in records:
        name=r['image_sha256'][:16]
        for kind,suffix,digest in [('images','.jpg',repairs.get(r['image_sha256'],r['image_sha256'])),('labels','.txt',r['label_sha256'])]:
            p=ROOT/'data'/kind/r['split']/(name+suffix)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Data changed: {p}'
        for key, value in [(groups,r['group']),(pixels,r['pixel_sha256'])]:
            assert key.setdefault(value,r['split'])==r['split'], 'Split leakage'
    for i,r in enumerate(records):
        for s in records[:i]:
            if r['split']!=s['split']:
                assert (r['dhash']^s['dhash']).bit_count()>4, 'Near duplicate leakage'
    final=json.loads((ROOT/'final_metrics.json').read_text())
    weights=ROOT.parents[1]/'models/drone_yolov8n_v1.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==final['model_sha256']
    model=YOLO(str(weights))
    assert model.task=='detect' and model.names=={0:'drone'}
    example=next((ROOT/'data/images/test').glob('*.jpg'))
    result=model(str(example),conf=.25,device='cpu',verbose=False)[0]
    assert result.boxes is not None
    for b in result.boxes:
        assert len(b.xyxy[0])==4 and int(b.cls[0])==0
    print(json.dumps({'integrity':'pass','split_isolation':'pass','task':model.task,'names':model.names,
                     'inference':'pass','boxes':len(result.boxes)},indent=2))

if __name__=='__main__': main()
