"""Exercise the existing Flask model selection/load/frame routes without a database."""
import argparse,base64,json,os,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--weights',type=Path)
    args=ap.parse_args()
    repo=args.repo.resolve()
    explicit_weights=args.weights.resolve() if args.weights else None
    root=Path(__file__).resolve().parent
    (root/'cache/ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(root/'cache/ultralytics')
    os.environ['MPLCONFIGDIR']=str(root/'cache/matplotlib')
    sys.path.insert(0,str(repo))
    sandbox=root/'cache/platform-smoke'; sandbox.mkdir(parents=True,exist_ok=True)
    os.chdir(sandbox)
    import app
    import torch,ultralytics
    weights=explicit_weights or root.parents[1]/'models/drone_yolov8n_v1.pt'
    app.MODELS_DIR=str(weights.parent)
    client=app.app.test_client()
    available=client.get('/api/models/scenario/drone')
    assert available.status_code==200,available.json
    assert weights.name in json.dumps(available.json),available.json
    loaded=client.post('/api/models/load',json={'model_path':str(weights)})
    assert loaded.status_code==200 and loaded.json['success'],loaded.json
    app.model.to('cpu')
    sample=sorted((root/'data/images/test').glob('*.jpg'))[0]
    frame=client.post('/api/process_frame',json={'image':'data:image/jpeg;base64,'+base64.b64encode(sample.read_bytes()).decode(),
                      'model_path':str(weights),'confidence':.25})
    assert frame.status_code==200 and frame.json['success'],frame.json
    assert app.model.names=={0:'drone'}
    try:
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit='unavailable'
    response=frame.get_json()
    response['model_path']=weights.name
    report={'repository_commit':commit,'ultralytics':ultralytics.__version__,'torch':torch.__version__,
            'scenario_list':'pass','model_load':'pass','frame_inference':'pass',
            'names':app.model.names,'response':response,
            'scope':'Flask test client; real CPU inference; no MySQL or browser/camera hardware validation'}
    (root/'platform_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
