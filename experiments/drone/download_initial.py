"""Fetch the pinned pretrained model, rejecting partial or changed downloads."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent

def download_initial():
    metadata = json.loads((ROOT / 'initial_weights.json').read_text())
    target = ROOT / 'yolov8n.pt'
    if not target.exists():
        temporary = target.with_suffix('.pt.part')
        urlretrieve(metadata['url'], temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('Downloaded initial weights checksum mismatch')
        temporary.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Initial weights checksum mismatch')
    return target

if __name__ == '__main__':
    print(download_initial())
