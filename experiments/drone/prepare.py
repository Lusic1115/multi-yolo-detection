"""Download version 1, validate boxes, group related images and freeze the split."""
import argparse, hashlib, json, random, re, shutil, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
URL = 'https://www.kaggle.com/api/v1/datasets/download/muki2003/yolo-drone-detection-dataset?datasetVersionNumber=1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(source=None, verify_only=False):
    if source is None:
        archive = ROOT / 'cache/source-v1.zip'
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            temporary=archive.with_suffix('.zip.part')
            urllib.request.urlretrieve(URL, temporary)
            expected=ROOT/'source_sha256.txt'
            if expected.exists() and sha(temporary)!=expected.read_text().strip():
                raise ValueError('Downloaded source archive checksum mismatch')
            temporary.replace(archive)
        expected = ROOT / 'source_sha256.txt'
        digest = sha(archive)
        if expected.exists() and digest != expected.read_text().strip():
            raise ValueError('Source archive checksum mismatch')
        expected.write_text(digest + '\n')
        source = ROOT / 'cache/source'
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest = (source / member.filename).resolve()
                if not dest.is_relative_to(source.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(source)
    source = Path(source)
    records, seen, duplicates = [], {}, []
    for p in sorted(source.rglob('*.jpg')):
        label = p.parent.parent / 'labels' / (p.stem + '.txt')
        if not label.exists():
            raise ValueError(f'Missing label: {p}')
        boxes = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or values[0] != 0 or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                raise ValueError(f'Invalid box: {label}: {line}')
            _, x, y, w, h = values
            if x-w/2 < -0.001 or y-h/2 < -0.001 or x+w/2 > 1.001 or y+h/2 > 1.001:
                raise ValueError(f'Box outside image: {label}')
            boxes.append(values)
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            gray = list(rgb.convert('L').resize((9, 8)).getdata())
            dh = sum((gray[y*9+x] > gray[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        if pixel_sha in seen:
            duplicates.append({'removed': str(p.relative_to(source)), 'kept': seen[pixel_sha]})
            continue
        relative = p.relative_to(source).as_posix()
        seen[pixel_sha] = relative
        match = re.match(r'^(.*?)(\d+)$', p.stem)
        block = f'{match[1]}:{int(match[2])//20}' if match else p.stem
        records.append(dict(source=relative, image_sha256=sha(p), label_sha256=sha(label),
                            pixel_sha256=pixel_sha, dhash=dh, block=block, boxes=len(boxes)))
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        parents[find(j)] = find(i)
    blocks = {}
    near_pairs = 0
    for i, r in enumerate(records):
        if r['block'] in blocks:
            union(i, blocks[r['block']])
        blocks[r['block']] = i
        for j in range(i):
            if (r['dhash'] ^ records[j]['dhash']).bit_count() <= 4:
                union(i, j)
                near_pairs += 1
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    ordered = list(groups.values())
    random.Random(20261013).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    counts = Counter()
    targets = {'train': .70*len(records), 'val': .15*len(records), 'test': .15*len(records)}
    for group in ordered:
        split = max(targets, key=lambda s: targets[s]-counts[s])
        for i in group:
            records[i]['split'] = split
            records[i]['group'] = min(group)
        counts[split] += len(group)
    manifest = ROOT / 'split_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != records:
        raise ValueError('Prepared data differs from the frozen manifest')
    if verify_only:
        if not manifest.exists():
            raise ValueError('No frozen manifest to verify')
        print(f'Frozen manifest reproduced exactly: {len(records)} images, {dict(counts)}')
        return
    manifest.write_text(json.dumps(records, indent=2))
    for r in records:
        p = source / r['source']
        name = r['image_sha256'][:16]
        for kind, src, suffix in [('images', p, '.jpg'), ('labels', p.parent.parent/'labels'/(p.stem+'.txt'), '.txt')]:
            target = ROOT / 'data' / kind / r['split'] / (name + suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            if kind=='images':
                with Image.open(target) as image:
                    is_jpeg=image.format.lower() in {'jpg','jpeg'}
                if is_jpeg and target.read_bytes()[-2:]!=b'\xff\xd9':
                    # Same repair applied by Ultralytics 8.3.226 during the original run.
                    ImageOps.exif_transpose(Image.open(target)).save(target,'JPEG',subsampling=0,quality=100)
    repairs_file=ROOT/'image_repairs.json'
    if repairs_file.exists():
        for repair in json.loads(repairs_file.read_text()):
            p=ROOT/'data/images'/repair['split']/(repair['original'][:16]+'.jpg')
            if sha(p)!=repair['current']:
                raise ValueError('Repaired image differs from recorded training input')
    (ROOT/'data.yaml').write_text('path: ' + (ROOT/'data').as_posix() + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: drone\n')
    summary = dict(images=dict(counts), boxes={s:sum(r['boxes'] for r in records if r['split']==s) for s in targets},
                   groups=len(groups), largest_group=max(map(len, ordered)), exact_duplicates_removed=duplicates,
                   near_duplicate_pairs_grouped=near_pairs, seed=20261013,
                   grouping='same filename prefix and numeric ID // 20; transitive dHash Hamming distance <=4')
    (ROOT/'data_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args(); prepare(args.source,args.verify_only)
