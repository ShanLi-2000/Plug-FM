"""Download the official Centro file or the 17-file teacher pretraining subset."""
import argparse
import hashlib
import json
from pathlib import Path


DATA = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pretraining', action='store_true', help='Download the larger 17-file subset instead of Centro.')
    args = p.parse_args()
    from huggingface_hub import hf_hub_download

    name = 'pretraining_subset.json' if args.pretraining else 'scene.json'
    spec = json.loads((DATA / name).read_text(encoding='utf-8'))
    for item in spec['files'] if args.pretraining else [spec]:
        local = Path(hf_hub_download(repo_id=spec['dataset_repo'], repo_type='dataset',
                                     filename=item['path'], revision=spec.get('revision'), local_dir=DATA))
        if digest(local) != item['sha256']:
            raise ValueError(f'Checksum mismatch for {local}. See data/README.md.')
        print(f'Verified {local}')


if __name__ == '__main__':
    main()
