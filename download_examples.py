"""Download optional author exemplars. Images stay outside version control."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
import argparse
import hashlib
import io
import json
from PIL import Image

BASE = 'https://people.eecs.berkeley.edu/~efros/research/NPS/images/'


def download_examples(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ['D1', 'bread']:
        target = directory / f'{name}.gif'
        url = BASE + target.name
        if target.exists():
            payload = target.read_bytes()
        else:
            request = Request(url, headers={'User-Agent': 'ce7491-texture-synthesis-codebase'})
            with urlopen(request, timeout=30) as response:
                payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        target.write_bytes(payload)
        records.append({'file': target.name, 'url': url,
                        'sha256': hashlib.sha256(payload).hexdigest()})
        print(f'Available: {target}')
    (directory / 'sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    try:
        download_examples(args.out)
    except (URLError, TimeoutError) as error:
        parser.exit(1, f'Download failed: {error}\nPlace D1.gif and bread.gif in {args.out} and rerun, '
                    'or use python run_experiments.py --synthetic-only.\n')
