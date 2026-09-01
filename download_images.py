"""Download university website images into a local folder for later face scanning."""
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from hashlib import sha1
import json, mimetypes

from server import crawl

ROOT = 'https://www.dusit.ac.th/home/'
OUT = Path(__file__).resolve().parent / 'downloaded_images'
META = OUT / 'metadata.json'

def main():
    OUT.mkdir(exist_ok=True)
    index_file=Path(__file__).resolve().parent / 'face_index.json'
    if index_file.exists():
        cached=json.loads(index_file.read_text(encoding='utf-8'))
        urls=[item['url'] for item in cached.get('entries',[])]
        pages=cached.get('pages',0)
        print(f'using cached face index: {len(urls)} image URLs')
    else:
        urls, pages = crawl(ROOT, full=True)
    rows=[]; downloaded=0
    for i, url in enumerate(urls, 1):
        try:
            req=Request(url, headers={'User-Agent':'ScanFace student project/1.0'})
            with urlopen(req, timeout=30) as r:
                data=r.read(8_000_000); content_type=r.headers.get('content-type','')
            if not content_type.startswith('image/'): continue
            ext=Path(urlparse(url).path).suffix.lower()
            if ext not in ('.jpg','.jpeg','.png','.webp','.gif','.avif'):
                ext=mimetypes.guess_extension(content_type.split(';')[0]) or '.img'
            name=f'{i:05d}_{sha1(url.encode()).hexdigest()[:10]}{ext}'
            (OUT/name).write_bytes(data)
            rows.append({'file':name,'url':url,'bytes':len(data)})
            downloaded+=1
        except Exception as e:
            rows.append({'url':url,'error':str(e)})
        if i % 50 == 0: print(f'processed {i}/{len(urls)} | downloaded {downloaded}')
    META.write_text(json.dumps({'source':ROOT,'pages':pages,'requested':len(urls),'downloaded':downloaded,'items':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'complete: {downloaded} images from {pages} pages')
    print(f'folder: {OUT}')
    print(f'metadata: {META}')

if __name__=='__main__': main()
