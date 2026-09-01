from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import base64, json, os, shutil, tempfile
from hashlib import sha1
from PIL import Image
import io
import cv2, numpy as np
try:
    from insightface.app import FaceAnalysis
    INSIGHT_APP=FaceAnalysis(name='buffalo_l',providers=['CPUExecutionProvider']); INSIGHT_APP.prepare(ctx_id=0,det_size=(320,320)); BEST_AI=True
except Exception as e:
    print('InsightFace unavailable:',e); INSIGHT_APP=None; BEST_AI=False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
YUNET_FILE = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_FILE = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')

def ensure_model(path, url):
    if os.path.exists(path): return True
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with urlopen(Request(url, headers={'User-Agent':'ScanFace student project/1.0'}), timeout=60) as r:
            with open(path, 'wb') as f: f.write(r.read())
        return True
    except Exception as e:
        print('model download failed:', e)
        return False

REAL_AI = ensure_model(YUNET_FILE, YUNET_URL) and ensure_model(SFACE_FILE, SFACE_URL)
if REAL_AI:
    # OpenCV's Windows model loader needs an ASCII path; the workspace path contains Thai characters.
    RUNTIME_YUNET = os.path.join(tempfile.gettempdir(), 'scanf_yunet.onnx')
    RUNTIME_SFACE = os.path.join(tempfile.gettempdir(), 'scanf_sface.onnx')
    shutil.copyfile(YUNET_FILE, RUNTIME_YUNET); shutil.copyfile(SFACE_FILE, RUNTIME_SFACE)
    try:
        DETECTOR = cv2.FaceDetectorYN_create(RUNTIME_YUNET, '', (320, 320), 0.85, 0.3, 5000)
        RECOGNIZER = cv2.FaceRecognizerSF_create(RUNTIME_SFACE, '')
    except Exception as e:
        print('AI model initialization failed:', e); REAL_AI=False; DETECTOR=None; RECOGNIZER=None
else:
    DETECTOR=None; RECOGNIZER=None

ROOT = 'https://www.dusit.ac.th/home/2024/1338773.html'
INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arcface_index.json')
MAX_PAGES, MAX_IMAGES = 12, 120
FULL_PAGES, FULL_IMAGES = 300, 3000
MATCH_THRESHOLD = 0.50
MAX_MATCH_RESULTS = 30

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.urls=[]
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        for key in ('href','src','data-src','data-lazy-src','data-original','data-image'):
            if d.get(key): self.urls.append(d[key])
        if d.get('srcset'):
            self.urls.extend(x.strip().split(' ')[0] for x in d['srcset'].split(','))

def get(url):
    req=Request(url, headers={'User-Agent':'ScanFace student project/1.0'})
    with urlopen(req, timeout=12) as r: return r.read(2_000_000), r.headers.get('content-type','')

def crawl(start, full=False):
    host=urlparse(start).hostname or ''; page_limit=FULL_PAGES if full else MAX_PAGES; image_limit=FULL_IMAGES if full else MAX_IMAGES
    queue=[start, 'https://www.dusit.ac.th/home/', 'https://www.dusit.ac.th/home/personal']; pages=[]; images=[]; seen=set()
    def allowed(url):
        h=urlparse(url).hostname or ''
        return h==host or h.endswith('.dusit.ac.th')
    while queue and len(pages)<page_limit:
        url=queue.pop(0)
        if url in seen or not allowed(url): continue
        seen.add(url)
        try: raw,typ=get(url)
        except Exception: continue
        if 'html' not in typ and not url.endswith('/'): continue
        p=Links(); p.feed(raw.decode('utf-8','ignore')); pages.append(url)
        for u in p.urls:
            absolute=urljoin(url,u).split('#')[0]
            path=urlparse(absolute).path.lower()
            if (any(path.endswith(x) for x in ('.jpg','.jpeg','.png','.webp','.gif','.avif')) or 'image' in u.lower()) and len(images)<image_limit: images.append(absolute)
            elif allowed(absolute) and absolute not in seen and len(queue)<page_limit*3: queue.append(absolute)
    return list(dict.fromkeys(images))[:image_limit], len(pages)

def faces_from(data):
    arr=np.frombuffer(data,np.uint8); img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: return None, []
    original=img
    scale=1.0
    max_side=max(img.shape[:2])
    if max_side>1200:
        scale=1200.0/max_side
        img=cv2.resize(img,(max(1,int(img.shape[1]*scale)),max(1,int(img.shape[0]*scale))),interpolation=cv2.INTER_AREA)
    if BEST_AI:
        faces=[]
        for f in INSIGHT_APP.get(img):
            x,y,w,h=map(int,[f.bbox[0]/scale,f.bbox[1]/scale,(f.bbox[2]-f.bbox[0])/scale,(f.bbox[3]-f.bbox[1])/scale])
            faces.append({'x':x,'y':y,'w':w,'h':h,'embedding':f.embedding.reshape(-1).astype(float).tolist()})
        return original, faces
    if not REAL_AI:
        return img, []
    h,w=img.shape[:2]; DETECTOR.setInputSize((w,h)); _, found=DETECTOR.detect(img)
    faces=[]
    for face in (found if found is not None else []):
        x,y,fw,fh=map(int,face[:4]); faces.append({'x':x,'y':y,'w':fw,'h':fh,'raw':face})
    return original, faces

def feature(img, face):
    if 'embedding' in face: return np.array(face['embedding'],dtype=np.float32).reshape(1,-1)
    crop=RECOGNIZER.alignCrop(img, face['raw']); return RECOGNIZER.feature(crop)

def make_index(source=None):
    # Prefer the already-downloaded, face-filtered local collection. This makes
    # scanning deterministic and avoids waiting for the university site again.
    local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_images_faces')
    metadata_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_images', 'metadata.json')
    if os.path.isdir(local_dir) and not source:
        url_by_file={}
        try:
            meta=json.load(open(metadata_file,encoding='utf-8'))
            url_by_file={str(x.get('file')):x.get('url') for x in meta.get('items',[]) if x.get('file')}
        except Exception: pass
        entries=[]; seen_files=set(); known_names=set(); known_urls=set()
        try:
            old_index=json.load(open(INDEX_FILE,encoding='utf-8'))
            entries=list(old_index.get('entries',[]))
            for old_item in entries:
                old_url=str(old_item.get('url',''))
                known_urls.add(old_url)
                if old_url.startswith('/local-images/'): known_names.add(os.path.basename(old_url))
                elif old_url.startswith('local://'): known_names.add(os.path.basename(old_url[8:]))
        except Exception: pass
        # downloaded_images is the raw archive; clean_images.py has already
        # filtered it into local_dir, so index only the face-positive set.
        scan_dirs=[local_dir]
        for scan_dir in scan_dirs:
          if not os.path.isdir(scan_dir): continue
          for name in sorted(os.listdir(scan_dir)):
            path=os.path.join(scan_dir,name)
            if not os.path.isfile(path) or name=='metadata.json' or not name.lower().endswith(('.jpg','.jpeg','.png','.webp','.gif','.avif')): continue
            if name in known_names or (url_by_file.get(name) and url_by_file.get(name) in known_urls): continue
            try:
                with open(path,'rb') as f: raw=f.read()
                digest=sha1(raw).hexdigest()
                if digest in seen_files: continue
                seen_files.add(digest)
                img,faces=faces_from(raw); saved=[]
                for face in faces:
                    if BEST_AI or REAL_AI:
                        fvec=feature(img,face); face.pop('raw',None); face['embedding']=fvec.reshape(-1).astype(float).tolist(); saved.append(face)
                if saved:
                    entries.append({'url':url_by_file.get(name, '/local-images/'+name), 'faces':saved})
            except Exception as e: print('local index skip',name,e)
        payload={'version':2,'source':'downloaded_images_faces','pages':0,'scanned':len(entries),'entries':entries}
        with open(INDEX_FILE,'w',encoding='utf-8') as f: json.dump(payload,f)
        return payload
    # A user-supplied page is an incremental source: scan its nearby pages only.
    # The local collection remains the full baseline and is not removed.
    urls,pages=crawl(source or 'https://www.dusit.ac.th/home/', full=not bool(source)); entries=[]
    if source and pages == 0:
        # Keep the app usable when Windows/network policy blocks the external site.
        # Rebuild from both local folders instead of returning a misleading success.
        return make_index(None)
    download_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_images')
    face_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_images_faces')
    os.makedirs(download_dir,exist_ok=True); os.makedirs(face_dir,exist_ok=True)
    new_meta=[]
    for u in urls:
        try:
            raw,typ=get(u)
            if not typ.startswith('image/'): continue
            img,faces=faces_from(raw); saved=[]
            for face in faces:
                if BEST_AI or REAL_AI:
                    f=feature(img,face); face.pop('raw',None); face['embedding']=f.reshape(-1).astype(float).tolist(); saved.append(face)
            if saved:
                ext=os.path.splitext(urlparse(u).path)[1].lower()
                if ext not in ('.jpg','.jpeg','.png','.webp','.gif','.avif'): ext='.jpg'
                name=sha1(u.encode()).hexdigest()[:16]+ext
                with open(os.path.join(download_dir,name),'wb') as f: f.write(raw)
                with open(os.path.join(face_dir,name),'wb') as f: f.write(raw)
                new_meta.append({'file':name,'url':u,'bytes':len(raw)})
                entries.append({'url':u,'faces':saved})
        except Exception: pass
    if source and os.path.exists(INDEX_FILE):
        try:
            old=json.load(open(INDEX_FILE,encoding='utf-8'))
            known={x.get('url') for x in old.get('entries',[])}
            entries=old.get('entries',[])+[x for x in entries if x.get('url') not in known]
            pages += old.get('pages',0)
        except Exception: pass
    payload={'version':2,'pages':pages,'scanned':len(entries),'entries':entries}
    with open(INDEX_FILE,'w',encoding='utf-8') as f: json.dump(payload,f)
    if new_meta:
        meta_path=os.path.join(download_dir,'metadata.json'); old_meta={}
        try: old_meta=json.load(open(meta_path,encoding='utf-8'))
        except Exception: pass
        old_items=old_meta.get('items',[]) if isinstance(old_meta,dict) else []
        known={x.get('url') for x in old_items}
        old_items += [x for x in new_meta if x.get('url') not in known]
        with open(meta_path,'w',encoding='utf-8') as f: json.dump({'source':source or ROOT,'pages':pages,'requested':len(old_items),'downloaded':len(old_items),'items':old_items},f,ensure_ascii=False,indent=2)
    return payload

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            try:
                data=json.load(open(INDEX_FILE,encoding='utf-8')) if os.path.exists(INDEX_FILE) else {}
                self.json_response({'ok':True,'pages':data.get('pages',0),'scanned':len(data.get('entries',[])),'faces':sum(len(x.get('faces',[])) for x in data.get('entries',[]))})
            except Exception as e: self.json_response({'ok':False,'error':str(e)},500)
            return
        if self.path.startswith('/local-images/'):
            name=os.path.basename(self.path.split('?',1)[0])
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_images_faces', name)
            if os.path.isfile(path):
                data=open(path,'rb').read(); self.send_response(200); self.send_header('Content-Type','image/*'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404); return
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/index':
            try:
                length=int(self.headers.get('content-length','0'))
                body=json.loads(self.rfile.read(length)) if length else {}
                source=(body.get('sourceUrl') or '').strip() or None
                payload=make_index(source); self.json_response({'ok':True,'pages':payload['pages'],'scanned':payload['scanned'],'indexed':len(payload['entries'])})
            except Exception as e: self.json_response({'ok':False,'error':str(e)},500)
            return
        if self.path != '/api/scan': self.send_error(404); return
        try:
            length=int(self.headers.get('content-length','0')); body=json.loads(self.rfile.read(length))
            ref=base64.b64decode(body['reference'].split(',',1)[-1]); source=body.get('sourceUrl') or ROOT
            if source.rstrip('/') == 'https://www.dusit.ac.th/home': source=ROOT
            ref_img, ref_faces=faces_from(ref)
            ref_features=[feature(ref_img,f) for f in ref_faces] if ref_faces and (BEST_AI or REAL_AI) else []
            ref_feature=ref_features[0] if ref_features else None
            if (BEST_AI or REAL_AI) and ref_feature is None:
                out={'ok':False,'pages':0,'scanned':0,'matches':[],'faces':0,'referenceFaces':0,'ai':True,'error':'ไม่พบใบหน้าในรูปต้นแบบ กรุณาใช้รูปที่เห็นใบหน้าชัดเจน'}
                data=json.dumps(out,ensure_ascii=False).encode(); self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data); return
            results=[]; total=0
            index=None
            if os.path.exists(INDEX_FILE):
                try: index=json.load(open(INDEX_FILE,encoding='utf-8'))
                except Exception: index=None
            if index and ref_features:
                for item in index.get('entries',[]):
                    matches=[]
                    for face in item.get('faces',[]):
                        target=np.array(face['embedding'],dtype=np.float32).reshape(1,-1)
                        score=max(float(np.dot(rf.reshape(-1),target.reshape(-1))/(np.linalg.norm(rf)*np.linalg.norm(target))) for rf in ref_features)
                        if score>=MATCH_THRESHOLD:
                            matches.append({k:v for k,v in face.items() if k!='embedding'}|{'score':round(max(0,min(100,score*100)),1)})
                    if matches:
                        matches.sort(key=lambda x:x['score'], reverse=True)
                        display_url=item['url']
                        if display_url.startswith('local://'): display_url='/local-images/'+display_url[8:]
                        results.append({'url':display_url,'faces':matches}); total+=len(matches)
                pages=index.get('pages',0); image_urls=[x['url'] for x in index.get('entries',[])]
            else:
                image_urls,pages=crawl(source)
            for u in image_urls:
                try:
                    raw,typ=get(u)
                    if not typ.startswith('image/'): continue
                    img, faces=faces_from(raw); matches=[]
                    for face in faces:
                        target=feature(img,face); score=max((float(np.dot(rf.reshape(-1),target.reshape(-1))/(np.linalg.norm(rf)*np.linalg.norm(target))) for rf in ref_features),default=0)
                        if ref_feature is None or score >= MATCH_THRESHOLD:
                            face['score']=round(max(0,min(100,score*100)),1); face.pop('raw',None); matches.append(face)
                    if matches:
                        matches.sort(key=lambda x:x['score'], reverse=True); results.append({'url':u,'faces':matches}); total += len(matches)
                except Exception: pass
            results.sort(key=lambda x:max(f['score'] for f in x['faces']), reverse=True); results=results[:MAX_MATCH_RESULTS]
            out={'ok':True,'pages':pages,'scanned':len(image_urls),'matches':results,'faces':total,'referenceFaces':len(ref_faces),'ai':REAL_AI,'threshold':MATCH_THRESHOLD*100,'note':'คะแนนเป็นเปอร์เซ็นต์ความคล้ายคลึงของใบหน้า ไม่ใช่ความน่าจะเป็น 100%'}
            self.json_response(out)
        except Exception as e: self.send_error(500,str(e))

    def json_response(self, obj, status=200):
        data=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)

if __name__=='__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print('ScanFace running at http://localhost:8005')
    ThreadingHTTPServer(('127.0.0.1',8005),Handler).serve_forever()
