"""Keep only images with detected faces and remove visual duplicates."""
from pathlib import Path
import json, shutil, tempfile, os
import cv2, numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'downloaded_images'
DEST=ROOT/'downloaded_images_faces'
MODEL=ROOT/'models'/'face_detection_yunet_2023mar.onnx'

def dhash(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); small=cv2.resize(gray,(9,8)); return bytes((small[:,1:]>small[:,:-1]).astype(np.uint8).flatten())

def main():
    DEST.mkdir(exist_ok=True)
    runtime=os.path.join(tempfile.gettempdir(),'scanf_clean_yunet.onnx'); shutil.copyfile(MODEL,runtime)
    detector=cv2.FaceDetectorYN_create(runtime,'',(320,320),0.85,0.3,5000)
    metadata=[]; hashes=set(); kept=0; skipped=0; duplicate=0
    for path in sorted(SRC.iterdir()):
        if not path.is_file() or path.name=='metadata.json': continue
        img=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if img is None: skipped+=1; continue
        h,w=img.shape[:2]; detector.setInputSize((w,h)); _,faces=detector.detect(img)
        if faces is None or len(faces)==0: skipped+=1; continue
        key=dhash(img)
        if key in hashes: duplicate+=1; continue
        hashes.add(key); out=DEST/path.name; shutil.copy2(path,out)
        metadata.append({'file':path.name,'faces':len(faces),'source_file':str(path)}); kept+=1
    (DEST/'metadata.json').write_text(json.dumps({'kept':kept,'skipped_no_face':skipped,'duplicates_removed':duplicate,'items':metadata},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'kept={kept}, no_face={skipped}, duplicates_removed={duplicate}')
    print(f'folder={DEST}')

if __name__=='__main__': main()
