# quick check: show bounding box and crop
from ml_module.detector import Facedetector  # (or however your detector returns faces)
import cv2
img = cv2.imread("images/test3_w.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = raw_detect(img_rgb)   # must return list of dicts with 'box'
print("faces:", faces)
# draw box and save
x,y,w,h = faces[0]['box']
cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imwrite("debug_bbox.jpg", img)
cv2.imwrite("debug_crop.jpg", img[y:y+h, x:x+w])
