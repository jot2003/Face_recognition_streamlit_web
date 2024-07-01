import numpy as np
import cv2
import insightface
import onnxruntime
from insightface.app import FaceAnalysis
app_l = FaceAnalysis(name='buffalo_l',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
app_l.prepare(ctx_id=0, det_size=(640,640))
app_sc = FaceAnalysis(name='buffalo_sc',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
app_sc.prepare(ctx_id=0, det_size=(640,640))
img =cv2.imread('test-image-2.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindow()