import cv2
import numpy as np
import os


imgs = os.listdir('data')
total = len(imgs)
cnt = 0
for i in imgs:
    img_path = os.path.join('data', i)
    output_path = os.path.join('transposed', i)
    img = cv2.imread(img_path)
    img = np.transpose(img, (1,0,2))
    img = np.flip(img, 1)
    cv2.imwrite(output_path, img)
    cnt += 1
    print('{}/{}'.format(cnt, total))
