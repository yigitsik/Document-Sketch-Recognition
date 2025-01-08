import cv2
from PIL import Image
import numpy as np
import timeit

ui = GET_YOUR_MODEL()

vid = cv2.VideoCapture(1)

while True:
    ret, frame = vid.read()
    

    if cv2.waitKey(113) == ord('q'):
        start = timeit.default_timer()
        frame = ui.scan_image(frame)
        frame = Image.fromarray(frame)
        img, texts, boxes,labels,scores = ui.predict_single(frame, ocr=True)
        img = np.array(img)
        stop = timeit.default_timer()
        cv2.imshow("frame", img)
        print('Time: ', stop - start) 
        print(texts) 
        while not cv2.waitKey(113) == ord('q'):
            pass
    else:
        cv2.imshow("frame", frame)





