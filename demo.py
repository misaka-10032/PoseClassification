#!/usr/bin/env python
import os
import cv2
import caffe
import numpy as np
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(CURRENT_DIR, 'haarcascade_frontalface_alt.xml')
NET_MODEL = os.path.join(CURRENT_DIR, 'xiyuan_quick_iter_30000.caffemodel')
NET_DEPLOY = os.path.join(CURRENT_DIR, 'xiyuan_quick_deploy.proto')
MEAN = os.path.join(CURRENT_DIR, 'mean_uni-30x30.npy')
#VIDEO = '-0321_2.mp4'
VIDEO = 0

W_WIN, H_WIN = 30, 30
RATIO = 0.6

def main():
    cascade = cv2.CascadeClassifier(CASCADE)
    net = caffe.Net(NET_DEPLOY, NET_MODEL, caffe.TEST)
    mean_img = np.load(MEAN)
    cap = cv2.VideoCapture(VIDEO)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            print 'frame is None!'
            continue
        h_frame, w_frame = frame.shape[:2]
        w_frame, h_frame = int(w_frame * RATIO), int(h_frame * RATIO)
        frame = cv2.resize(frame, (w_frame, h_frame))

        tic = time.time()
        rects = cascade.detectMultiScale(frame)
        toc = time.time()
        # print 'cascade spends %ds' % (toc - tic)

        for x, y, w, h in rects:
            dw = int(0.1 * w)
            dh = int(0.1 * h)
            if y - 3*dh >= 0 and\
                    y + dh < frame.shape[0] and\
                    x - 2*dw >= 0 and\
                    x + 2*dw < frame.shape[1]:
                x, y, w, h = x - 2*dw, y - 3*dh, w + 4*dw, h + 4*dh
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (W_WIN, H_WIN))

            tic = time.time()
            caffe_img = preprocess(face, mean_img)
            input = caffe_img.reshape((1, ) + caffe_img.shape)
            probs = net.forward_all(data=input)['prob'][0]
            toc = time.time()
            # print 'caffe spends %ds' % (toc - tic)

            up = probs[0] > 0.9
            down = probs[1] > 0.9
            if up:
                color = (255, 0, 0)
            elif down:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            #if probs[0, 0, 0] > probs[1, 0, 0]:
            #    color = (0, 255, 0)
            #else:
            #    color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess(cv2_img, mean_caffe_img):
        img = cv2_img.transpose(2, 0, 1)  # h, w, c -> c, h, w
        img = img.astype(np.float32, copy=False)
        img -= mean_caffe_img
        img = img[(2, 1, 0), :, :]
        return img

if __name__ == '__main__':
    main()
