import numpy as np
import cv2
import threading

def colortrans_709_2020(img, gamma):
    img = img**gamma
    h,w,c = img.shape
    m1 = np.array([[0.6274, 0.3293, 0.0433],
       [0.0691, 0.9195, 0.0114],
       [0.0164, 0.0880, 0.8956]])
    imgnew = img.transpose((2, 0, 1))
    imgnew = imgnew.reshape((c,-1))
    imgnew = np.dot(m1,imgnew)
    imgnew = np.clip(imgnew, 0, 1)
    imgnew = imgnew.reshape((c,h,w))
    img = imgnew.transpose((1, 2, 0))
    img = img**(1/gamma)
    return img

def save_image(imgtensor, filename):
    _tensor = imgtensor.float().detach().cpu().clamp_(0, 1) * 65535
    img_np = _tensor.numpy().astype('uint16')
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    cv2.imwrite(str(filename), img_np)


class ImgSaver(threading.Thread):

    def __init__(self, save_func, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.save_func = save_func

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break
            self.save_func(msg['img'], msg['path'])
        # print(f'{self.qid} quit.')

