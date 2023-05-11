import numpy as np
import cv2
from registry import SAVEIMG_REGISTRY
import utils


@SAVEIMG_REGISTRY.register('sdr')
def saveimg_sdr(img, path):
    # torchvision.utils.save_image is very slow
    # return torchvision.utils.save_image(img, path)
    img = img.cpu().detach().clamp_(0, 1).numpy().transpose((1, 2, 0)) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)

@SAVEIMG_REGISTRY.register('hlg')
def saveimg_hlg(img, path):

    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    img = img / 1023.0
    img[img>1]=1
    img[img<0]=0
    img = utils.colortrans_709_2020(img, 2.4)
    img = (img * 65535.0).astype(np.uint16)
    img = img[:,:,::-1].copy()
    cv2.imwrite(str(path), img)
