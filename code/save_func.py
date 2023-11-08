import numpy as np
import cv2
from registry import SAVEIMG_REGISTRY
import utils


@SAVEIMG_REGISTRY.register('sdr')
def saveimg_sdr(img, path, datainfo=None):
    # torchvision.utils.save_image is very slow
    # return torchvision.utils.save_image(img, path)
    img = img.cpu().detach().clamp_(0, 1).numpy().transpose((1, 2, 0)) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)

@SAVEIMG_REGISTRY.register('hlg')
def saveimg_hlg(img, path, datainfo=None):

    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    img = img / 1023.0
    img[img>1]=1
    img[img<0]=0
    img = utils.colortrans_709_2020(img, 2.4)
    img = (img * 65535.0).astype(np.uint16)
    img = img[:,:,::-1].copy()
    cv2.imwrite(str(path), img)


@SAVEIMG_REGISTRY.register('int16')
def saveimg_int16(img, path, datainfo=None):
    img = img.cpu().detach().clamp_(0,1).numpy().transpose((1, 2, 0))
    img = (img * 65535.0).astype(np.uint16)
    img = img[:,:,::-1].copy()
    cv2.imwrite(str(path), img)


@SAVEIMG_REGISTRY.register('colorenhance')
def saveimg_colorenhance(img, path, datainfo=None):
    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0))
        return img
    Y, UV = datainfo['Y'], datainfo['UV']
    predict = img
    predict = 0.7 * predict + 0.3 * UV
    predict = tensor_to_np(predict)
    Y = tensor_to_np(Y)
    img = cv2.merge([Y, predict])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    cv2.imwrite(str(path), img)
