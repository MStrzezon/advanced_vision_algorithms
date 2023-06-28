import os
import numpy as np
import cv2

TAG_FLOAT = 202021.25

def readflo(file):
    """ Read a velocity file from disk """
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    f.close()
    return flow


def vis_flow(flow):
    hsv = np.ones((flow[..., 0].shape[0], flow[..., 0].shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:, :, 0] = (ang * 90 / np.pi)
    hsv[:, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[:, :, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imshow('rgb', rgb)
    cv2.waitKey(0)

flow = readflo('spynet/out_spynet.flo')
vis_flow(flow)
flow = readflo('liteflownet/out_liteflownet.flo')
vis_flow(flow)
cv2.destroyAllWindows()




