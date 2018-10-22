from os.path import abspath
from ctypes import CDLL, RTLD_GLOBAL, POINTER, Structure
from ctypes import c_float, c_int, c_void_p, c_char_p, pointer

lib = CDLL(abspath('libdarknet.so'), RTLD_GLOBAL)


class BOX(Structure):
    _fields_ = (
        ("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)
    )


class DETECTION(Structure):
    _fields_ = (
        ("bbox", BOX), ("classes", c_int), ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)), ("objectness", c_float),
        ("sort_class", c_int)
    )


class IMAGE(Structure):
    _fields_ = (
        ("w", c_int), ("h", c_int), ("c", c_int), ("data", POINTER(c_float))
    )


class METADATA(Structure):
    _fields_ = (("classes", c_int), ("names", POINTER(c_char_p)))


predict = lib.network_predict_image
predict.argtypes = (c_void_p, IMAGE)

load_net = lib.load_network
load_net.restype = c_void_p

load_meta = lib.get_metadata
load_meta.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = (c_char_p, c_int, c_int)
load_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]

free_detections = lib.free_detections
free_detections.argtypes = (POINTER(DETECTION), c_int)

do_nms = lib.do_nms_obj
do_nms.argtypes = (POINTER(DETECTION), c_int, c_int, c_float)

get_boxes = lib.get_network_boxes
get_boxes.argtypes = (
    c_void_p, c_int, c_int, c_float, c_float,
    POINTER(c_int), c_int, POINTER(c_int)
)
get_boxes.restype = POINTER(DETECTION)


class Darknet:

    def __init__(self, meta, cfg, weights):
        self.net = load_net(cfg.encode(), weights.encode(), 0)
        self.meta = load_meta(meta.encode())

    def name(self, i):
        return self.meta.names[i].decode()

    def __call__(self, fn, thresh=.1, hier=.5, nms=.45):
        img = load_image(fn.encode(), 0, 0)
        w = img.w
        h = img.h
        n = pointer(c_int(0))

        predict(self.net, img)
        free_image(img)

        dets = get_boxes(self.net, w, h, thresh, hier, None, 0, n)
        num_dets = n[0]

        if nms:
            do_nms(dets, num_dets, self.meta.classes, nms)

        for j in range(num_dets):
            d = dets[j]
            for i in range(self.meta.classes):
                conf = d.prob[i]
                if not conf:
                    continue
                b = d.bbox
                yield conf, i, (b.x / w, b.y / h, b.w / w, b.h / h)

        free_detections(dets, num_dets)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: %s netname file')
        sys.exit(1)

    net = sys.argv[1]
    fn = sys.argv[2]

    dn = Darknet(*(a % net for a in ('%s.data', '%s.cfg', '%s.weights')))
    for c, i, b in sorted(dn(fn), reverse=True):
        print('\t%s (%.3f)' % (dn.name(i), c))
        print('%d %s' % (i, ' '.join('%.6f' % p for p in b)))

    print('\nDone')
