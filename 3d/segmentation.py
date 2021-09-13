from torchvision import models
from torchvision import transforms as T
import PIL
import numpy as np
import torch
import cv2
from realsense import RealSense

segmentation = models.segmentation.fcn_resnet101(pretrained=True).eval().to("cuda:0")
cam = RealSense()


class Segmentation:
    def __init__(self):
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boar", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                        "tv/monitor"]

    def normalize(self, x):
        trf = T.Compose([T.Resize(256), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        x = PIL.Image.fromarray(x)
        return trf(x).unsqueeze(0).to("cuda:0")


    def postprocessing(self, y):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        category_map = torch.argmax(y.squeeze(), dim=0).detach().cpu().numpy()

        r = np.zeros_like(category_map).astype(np.uint8)
        g = np.zeros_like(category_map).astype(np.uint8)
        b = np.zeros_like(category_map).astype(np.uint8)

        for l in range(len(label_colors)):
            idx = category_map == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        segmented_map = np.stack([r, g, b], axis=2)
        segmented_map = np.array(T.Resize(480, interpolation=PIL.Image.NEAREST)(T.ToPILImage()(segmented_map)))

        category_map = np.array(T.Resize((480, 640), interpolation=PIL.Image.NEAREST)(
            T.ToPILImage()(category_map.astype(np.float32)))).astype(int)

        return segmented_map, category_map

    def segment(self, color):
        img = self.normalize(color)
        y = segmentation(img)["out"]
        seg, cat = self.postprocessing(y)
        return seg, cat


if __name__ == "__main__":
    seg = Segmentation()
    while True:
        c, d = cam.read()
        s, _ = seg.segment(c)
        cv2.imshow("window", s)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()
