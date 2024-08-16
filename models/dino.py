import cv2
import numpy as np
import torch
import torchvision.transforms as T

from tools.utils import get_affine_transform


class Dino:
    def __init__(self, model_ver):
        self.dino = torch.hub.load('facebookresearch/dinov2', model_ver)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def inference(self, img, bbox):
        h, w = img.shape[:2]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

        s = 1
        r = 0
        c = np.array([cx, cy])
        origin_size = min(h, w)
        trans = get_affine_transform(c, s, r, origin_size, [224, 224])
        img = cv2.warpAffine(
            img,
            trans,
            (224, 224),
            flags=cv2.INTER_LINEAR)
        img_crop = img.copy()

        img = img[:, :, ::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]

        img = self.transform(img)
        img = img.unsqueeze(0)

        out = self.dino(img, is_training=True)['x_norm_patchtokens']
        out = out.squeeze(0).detach().cpu().numpy().reshape(h//14, w//14, -1)
        return img_crop, out
