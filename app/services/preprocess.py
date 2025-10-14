from PIL import Image
import numpy as np
import cv2
from typing import Tuple

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # PIL -> RGB array
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# базовая очистка
def clean_image(image: Image.Image, blur_ksize: Tuple[int, int]=(3,3)) -> Image.Image:
    img_cv = pil_to_cv(image)
    img_cv = cv2.GaussianBlur(img_cv, blur_ksize, 0) # шумоподавление
    img_cv = cv2.convertScaleAbs(img_cv, alpha=1.05, beta=3) # небольшая коррекция контраста/яркости: alpha, beta
    return cv_to_pil(img_cv)
