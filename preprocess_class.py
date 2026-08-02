import numpy as np
import cv2
from scipy import ndimage
import timm
from timm.data import resolve_model_data_config

"""
Following contrast types available
- No contrast (none) (default)
- Histogram EQ (he)
- CLAHE (clahe)
"""

# Python retardation made me do this 
class TimmNormalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, image):
        x = image.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.astype(np.float32)


def get_preprocessing_fn_timm(model_name):
    model = timm.create_model(model_name, pretrained=False)
    data_cfg = resolve_model_data_config(model)
    mean = np.array(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
    std = np.array(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)

    return TimmNormalize(mean=mean, std=std)


class PreProObj:
    def __init__(self, args):
        self.args = args
        
    def setup_stack(self, stack):
        self.prepro_stack = stack
        model_name = getattr(self.args, "model_name", None)
        if not model_name:
            model_name = getattr(self.args, "encoder_name", "resnet18")
        self.prepro_stack.append(get_preprocessing_fn_timm(model_name))
        
    def process(self, image):
        for fn in self.prepro_stack:
            image = fn(image)
        return image.astype(np.float32)
    
    def apply_contrast(self, image):
        match self.args.contrast:
            case 'he':
                return self.apply_contrast_he(image)
            case 'clahe':
                return self.apply_contrast_clahe(image)
            case 'none':
                return self.identity(image)

    # ---------THESE FUNCTIONS EXPECT RGB-----------
    # returns histogram equalized image as numpy array in RGB colorspace
    def apply_contrast_he(self, image):
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])
        
        equ = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
        
        return equ

    # returns clahe as np arr, RGB colorspace
    # Expects a clip limit and a 
    def apply_contrast_clahe(self, image, tile_size = (8, 8)): # RIGHT NOW THESE ARGS ARE UNUSED, maybe add params later    
        clip_limit = self.args.clahe_clip
        
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
        
        lum = ycrcb_image[:, :, 0]
        processed_lum = np.clip(clahe.apply(lum), 0, 255).astype(np.uint8)
        ycrcb_image[:, :, 0] = processed_lum    
        
        clahe_img = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
        return clahe_img
    
    def identity(self, image):
        return image
    
class PostProObj:
    def __init__(self, args):
        self.args = args
        
    def setup_stack(self, stack):
        self.post_pro_stack = stack
        
    def process(self, image):
        for fn in self.post_pro_stack:
            image = fn(image)
        return image
    
    # Requires binary image with floats like 1.0 or 0.0
    def fill_holes(self, image):
        image = ndimage.binary_fill_holes(image)
        return image

"""
img = Image.open('filt_res_data/raw/Alcelaphini raw/images/Extant/Alcelaphini/Alcelaphus/buselaphus/LM1/DSCN2871.JPG').convert('RGB')
x = np.array(img, dtype = np.uint8)
x = z_normalize(apply_contrast_clahe(x))

res = Image.fromarray(x)
res.save("updateznormetest.jpg")
"""