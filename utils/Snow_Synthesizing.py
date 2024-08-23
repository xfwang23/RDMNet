import os
import random
import numpy as np
import cv2
from tqdm import tqdm


def read_img(path):
    img = cv2.imread(path)  # [:, :, ::-1]
    return img


def add_snow(clean_path, snow_mask_path, atmosphere_light=1):
    clean = read_img(clean_path)  # .astype(np.float32)
    snow_mask = read_img(snow_mask_path)  # .astype(np.float32)
    snow_mask = cv2.resize(snow_mask, (clean.shape[1], clean.shape[0]))
    cv2.imwrite('clean.png', clean)
    # snow_mask = np.expand_dims(snow_mask, axis=2)
    # corrupt = clean * (1 - snow_mask) + atmosphere_light * snow_mask

    assert snow_mask.shape == clean.shape
    beta = random.randint(5, 10) * 0.1
    corrupt = cv2.addWeighted(clean, 1.0, snow_mask, beta, 0)

    corrupt = np.uint8(corrupt.clip(0., 255.))
    return corrupt


if __name__ == '__main__':
    clean_dir = r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter4\Datasets\VOC_Snow\train\CleanImages"
    snow_mask_dir = r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter4\Datasets\CSD\Train\Mask"
    save_dir = r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter4\Datasets\VOC_Snow\train\SnowyImages"
    cleans = os.listdir(clean_dir)
    snow_masks = os.listdir(snow_mask_dir)
    for clean_name in tqdm(cleans):
        clean_path = os.path.join(clean_dir, clean_name)
        snow_mask_path = os.path.join(snow_mask_dir, random.sample(snow_masks, 1)[0])
        snow_image = add_snow(clean_path, snow_mask_path)
        cv2.imwrite(save_dir + '\\' + clean_name, snow_image)
    # clean_path = r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter4\Datasets\VOC_Snow\test\CleanImages\2007_002262.jpg"
    # snow_mask_path = r"D:\WorkofMaster\GraduationThesis\Experiments\Chapter4\Datasets\CSD\Train\Mask\4.tif"
    # snow_image = add_snow(clean_path, snow_mask_path)
    # cv2.imwrite('snow_image.png', snow_image)
