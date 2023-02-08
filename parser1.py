import os
import numpy as np
from patchify import patchify
from PIL import Image
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def psr(root_directory='Semantic segmentation dataset', patch_size=256, scaler = MinMaxScaler()):
    image_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':  # Find all 'images'
            images = os.listdir(path)  # List of all image names in this subdirectory
            for i, image_name in enumerate(images):
                if image_name.endswith(".jpg"):

                    image = cv2.imread(path + "/" + image_name, 1)  # Read each image as BGR
                    SIZE_X = (image.shape[1] // patch_size) * patch_size  # Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))   # Crop from top left corner
                    image = np.array(image)

                    print("Now patchifying image:", path + "/" + image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3),
                                           step=patch_size)

                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i, j, :, :]

                            single_patch_img = scaler.fit_transform(
                                single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

                            single_patch_img = single_patch_img[0]
                            image_dataset.append(single_patch_img)

    # Now same for masks
    mask_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        # print(path)
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':  # Find all 'images'
            masks = os.listdir(path)
            for i, mask_name in enumerate(masks):
                if mask_name.endswith(".png"):

                    mask = cv2.imread(path + "/" + mask_name, 1)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1] // patch_size) * patch_size
                    SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                    mask = np.array(mask)

                    print("Now patchifying mask:", path + "/" + mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3),
                                            step=patch_size)

                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            single_patch_mask = patches_mask[i, j, :, :]
                            single_patch_mask = single_patch_mask[0]
                            mask_dataset.append(single_patch_mask)

    return image_dataset, mask_dataset, single_patch_mask