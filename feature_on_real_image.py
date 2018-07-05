import cv2
import numpy as np
from typing import Any
from torchvision import transforms
from PIL import Image
from fastai.core import create_variable


def get_loader(image_size: int) -> Any:
    loader = transforms.Compose([transforms.Resize((image_size, image_size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    return loader


def image_loader(image_name: str, image_size: int):
    """load image, returns cuda tensor"""
    loader = get_loader(image_size)
    image = Image.open(image_name)
    image = loader(image).float()
    image = create_variable(image, True, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def save_features(model, layer, image_path: str, image_size: int=224, saved_folder: str='./'):

    def get_data(m, i, o):
        img = o.data.squeeze(0)
        reshaped_image = img.transpose(2, 0)
        reshaped_image = reshaped_image.transpose(0, 1)
        reshaped_image = reshaped_image.numpy()

        for i in range(reshaped_image.shape[-1]):

            one_img = reshaped_image[:, :, i]
            delta = abs(255 / one_img.max())
            one_img *= delta
            one_img[one_img < 0] = 0

            zoom_kernel = cv2.resize(one_img, (image_size, image_size))

            final_image = cv2.addWeighted(original_image, 0.1, zoom_kernel, 0.9, 0)

            cv2.imwrite(f"{saved_folder}/{i}.jpg", final_image)

    image = image_loader(image_path, image_size)
    original_image = cv2.resize(cv2.imread(image_path, 0), (224, 224))
    original_image = np.array(original_image, np.float32)

    hook = layer.register_forward_hook(get_data)
    model(image).detach().numpy()
    hook.remove()
