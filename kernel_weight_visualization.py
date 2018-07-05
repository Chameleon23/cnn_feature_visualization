from matplotlib import pyplot as plt
import torchvision.transforms as transforms


def save_kernels_images(layer, num_rows: int=8, num_cols: int=8, image_name: str="test.png"):

    picture_number = num_cols * num_rows
    tensor = layer.weight.data.clone().cpu()

    # Normalise
    max_element = tensor.max()
    min_element = abs(tensor.min())
    max_value = max(max_element, min_element)
    tensor = tensor / max_value
    tensor = tensor / 2
    tensor = tensor + 0.5

    fig = plt.figure(figsize=(num_cols, num_rows))

    for i, each in enumerate(tensor):

        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        transformation = transforms.ToPILImage()
        image = transformation(each)
        ax1.imshow(image, interpolation='none')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        if i == picture_number:
            break
    plt.savefig(image_name)

