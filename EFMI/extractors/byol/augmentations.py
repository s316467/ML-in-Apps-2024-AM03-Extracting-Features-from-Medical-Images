import torch
import torchvision.transforms as transforms
import kornia.augmentation as K
import kornia
import config

IMAGE_SIZE = config.IMAGE_SIZE

def get_train_augmentations(image_size=IMAGE_SIZE, ):
    s = 1
    color_jitter = K.ColorJitter(
        brightness=0.8 * s, 
        contrast=0.8 * s, 
        saturation=0.8 * s, 
        hue=0.2 * s, 
        p=0.2
    )
    
    transform1 = torch.nn.Sequential(
        K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
        K.RandomHorizontalFlip(),
        color_jitter,
        # K.RandomGaussianBlur((IMAGE_SIZE // 20 * 2 + 1, IMAGE_SIZE // 20 * 2 + 1), (0.1, 2.0), p=0.1),
        K.RandomGrayscale(p=0.2),
        kornia.augmentation.Normalize(mean=torch.tensor([-0.3797, -0.0606, -0.1343]), std=torch.tensor([1.1741, 1.0244, 1.0651]))
    )

    # Kornia transformations equivalent to train_transform2
    transform2 = torch.nn.Sequential(
        K.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
        K.RandomHorizontalFlip(),
        color_jitter,
        # K.RandomGaussianBlur((IMAGE_SIZE // 20 * 2 + 1, IMAGE_SIZE // 20 * 2 + 1), (0.1, 2.0), p=0.1),
        K.RandomGrayscale(p=0.2),
        kornia.augmentation.Normalize(mean=torch.tensor([-0.3797, -0.0606, -0.1343]), std=torch.tensor([1.1741, 1.0244, 1.0651]))
    )
    return transform1, transform2

def get_eval_augmentations(image_size=IMAGE_SIZE):
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    return eval_transform



