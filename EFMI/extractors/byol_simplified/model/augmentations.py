# 1. Data Augmentation
class BYOLTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    512, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                T.RandomRotation(degrees=(-180, 180)),
                # TODO: normalize with ds mean and std?
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return y1, y2
