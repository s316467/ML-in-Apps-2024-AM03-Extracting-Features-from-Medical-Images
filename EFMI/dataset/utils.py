from torch.utils.data import random_split


def get_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, b, _, _ in dataloader:  # Assuming (images, labels) are returned
        batch_samples = images.size(0)  # batch size (number of images in the batch)
        images = images.view(
            batch_samples, images.size(1), -1
        )  # reshape to (batch, channels, pixels)
        mean += images.mean(2).sum(0)  # sum over pixels and batches
        std += images.std(2).sum(0)  # sum over pixels and batches
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std


def train_test_split(full_dataset, train_ratio):
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset
