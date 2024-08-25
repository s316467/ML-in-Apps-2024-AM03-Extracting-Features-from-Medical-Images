import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import numpy as np
from PatchedDataset import PatchedDataset
from torch.utils.data import DataLoader

def analyze_dataset(dataset, dataloader):
    # Get total number of samples
    total_samples = len(dataset)
    print(f"Total number of samples: {total_samples}")

    # Count samples for each label
    label_counts = Counter(dataset.labels)
    print("Number of samples for each label:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    # Print an example from both classes
    class_examples = {0: None, 1: None}
    for image, label, patient_id, coordinates in dataloader:
        for i in range(len(label)):
            if class_examples[label[i].item()] is None:
                class_examples[label[i].item()] = image[i]
            if all(example is not None for example in class_examples.values()):
                break
        if all(example is not None for example in class_examples.values()):
            break

    # Plot examples
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (label, img) in enumerate(class_examples.items()):
        img = img.numpy().transpose((1, 2, 0))  # Change from CxHxW to HxWxC
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        axes[i].imshow(img)
        axes[i].set_title(f"Class {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Print some additional statistics
    print("\nAdditional Statistics:")
    print(f"Number of unique patients: {len(set(dataset.patients))}")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"Label type: {type(dataset[0][1])}")


def main(args):
  dataset = PatchedDataset(root_dir=args.root_dir, num_images=24)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
  analyze_dataset(dataset, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches features with pre-trained nets"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to directory containing the patches folders",
    )
    args = parser.parse_args()
    main(args)