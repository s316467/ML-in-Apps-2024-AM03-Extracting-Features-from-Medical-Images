import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_baseline  # networks with zero padding
import models as models_partial  # partial conv based padding
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_generated_images(images, epoch, batch_idx, output_dir='generated_images'):
    """
    Save a batch of generated images to disk.

    Args:
        images (torch.Tensor): Batch of images to save.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        output_dir (str): Directory where images will be saved.
    """
    if not os.path.exists(output_dir):
        print("no dir found")
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f'epoch_{epoch}_batch_{batch_idx}.png')
    vutils.save_image(images, file_path, normalize=True)
    print(f'Saved generated images to {file_path}')

class RandomHole:
    def __init__(self, hole_size_range=(20, 50), num_holes=1):
        self.hole_size_range = hole_size_range
        self.num_holes = num_holes

    def __call__(self, img):
        img = img.copy()
        w, h = img.size
        pixels = img.load()

        for _ in range(self.num_holes):
            hole_w = random.randint(self.hole_size_range[0], self.hole_size_range[1])
            hole_h = random.randint(self.hole_size_range[0], self.hole_size_range[1])
            x = random.randint(0, w - hole_w)
            y = random.randint(0, h - hole_h)

            for i in range(x, x + hole_w):
                for j in range(y, y + hole_h):
                    pixels[i, j] = (0, 0, 0)

        return img

def generate_tsne(embeddings, labels, output_file='tsne_plot.png'):
    # Reduce dimensions with PCA if embeddings have more than 50 features
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE of PCA-reduced features')
    plt.savefig(output_file)
    plt.show()


model_baseline_names = sorted(name for name in models_baseline.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(models_baseline.__dict__[name]))

model_partial_names = sorted(name for name in models_partial.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models_partial.__dict__[name]))

model_names = model_baseline_names + model_partial_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_train', metavar='DIRTRAIN',
                    help='path to training dataset')
parser.add_argument('--data_val', metavar='DIRVAL',
                    help='path to validation dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=192, type=int,
                    metavar='N', help='mini-batch size (default: 192)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--prefix', default='', type=str)
parser.add_argument('--ckptdirprefix', default='', type=str)
parser.add_argument('--extract-embeddings', dest='extract_embeddings', action='store_true',
                    help='extract embeddings for training an SVM')
parser.add_argument('--save-model', default='', type=str, metavar='PATH',
                    help='path to save the trained model in .pth format (default: none)')
parser.add_argument('--generate-tsne', dest='generate_tsne', action='store_true',
                    help='generate t-SNE plot of PCA-reduced features')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    checkpoint_dir = args.ckptdirprefix + 'checkpoint_' + args.arch + '_' + args.prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.logger_fname = os.path.join(checkpoint_dir, 'loss.txt')

    with open(args.logger_fname, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
        log_file.write('world size: %d\n' % args.world_size)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch](pretrained=True)
        else:
            model = models_partial.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch in models_baseline.__dict__:
            model = models_baseline.__dict__[args.arch]()
        else:
            model = models_partial.__dict__[args.arch]()

    with open(args.logger_fname, "a") as log_file:
        log_file.write('model created\n')

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir = args.data_train
    valdir = args.data_val
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomHole(hole_size_range=(20, 50), num_holes=3),  # Add the RandomHole transform
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    with open(args.logger_fname, "a") as log_file:
        log_file.write('training/val dataset created\n')

    with open(args.logger_fname, "a") as log_file:
        log_file.write('started training\n')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, foldername=checkpoint_dir, filename='checkpoint.pth.tar')

        if epoch >= 94:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, False, foldername=checkpoint_dir, filename='epoch_' + str(epoch) + '_checkpoint.pth.tar')
    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    if args.extract_embeddings:
        extract_and_save_embeddings(train_loader, val_loader, model, args)
        return


def extract_and_save_embeddings(train_loader, val_loader, model, args):
    model.eval()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    def process_loader(loader, embeddings, labels, loader_name=""):
        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # Forward pass to get features
                if isinstance(model, torch.nn.DataParallel):
                    _, features = model(input)
                else:
                    _, features = model.module(input)
                
                features = features.view(features.size(0), -1)
                embeddings.append(features.cpu().numpy())
                labels.append(target.cpu().numpy())
                torch.cuda.empty_cache()  # Clear cache to free memory
                if i % 10 == 0:
                    print(f"Processed {i} batches for {loader_name}")

    process_loader(train_loader, train_embeddings, train_labels, "train")
    process_loader(val_loader, val_embeddings, val_labels, "validation")

    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_embeddings = np.concatenate(val_embeddings, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    
    print("Train embeddings shape:", train_embeddings.shape)
    print("Train labels shape:", train_labels.shape)
    print("Val embeddings shape:", val_embeddings.shape)
    print("Val labels shape:", val_labels.shape)

    # Handle NaN values
    train_embeddings = imputer.fit_transform(train_embeddings)
    val_embeddings = imputer.transform(val_embeddings)
    print("NaN values handled.")
    
    """
    # Apply PCA to reduce the number of embeddings to 128
    pca = PCA(n_components=128)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    val_embeddings_pca = pca.transform(val_embeddings)
    print("Train embeddings PCA shape:", train_embeddings_pca.shape)
    print("Val embeddings PCA shape:", val_embeddings_pca.shape)
    np.save('train_embeddings_pca.npy', train_embeddings_pca)
    np.save('train_labels.npy', train_labels)
    np.save('val_embeddings_pca.npy', val_embeddings_pca)
    np.save('val_labels.npy', val_labels)
    """
    
    # Incremental learning for large datasets
    clf = SGDClassifier(loss='hinge')
    for i in range(0, len(train_embeddings), args.batch_size):
        end = i + args.batch_size if i + args.batch_size < len(train_embeddings) else len(train_embeddings)
        clf.partial_fit(train_embeddings[i:end], train_labels[i:end], classes=np.unique(train_labels))
    
    # Save the classifier
    joblib.dump(clf, 'svm_classifier.pkl')

    # Evaluate on validation set
    val_pred = clf.predict(val_embeddings)
    accuracy = accuracy_score(val_labels, val_pred)
    print("Validation Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(val_labels, val_pred))

    if args.generate_tsne:
        generate_tsne(train_embeddings, train_labels)




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)  # Ensure this output contains the images you want to save

        if isinstance(output, tuple):
            output = output[0]  # Unpack the first element if the model returns a tuple

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

            with open(args.logger_fname, "a") as log_file:
                log_file.write('Epoch: [{0}][{1}/{2}]\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            # Save generated images for current batch
            # save_generated_images(input, epoch, i)  # Save the input images for reference
            # save_generated_images(output, epoch, i)  # Save the output images

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)  # Ensure this output contains the images you want to save

            if isinstance(output, tuple):
                output = output[0]  # Unpack the first element if the model returns a tuple

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

                with open(args.logger_fname, "a") as log_file:
                    log_file.write('Test: [{0}/{1}]\t'
                                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        with open(args.logger_fname, "a") as final_log_file:
            final_log_file.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                                 .format(top1=top1, top5=top5))
        
        # Save generated images at the end of validation
        # save_generated_images(input, 'val', 0)  # Save the input images for reference
        # save_generated_images(output, 'val', 0)  # Save the output images

    return top1.avg


    return top1.avg

def save_checkpoint(state, is_best, foldername='experiment_1', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(foldername, filename))
    if is_best:
        shutil.copyfile(os.path.join(foldername, filename), os.path.join(foldername, 'model_best.pth.tar'))
    if args.save_model:
        model_path = os.path.join(foldername, args.save_model)
        torch.save(state['state_dict'], model_path)
        print(f"Model saved to {model_path}")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
