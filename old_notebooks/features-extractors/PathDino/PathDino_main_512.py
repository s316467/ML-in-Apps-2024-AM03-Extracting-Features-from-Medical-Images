import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchinfo import summary
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from PathDino import get_pathDino_model

from datafolders import CustomDatasetFolders

import utils

import PathDino as vits
from PathDino import DINOHead

output_col = ["input_size","output_size","num_params"]

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='pathdino', type=str,
        choices=['pathdino'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=3, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.0004, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    # parser.add_argument('--batch_size_per_gpu', default=64, type=int,
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=10, type=int, help="""Number of epochs   
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-5, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,  
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/histology/image/folder', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')  
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.') 
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

def plot_pca_variance(pca, output_dir):
    # Plot variance explained by each component
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.title('PCA - Variance Explained by Components')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'pca_variance.png'))
    plt.close()

def plot_tsne(features, labels, title, filename):
    # Ensure that features is a 2D array
    features = np.array(features)
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    # Check the number of samples and features
    n_samples, n_features = features.shape
    if n_samples < 2 or n_features < 2:
        print(f"Insufficient data for t-SNE: n_samples={n_samples}, n_features={n_features}")
        return

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.savefig(filename)
    plt.close()

def plot_loss(training_losses, output_dir):
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

def extract_embeddings(data_loader, model):
    model.eval()
    embeddings = []
    labels = []

    # Handle DDP/DataParallel models
    if hasattr(model, 'module'):
        model_to_use = model.module
    else:
        model_to_use = model

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            print(f"Processing batch {i + 1}/{len(data_loader)}")
            # Move each image tensor to the GPU
            images = [image.cuda(non_blocking=True) for image in images]
            # Forward pass
            output = model_to_use(images)
            print(f"Output shape: {output.shape}")  # Debug statement
            print(f"Target shape: {target.shape}")  # Debug statement
            
            # Check dimensions of output and target
            if isinstance(output, list):
                # Assuming the output is a list of embeddings from multiple crops
                output = torch.cat(output, dim=0)
            embeddings.append(output.cpu().numpy())

            # Repeat labels to match the number of embeddings
            repeated_labels = np.repeat(target.cpu().numpy(), output.shape[0] // target.shape[0])
            labels.append(repeated_labels)

            # Process in batches to avoid memory issues
            if len(embeddings) >= 10:  # Process every 10 batches
                yield np.vstack(embeddings), np.hstack(labels)
                embeddings, labels = [], []

    # Yield remaining data
    if embeddings:
        yield np.vstack(embeddings), np.hstack(labels)


def incremental_pca_svm_train(data_loader, model, n_components=128):
    pca = IncrementalPCA(n_components=n_components)
    svm = SVC(kernel='linear')
    scaler = StandardScaler()

    # First pass: Fit PCA
    for embeddings, labels in extract_embeddings(data_loader, model):
        pca.partial_fit(embeddings)
    
    # Second pass: Transform data and fit SVM
    for embeddings, labels in extract_embeddings(data_loader, model):
        reduced_embeddings = pca.transform(embeddings)
        reduced_embeddings = scaler.fit_transform(reduced_embeddings)
        svm.fit(reduced_embeddings, labels)

    return svm, pca, scaler


def incremental_pca_svm_evaluate(data_loader, model, svm, pca, scaler):
    all_embeddings = []
    all_labels = []
    
    for embeddings, labels in extract_embeddings(data_loader, model):
        reduced_embeddings = pca.transform(embeddings)
        reduced_embeddings = scaler.transform(reduced_embeddings)
        all_embeddings.append(reduced_embeddings)
        all_labels.append(labels)
        
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    
    predictions = svm.predict(all_embeddings)
    accuracy = accuracy_score(all_labels, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(all_labels, predictions))
    return all_labels, predictions

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for data in loader:
        data = data[0]  # data[0] because data is a tuple (images, labels)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = CustomDatasetFolders(args.data_path, transform=transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
        
    summary(student, input_size=(1, 3, 512, 512), depth=5)

    print("Student and Teacher are loaded...")

    # Load initial weights
    weights_path = os.path.join('inference', 'PathDino512.pth')
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        student.load_state_dict(state_dict, strict=False)
        teacher.load_state_dict(state_dict, strict=False)
        print(f"Loaded initial weights from {weights_path}")
    else:
        print(f"Warning: Initial weights file {weights_path} not found. Proceeding without loading weights.")
    
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    summary(student, input_size=(1, 3, 512, 512), depth=5)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    # Initialize list to store training losses
    training_losses = []

    
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # Append current epoch loss to the list
        training_losses.append(train_stats['loss'])

        # Plot the loss
        plot_loss(training_losses, args.output_dir)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Extract embeddings using the trained teacher model (should be better than student)
    svm_classifier, pca, scaler = incremental_pca_svm_train(data_loader, teacher)

    # Evaluate the classifier
    y_test, y_pred = incremental_pca_svm_evaluate(data_loader, teacher, svm_classifier, pca, scaler)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    plot_tsne(y_test, y_pred, title="t-SNE plot of SVM predictions", filename=os.path.join(args.output_dir, "tsne_plot.png"))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # print(type(images))
        images = [im.cuda(non_blocking=True) for im in images]
        # print(type(images))
        # print(len(images[:2]))
        # print(images[:2][0].shape)
        # print(images[:2][1].shape)
        # print(images[2].shape)
        # print(len(images))
        # print(type(images[0]))
        # print(images[0].shape)
        # print(images[-1].shape)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            # print(type(teacher_output))
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
class ExactRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.8688, 0.7047, 0.8022), (0.0803, 0.1345, 0.0987)),
        ])
        
        rotator = transforms.RandomRotation(degrees=(0,360))
        rotator_3digrees = ExactRotation([90, 180, 270, 360])
        
        
        cropper = transforms.RandomResizedCrop(600, scale=(0.4, 7.0), interpolation=Image.BICUBIC)
            
        #  transforms for the images with 1024x1024 dimension =========================================================
        # first global crop
        self.global_transfo1 = transforms.Compose([
            rotator,
            cropper,
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            rotator, 
            cropper,
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            rotator,
            cropper,
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        
        #  transforms for the images with 512x512 dimension =========================================================
        # first global crop
        self.global_transfo1_512 = transforms.Compose([
            rotator_3digrees,
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2_512 = transforms.Compose([
            rotator_3digrees,
            transforms.RandomResizedCrop(512, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_transfo_512 = transforms.Compose([
            rotator,
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        
    def __call__(self, image):
        crops = []
        
        if image.size[0] == 1024:
            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image))
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image))
            # print('==========================================================   1024')
        else:
            crops.append(self.global_transfo1_512(image))
            crops.append(self.global_transfo2_512(image))
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo_512(image))
            # print('==========================================================   512')
            
        # # Plotting the original and rotated image # to check the rotation -------------------------------------------------------
        # img_to_plot = []
        # for cr in crops:
        #     image_array = cr.numpy()
        #     image_array = np.transpose(image_array, (1, 2, 0))
        #     img_to_plot.append(image_array)
        # plt.figure(figsize=(15, 3))
        # plt.subplot(1, 11, 1)
        # plt.imshow(image)
        # plt.title('Original')
        # plt.axis('off')
        # plt.subplot(1, 11, 2)
        # plt.imshow(img_to_plot[0])
        # plt.title('global1')
        # plt.axis('off')
        # plt.subplot(1, 11, 3)
        # plt.imshow(img_to_plot[1])
        # plt.title('global2')
        # plt.axis('off')
        # plt.subplot(1, 11, 4)
        # plt.imshow(img_to_plot[2])
        # plt.title('local1')
        # plt.axis('off')
        # plt.subplot(1, 11, 5)
        # plt.imshow(img_to_plot[3])
        # plt.title('local2')
        # plt.axis('off')
        # plt.subplot(1, 11, 6)
        # plt.imshow(img_to_plot[4])
        # plt.title('local3')
        # plt.axis('off')
        # plt.subplot(1, 11, 7)
        # plt.imshow(img_to_plot[5])
        # plt.title('local4')
        # plt.axis('off')
        # plt.subplot(1, 11, 8)
        # plt.imshow(img_to_plot[6])
        # plt.title('local5')
        # plt.axis('off')
        # plt.subplot(1, 11, 9)
        # plt.imshow(img_to_plot[7])
        # plt.title('local6')
        # plt.axis('off')
        # plt.subplot(1, 11, 10)
        # plt.imshow(img_to_plot[8])
        # plt.title('local7')
        # plt.axis('off')
        # plt.subplot(1, 11, 11)
        # plt.imshow(img_to_plot[9])
        # plt.title('local8')
        # plt.axis('off')
        # plt.show()
        # plt.savefig(f'samples_sampled_for_512_another_1024.png', dpi=300, bbox_inches='tight')
        # plt.pause(2)
        # # # #to check the rotation -------------------------------------------------------
        
        return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute mean and std
    dataset = datasets.ImageFolder(args.data_path, transform=transforms.ToTensor())
    mean, std = compute_mean_std(dataset)
    print(f"Computed mean: {mean}, std: {std}")
    
    train_dino(args)
