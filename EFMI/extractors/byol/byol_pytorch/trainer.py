from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm
import torch.backends.cudnn as cudnn

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from byol_pytorch.byol_pytorch import BYOL

from beartype import beartype
from beartype.typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import os
# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# functions

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# class

class MockDataset(Dataset):
    def __init__(self, image_size, length):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(3, self.image_size, self.image_size)

# main trainer

class BYOLTrainer(Module):
    @beartype
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        hidden_layer: str,
        learning_rate: float,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int = 16,
        optimizer_klass = Adam, 
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        resume_from_checkpoint = False,
        byol_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        
    ):
        super().__init__()


        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.rank = 0
        self.world_size = 1 
        ngpus_per_node = torch.cuda.device_count()

        self.rank = self.rank * ngpus_per_node
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001', world_size=self.world_size, rank=self.rank)
        

       


        # Set the device for the current process
        self.device = torch.device(f'cuda:{self.accelerator.local_process_index}')
        torch.cuda.set_device(self.device)
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        self.net = net.to(self.device)

        self.byol = BYOL(net, image_size=image_size, hidden_layer=hidden_layer, **byol_kwargs)
        self.byol = self.byol.cuda()

        self.byol = DDP(self.byol, find_unused_parameters=True)

        self.optimizer = optimizer_klass(self.byol.parameters(), lr=learning_rate, **optimizer_kwargs)
        cudnn.benchmark = True
        # Use DistributedSampler
        self.sampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, sampler=self.sampler, num_workers=4, )

        self.num_train_steps = num_train_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # prepare with accelerate

        (
            self.byol,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.byol,
            self.optimizer,
            self.dataloader
        )

        self.register_buffer('step', torch.tensor(0))
        self.epoch = 0
        self.load_checkpoint()

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def load_checkpoint(self):
        checkpoints = list(self.checkpoint_folder.glob('checkpoint.*.pt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('.')[1]))
            self.print(f"Loading checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
        else:
            self.print("No checkpoint found, starting from scratch.")
    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoint_num = self.step.item() // self.checkpoint_every
            checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
            torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step': self.step,
                'epoch': self.epoch
            }, str(checkpoint_path))
    def forward(self):
        step = self.step.item()
        data_it = cycle(self.dataloader)

        for epoch in tqdm(range(self.epoch, self.num_train_steps), desc="Training Progress", leave=True):
            pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f'Epoch {epoch + 1}', leave=True)

            for i, images in enumerate(self.dataloader):

                for image in images:

                    with self.accelerator.autocast():
                          
                          loss = self.byol(image)
                          self.accelerator.backward(loss)

                if i % 30 == 0:
                    pbar.set_description(f'Epoch {epoch + 1} [Batch {i}/{len(self.dataloader)}] Loss: {loss.item():.3f}')
                    self.print(f'Epoch {epoch + 1} Batch {i}: Loss {loss.item():.3f}')
                

                self.optimizer.zero_grad()
                self.optimizer.step()

                self.wait()

                self.byol.module.update_moving_average()

                self.wait()

                if not (step % self.checkpoint_every) and self.accelerator.is_main_process:
                    self.save_checkpoint()

                step += 1
                self.step += 1
            self.epoch += 1

            self.print('training complete')
    def get_features(self):
        self.byol.module.eval()
        embeddings_with_labels = []

        with torch.no_grad():
            for data, labels in tqdm(self.dataloader):
                projection, embedding = self.byol.module(data, return_embedding = True)
                
                embeddings_with_labels.append((projection, labels))
        torch.save(embeddings_with_labels, './embeddings/eval_embeddings.pt')
        self.print("Embeddings saved")

