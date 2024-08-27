import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from byol_pytorch.byol_pytorch import BYOL
from beartype import beartype
from typing import Optional
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import config



# Constants
DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)

# Utility functions
def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# Main Trainer Class
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
        # dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001', world_size=self.world_size, rank=self.rank)

        # Set the device for the current process
        self.device = torch.device(f'cuda:{self.accelerator.local_process_index}')
        torch.cuda.set_device(self.device)
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            net = SyncBatchNorm.convert_sync_batchnorm(net)

        self.net = net.to(self.device)
        self.image_size = image_size    

        self.byol = BYOL(net, image_size=image_size, hidden_layer=hidden_layer, **byol_kwargs)
        self.byol = self.byol.cuda()

        #self.byol = DDP(self.byol, find_unused_parameters=True)

        self.optimizer = optimizer_klass(self.byol.parameters(), lr=learning_rate, **optimizer_kwargs)
        cudnn.benchmark = True

        # Use DistributedSampler
        #self.sampler = DistributedSampler(dataset)
        #self.dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, sampler=self.sampler, num_workers=4)
        self.dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

        self.num_train_steps = num_train_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        self.losses = []  # Lista per salvare le perdite
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

    def load_checkpoint(self):
    # Cerca i checkpoint che contengono il transform_descriptor
        checkpoints = list(self.checkpoint_folder.glob(f'checkpoint.*.pt'))
        
        if checkpoints:
            # Trova il checkpoint più recente
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('.')[1]))
            self.print(f"Loading checkpoint from {latest_checkpoint}")
            
            # Carica solo i pesi della rete dal checkpoint
            self.net.load_state_dict(torch.load(latest_checkpoint))
            
            # Deduce l'epoca dal nome del file
            checkpoint_num = int(latest_checkpoint.stem.split('.')[1])
            self.epoch = checkpoint_num
            self.print(f"Resuming from epoch {self.epoch}, step {self.step}")
        else:
            self.print("No checkpoint found for the given transform descriptor, starting from scratch.")
            self.epoch = 0


    def save_losses(self):
        losses_path = self.checkpoint_folder / f'losses/losses.pt'
        os.makedirs(losses_path.parent, exist_ok=True)
        torch.save(self.losses, losses_path)
    def plot_losses(self):
        plt.figure()
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Time")
        plt.legend()
        plt.savefig(self.checkpoint_folder / "loss_plot.png")
        plt.close()
    
    def print_training_info(self):
        self.print("=== Training Setup ===")
        self.print(f"Model: {self.net.__class__.__name__}")
        self.print(f"Image Size: {self.image_size}")
        self.print(f"Hidden Layer: {self.byol.hidden_layer}")
        self.print(f"Learning Rate: {config.LR}")
        self.print(f"Batch Size: {config.BATCH_SIZE}")
        self.print(f"Number of Training Steps: {self.num_train_steps}")

        self.print("=======================")

    def forward(self):
        step = self.step.item()
        data_it = cycle(self.dataloader)
        start_epoch = self.epoch
        self.print_training_info()

        for epoch in range(start_epoch, self.num_train_steps):
            self.print(f'Starting epoch {epoch + 1}/{self.num_train_steps}')

            # Crea la barra di progresso per l'epoca corrente
            progress_bar = tqdm(range(len(self.dataloader)), desc=f"Epoch {epoch + 1}/{self.num_train_steps}", unit="batch")

            for _ in progress_bar:
                images = next(data_it)

                with self.accelerator.autocast():
                    loss = self.byol(images)
                    self.accelerator.backward(loss)

                self.losses.append(loss.item())
                progress_bar.set_postfix(loss=loss.item())  # Aggiorna la barra con la perdita corrente

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.wait()

                self.byol.update_moving_average()

                self.wait()

                self.step += 1

            # Salva i pesi del modello alla fine di ogni epoca
            if self.accelerator.is_main_process:
                checkpoint_num = epoch + 1
                checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
                torch.save(self.net.state_dict(), str(checkpoint_path))
                self.print(f'Checkpoint saved at {checkpoint_path}')

            self.print(f'Epoch {epoch + 1} complete')
        
        self.save_losses()
        self.plot_losses()

        self.print('Training complete')
