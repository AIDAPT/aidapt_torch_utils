"""Contains the utils needed to save/load checkpoints for PyTorch models."""

from dataclasses import dataclass
from typing import Optional
import torch
import os

@dataclass
class CheckpointData:
    """Class which contains data from a loaded checkpoint.
    
    Attributes:
        arch (str): The name of the model or architecture
        model (torch.nn.Module): The model with the state_dict loaded from the checkpoint
        optimizer (Optional[torch.optim.Optimizer]): If present, the optimizer with the state_dict loaded from the checkpoint
        epoch (int): The epoch relative to the checkpoint
        
    """

    arch: str
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    epoch: int

class CheckpointHandler:
    """Class which handles checkpoints for a PyTorch model.
    
    This class contains the methods used to save a checkpoint, or load one. The checkpoints will contain
    the model and the optimizer with the new weights, which can be used to resume training.

    Attributes:
        device (torch.device): Used by the CheckpointHandler to load the checkpoint on the correct device
        checkpoint_dir (str): Path where the checkpoints will be saved or loaded
        interval (int): Epoch interval, which dictates how often the checkpoints may be saved
    """

    def __init__(self, device: torch.device, interval: int, checkpoint_dir: str = "") -> None:
        """Initialize an instance of the class CheckpointHandler.
        
        Attributes:
            device (torch.device): Used by the CheckpointHandler to load the checkpoint on the correct device
            checkpoint_dir (str): Path where the checkpoints will be saved or loaded
            interval (int): Epoch interval, which dictates how often the checkpoints may be saved
        """
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def load_checkpoint(
            self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, resume_path: str = "checkpoint.pth") -> CheckpointData:
        """
        Return the data from a checkpoint which path is specified by the user.

        Attributes:
            resume_path (str): Path where the checkpoint is stored
            model (torch.nn.Module): The PyTorch model where the checkpoint's state_dict will be loaded
            optimizer (Optional[torch.optim.Optimizer]): If present, the PyTorch optimizer where the checkpoint's state_dict will be loaded

        Returns:
            An instance of CheckpointData, which contains data from a loaded checkpoint
        """
        checkpoint = torch.load(resume_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return CheckpointData(checkpoint["arch"], model, optimizer, checkpoint["epoch"])

    def load_latest_checkpoint(
            self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> CheckpointData:
        """
        Return the data from the checkpoint relative to the latest epoch.

        Attributes:
            model (torch.nn.Module): The PyTorch model where the checkpoint's state_dict will be loaded
            optimizer (Optional[torch.optim.Optimizer]): If present, the PyTorch optimizer where the checkpoint's state_dict will be loaded

        Returns:
            An instance of CheckpointData, which contains data from a loaded checkpoint
        """
        chosen_checkpoint_index = sorted(map(lambda file_name: int(file_name.split(
            ".")[0].split("-")[1]), os.listdir(self.checkpoint_dir)), reverse=True)[0]
        return self.load_checkpoint(model, optimizer, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth'.format(chosen_checkpoint_index)))

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """Save a checkpoint.
        
        Save a checkpoint, given the current model, optimizer, and epoch, to the CheckpointHandler's dir.

        Attributes:
            model (torch.nn.Module): The PyTorch model which weigths will be saved into the checkpoint
            optimizer (torch.optim.Optimizer): The PyTorch optimizer which weigths will be saved into the checkpoint
            epoch (int): The epoch associated to the checkpoint to be saved
        """
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir,
                                'checkpoint-{}.pth'.format(epoch))
        torch.save(state, filename)

    def save_checkpoint_interval(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """Save a checkpoint every each interval.
        
        Save a checkpoint, given the current model, optimizer, and epoch, to the CheckpointHandler's dir. The checkpoint will be saved only if the current epoch matches the interval given to the CheckpointHandler.

        Attributes:
            model (torch.nn.Module): The PyTorch model which weigths will be saved into the checkpoint
            optimizer (torch.optim.Optimizer): The PyTorch optimizer which weigths will be saved into the checkpoint
            epoch (int): The epoch associated to the checkpoint to be saved
        """
        if epoch % self.interval == 0:
            self.save_checkpoint(model, optimizer, epoch)
