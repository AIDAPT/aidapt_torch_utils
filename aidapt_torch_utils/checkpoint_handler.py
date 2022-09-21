import torch
import os

class CheckpointHandler:
    def __init__(self, device: torch.device, checkpoint_epoch_interval: int, checkpoint_dir: str = "") -> None:
        self.interval = checkpoint_epoch_interval
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def load_checkpoint(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, resume_path: str | None = None):

        checkpoint = torch.load(resume_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return {
            "arch": checkpoint["arch"],
            "model": model,
            "optimizer": optimizer,
            "epoch": checkpoint["epoch"]
        }
    
    def load_latest_checkpoint(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
        chosen_checkpoint_index = sorted(map(lambda file_name: int(file_name.split(".")[0].split("-")[1]), os.listdir(self.checkpoint_dir)), reverse=True)[0]
        return self.load_checkpoint(model, optimizer, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth'.format(chosen_checkpoint_index)))

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth'.format(epoch))
        torch.save(state, filename)
    
    def save_checkpoint_interval(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        if epoch % self.interval == 0:
            self.save_checkpoint(model, optimizer, epoch)