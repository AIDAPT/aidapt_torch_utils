from torch.utils.tensorboard import SummaryWriter
import datetime

class TensorboardWriter:
    def __init__(self, log_dir: str | None = None) -> None:
        self.writer = SummaryWriter(log_dir)
        self.tensorboard_writer_fns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_histogram', 'add_embedding'
        }

    def get_writer(self) -> SummaryWriter:
        return self.writer

    def record_epoch_loss(
        self, epoch: int, train_loss: float | None = None, validation_loss: float | None = None, test_loss: float | None = None):
        if test_loss is not None:
            self.writer.add_scalar("Loss/test", test_loss, epoch)
        
        if validation_loss is not None:
            self.writer.add_scalar("Loss/validation", validation_loss, epoch)
        
        if train_loss is not None:
            self.writer.add_scalar("Loss/train", train_loss, epoch)

    def record_epoch_accuracy(
        self, epoch: int, train_accuracy: float | None = None, validation_accuracy: float | None = None, test_accuracy: float | None = None):
        if test_accuracy is not None:
            self.writer.add_scalar("Accuracy/test", test_accuracy, epoch)
        
        if validation_accuracy is not None:
            self.writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)
        
        if train_accuracy is not None:
            self.writer.add_scalar("Accuracy/train", train_accuracy, epoch)

    # TODO: create method which maps generic dict to writer calls

    def __getattr__(self, name):
        if name in self.tensorboard_writer_fns:
            add_data = getattr(self.writer, name, None)

            if add_data is not None:
                # add_data(tag, data, global_step, *args, **kwargs)
                return add_data
        
        raise AttributeError("Class TensorboardWriter has no attribute '{}'".format(name))