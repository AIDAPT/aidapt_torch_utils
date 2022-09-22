"""Contains the utils needed to record new data for consultation in TensorBoard."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TBItemData:
    """
    Class which describes an item which can be recorded inside of tensorboard.

    Attributes:
        type (str): Denotes the type of the item. It can be one of the following: scalar, scalars, image, figure, audio, video, text, histogram, graph
        data (Any): Field which contains the data to be recorded. The type depends on the value supplied to the type field
        step (Optional[int]): An optional field which indicates the step at which the data is recorded. If not present, an internal incremental value is used instead
    """

    type: str
    data: Any
    step: Optional[int] = None

class TensorboardWriter:
    """Class which can be used to record data for TensorBoard.

    Attributes:
        writer (SummaryWriter): Object which can be used to explicitly record data in TensorBoard
        tensorboard_writer_fns (List[str]): List of SummaryWriter functions supported by the class
        types (List[str]): List of supported types
        tag_counter (Dict[str, int]): Keeps track of the number of times a tag has been recorded
        tag_steps (Dict[str, Any]): Keeps track of the internal step value for each tag. 
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        """Initialize an instance of the class TensorboardWriter.

        Attributes:
            log_dir (Optional[str]): if specified, it denotes the directory where the TensorBoard logs are stored
        """
        self.writer = SummaryWriter(log_dir)
        self.tensorboard_writer_fns = [
            'add_scalar', 'add_scalars', 'add_image', 'add_figure', 'add_audio',
            'add_video', 'add_text', 'add_histogram', 'add_graph'
        ]
        self.types = [
            'scalar', 'scalars', 'image', 'figure', 'audio', 'video', 'text', 'histogram', 'graph' 
        ]
        self.tag_counter = {}
        self.tag_steps = {}

    def get_writer(self) -> SummaryWriter:
        """Get the writer used by the class instance to record values in TensorBoard.

        Returns:
            The SummaryWriter object associated with the class instance
        """
        return self.writer

    def record_data_from_dict(self, data_dict: Dict[str, TBItemData]) -> None:
        """Given a dictionary containing the items to be recorded, records the item in TensorBoard.

        The data_dict which is passed to this function has this structure:
        {
            "<tag_1>": TBItemData("<type>", <data>[, <step>]),
            "<tag_2>": ...,
            ...
        }
        where the keys refer to the tag name associated to the item, whereas the value is an instance of the
        TBItemData class, containing the information about the item which has to be recorded.

        Attributes:
            data_dict (Dict[str, TBItemData]): Dictionary with the items which will be recorded
        """
        for tag, item_data in data_dict.items():
            writer_fn_name = self.tensorboard_writer_fns[self.types.index(item_data.type)]
            add_data = getattr(self.writer, writer_fn_name, None)

            if item_data.type != "scalars":
                if item_data.step is not None:
                    step = item_data.step
                    self.tag_steps.update({ tag: step + 1 })
                else:
                    if tag not in self.tag_steps:
                        self.tag_steps[tag] = 0
                    step = self.tag_steps[tag]
                    self.tag_steps[tag] += 1
            else:
                if tag not in self.tag_steps:
                    self.tag_steps[tag] = {}

                if item_data.step is not None:
                    step = item_data.step
                    for scalar_tag in item_data.data.keys():
                        self.tag_steps[tag].update({ scalar_tag: step + 1 })
                else:
                    related_step_keys = list(filter(lambda scalar_tag: scalar_tag in self.tag_steps[tag], item_data.data.keys()))
                    if len(related_step_keys) == len(item_data.data.keys()):
                        related_step_values = list(map(lambda scalar_tag: self.tag_steps[tag][scalar_tag], related_step_keys))
                        if related_step_values.count(related_step_values[0]) == len(related_step_values):
                            step = related_step_values[0]
                        else:
                            raise Exception("Step mismatch for the scalars!")
                    elif len(related_step_keys) == 0:
                        step = 0
                    else:
                        raise Exception("Step mismatch for the scalars!")

                    for scalar_tag in item_data.data.keys():
                        self.tag_steps[tag].update({ scalar_tag: step + 1 })
            

            add_data(tag, item_data.data, step)

            counter = self.tag_counter.get(tag, 0)
            self.tag_counter.update({ tag: counter + 1 })

    def __getattr__(self, name):
        """Depending on the function name, return a function of the SummaryWriter which can be invoked to record the data."""
        if name in self.tensorboard_writer_fns:
            add_data = getattr(self.writer, name, None)

            if add_data is not None:
                # add_data(tag, data, global_step, *args, **kwargs)
                return add_data
        
        raise AttributeError("Class TensorboardWriter has no attribute '{}'".format(name))