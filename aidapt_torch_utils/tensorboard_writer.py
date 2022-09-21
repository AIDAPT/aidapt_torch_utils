from typing import Any, Dict, TypedDict
from torch.utils.tensorboard import SummaryWriter

TBItemDataDict = TypedDict("TBDataDict", {"type": str, "data": Any, "step": int})

class TensorboardWriter:
    def __init__(self, log_dir: str | None = None) -> None:
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
        return self.writer

    ''' dict structure:
        { "<tag>": { "type": "<type>", "data": <data> (, "step": <step> ) }, ... }
    '''

    def record_data_from_dict(self, data_dict: Dict[str, TBItemDataDict]):
        for tag, item_data in data_dict.items():
            writer_fn_name = self.tensorboard_writer_fns[self.types.index(item_data['type'])]
            add_data = getattr(self.writer, writer_fn_name, None)

            if item_data["type"] != "scalars":
                if "step" in item_data:
                    step = item_data["step"]
                    self.tag_steps.update({ tag: step + 1 })
                else:
                    if tag not in self.tag_steps:
                        self.tag_steps[tag] = 0
                    step = self.tag_steps[tag]
                    self.tag_steps[tag] += 1
            else:
                if tag not in self.tag_steps:
                    self.tag_steps[tag] = {}

                if "step" in item_data:
                    step = item_data["step"]
                    for scalar_tag in item_data['data'].keys():
                        self.tag_steps[tag].update({ scalar_tag: step + 1 })
                else:
                    related_step_keys = list(filter(lambda scalar_tag: scalar_tag in self.tag_steps[tag], item_data['data'].keys()))
                    if len(related_step_keys) == len(item_data['data'].keys()):
                        related_step_values = list(map(lambda scalar_tag: self.tag_steps[tag][scalar_tag], related_step_keys))
                        if related_step_values.count(related_step_values[0]) == len(related_step_values):
                            step = related_step_values[0]
                        else:
                            raise Exception("Step mismatch for the scalars!")
                    elif len(related_step_keys) == 0:
                        step = 0
                    else:
                        raise Exception("Step mismatch for the scalars!")

                    for scalar_tag in item_data['data'].keys():
                        self.tag_steps[tag].update({ scalar_tag: step + 1 })
            

            add_data(tag, item_data['data'], step)

            counter = self.tag_counter.get(tag, 0)
            self.tag_counter.update({ tag: counter + 1 })

    def __getattr__(self, name):
        if name in self.tensorboard_writer_fns:
            add_data = getattr(self.writer, name, None)

            if add_data is not None:
                # add_data(tag, data, global_step, *args, **kwargs)
                return add_data
        
        raise AttributeError("Class TensorboardWriter has no attribute '{}'".format(name))