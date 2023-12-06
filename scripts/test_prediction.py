import os

ROOT_DIR = os.getcwd()

from src.helper.data import load_event
from src.training.model import UTimeModel

from src.submission.postprocessing import (
    add_predictions,
    labelize_event,
    select_best_night_candidates,
)

from src.helper.plot import plot

ckpt_path = "data/model.ckpt"

model = UTimeModel.load_from_checkpoint(ckpt_path)
model.freeze()
model.to("cpu")

event_data = load_event(0)

# Add model raw predictions
event_data = add_predictions(event_data, model)

# Gather prediction into event labels
event_data = labelize_event(event_data)

# Filter events
event_data = select_best_night_candidates(event_data)

plot(event_data)

print("OK")
