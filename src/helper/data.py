import os
import joblib
from tqdm import tqdm

from src.config import FT_COLUMNS, LABELS_COLUMN, EVENTS_FOLDER


def get_event_path(event_id):
    return EVENTS_FOLDER / f"event_{event_id}.pkl"


def load_event(event_id):
    event_path = get_event_path(event_id)
    return joblib.load(event_path)


def load_full_dataset(data_dir: str = EVENTS_FOLDER):
    event_id_list = [entry.name[6:-4] for entry in os.scandir(data_dir)]

    print("Loading events...")
    loaded_events = {}

    for event_id in tqdm(event_id_list):
        df = load_event(event_id)
        df = df[FT_COLUMNS + [LABELS_COLUMN]].copy()

        loaded_events[event_id] = df

    return loaded_events
