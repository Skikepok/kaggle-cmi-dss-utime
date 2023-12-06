import numpy as np
import pandas as pd


class RandomCrop1D:
    def __init__(self, size):
        self.size = size

    def __call__(self, x: pd.DataFrame, return_index=True):
        """
        Args:
            x (pd.DataFrame): Dataframe of data.
        Returns:
            pd.DataFrame: Randomly cropped along the rows axis.
        """

        length, _ = x.shape

        if length < self.size:
            return (x, 0) if return_index else x

        # Generate a random starting index for cropping
        start_idx = np.random.randint(0, length - self.size + 1, size=1)[0]

        # Crop the time series
        cropped_x = x.iloc[start_idx : start_idx + self.size, :]

        return (cropped_x, start_idx) if return_index else cropped_x
