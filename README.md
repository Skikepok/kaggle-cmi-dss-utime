# Introduction
This project is the result of my Pytorch Lightning implementation of the U-Time architechture for Kaggle Competiton [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states).

This competition was my first kaggle participation and I joined it only 20 days before the end of it. I therefore did not aimed for the leaderboard, but rather for having fun implementing a U-Time model for the first time and as a personnal challenge.

# The project stack

This project uses [Pytorch lightning](https://lightning.ai/docs/pytorch/stable/) for the model definition, [Optuna](https://optuna.readthedocs.io/en/stable/index.html) for the model fine tuning and [Neptune](https://neptune.ai/) for the fine tuning management.

# Results

In order to visualize the model results, you can run the following command

```bash
make test
```

<figure>
    <figcaption>Here you can see an example of the model segmentation on the raw event data.</figcaption>
    <img src="/assets/results_demo.png" alt="Results demo">
</figure>

A .onnx version of the model is also provided. You can use [Netron](https://netron.app/) to visualize it.

# Demo training

In order to test the model training, you can run the following command

```bash
make fine_tune
```

Note that this version only uses 6 events for training and therefore will not give any decent results.

<figure>
    <figcaption>Here you can see an example of the model training results in neptune dashboard</figcaption>
    <img src="/assets/neptune_1.png" alt="Neptune training example">
</figure>

<figure>
    <figcaption>Here you can see a debug plot of the validation event at the beginning of the training</figcaption>
    <img src="/assets/neptune_2.png" alt="Neptune training example">
</figure>

<figure>
    <figcaption>Here you can see a debug plot of the validation event at the end of the training</figcaption>
    <img src="/assets/neptune_3.png" alt="Neptune training example">
</figure>

