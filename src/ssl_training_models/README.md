
## SSL Training Models

### Input

Select the DAIC-WOZ dataset without labels to train with Wav2Vec2 in order to enable the models to extract and learn meaningful features from the data.

### Output

The Wav2Vec2 models will unfreeze the last 4 layers for training. This process will update the new indicators and parameters in Wav2Vec2.

## How to Train?

1. Upload all files from the `ssl_training_models` directory and the dataset to Kaggle.
2. Run the `wav2s.ipynb` notebook in Kaggle.
