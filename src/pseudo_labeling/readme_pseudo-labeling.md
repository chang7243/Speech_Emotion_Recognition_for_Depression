

## Pseudo-Labeling
#### Overview
The main functionalities of this notebook include:

1. Generating audio pseudo-labels using Wav2Vec 2.0 and CLAP models.
2. Generating audio pseudo-labels using the Hubert model.
3. Generating text pseudo-labels.
4. Saving the final labels with confidence level (if two of the three labels agree).

#### Environment
- Kaggle Notebook
- To run the notebook, you need to install the following libraries:
 `pip install torch librosa numpy pandas scipy tqdm transformers msclap` (or you can just run the code block step by step)
 

### Required inputs
- preprocessed datasets DAIC_WOZ and MELD


#### Output
- The notebook will generate a CSV file containing the pseudo-labels for the DAIC-WOZ dataset.

#### Usage Notes

- Change the input/output paths;
- Prepare datasets;
- Remember to add your **huggingface token.**