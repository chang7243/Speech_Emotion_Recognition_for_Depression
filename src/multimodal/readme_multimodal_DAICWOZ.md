
## Finetune multimodal model on DAIZ-WOZ
### Input
The input of the multimodal model are the data of DAIC-WOZ, the pseduo label of DAIC-WOZ, and the multimodal model pretained on MELD(the *.tar.gz model file).
### Output
The output of the multimodal model is the model that finetune on DAIZ-WOZ(the *.zip file)
### Point
- load all the input you need to the working file
- fun `match_label` to match the pseduo label
- class `DAIC_WOZ_Modal_Dataset` to get the tensor of your training and testing data
- class `MultimodalClassifier` to get the multimodal model
- `model.load_state_dict(torch.load(model_path))`to load the pretained weight on MELD.