[MICCAI2022 DRAC AI Challenge task3] This is repository to share training and inference codes of Team KT_Bio_Health

## **Directory Structure**

``` bash
# code directory contains executable files for preprocessing and training.
code
├── util
│   ├── custom_augmentation.py
│   ├── make_k_fold_dataset.py
│   ├── upper_sampling_aug_random.py 
├── control_data_coloring.py
├── control_model.py 
├── tools.py
├── tools_criterion.py
├── tools_scheduler.py
├── train.py
├── submission.py
├── find_segmentation_file.py
└── voting_with_segmentation_new_rule.py

db

#db directory contains image files, csv about training data and submission.    
db
├── input
│   ├── train_images
│   ├── valid_images
├── submission_result.csv
└── train.csv
```

## Training & Submission

``` bash

# Pre-processing
python code/utils/make_k_fold_dataset.py   # 5-Fold
python code/upper_sampling_aug_random.py   # upsampling


# Training & Post-processing
python train.py # model=BeitLarge512, criterion=CELoss, optimizer=AdamW, scheduler=StepLR
python find_segmentation_file.py
python voting_with_segmentation_new_rule.py

```

if you want to change model, check the 'control_model.py' file

