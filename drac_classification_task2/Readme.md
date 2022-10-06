[MICCAI2022 DRAC AI Challenge task2] This is repository to share training and inference codes of Team KT_Bio_Health

# Directory Structure

```bash
#code directory contains executable files for preprocessing and training.
code
├── control_data_enhance.py
├── control_model.py 
├── tools.py
├── train_1st.py
├── train_2nd.py
├── submission_1st.py
├── submission_2nd.py
└── submission_ensemble.py


input

#input directory contains image files, csv about training data and submission.    
input
├── train_images_pseudo.csv
├── KT_bio_health.csv
└── test_8016.csv

```

# Training & Submission

```bash

# Training & Post-processing
python train_1st.py # model=BeitLarge512, criterion=CELoss, optimizer=AdamW, scheduler=StepLR
python train_2nd.py # model=NFNet_f6, criterion=CELoss, optimizer=AdamW, scheduler=StepLR
python submission_1st.py
python submission_2nd.py
python submission_ensemble.py
```

if you want to change model, check the 'control_model.py' file
