import os
import numpy as np
import pandas as pd

RAW_FOLDER = r"C:\AI\DRAC\task3\post_processing"
SEG_RESULT_1_CSV = os.path.join(RAW_FOLDER, "seg_result_1.csv")
SEG_RESULT_2_CSV = os.path.join(RAW_FOLDER, "seg_result_2.csv")
SEG_RESULT_3_CSV = os.path.join(RAW_FOLDER, "seg_result_3.csv")
TARGET_CSV = os.path.join(RAW_FOLDER, "KT_Bio_Health_8658.csv")
RESULT_CSV = os.path.join(RAW_FOLDER, "KT_Bio_Health_seg_new_rule.csv")
LOG_FILE = os.path.join(RAW_FOLDER, "log_new_rule.txt")
THRESHOLD = 0.12#0.12
THRESHOLD_CHECK = [0.50, 0.60]
IMAGE_ID = "image_id"

ID_CASE = "case"
ID_CLASS = "class"
ID_P0 = "P0"
ID_P1 = "P1"
ID_P2 = "P2"

seg_result_1 = pd.read_csv(SEG_RESULT_1_CSV)[IMAGE_ID].values
seg_result_2 = pd.read_csv(SEG_RESULT_2_CSV)[IMAGE_ID].values
seg_result_3 = pd.read_csv(SEG_RESULT_3_CSV)[IMAGE_ID].values
target_df = pd.read_csv(TARGET_CSV)
target_case = target_df[ID_CASE].values
target_class = target_df[ID_CLASS].values

target_p0 = target_df[ID_P0].values
target_p1 = target_df[ID_P1].values
target_p2 = target_df[ID_P2].values

result_class = []
with open(LOG_FILE, "w") as file:
    for index, current_case in enumerate(target_case):
        current_class = target_class[index]
        current_result_class = current_class
        if current_case in seg_result_3:
            if current_class != 2:
                if target_p2[index] >= THRESHOLD:
                    current_result_class = 2
                    print(current_case + " change to 2")
                    file.write(current_case + " change to 2\n")
        if current_case in seg_result_1:
            if current_class == 0:
                if target_p1[index] >= THRESHOLD:
                    current_result_class = 1
                    print(current_case + " change to 1")
                    file.write(current_case + " change to 1\n")
        if current_class == 1:
            if target_p1[index] >= THRESHOLD_CHECK[0] and target_p1[index] < THRESHOLD_CHECK[1]:
                if not current_case in seg_result_1:
                    if not current_case in seg_result_2:
                        if not current_case in seg_result_3:
                            print(current_case + "is low probability")
                            current_result_class = 0
        result_class.append(current_result_class)

result_df = pd.DataFrame({ID_CASE: target_case, ID_CLASS: result_class, ID_P0: target_p0, ID_P1: target_p1, ID_P2: target_p2})
result_df.to_csv(RESULT_CSV, header=True, index=False)
print("finish")
