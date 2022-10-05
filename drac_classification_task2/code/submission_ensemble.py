
import os
import numpy as np
import pandas as pd


CSV_1ST_PATH = '../input/v9.99.csv'
CSV_2ND_PATH = '../input/v10.4.csv'
RESULT_CSV_NAME = 'KT_bio_health.csv'
CSV_IMAGE_ID = 'case'
CSV_LABEL_ID = 'class'
CSV_PO_ID = 'P0'
CSV_P1_ID = 'P1'
CSV_P2_ID = 'P2'


DF_1st = pd.read_csv(CSV_1ST_PATH)
DF_2nd = pd.read_csv(CSV_2ND_PATH)
proba_1 = DF_1st.iloc[:, -3:]
proba_2 = DF_2nd.iloc[:, -3:]


weighted_sum = proba_1*0.55 + proba_2*0.45
prediction = [np.argmax(i) for i in weighted_sum.values]


name = DF_1st.iloc[:, 0]
P0 = weighted_sum.iloc[:, 0]
P1 = weighted_sum.iloc[:, 1]
P2 = weighted_sum.iloc[:, 2]


result_data_frame = pd.DataFrame(
    {CSV_IMAGE_ID: name, CSV_LABEL_ID: prediction, CSV_PO_ID: P0, CSV_P1_ID: P1, CSV_P2_ID: P2})

result_data_frame.to_csv(RESULT_CSV_NAME, header=True, index=False)
print("Finish")
