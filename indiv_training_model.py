import numpy as np
import pandas as pd
from joblib import load

df_train = pd.read_csv("dataset\\training_data.csv")
df_test = pd.read_csv("dataset\\testing_data.csv", cache_dates=False)

def adjust_tumor_size(df):
    return df * 10

def age_recode(df):
    if df >= 15 and df <= 19:
        return "15-19 years"
    elif df >= 20 and df <= 24:
        return "20-24 years"
    elif df >= 25 and df <= 29:
        return "25-29 years"
    elif df >= 30 and df <= 34:
        return "30-34 years"
    elif df >= 35 and df <= 39:
        return "35-39 years"
    elif df >= 40 and df <= 44:
        return "40-44 years"
    elif df >= 45 and df <= 49:
        return "45-49 years"
    elif df >= 50 and df <= 54:
        return "50-54 years"
    elif df >= 55 and df <= 59:
        return "55-59 years"
    elif df >= 60 and df <= 64:
        return "60-64 years"
    elif df >= 65 and df <= 69:
        return "65-69 years"
    elif df >= 70 and df <= 74:
        return "70-74 years"
    elif df >= 75 and df <= 79:
        return "75-79 years"
    elif df >= 80 and df <= 84:
        return "80-84 years"
    elif df >= 85:
        return "85+ years"
    else:
        raise ValueError("Invalid parameter map_func.")


def encode_event(df):
    if (df == "Dead"):
        return True
    else:
        return False


df_test["Age recode with <1 year olds"] = df_test["Age recode with <1 year olds"].apply(age_recode)
df_total = df_test.append(df_train)
df_total = pd.get_dummies(df_total, prefix=["Surg Prim Site",
                                            "T",
                                            "N",
                                            "M", "Hist/behav",
                                            "Age recode", "Sex",
                                            "Grade"],
                          columns=["RX Summ--Surg Prim Site (1998+)",
                                   "Derived AJCC T, 6th ed (2004-2015)",
                                   "Derived AJCC N, 6th ed (2004-2015)",
                                   "Derived AJCC M, 6th ed (2004-2015)", "ICD-O-3 Hist/behav",
                                   "Age recode with <1 year olds", "Sex",
                                   "Grade (thru 2017)"])
df_total["End Calc Vital Status (Adjusted)"] = df_total["End Calc Vital Status (Adjusted)"].apply(encode_event)

df_test_count = df_test.shape[0]
df_train = df_total[df_test_count:]
df_test = df_total[:df_test_count]

get_target = lambda df: (df['Number of Intervals (Calculated)'].values.astype(np.float16),
                         df['End Calc Vital Status (Adjusted)'].values.astype(bool))

zip_arrays = lambda arr1, arr2: list(zip(arr1, arr2))

def list_to_nparr(arr):
    out = np.empty(len(arr), dtype=[('cens', '?'), ('time', '<f8')])
    out[:] = arr
    return out

durations_train, events_train = get_target(df_train)
y_train = list_to_nparr(zip_arrays(events_train, durations_train))
x_train = df_train.drop("Number of Intervals (Calculated)", axis=1).drop("End Calc Vital Status (Adjusted)", axis=1)
durations_test, events_test = get_target(df_test)
y_test = list_to_nparr(zip_arrays(events_test, durations_test))
x_test = df_test.drop("Number of Intervals (Calculated)", axis=1).drop("End Calc Vital Status (Adjusted)", axis=1)

model = load("model\\rsf_model.joblib")
score = model.score(x_test, y_test)
# The test score for this model is 0.739344262295082
print("testing score is ", score)
