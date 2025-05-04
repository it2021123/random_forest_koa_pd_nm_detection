#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:11:53 2025

@author: poulimenos
"""

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from collections import Counter

# === Χρήσιμη συνάρτηση ===
def convert_commas_to_periods(df):
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].str.replace(',', '.', regex=False)
    return df

# === Φόρτωση αρχείου ===
df = pd.read_csv("/home/poulimenos/project/rand_forest2/all_features.csv")



# === Δημιουργία Id ===
df['Id'] = (
    df['ID'].astype(str) +
    df['Disease'].astype(str) +
    df.get('Level', 'NA').astype(str) 
)

# === Καθαρισμός & Προετοιμασία ===
df = df.drop(columns=['ID'], errors='ignore')
df = df.drop(columns=['Source_File'], errors='ignore')

df = convert_commas_to_periods(df)

# === Δημιουργία πεδίων ετικετών ===
df['Disease_Level'] = df['Disease'].astype(str)
if 'Level' in df.columns:
    df['Disease_Level'] = df['Disease'] + "_" + df['Level'].astype(str)

# === Label Encoding ===
le_disease = LabelEncoder()
le_disease_level = LabelEncoder()

df['Disease_encoded'] = le_disease.fit_transform(df['Disease'])
df['Disease_Level_encoded'] = le_disease_level.fit_transform(df['Disease_Level'])

# === Κανονικοποίηση χαρακτηριστικών ===
features = df.drop(columns=['Disease', 'Level', 'Disease_Level', 'Disease_encoded', 'Disease_Level_encoded', 'Id'], errors='ignore')
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# === Τελικά δεδομένα ===
y_disease = df['Disease_encoded']
y_disease_level = df['Disease_Level_encoded']
groups = df['Id']



# === Συνάρτηση για LOSO ===

import pandas as pd


def loso_cross_validation(x, y, groups, message, num_classes):
    print(message)
    logo = LeaveOneGroupOut()
    all_results = []

    accuracy_list = []
    f1_list = []

    all_y_true = []
    all_y_pred = []

    for i, (train_idx, test_idx) in enumerate(logo.split(x, y, groups=groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        group_label = groups.iloc[test_idx].unique()[0]



        clf = RandomForestClassifier(
            n_estimators=500, max_depth=20,
            min_samples_split=10, min_samples_leaf=5,
            max_features='log2', bootstrap=True
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")  

        accuracy_list.append(acc)
        f1_list.append(f1)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        for idx, true_label, pred_label in zip(x_test.index, y_test, y_pred):
            result = {
                'subject': group_label,
                'sample_index': idx,
                'accuracy': acc,
                'f1-score': f1,
                'Target': true_label
            }

            predicted_class_counts = dict(Counter(y_pred))
            for class_id in range(num_classes):
                result[f"Predicted class: {class_id}"] = predicted_class_counts.get(class_id, 0)
            
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    # Υπολογισμός Precision, Recall, F1 per class (χειροκίνητα)
    precision_list = []
    recall_list = []
    f1_list_per_class = []

    for cls in range(num_classes):
        tp = results_df.loc[results_df['Target'] == cls, f'Predicted class: {cls}'].sum()
        fp = results_df.loc[results_df['Target'] != cls, f'Predicted class: {cls}'].sum()
        fn = results_df.loc[(results_df['Target'] == cls), [f'Predicted class: {i}' for i in range(num_classes) if i != cls]].sum(axis=1).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list_per_class.append(f1)

    # Υπολογισμός γενικών μετρικών
    macro_f1 = np.mean(f1_list_per_class)
    avg_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    avg_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)


    ## Εκτύπωση αποτελεσμάτων με όμορφη μορφοποίηση
    print(f"\n*** Αποτελέσματα Διασταυρούμενης Αξιολόγησης (LOSO) ***\n")

    # Γενικά Αποτελέσματα
    print(f"Μέση Ακρίβεια (Accuracy): {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Μέσο F1-Score (weighted): {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro F1 (χειροκίνητος υπολογισμός): {macro_f1:.4f}")
    


    # Αποτελέσματα ανά Κλάση
    print("\n*** Αποτελέσματα ανά Κλάση (Precision, Recall, F1) ***")
    for i in range(num_classes):
      print(f"\nΚλάση {i}:")
      print(f"  - Precision: {precision_list[i]:.4f}")
      print(f"  - Recall: {recall_list[i]:.4f}")
      print(f"  - F1-Score: {f1_list_per_class[i]:.4f}")

    return results_df


# === Εκτελέσεις ===

# ➤ Πρόβλεψη Νόσου (Disease)
loso_cross_validation(x, y_disease, groups, "Random Forest - Predicting Disease",3)

# ➤ Πρόβλεψη Νόσου + Στάδιο (Disease_Level)
loso_cross_validation(x, y_disease_level, groups, "Random Forest - Predicting Disease + Stage",5)
