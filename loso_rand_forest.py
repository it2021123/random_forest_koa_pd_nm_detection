#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:11:53 2025

@author: poulimenos
"""

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

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

print(y_disease.head())
print(y_disease_level.head())

# === Συνάρτηση για LOSO ===
def loso_cross_validation(x, y, groups, message, class_labels=None):
    logo = LeaveOneGroupOut()
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'f1_macro': [], 'f1_micro': []}

    for train_idx, test_idx in logo.split(x, y, groups=groups):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(
            n_estimators=500, max_depth=20,
            min_samples_split=10, min_samples_leaf=5,
            max_features='log2', bootstrap=True
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        metrics['f1_micro'].append(f1_score(y_test, y_pred, average='micro'))

    # === Εκτύπωση αποτελεσμάτων ===
    print(f"\n{message}")
    for key, values in metrics.items():
      mean = np.mean(values)
      std = np.std(values)
      print(f"{key.capitalize():<15} -> Mean: {mean:.4f} | Std: {std:.4f}")

# === Εκτελέσεις ===

# ➤ Πρόβλεψη Νόσου (Disease)
loso_cross_validation(x, y_disease, groups, "Random Forest - Predicting Disease")

# ➤ Πρόβλεψη Νόσου + Στάδιο (Disease_Level)
loso_cross_validation(x, y_disease_level, groups, "Random Forest - Predicting Disease + Stage")
