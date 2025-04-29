
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:11:53 2025

@author: poulimenos
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

df['Disease_Level'] = df['Disease'] + "_" + df['Level'].astype(str)
    
print(df['Disease'])
print(df['Disease_Level'])


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
from sklearn.model_selection import GroupShuffleSplit



def train_random_forest_grouped(x, y, groups, message, class_labels=None, show_confusion=True):
    """
    Εκπαίδευση Random Forest με train-test split που σέβεται τα group IDs.

    Params:
    - x: DataFrame με χαρακτηριστικά.
    - y: Series με ετικέτες.
    - groups: Ομαδοποιήσεις (π.χ. ασθενείς).
    - message: Μήνυμα για εκτύπωση.
    - class_labels: Προαιρετικά labels για confusion matrix.
    - show_confusion: Αν θα εμφανιστεί confusion matrix.

    Returns:
    - clf: Εκπαιδευμένος ταξινομητής.
    - metrics: Dict μετρικών.
    """

    # Χωρισμός βάσει ομάδων
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(x, y, groups=groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=20, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        max_features='log2', 
        bootstrap=True,
        random_state=42
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Μετρικές
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_micro": f1_score(y_test, y_pred, average='micro'),
    }

    # Εκτύπωση
    print(f"\n{message}")
    for k, v in metrics.items():
        print(f"{k.capitalize():<13}: {v:.4f}")

    # Confusion Matrix
    if show_confusion:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {message}")
        plt.tight_layout()
        plt.show()

    return clf, metrics


# === Εκτελέσεις ===

# Πρόβλεψη Disease
train_random_forest_grouped(x, y_disease, groups, "Random Forest - Predicting Disease", class_labels=le_disease.classes_)

# Πρόβλεψη Disease + Level
train_random_forest_grouped(x, y_disease_level, groups, "Random Forest - Predicting Disease + Level", class_labels=le_disease_level.classes_)

