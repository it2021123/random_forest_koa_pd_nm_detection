import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from math import atan2, degrees
from pathlib import Path
import re

def calculate_angle(a, b, c):
    """Calculate angle between 3 points in degrees (a-b-c)"""
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return degrees(np.arccos(np.clip(cosine_angle, -1, 1)))

def calculate_similarity(left_vals, right_vals):
    """Calculate similarity metrics between bilateral features"""
    if len(left_vals) == 0 or len(right_vals) == 0:
        return {'mean': np.nan, 'max': np.nan, 'std': np.nan}
    
    # Absolute differences frame-by-frame
    diffs = np.abs(np.array(left_vals) - np.array(right_vals))
    
    return {
        'mean': np.nanmean(diffs),
        'max': np.nanmax(diffs),
        'std': np.nanstd(diffs)
    }

def process_window(window, window_num, nm):
    """Process a single time window of gait data"""
    features = {}
    columns = ['ID', 'Disease', 'Side'] if nm == 'nm' else ['ID', 'Disease', 'Side', 'Level']
    # Preserve metadata
    for col in  columns:
        if col in window.columns:
            features[col] = window[col].iloc[0]
    
    features['Window'] = window_num
    
    # Initialize containers for bilateral comparisons
    left_hip, right_hip = [], []
    left_knee, right_knee = [], []
    left_ankle, right_ankle = [], []
    left_heel, right_heel = [], []
    step_lengths = []
    
    for _, row in window.iterrows():
        # Process left side
        left_hip.append(calculate_angle(
            [row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']],
            [row['LEFT_HIP_x'], row['LEFT_HIP_y']],
            [row['LEFT_KNEE_x'], row['LEFT_KNEE_y']]
        ))
        left_knee.append(calculate_angle(
            [row['LEFT_HIP_x'], row['LEFT_HIP_y']],
            [row['LEFT_KNEE_x'], row['LEFT_KNEE_y']],
            [row['LEFT_ANKLE_x'], row['LEFT_ANKLE_y']]
        ))
        left_ankle.append(degrees(atan2(
            row['LEFT_ANKLE_x'] - row['LEFT_KNEE_x'],
            row['LEFT_ANKLE_y'] - row['LEFT_KNEE_y']
        )))
        left_heel.append(row['LEFT_ANKLE_y'])
        
        # Process right side
        right_hip.append(calculate_angle(
            [row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']],
            [row['RIGHT_HIP_x'], row['RIGHT_HIP_y']],
            [row['RIGHT_KNEE_x'], row['RIGHT_KNEE_y']]
        ))
        right_knee.append(calculate_angle(
            [row['RIGHT_HIP_x'], row['RIGHT_HIP_y']],
            [row['RIGHT_KNEE_x'], row['RIGHT_KNEE_y']],
            [row['RIGHT_ANKLE_x'], row['RIGHT_ANKLE_y']]
        ))
        right_ankle.append(degrees(atan2(
            row['RIGHT_ANKLE_x'] - row['RIGHT_KNEE_x'],
            row['RIGHT_ANKLE_y'] - row['RIGHT_KNEE_y']
        )))
        right_heel.append(row['RIGHT_ANKLE_y'])
        
        # Calculate step length (when both feet are on ground)
        step_lengths.append(abs(row['LEFT_ANKLE_x'] - row['RIGHT_ANKLE_x']))
    
    # Calculate bilateral similarities
    for joint, left, right in zip(
        ['Hip', 'Knee', 'Ankle', 'Heel'],
        [left_hip, left_knee, left_ankle, left_heel],
        [right_hip, right_knee, right_ankle, right_heel]
    ):
        sim = calculate_similarity(left, right)
        features[f'{joint}_Similarity_Mean'] = sim['mean']
        features[f'{joint}_Similarity_Max'] = sim['max']
        features[f'{joint}_Similarity_Std'] = sim['std']
    
    # Step length statistics
    features['Step_Length_Mean'] = np.nanmean(step_lengths)
    features['Step_Length_Max'] = np.nanmax(step_lengths)
    features['Step_Length_Std'] = np.nanstd(step_lengths)
    
    # Individual side statistics
    for side, hip, knee, ankle, heel in zip(
        ['Left', 'Right'],
        [left_hip, right_hip],
        [left_knee, right_knee],
        [left_ankle, right_ankle],
        [left_heel, right_heel]
    ):
        features[f'{side}_Hip_Mean'] = np.nanmean(hip)
        features[f'{side}_Hip_Max'] = np.nanmax(hip)
        features[f'{side}_Hip_Std'] = np.nanstd(hip)
        
        features[f'{side}_Knee_Mean'] = np.nanmean(knee)
        features[f'{side}_Knee_Max'] = np.nanmax(knee)
        features[f'{side}_Knee_Std'] = np.nanstd(knee)
        
        features[f'{side}_Ankle_Mean'] = np.nanmean(ankle)
        features[f'{side}_Ankle_Max'] = np.nanmax(ankle)
        features[f'{side}_Ankle_Std'] = np.nanstd(ankle)
        
        features[f'{side}_Heel_Max'] = np.nanmax(heel)
        features[f'{side}_Heel_Min'] = np.nanmin(heel)
    
    return features

def process_group(input_path, output_path, pattern ,nm):
    """Process all files in a patient group"""
    all_data = []
    
    for csv_file in Path(input_path).glob('*.csv'):
        try:
            # Verify filename pattern
            if not re.search(pattern, csv_file.name):
                continue
                
            df = pd.read_csv(csv_file)
            print(f"Processing {csv_file.name} ({len(df)} frames)")
            
            # Process in 25-frame windows with 12-frame overlap
            window_size = 25
            step_size = 12
            
            for i, window_num in enumerate(range(0, len(df)-window_size+1, step_size)):
                window = df.iloc[window_num:window_num+window_size]
                window_features = process_window(window, i, nm)
                window_features['Source_File'] = csv_file.name
                all_data.append(window_features)
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    # Save results
    if all_data:
        pd.DataFrame(all_data).to_csv(output_path, index=False)
        print(f"Saved {len(all_data)} windows to {output_path}")

# Configuration for each patient group
groups = [
    {
        'name': 'NM',
        'input': '/home/poulimenos/output/NM/',
        'output': '/home/poulimenos/nm_features2.csv',
        'pattern': r"(\d{3})_(\w+)_(\d{2})",
        'nm':'nm'
    },
    {
        'name': 'KOA_EL',
        'input': '/home/poulimenos/output/KOA/KOA_EL',
        'output': '/home/poulimenos/koa_El_features2.csv',
        'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
        'nm':'koa'
    },
    {
        'name': 'KOA_MD',
        'input': '/home/poulimenos/output/KOA/KOA_MD',
        'output': '/home/poulimenos/koa_MD_features2.csv',
        'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
        'nm':'koa'
    },
    {
        'name': 'KOA_SV',
        'input': '/home/poulimenos/output/KOA/KOA_SV',
        'output': '/home/poulimenos/koa_SV_features2.csv',
        'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
        'nm':'koa'
    },
    {
        'name': 'PD',
        'input': '/home/poulimenos/output/PD/',
        'output': '/home/poulimenos//pd_features2.csv',
        'pattern': r"(\d{3})_(\w+)_(\d{2})_(\w+)",
        'nm':'pd'
        
    }
    
]

# Process all groups
for group in groups:
    print(f"\nProcessing {group['name']} group...")
    process_group(group['input'], group['output'], group['pattern'],group['nm'])

print("\nAnalysis complete for all groups!")