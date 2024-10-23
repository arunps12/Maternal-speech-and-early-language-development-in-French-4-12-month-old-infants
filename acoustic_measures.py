import os
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from utils import Hz_to_semitones

def pitch_in_st(df, pitch_Hz_col_name,pitch_st_col_name,ref_in_Hz = 10):
    df = df.dropna().reset_index(drop=True)
    pitch_Hz = df[pitch_Hz_col_name]
    pitch_st = Hz_to_semitones(pitch_Hz, ref_in_Hz)
    df[pitch_st_col_name] = pitch_st
    return df

def range_in_st(df, min_pitch_Hz_col_name, max_pitch_Hz_col_name,range_st_col_name, ref_in_Hz = 10):
    df = df.dropna().reset_index(drop=True)
    min_pitch_Hz = df[min_pitch_Hz_col_name]
    min_pitch_st = Hz_to_semitones(min_pitch_Hz, ref_in_Hz)
    max_pitch_Hz = df[max_pitch_Hz_col_name]
    max_pitch_st = Hz_to_semitones(max_pitch_Hz, ref_in_Hz)
    range_st = max_pitch_st - min_pitch_st
    df[range_st_col_name] = range_st
    return df

def duration_in_ms(df, duration_sec_col_name, duration_ms_col_name):
    df = df.dropna().reset_index(drop=True)
    duration_sec = df[duration_sec_col_name]
    #duration_ms = 1000 * duration_sec
    df[duration_ms_col_name] = 1000 * duration_sec
    return df

def vowel_space_expansion(df, group_col, cat_col, feature_col):
    df = df.dropna().reset_index(drop=True)
    results = []
    for (spkid, age), group in df.groupby(group_col):
        if group[cat_col].nunique() <= len(feature_col):
            print(f"Skipping speaker {spkid}, age {age} due to number of categories less than number of features.")
            continue
        try:
            mean_group_col = group_col + [cat_col]
            mean_features = group.groupby(mean_group_col)[feature_col].mean().reset_index()
            feature_values = mean_features[feature_col].values
            area = ConvexHull(feature_values).volume
            #print(area)
            results.append({'SPK_id': spkid, 'AgeInDays': age, 'ConvexHullArea': round(area)})           
            #print(results)
        except Exception as e:
            print(f"Error during Creating ConvexHull for speaker {spkid}, age {age}: {e}")
            continue
    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna().reset_index(drop=True)
    #results_df.dropna(inplace=True).reset_index(drop=True)
    print("Convex hull areas have been saved")
    return results_df

def vowel_variability(df, group_col, feature_col):
    df = df.dropna().reset_index(drop=True)
    results = []
    for (spkid, age, vow), group in df.groupby(group_col):
        try:
                
            # Compute the mean and std for each group
            mean_features = group[feature_col].mean()
            std_features = group[feature_col].std()

            # Filter data to include only points within one standard deviation from the mean
            condition = True
            for feature in feature_col:
                condition &= (group[feature] >= (mean_features[feature] - std_features[feature])) & \
                                     (group[feature] <= (mean_features[feature] + std_features[feature]))
                    
                filtered_group = group[condition]
                feature_values = filtered_group[feature_col].values

            if len(feature_values) < 3:
                print(f"Skipping speaker {spkid}, age {age}, category {vow} due to less than 3 samples.")
                continue

                # Compute the area of the ellipse using the standard deviations of F1 and F2
            sigmaF1 = feature_values[0].std()
            sigmaF2 = feature_values[1].std()
            area = np.pi * sigmaF1 * sigmaF2  # Area of ellipse

            results.append({
                        'SPK_id': spkid,
                        'AgeInDays': age,
                        'vowels': vow,
                        'no_Samples': len(feature_values),
                        'variability': round(area, 2)
            })
                    
        except Exception as e:
            print(f"Error during processing for speaker {spkid}, age {age}: {e}")
            continue
    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna().reset_index(drop=True)
    #results_df.dropna(inplace=True).reset_index(drop=True)
    print("Variabilities have been saved")
    return results_df

def vowel_distinctiveness(df, group_col, cat_col, feature_col):
    
    
    results = []

    # Group data by participant and register
    for (spkid, age), group in df.groupby(group_col):
        # Filter out vowel categories with fewer than 3 samples
        valid_groups = [vowel_group for vowel, vowel_group in group.groupby(cat_col) if len(vowel_group) >= 3]
        
        # If there are no valid vowel groups, skip this participant and age
        if len(valid_groups) == 0:
            print(f"Skipping participant {spkid}, age {age} due to no valid vowel groups.")
            continue
        
        # Compute total variance (both between and within-cluster variance) for the remaining valid vowel groups
        valid_data = pd.concat(valid_groups)
        total_var = np.sum(np.var(valid_data[feature_col], axis=0, ddof=1))

        # Compute within-cluster variance
        within_cluster_var = 0
        for vowel_group in valid_groups:
            #print(vowel_group[cat_col])
            # Compute variance within the vowel group
            cluster_var = np.sum(np.var(vowel_group[feature_col], axis=0, ddof=1)) * len(vowel_group)
            within_cluster_var += cluster_var

        # Normalize within-cluster variance by the total number of samples
        within_cluster_var /= len(valid_data)

        
        # Between-cluster variance
        between_cluster_var = total_var - within_cluster_var

        # Compute vowel distinctiveness as the ratio of between-cluster variance to total variance
        distinctiveness = max(0, between_cluster_var / total_var) if total_var > 0 else 0
        
        # Append the results
        results.append({
            'SPK_id': spkid,
            'AgeInDays': age,
            'Vowel_Dist': f'{distinctiveness:.4f}'
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna().reset_index(drop=True)
    print("Distinctiveness have been saved")
    return results_df



