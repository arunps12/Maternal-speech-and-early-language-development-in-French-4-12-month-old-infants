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
            sigmaF1 = std_features[feature_col[0]]
            sigmaF2 = std_features[feature_col[1]]
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

    # Group data by participant and register (or any columns provided in group_col)
    for key, group in df.groupby(group_col):
        # Extract keys from the group for easier reference
        spkid, age = key

        # Filter out vowel categories with fewer than 3 samples
        valid_groups = [vowel_group for vowel, vowel_group in group.groupby(cat_col) if len(vowel_group) >= 3]
        
        # If there are no valid vowel groups, skip this participant and age
        if len(valid_groups) == 0:
            print(f"Skipping participant {spkid}, age {age} due to no valid vowel groups.")
            continue
        
        # Combine the valid vowel groups into a single DataFrame
        valid_data = pd.concat(valid_groups)
        valid_data[feature_col[0]] = (valid_data[feature_col[0]] - valid_data[feature_col[0]].mean()) / valid_data[feature_col[0]].std()
    
        valid_data[feature_col[1]] = (valid_data[feature_col[1]] - valid_data[feature_col[1]].mean()) / valid_data[feature_col[1]].std()
        # Calculate the overall centroid for the vowel space (F1 and F2)
        overall_centroid = valid_data[feature_col].mean().values
        
        # Compute total sum of squares (TSS): squared distances of individual vowels from the overall centroid
        total_ss = np.sum(np.sum((valid_data[feature_col] - overall_centroid) ** 2, axis=1))
        
        if total_ss == 0:
            print(f"Skipping participant {spkid}, age {age} due to zero total variance.")
            continue
        
        # Compute the centroid for each vowel category
        vowel_centroids = valid_data.groupby(cat_col)[feature_col].mean()

        # Compute between-vowel category sum of squares (BSS): squared distances of vowel category centroids from the overall centroid
        between_ss = 0
        for vowel, centroid in vowel_centroids.iterrows():
            n_vowel = len(valid_data[valid_data[cat_col] == vowel])  # number of tokens in this vowel category
            centroid_values = centroid.values  # F1 and F2 values for the centroid
            # Calculate the squared distance between the vowel centroid and the overall centroid, then multiply by the number of tokens
            between_ss += n_vowel * np.sum((centroid_values - overall_centroid) ** 2)

        # Compute vowel distinctiveness as the ratio of between-category variance to total variance
        distinctiveness = between_ss / total_ss
        
        # Append the results
        results.append({
            'SPK_id': spkid,
            'AgeInDays': age,
            'Vowel_Dist': f'{distinctiveness:.4f}'
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna().reset_index(drop=True)
    
    print("Distinctiveness has been saved")
    return results_df


