import pandas as pd

def ensemble_preds(pred_file_list):
    dfs = [pd.read_csv(file) for file in pred_file_list]
    for i in range(1, len(dfs)):
        dfs[i] = dfs[i].drop(columns=['id'])
    
    new_df = pd.concat(dfs, axis=1)

    new_df['label'] = new_df.iloc[:, 1:].mean(axis=1)
    new_df = new_df.iloc[:, [0, -1]]  
    
    return new_df

if __name__ == "__main__":
    pred_file_list = [
        "submission_seresnext101_32x4d_20epochs.csv",
        "submission_densenet169_24epochs_final.csv",
        "submission_swin_small_patch4_window7_224_24epochs_final.csv",
        "submission_dino_vits8_20epochs.csv",
        "submission_swin_tiny_patch4_window7_224_16epochs_2.csv",
    ]
    l = len(pred_file_list)
    ensemble_df = ensemble_preds(pred_file_list)
    ensemble_df.to_csv(f"submission_ensemble_final_{l}.csv", index=False)