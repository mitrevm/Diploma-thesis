import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from functools import lru_cache
from joblib import Parallel, delayed
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA, TruncatedSVD

feature_cols = [
    "air", "bsa", "desolv" , "elec", "total", "vdw", "score", "lrmsd", "ilrmsd", "dockq"
]

list_bad = []



# ---------------------- Helper Functions ----------------------
def find_related_rows(df, n=15):
    results = []
    pair_rows = df[df['pair'] == 1]

    for _, row in pair_rows.iterrows():
        trgb1 = row['trgb1_gen']
        trgc1 = row['trgc1_gen']

        candidates = df[(df['trgb1_gen'] != trgb1) &
                        (df['trgc1_gen'] != trgc1) &
                        (df['pair'] == 0)]

        candidates = candidates.drop_duplicates(
            subset=['trgb1_gen', 'trgc1_gen']
        ).sample(n=n, random_state=42)##head(n)
          # random_state optional


        related = [tuple(x) for x in candidates.to_numpy()]
        original = tuple(row.to_numpy())
        results.append((original, related))

    return results


def split_pairs(df: pd.DataFrame, pos_name: tuple = None,
                neg_name: tuple = None, percent: float = 0.6):

    if pos_name:
        pos_row = df[(df["trgb1_gen"] == pos_name[0]) &
                     (df["trgc1_gen"] == pos_name[1]) &
                     (df["pair"] == 1)].iloc[0]

    if neg_name:
        neg_row = df[(df["trgb1_gen"] == neg_name[0]) &
                     (df["trgc1_gen"] == neg_name[1]) &
                     (df["pair"] == 0)].iloc[0]

    selected = pd.DataFrame([pos_row, neg_row])
    exclude = {pos_row["trgb1_gen"], pos_row["trgc1_gen"],
               neg_row["trgb1_gen"], neg_row["trgc1_gen"]}

    remaining = df[~df["trgb1_gen"].isin(exclude) &
                   ~df["trgc1_gen"].isin(exclude)]

    if not remaining.empty and percent < 1.0:
        pos_df = remaining[remaining["pair"] == 1]
        neg_df = remaining[remaining["pair"] == 0].sample(frac=percent,
                                                          random_state=42)
        remaining = pd.concat([pos_df, neg_df], ignore_index=True)

    return selected, remaining


def create_data_set(selected_df: pd.DataFrame,
                    remaining_df: pd.DataFrame,
                    n_in_training=50, n_in_test=50, model= ""):

    required_columns = feature_cols

    def process_df(input_df, is_training=False, model= ""):
        all_data = []
        #embeded_cols = set()  # Use a set to avoid duplicates

        for _, row in input_df.iterrows():
            ##Here we select from which f/
           #Old_Model_kako_og_samo_trg_extra_active
           #Old_Model_pattern_i_extra_active_trgb
            capri_ss_path = (f"/d/hpc/home/mm5129/Diplomska/runs/collected_caprieval/07_caprieval_run_"
                             f"{row['trgb1_gen']}_{row['trgc1_gen']}/capri_ss.tsv")
            # capri_ss_path = (f"/d/hpc/home/mm5129/Diplomska/runs/collected_caprieval/07_caprieval_run_"
            #                  f"{row['trgb1_gen']}_{row['trgc1_gen']}/capri_ss.tsv")
            try:
                n_limit = n_in_training if is_training else n_in_test
                full_df = pd.read_csv(capri_ss_path, sep='\t').head(n_limit)

                if not all(col in full_df.columns for col in required_columns):
                    continue

                df_slice = full_df[required_columns].mean().to_frame().T

                # Assign labels
                df_slice['compatible'] = row['pair']
                
                # 4. Use only the Z-scored columns for your features
                df_selected = df_slice.copy()
                
                all_data.append(df_selected)
            except Exception as e:
                print(f"Error processing {capri_ss_path}: {e}")
                continue
        if all_data:
            return pd.concat(all_data, ignore_index=True)#, list(embeded_cols)
        else:
            print(f"HH")
            return pd.DataFrame(columns=required_columns + ['compatible'])# +


    selected_all_data = process_df(selected_df, is_training=False, model=model)
    remaining_all_data = process_df(remaining_df, is_training=True, model=model)

    return selected_all_data, remaining_all_data, 


def train_logreg_and_predict(selected_df: pd.DataFrame,
                             remaining_df: pd.DataFrame,
                             c: float):

    target_col = 'compatible'

    X_train_num = remaining_df[feature_cols].values          # HADDOCK features
    y_train = remaining_df[target_col].values

    X_selected_num = selected_df[feature_cols].values
    y_selected = selected_df[target_col].values

   

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_selected_num = scaler.transform(X_selected_num)

    if c is not None:
        clf = LogisticRegression(max_iter=5000, C=c)#, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=5000)
    
    #print(X_train_num.shape, X_selected_num.shape)
    clf.fit(X_train_num, y_train)
    #for name, coef in zip(new_cols, clf.coef_[0]):
     #   print(f"Feature: {name:10} Coefficient: {coef:.4f}")
    probs = clf.predict_proba(X_selected_num)[:, 1] 

    avg_prob_1 = probs[y_selected == 1].mean()
    avg_prob_0 = probs[y_selected == 0].mean()

    return avg_prob_1, avg_prob_0

# ---------------------- Parallel Wrapper ----------------------
def run_pipeline(original, i, percent, n_in_training, n_in_test, df, c, model):

    selected_df, remaining_df = split_pairs(df,
                                            pos_name=original,
                                            neg_name=i,
                                            percent=percent)
    
    
    en, dva = create_data_set(selected_df, remaining_df,
                                            n_in_training=n_in_training,
                                            n_in_test=n_in_test, model=model)
    if en.empty or dva.empty:
        print(f"Empty data set for {original} and {i} with percent {percent}")
        return None

    #print(en.shape, dva.shape)
    avg_prob_1, avg_prob_0 = train_logreg_and_predict(en, dva, c)

    return {
        "original": original,
        "i": i,
        "avg_predicted_for_compatible_1": avg_prob_1,
        "avg_predicted_for_compatible_0": avg_prob_0,
        "percent": percent
    }
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5, help="Number of training samples")
    parser.add_argument("--test", type=int, default=2, help="Number of test samples")
    parser.add_argument("--model", type=str, help="Model name")
    return parser.parse_args()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    args = parse_args()
    n_in_training = args.train
    n_in_test = args.test
    c = 1.0
    model = args.model
    df = pd.read_csv("pairs_fixed.csv")
    r = find_related_rows(df)
    


    for n_in_training in [n_in_training]:
        for n_in_test in [n_in_test]:
            tasks = []
            for original, related in r:
                for i in related:
                    for percent in [0.1,0.2,0.3,0.4,0.5,
                                    0.6,0.7,0.8,0.9,1.0]:
                        tasks.append((original, i, percent))

            results = Parallel(n_jobs=10, verbose=10)(
                delayed(run_pipeline)(original, i, percent,
                                      n_in_training, n_in_test, df, c, model)
                for (original, i, percent) in tasks
            )

            results = [res for res in results if res is not None]
            results_df = pd.DataFrame(results)
            folder = f"/d/hpc/home/mm5129/A_pairs_fixed/OnlyHaddock/mean_first"
            os.makedirs(folder, exist_ok=True)            
            for percent, group in results_df.groupby("percent"):
                filename = (f"{folder}/n_in_training_{n_in_training}"
                            f"_n_in_test_{n_in_test}_results_pipeline_percent_{percent}.csv")
                group.drop(columns="percent").to_csv(filename, index=False)
                print(f"Saved {filename}")
