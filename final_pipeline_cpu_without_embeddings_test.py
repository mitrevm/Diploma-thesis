import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from functools import lru_cache
from joblib import Parallel, delayed
import os


feature_cols = [
    #"air", 
    #"angles", 
    #"bonds", 
    #"bsa", 
    #"cdih", 
    #"coup", 
    #"dani",
    "desolv", 
    #"dihe", 
    #"elec", 
    #"improper",
    "dockq",
    #  "rdcs", 
    #  "rg", 
    #  "sym",
    # "total", 
    #"vdw", 
    # "vean", 
    # "xpcs", ]#"caprieval_rank"
]

list_bad = []



# ---------------------- Helper Functions ----------------------
def find_related_rows(df, n=50):
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
        )#.head(n)

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
            capri_ss_path = (f"/d/hpc/home/mm5129/Diplomska/Old_Model_new_restraintments/collected_caprieval/07_caprieval_run_new_model_0_"
                             f"{row['trgb1_gen']}_{row['trgc1_gen']}/capri_ss.tsv")
            # capri_ss_path = (f"/d/hpc/home/mm5129/Diplomska/runs/collected_caprieval/07_caprieval_run_"
            #                  f"{row['trgb1_gen']}_{row['trgc1_gen']}/capri_ss.tsv")
            try:
                n_limit = n_in_training if is_training else n_in_test
                df = pd.read_csv(capri_ss_path, sep='\t').head(n_limit)

                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: Missing required columns in {capri_ss_path}")
                    continue

                if (df['desolv'] == 0).any() or \
                   (df['vdw'] == 0).any() or \
                   (df['elec'] == 0).any():
                    if capri_ss_path not in list_bad:
                        list_bad.append(capri_ss_path)
                        print(f"Warning: Zero values in {capri_ss_path}")

                df['compatible'] = row['pair']


            except Exception as e:
                print(f"Error processing {capri_ss_path}: {e}")
                continue
            
            epsilon = 1e-9
            df['vdw_density'] = df['vdw'] / (df['bsa'] + epsilon)
            df['elec_density'] = df['elec'] / (df['bsa'] + epsilon)
            df['desolv_density'] = df['desolv'] / (df['bsa'] + epsilon)
            df_selected = df[required_columns + ['compatible', 'vdw_density', 'elec_density', 'desolv_density']].copy()
            all_data.append(df_selected)

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

    X_selected_num = selected_df[feature_cols ].values
    y_selected = selected_df[target_col].values

    #print(feature_cols)
    #print(target_col)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_selected_num_scaled = scaler.transform(X_selected_num)

    X_train_final = np.hstack([X_train_num_scaled])
    X_selected_final = np.hstack([X_selected_num_scaled])
    if c is not None:
        clf = LogisticRegression(max_iter=5000, C=c)#, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=5000)
    
    #print(X_selected_num)
   # print(y_train)
    clf.fit(X_train_final, y_train)
    #print(clf.predict_proba(X_selected_final))
    probs = clf.predict_proba(X_selected_final)[:, 1] 

    avg_prob_1 = probs[y_selected == 1].mean()
    avg_prob_0 = probs[y_selected == 0].mean()
    #print(probs.shape)
   # print(y_selected.shape)
   # print(y_selected)
   # print(probs[y_selected == 1])
    #print(probs[y_selected == 0])
    #print(avg_prob_1, avg_prob_0)
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
    model = 0
    df = pd.read_csv("parovi.csv")
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
            folder = f"/d/hpc/home/mm5129/AlphaFoldModelTesting/Old_Model_new_restraintment_both_only_active_testing_only_desolv,dockq/csv_onlyhaddock"
            os.makedirs(folder, exist_ok=True)            
            for percent, group in results_df.groupby("percent"):
                filename = (f"{folder}/n_in_training_{n_in_training}"
                            f"_n_in_test_{n_in_test}_results_pipeline_percent_{percent}.csv")
                group.drop(columns="percent").to_csv(filename, index=False)
                print(f"Saved {filename}")

