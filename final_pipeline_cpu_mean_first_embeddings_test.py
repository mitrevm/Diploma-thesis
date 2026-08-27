import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from functools import lru_cache
from joblib import Parallel, delayed
import os
from sklearn.metrics.pairwise import cosine_similarity
global_var = 0

feature_cols = [
    "air", "angles", "bonds", "bsa", "cdih", "coup", "dani",
    "desolv", "dihe", "elec", "improper", "rdcs", "rg", "sym",
    "total", "vdw", "vean", "xpcs"
]

list_bad = []

# ---------------------- Embedding Cache ----------------------
def load_embedding(name: str, model: str):
    """Load embedding from disk (no lru_cache, for joblib compatibility)."""
    path = f"/d/hpc/home/mm5129/np_embeddings_{model}/{name.upper()}.npy"
    return np.load(path)

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
        ).sample(n, random_state=30)#.sample(n, random_state=42)#.head(n)

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
                    n_in_training=50, n_in_test=50,
                    model=""):

    required_columns = feature_cols  # your list of feature columns

    def process_df(input_df, is_training=False, model=""):
        all_data = []

        # Determine number of embedding dimensions by inspecting first embedding
        first_row = input_df.iloc[0]
        trgb1_emb_len = len(load_embedding(first_row['trgb1_gen'], model))
        trgc1_emb_len = len(load_embedding(first_row['trgc1_gen'], model))
        
        # Global embedding column names
        embeded_cols = [f"trgb1_emb_{i}" for i in range(trgb1_emb_len)] + [f"trgc1_emb_{i}" for i in range(trgc1_emb_len)]

        for _, row in input_df.iterrows():
            capri_ss_path = (
                f"/d/hpc/home/mm5129/Diplomska/runs/collected_caprieval/"
                f"07_caprieval_run_{row['trgb1_gen']}_{row['trgc1_gen']}/capri_ss.tsv"
            )

            try:
                n_limit = n_in_training if is_training else n_in_test
                df_full = pd.read_csv(capri_ss_path, sep='\t').head(n_limit)

                # Keep only existing required columns
                existing_cols = [c for c in required_columns if c in df_full.columns]
                if not existing_cols:
                    print(f"No required columns in {capri_ss_path}, skipping...")
                    continue

                # Aggregation: mean, min, max, std
                agg = df_full[existing_cols].agg(['mean', 'min', 'max', 'std'])
                # legacy stack(dropna=True) dropped NaN std (1-row case); future_stack keeps full width
                df_row = agg.T.stack().to_frame().T
                
                #df_row = df_full[required_columns].copy()
               # df_row = agg.T.stack(dropna=False).to_frame().T
                df_row.columns = [f"{col}_{stat}" for col in existing_cols
                                  for stat in ['mean', 'min', 'max', 'std']]
                
                df_row['compatible'] = row['pair']

                # Load embeddings
                trgb1_emb = load_embedding(row['trgb1_gen'], model)
                trgc1_emb = load_embedding(row['trgc1_gen'] if "nc" not in row['trgc1_gen']
                                           else f"TRGC_{row['trgc1_gen']}", model)

                emb_trb1_flat = trgb1_emb.ravel()
                emb_trgc1_flat = trgc1_emb.ravel()
                cos_sim = float(cosine_similarity([emb_trb1_flat], [emb_trgc1_flat])[0, 0])
                euclidean = float(np.linalg.norm(emb_trb1_flat - emb_trgc1_flat))
                dot_prod = float(np.dot(emb_trb1_flat, emb_trgc1_flat))
                # assign scalars so pandas broadcasts to every row
                df_row["cos_sim"] = cos_sim
                df_row["euclidean"] = euclidean
                df_row["dot_prod"] = dot_prod
                #trgb1_gen,trgc1_gen,pair,iptm,ptm,ranking_score,pae_0_1,pae_1_0
                df_row["iptm"] = row["iptm"]
                df_row["ptm"] = row["ptm"]
                df_row["ranking_score"] = row["ranking_score"]
                df_row["pae_0_1"] = row["pae_0_1"]
                df_row["pae_1_0"] = row["pae_1_0"]
                df_row["iptm_0_1"] = row["iptm_0_1"]
                df_row["iptm_1_0"] = row["iptm_1_0"]
                df_row["iptm_0_0"] = row["iptm_0_0"]
                df_row["iptm_1_1"] = row["iptm_1_1"]
                # #interface_min_pae,interface_mean_pae,interface_max_contact_prob
                # df_row["interface_min_pae"] = row["interface_min_pae"]
                # df_row["interface_mean_pae"] = row["interface_mean_pae"]
                # df_row["interface_max_contact_prob"] = row["interface_max_contact_prob"]
               # df_row["pae"] = (row["pae_0_1"] + row["pae_1_0"])/2

                #diff = trgb1_emb - trgc1_emb
                # Combine embeddings into DataFrame
                emb_df = pd.DataFrame([trgb1_emb.tolist() + trgc1_emb.tolist()],
                                       columns=embeded_cols)
                #emb_df = pd.DataFrame([diff.tolist()], columns=embeded_cols)
                # Combine aggregated features + embeddings
                #df_selected = pd.concat([df_row.reset_index(drop=True), emb_df], axis=1)
                df_selected = df_row.reset_index(drop=True)

                all_data.append(df_selected)

            except Exception as e:
                print(f"Error processing {capri_ss_path}: {e}")
                continue

        if all_data:
            main_df = pd.concat(all_data, ignore_index=True)
            return main_df, embeded_cols
        else:
            # Return empty DataFrame with proper columns
            columns = [f"{col}_{stat}" for col in required_columns
                       for stat in ['mean', 'min', 'max', 'std']] + ['compatible'] + embeded_cols
            return pd.DataFrame(columns=columns), embeded_cols

    # Process both selected and remaining
    selected_all_data, embeded_cols = process_df(selected_df, is_training=False, model=model)
    remaining_all_data, _ = process_df(remaining_df, is_training=True, model=model)

    return selected_all_data, remaining_all_data, embeded_cols



def train_logreg_and_predict(selected_df: pd.DataFrame,
                             remaining_df: pd.DataFrame,
                             embeded_cols: list, c: float, pca:int):

    target_col = 'compatible'

    #trgb1_gen,trgc1_gen,pair,iptm,ptm,ranking_score,pae_0_1,pae_1_0


    X_train_num = remaining_df[
        #['euclidean', 'dot_prod'] + 
        ['desolv_mean', 'vdw_mean', 'elec_mean', 'improper_mean', 'rdcs_mean', 'rg_mean', 'sym_mean', 'total_mean'
        , 'vdw_min', 'vdw_max', 'vdw_std', 
    'elec_min', 'elec_max', 'elec_std', 'improper_min', 'improper_max', 'improper_std', 'rdcs_min', 'rdcs_max', 
    'rdcs_std', 'rg_min', 'rg_max', 'rg_std', 'sym_min', 'sym_max', 'sym_std', 'total_min', 'total_max', 'total_std'
   ]
#    ["air", "angles", "bonds", "bsa", "cdih", "coup", "dani",
#     "desolv", "dihe", "elec", "improper", "rdcs", "rg", "sym",
#     "total", "vdw", "vean", "xpcs"]
    +
    ['pae_0_1', 'pae_1_0']
   # +['pae']
    ].values          # HADDOCK features
    X_train_emb = remaining_df["cos_sim"].values.reshape(-1, 1) 
    #X_train_emb = remaining_df[embeded_cols].values          # embeddings
    y_train = remaining_df[target_col].values

    X_selected_num = selected_df[
       # ['euclidean', 'dot_prod'] + 
        ['desolv_mean', 'vdw_mean', 'elec_mean', 'improper_mean', 'rdcs_mean', 'rg_mean', 'sym_mean', 'total_mean'
        , 'vdw_min', 'vdw_max', 'vdw_std', 'elec_min', 'elec_max', 'elec_std', 'improper_min', 'improper_max', 'improper_std', 
    'rdcs_min', 'rdcs_max', 'rdcs_std', 'rg_min', 'rg_max', 'rg_std', 'sym_min', 'sym_max', 'sym_std', 'total_min', 'total_max', 'total_std'
    ]
    # ["air", "angles", "bonds", "bsa", "cdih", "coup", "dani",
    # "desolv", "dihe", "elec", "improper", "rdcs", "rg", "sym",
    # "total", "vdw", "vean", "xpcs"]
    +
    ['pae_0_1', 'pae_1_0']
   # +['pae']
    ].values
    X_selected_emb = selected_df["cos_sim"].values.reshape(-1, 1) 
    #X_selected_emb = selected_df[embeded_cols].values
    y_selected = selected_df[target_col].values

    # Scale only HADDOCK features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_selected_num_scaled = scaler.transform(X_selected_num)

   # pca_r = TruncatedSVD(n_components=pca, random_state=42)    
    #X_train_emb_pca = pca_r.fit_transform(X_train_emb)
    #X_selected_emb_pca = pca_r.transform(X_selected_emb)
    extra = remaining_df[['iptm', 'ptm', 'iptm_0_1', 'iptm_1_0', 'iptm_0_0', 'iptm_1_1', 'ranking_score']].values
    extr_selected = selected_df[['iptm', 'ptm', 'iptm_0_1', 'iptm_1_0', 'iptm_0_0', 'iptm_1_1', 'ranking_score']].values
    # Concatenate scaled numeric features + raw embeddings
    X_train_final = np.hstack([X_train_num_scaled,
    #X_train_emb,
     extra])
    X_selected_final = np.hstack([X_selected_num_scaled, 
   # X_selected_emb,
    extr_selected])
    if c is not None:
        clf = LogisticRegression(max_iter=5000, C=c, penalty='l1', solver='liblinear')#, n_jobs=-1)
        #clf = LogisticRegression(max_iter=5000, C=c)#, penalty='l1', solver='liblinear')#, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=5000)
        
    clf.fit(X_train_final, y_train)
    probs = clf.predict_proba(X_selected_final)[:, 1] 

    avg_prob_1 = probs[y_selected == 1].mean()
    avg_prob_0 = probs[y_selected == 0].mean()

    return avg_prob_1, avg_prob_0

# ---------------------- Parallel Wrapper ----------------------
def run_pipeline(original, i, percent, n_in_training, n_in_test, df, c, model, pca):
    global global_var
    global_var += 1

    selected_df, remaining_df = split_pairs(df,
                                            pos_name=original,
                                            neg_name=i,
                                            percent=percent)
    
    
    en, dva, embeded_cols = create_data_set(selected_df, remaining_df,
                                            n_in_training=n_in_training,
                                            n_in_test=n_in_test, model=model)
    if en.empty or dva.empty:
        return None

    avg_prob_1, avg_prob_0 = train_logreg_and_predict(en, dva, embeded_cols, c, pca)

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
    parser.add_argument("--c", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--pca", type=int, default=10, help="Number of PCA components")
    return parser.parse_args()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    args = parse_args()
    n_in_training = args.train
    n_in_test = args.test
    c = args.c
    model = args.model
    #df = pd.read_csv("pairs_fixed.csv") 
    df = pd.read_csv("pairs_with_scores_add.csv")
    pca = args.pca
    #df = pd.read_csv("parovi.csv")
    r = find_related_rows(df)
    

    # for n_in_training in [5]:
    #     for n_in_test in [2]:
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
                                      n_in_training, n_in_test, df, c, model, pca)
                for (original, i, percent) in tasks
            )

            results = [res for res in results if res is not None]
            results_df = pd.DataFrame(results)
            folder = f"/d/hpc/home/mm5129/A_pairs_fixed/Final/mean/model_{model}/Regularization/L1/no_emb_features&bez_haddock&alphafold&iptm_for_each_c={c}&random_state=30/csv"
            os.makedirs(folder, exist_ok=True)            
            for percent, group in results_df.groupby("percent"):
                filename = (f"{folder}/n_in_training_{n_in_training}"
                            f"_n_in_test_{n_in_test}_results_pipeline_percent_{percent}.csv")
                group.drop(columns="percent").to_csv(filename, index=False)
                print(f"Saved {filename}")
