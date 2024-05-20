import wandb
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Logger configuration -------
logging.basicConfig(
    filename='./example3.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df = df.sort_values(["YrSold", "MoSold"]).reset_index(drop=True)
    cols_to_keep = [
        "MoSold",
        "YrSold",
        "KitchenAbvGr",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "FullBath",
        "LotArea",
        "BldgType",
        "YearBuilt",
    ]
    target_var = "SalePrice"

    X = df[cols_to_keep].copy().reset_index(drop=True)
    y = df[target_var].copy().reset_index(drop=True)

    _tfm_dict = {k: i for i, k in enumerate(X.BldgType.unique())}
    X["BldgType"] = X.BldgType.map(_tfm_dict)

    n = int(len(X) * 0.8)
    X_train, y_train = X[:n], y[:n]
    X_test, y_test = X[n:], y[n:]
    return X_train, X_test, y_train, y_test

def within_10(model, X_test, y_test):
    preds = model.predict(X_test)
    err = np.abs((preds / y_test) - 1)
    w10 = err < 0.1
    return w10.mean().round(3)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to the data file')
    parser.add_argument('--n_estimators',
                        type=int,
                        required=True,
                        help='Number of estimators for the RandomForest')
    parser.add_argument('--max_depth',
                        type=int,
                        required=True,
                        help='Max depth of the RandomForest')
    parser.add_argument('--min_samples_split',
                        type=int,
                        required=True,
                        help='Min samples split for the RandomForest')
    parser.add_argument('--min_samples_leaf',
                        type=int,
                        required=True,
                        help='Min samples leaf for the RandomForest')
    args = parser.parse_args()

    run = wandb.init(project='RF')
    config = wandb.config

    X_train, X_test, y_train, y_test = prepare_data(args.data_path)

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    w10 = within_10(model, X_test, y_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    wandb.log(
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "within_10": w10,
            "r2": r2,
            "mae": mae
        }
    )

    logger.info(f'n_estimators: {args.n_estimators}, max_depth: {args.max_depth}, min_samples_split: {args.min_samples_split}, min_samples_leaf: {args.min_samples_leaf}')
    logger.info(f'within_10: {w10}, r2: {r2}, mae: {mae}')

    with open('results.txt', 'a') as f:
        f.write(f'n_estimators: {args.n_estimators}, max_depth: {args.max_depth}, min_samples_split: {args.min_samples_split}, min_samples_leaf: {args.min_samples_leaf}\n')
        f.write(f'within_10: {w10}, r2: {r2}, mae: {mae}\n\n')

if __name__ == "__main__":
    main()
