#!/usr/bin/env python

import argparse
import pickle
import random
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

COLS = ['ts', 'next_ts', 'ttl', 'create_ts', 'freq', 'last_ts', 'update_ts', 'revals', 'good_rv', 'bad_rv', 'mime', 'size']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train cache expiry ML model')
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='Input CSV/ZST trace files to use for training')
    parser.add_argument('-o', '--output', required=True,
                        help='Output model path prefix (without extension); .pkl and .onnx will be written')
    parser.add_argument('--sample-frac', type=float, default=0.33,
                        help='Fraction of rows to sample from each input file (0 < f <= 1)')
    return parser.parse_args()


def read_trace(path: str, sample_frac: float) -> pd.DataFrame:
    if not 0 < sample_frac <= 1:
        raise ValueError('sample-frac must be in (0, 1].')

    # Skip rows on the fly to avoid loading huge files when subsampling.
    skip_fn = None if sample_frac == 1 else (lambda i: random.random() > sample_frac)
    return pd.read_csv(path, names=COLS, skiprows=skip_fn)


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    df['should_reval'] = (df['ts'] <= df['next_ts']) & (df['next_ts'] <= (df['ts'] + df['ttl'])) & (df['ttl'] < 7 * 86400)
    df['generations'] = ((df['ts'] - df['create_ts'] - 1) / df['ttl']).round()
    df['t_since_last'] = df['ts'] - df['last_ts']
    df['t_since_last_frac'] = (df['t_since_last'] - 1) / df['ttl']
    return df


def load_data(paths: Iterable[str], sample_frac: float) -> pd.DataFrame:
    frames = [read_trace(p, sample_frac) for p in paths]
    return calc_features(pd.concat(frames, ignore_index=True))


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df['should_reval']
    x = df[['ttl', 'freq', 'generations', 't_since_last', 't_since_last_frac', 'mime', 'size']]
    return x, y


def make_rfc(x_train, y_train, x_test, y_test) -> RandomForestClassifier:
    rfc = RandomForestClassifier(
        n_estimators=32,
        criterion='gini',
        min_samples_leaf=1 / 2000,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    print(f'Train acc: {accuracy_score(y_train, rfc.predict(x_train)):0.4f}')
    print(f'Test acc: {accuracy_score(y_test, y_pred):0.4f}')
    print(classification_report(y_test, y_pred))
    return rfc


def save_model(model: RandomForestClassifier, output_prefix: str, num_features: int) -> None:
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = base.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

    onnx_model = skl2onnx.to_onnx(
        model,
        initial_types=[('in', FloatTensorType([None, num_features]))], # type: ignore
        options={'zipmap': False},
        verbose=1,
    )
    onnx_path = base.with_suffix('.onnx')
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString()) # type: ignore

    print(f'Saved pickle model to {pkl_path}')
    print(f'Saved ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)')


def main() -> None:
    args = parse_args()
    df = load_data(args.inputs, args.sample_frac)

    x, y = split_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    x_train_over, y_train_over = RandomOverSampler(random_state=42).fit_resample(x_train, y_train) # type: ignore
    model = make_rfc(x_train_over, y_train_over, x_test, y_test)
    save_model(model, args.output, x_train.shape[1])


if __name__ == '__main__':
    main()
