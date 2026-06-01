"""
lstm.py
-------
LSTM deep learning model for hourly Power_Load_kW forecasting.

Input data
----------
Sequences built from the full feature matrix of hourly-resampled
PowerLoad_Dataset.csv. All 8 original dataset columns + engineered
features are used. Target (Power_Load_kW) is column 0 of the scaled array.

Architecture options
--------------------
'stacked'  – stacked (Bi-)LSTM with optional attention
'seq2seq'  – encoder-decoder seq2seq for multi-step output

Training
--------
- MinMaxScaler fitted on training data
- EarlyStopping + ReduceLROnPlateau
- ModelCheckpoint saves the best epoch
- TensorBoard logging (optional)
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_data, train_test_split
from data.preprocessing import build_feature_matrix, create_sequences

TARGET     = "Power_Load_kW"
MODEL_PATH = Path(__file__).parent / "lstm_model.keras"
SCALER_PATH= Path(__file__).parent / "lstm_scaler.pkl"

# ── TF import (soft) ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow not installed — LSTMForecaster unavailable.")


# ── Attention helper ──────────────────────────────────────────────────────────

def _attention_block(x):
    """Bahdanau-style soft attention over LSTM output sequence."""
    score   = layers.Dense(1, activation="tanh")(x)      # (batch, T, 1)
    weights = layers.Softmax(axis=1)(score)               # (batch, T, 1)
    context = x * weights
    return layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context)


# ── Architecture builders ─────────────────────────────────────────────────────

def build_stacked_lstm(
    seq_len:          int,
    n_features:       int,
    horizon:          int,
    units:            list[int]  = [128, 64],
    dropout:          float      = 0.2,
    recurrent_dropout:float      = 0.1,
    bidirectional:    bool       = True,
    attention:        bool       = True,
) -> "keras.Model":
    inp = layers.Input(shape=(seq_len, n_features), name="input")
    x   = inp

    for i, u in enumerate(units):
        lstm_layer = layers.LSTM(
            u, return_sequences=True,
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            name=f"lstm_{i+1}",
        )
        x = (layers.Bidirectional(lstm_layer, name=f"bilstm_{i+1}")(x)
             if bidirectional else lstm_layer(x))
        x = layers.BatchNormalization()(x)

    x   = _attention_block(x) if attention else layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon, name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LSTM_Stacked")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse", metrics=["mae"],
    )
    return model


def build_seq2seq(
    seq_len:   int,
    n_features:int,
    horizon:   int,
    units:     int   = 128,
    dropout:   float = 0.2,
) -> "keras.Model":
    enc_inp = layers.Input(shape=(seq_len, n_features), name="encoder_input")
    _, state_h, state_c = layers.LSTM(
        units, return_state=True, dropout=dropout, name="encoder"
    )(enc_inp)

    dec_inp = layers.RepeatVector(horizon)(state_h)
    dec_out = layers.LSTM(units, return_sequences=True, dropout=dropout,
                          name="decoder")(dec_inp, initial_state=[state_h, state_c])
    dec_out = layers.TimeDistributed(layers.Dense(32, activation="relu"))(dec_out)
    output  = layers.TimeDistributed(layers.Dense(1))(dec_out)
    output  = layers.Reshape((horizon,))(output)

    model = keras.Model(inputs=enc_inp, outputs=output, name="LSTM_Seq2Seq")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


# ── Model class ───────────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    LSTM load forecaster using PowerLoad_Dataset features.

    Parameters
    ----------
    seq_len       : look-back window in hours (default 168 = 1 week)
    horizon       : forecast horizon in hours (default 24)
    architecture  : 'stacked' | 'seq2seq'
    units         : list of units (stacked) or single int (seq2seq)
    bidirectional : use Bidirectional LSTM (stacked only)
    attention     : add attention pooling (stacked only)
    dropout       : dropout rate
    batch_size    : mini-batch size
    max_epochs    : training epochs ceiling
    patience      : EarlyStopping patience
    """

    def __init__(
        self,
        seq_len:       int   = 168,
        horizon:       int   = 24,
        architecture:  str   = "stacked",
        units                = [128, 64],
        bidirectional: bool  = True,
        attention:     bool  = True,
        dropout:       float = 0.2,
        batch_size:    int   = 64,
        max_epochs:    int   = 100,
        patience:      int   = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("Install TensorFlow: pip install tensorflow")

        self.seq_len       = seq_len
        self.horizon       = horizon
        self.architecture  = architecture
        self.units         = units
        self.bidirectional = bidirectional
        self.attention     = attention
        self.dropout       = dropout
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience

        self.model_      = None
        self.scaler_     = MinMaxScaler(feature_range=(-1, 1))
        self.history_    = None
        self.n_features_ = None

    # ── data preparation ──────────────────────────────────────────────────────

    def _prepare(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series   | np.ndarray,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale then build (X_seq, y_seq) arrays.
        Power_Load_kW must be column 0 of the scaled array
        (used for inverse-transform during predict).
        """
        Xv = X.values if isinstance(X, pd.DataFrame) else X
        yv = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)

        # Stack target as column 0, features as columns 1..n
        data = np.hstack([yv, Xv])

        data = self.scaler_.fit_transform(data) if fit_scaler \
               else self.scaler_.transform(data)

        self.n_features_ = data.shape[1]
        return create_sequences(data, self.seq_len, self.horizon)

    def _inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """
        Inverse-transform the target column (col 0) only.
        scaled shape: (n_samples,) or (n_samples, horizon)
        """
        mn = self.scaler_.data_min_[0]
        mx = self.scaler_.data_max_[0]
        return scaled * (mx - mn) + mn

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series   | np.ndarray,
        X_val:   pd.DataFrame | np.ndarray | None = None,
        y_val:   pd.Series   | np.ndarray | None = None,
        log_dir: str | None = None,
    ) -> "LSTMForecaster":
        """Train the LSTM model on the PowerLoad feature matrix."""
        X_seq, y_seq = self._prepare(X_train, y_train, fit_scaler=True)
        n_feat = X_seq.shape[2]

        # Build model
        if self.architecture == "seq2seq":
            u = self.units if isinstance(self.units, int) else self.units[0]
            self.model_ = build_seq2seq(self.seq_len, n_feat, self.horizon,
                                        u, self.dropout)
        else:
            u_list = ([self.units] if isinstance(self.units, int)
                      else self.units)
            self.model_ = build_stacked_lstm(
                self.seq_len, n_feat, self.horizon,
                u_list, self.dropout, 0.1,
                self.bidirectional, self.attention,
            )

        self.model_.summary(line_length=80)

        # Validation sequences
        val_data = None
        if X_val is not None and y_val is not None:
            X_v, y_v = self._prepare(X_val, y_val, fit_scaler=False)
            val_data  = (X_v, y_v)

        # Callbacks
        monitor = "val_loss" if val_data else "loss"
        cb = [
            callbacks.EarlyStopping(monitor=monitor, patience=self.patience,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                        patience=5, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True,
                                      verbose=0),
        ]
        if log_dir:
            cb.append(callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

        print(f"\n[LSTM] Training {self.architecture}  "
              f"seq={self.seq_len}h  horizon={self.horizon}h  "
              f"samples={len(X_seq)} …")
        self.history_ = self.model_.fit(
            X_seq, y_seq,
            validation_data=val_data,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=1,
        )
        best_val = min(self.history_.history.get("val_loss", [float("inf")]))
        print(f"[LSTM] Training done.  Best val_loss={best_val:.6f}")
        return self

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series   | np.ndarray,
    ) -> np.ndarray:
        """
        Forecast the next `horizon` hours for each sliding window in (X, y).

        Returns
        -------
        np.ndarray  shape (n_windows, horizon)  — unscaled kW values
        """
        X_seq, _ = self._prepare(X, y, fit_scaler=False)
        raw = self.model_.predict(X_seq, verbose=0)   # (n, horizon)
        return self._inverse_target(raw)

    def predict_one_step(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series   | np.ndarray,
    ) -> np.ndarray:
        """Return only the first horizon step for each window."""
        return self.predict(X, y)[:, 0]

    # ── training history ─────────────────────────────────────────────────────

    def plot_history(self) -> None:
        if self.history_ is None:
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.history_.history["loss"], label="Train loss")
        if "val_loss" in self.history_.history:
            ax.plot(self.history_.history["val_loss"], label="Val loss")
        ax.set_title(f"LSTM ({self.architecture}) Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).parent / "lstm_training_history.png"
        fig.savefig(out, dpi=120)
        plt.close()
        print(f"[LSTM] History → {out}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        model_path:  str | Path = MODEL_PATH,
        scaler_path: str | Path = SCALER_PATH,
    ) -> None:
        self.model_.save(str(model_path))
        joblib.dump({
            "scaler": self.scaler_, "seq_len": self.seq_len,
            "horizon": self.horizon, "architecture": self.architecture,
            "n_features": self.n_features_,
        }, scaler_path)
        print(f"[LSTM] Model  → {model_path}")
        print(f"[LSTM] Scaler → {scaler_path}")

    @classmethod
    def load(
        cls,
        model_path:  str | Path = MODEL_PATH,
        scaler_path: str | Path = SCALER_PATH,
    ) -> "LSTMForecaster":
        s = joblib.load(scaler_path)
        inst = cls(seq_len=s["seq_len"], horizon=s["horizon"],
                   architecture=s["architecture"])
        inst.model_      = keras.models.load_model(str(model_path))
        inst.scaler_     = s["scaler"]
        inst.n_features_ = s["n_features"]
        print(f"[LSTM] Loaded ← {model_path}")
        return inst

    def summary(self) -> str:
        if self.model_ is None:
            return "LSTMForecaster — not built yet."
        lines = [f"LSTMForecaster({self.architecture})  "
                 f"seq={self.seq_len}h  horizon={self.horizon}h"]
        self.model_.summary(print_fn=lambda s: lines.append("  " + s))
        return "\n".join(lines)


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("Install TensorFlow to run this module.")
        sys.exit(1)

    df = load_data()
    train_df, test_df = train_test_split(df, test_months=3)
    X_train, y_train  = build_feature_matrix(train_df)
    X_test,  y_test   = build_feature_matrix(test_df)

    # 10 % of train as validation
    n_val = int(0.1 * len(X_train))
    X_v, y_v = X_train.iloc[-n_val:], y_train.iloc[-n_val:]
    X_t, y_t = X_train.iloc[:-n_val], y_train.iloc[:-n_val]

    lstm = LSTMForecaster(
        seq_len=168, horizon=24,
        architecture="stacked",
        units=[128, 64], bidirectional=True, attention=True,
        dropout=0.2, batch_size=64, max_epochs=50, patience=8,
    )
    lstm.fit(X_t, y_t, X_v, y_v)
    lstm.plot_history()

    preds = lstm.predict_one_step(X_test, y_test)
    offset = lstm.seq_len

    from evaluation.metrics import compute_metrics, print_metrics
    n = min(len(y_test.values) - offset, len(preds))
    m = compute_metrics(y_test.values[offset: offset + n], preds[:n])
    print_metrics(m, "LSTM")
    lstm.save()