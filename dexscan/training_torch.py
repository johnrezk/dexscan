import bz2
import gc
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import cattrs
import keras
import numpy as np
import orjson
import pandas as pd
import polars as pl
import tensorflow as tf
import torch
import torch.nn as nn
from attrs import frozen
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Input, LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers.schedules import LearningRateSchedule
from keras.regularizers import L2
from keras.utils import register_keras_serializable, to_categorical
from rich.progress import track
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset

from dexscan.constants import PROJECT_DIR
from dexscan.sampling import (
    SAMPLES_DIR,
    TOTAL_CLASSES,
    InputData,
    get_data_polars_schema,
    get_scores_dir,
)

# https://keras.io/examples/structured_data/structured_data_classification_with_feature_space/
# https://stackoverflow.com/questions/62484768/how-to-choose-the-number-of-units-for-the-dense-layer-in-the-convoluted-neural-n
# https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
# https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
# https://keras.io/getting_started/intro_to_keras_for_engineers/
# https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs
# https://www.quora.com/What-is-the-number-one-cause-of-low-accuracy-when-training-a-machine-learning-model
# https://christophm.github.io/interpretable-ml-book/feature-importance.html
# https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
# https://colab.research.google.com/drive/1QJjb6YtTqgsojYYtzqvVUCCSQtreEGbj?usp=sharing
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-binary-categorical-crossentropy-with-keras.md
# https://github.com/aldente0630/gauss-rank-scaler
# https://stats.stackexchange.com/questions/258166/good-accuracy-despite-high-loss-value/448033#448033
# https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy
# https://cs231n.github.io/neural-networks-2/

BATCH_SIZE = 48
MODEL_DIR = PROJECT_DIR / "model"
LGBM_MODEL_DIR = PROJECT_DIR / "model-lgbm"

# TYPES


FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


# CLR IMPLEMENTATION


@register_keras_serializable()
class CyclicalLearningRate(LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule."""

    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        step_size: FloatTensorLike,
        scale_fn: Callable,
        scale_mode: str = "cycle",
        name: str = "CyclicalLearningRate",
    ):
        """Applies cyclical schedule to the learning rate.

        See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


        ```python
        lr_schedule = tf.keras.optimizers.schedules.CyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_fn=lambda x: 1.,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.legacy.Optimizer` as the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial learning rate.
            maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The maximum learning rate.
            step_size: A scalar `float32` or `float64` `Tensor` or a
                Python number. Step size denotes the number of training iterations it takes to get to maximal_learning_rate.
            scale_fn: A function. Scheduling function applied in cycle
            scale_mode: ['cycle', 'iterations']. Mode to apply during cyclic
                schedule
            name: (Optional) Name for the operation.

        Returns:
            Updated learning rate value.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))  # type: ignore
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)  # type: ignore

            mode_step = cycle if self.scale_mode == "cycle" else step

            lr_dif = maximal_learning_rate - initial_learning_rate  # type: ignore
            lr_max = tf.maximum(tf.cast(0, dtype), (1 - x))  # type: ignore
            return initial_learning_rate + lr_dif * lr_max * self.scale_fn(mode_step)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


# MY CODE


class CustomDataset(Dataset):
    def __init__(self, features: torch.Tensor, target: torch.Tensor):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


def df_to_dataloader(df: pd.DataFrame) -> DataLoader:
    df = df.copy()

    y = df.pop("score").to_numpy(dtype="int32")
    x = df.to_numpy(dtype="float32")

    y_tensor = torch.tensor(y)
    x_tensor = torch.tensor(x)

    ds = CustomDataset(x_tensor, y_tensor)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def load_parquets(files: Iterable[Path]) -> pl.DataFrame:
    full_df: pl.DataFrame | None = None
    for parquet_file in track(files, "loading parquet files", transient=True):
        partial_df = pl.read_parquet(parquet_file)
        if full_df is None:
            full_df = partial_df
        else:
            full_df.vstack(partial_df, in_place=True)
    assert full_df is not None
    return full_df


def load_json_parquets(files: Iterable[Path]) -> pl.DataFrame:
    full_df: pl.DataFrame | None = None
    for parquet_file in track(files, "loading parquet files", transient=True):
        assert parquet_file.name.endswith(".json.bz2")
        partial_df = pl.DataFrame(
            data=orjson.loads(bz2.decompress(parquet_file.read_bytes())),
            schema=get_data_polars_schema(),
        )
        if full_df is None:
            full_df = partial_df
        else:
            full_df.vstack(partial_df, in_place=True)
    assert full_df is not None
    return full_df


@frozen
class InputNorm:
    mean: float
    std: float


@frozen
class Normalizer:
    map: dict[str, InputNorm]

    @classmethod
    def from_df(cls, df: pl.DataFrame) -> "Normalizer":
        m: dict[str, InputNorm] = {}
        skip_cols: set[str] = {"pair_addr", "block_num", "score"}
        for col_name, pl_type in df.schema.items():
            if col_name in skip_cols:
                continue
            assert pl_type == pl.Float32
            feat_series = df[col_name]
            mean = feat_series.mean()
            std = feat_series.std()
            assert isinstance(mean, float) and isinstance(std, float)
            m[col_name] = InputNorm(mean=mean, std=std)
        return cls(map=m)

    @classmethod
    def from_json_file(cls, p: Path) -> "Normalizer":
        return cls(
            map=cattrs.structure(orjson.loads(p.read_bytes()), dict[str, InputNorm])
        )

    def to_json_file(self, p: Path) -> None:
        p.write_bytes(orjson.dumps(cattrs.unstructure(self.map)))

    def preprocess(self, input_df: pl.DataFrame) -> pl.DataFrame:
        df = input_df.with_columns(
            **{
                col_name: ((pl.col(col_name) - tune.mean) / tune.std)
                for col_name, tune in self.map.items()
            }
        )

        valid_col_names = set(self.map.keys())
        valid_col_names.add("score")
        all_col_names = set(df.columns)
        invalid_col_names = all_col_names - valid_col_names
        for col_name in invalid_col_names:
            df.drop_in_place(col_name)

        return df


@frozen
class Prediction:
    chop: float
    bull: float
    bear: float

    @classmethod
    def placeholder(cls) -> "Prediction":
        return cls(0, 0, 0)

    def rolling_avg(self, p: "Prediction", n: int) -> "Prediction":
        if n < 1:
            raise ValueError("n must be >= 1")
        nm1 = n - 1
        return Prediction(
            chop=(self.chop * nm1 + p.chop) / n,
            bull=(self.bull * nm1 + p.bull) / n,
            bear=(self.bear * nm1 + p.bear) / n,
        )


@frozen
class KerasModelWrapper:
    model: keras.Model
    normalizer: Normalizer

    def predict(self, i: InputData) -> Prediction:
        norm_df = self.normalizer.preprocess(i.to_df())
        input_tensor = tf.constant(norm_df.to_numpy(), "float32")
        res = self.model(input_tensor, training=False)

        raw_values = [float(res[0][x]) for x in range(TOTAL_CLASSES)]  # type: ignore

        # dif = 0 - min(0, *raw_values)
        # denom = max(1, *raw_values) + dif

        # def norm(v: float) -> float:
        #     return (v + dif) / denom

        # norm_values = [norm(val) for val in raw_values]

        return Prediction(
            chop=raw_values[0],
            bull=raw_values[1],
            bear=raw_values[2],
        )


conv = cattrs.Converter()
conv.register_structure_hook(Decimal, lambda v, _: Decimal(v))
conv.register_unstructure_hook(Decimal, str)


def get_normalizer(model_name: str, raw_df: pl.DataFrame | None = None) -> Normalizer:
    normalizer_file = MODEL_DIR / f"{model_name}-norm.json"
    if normalizer_file.exists():
        return Normalizer.from_json_file(normalizer_file)
    if raw_df is None:
        raise RuntimeError(
            f"no normalizer found for model {model_name} and no raw_df provided"
        )
    normalizer = Normalizer.from_df(raw_df)
    normalizer.to_json_file(normalizer_file)
    return normalizer


@frozen
class LoadedData:
    training_df: pd.DataFrame
    validation_df: pd.DataFrame


def load_all_data(
    model_name: str,
    mod_func: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> LoadedData:
    score_files = list(get_scores_dir(model_name).glob("*.parquet"))
    score_df = load_parquets(score_files)

    print("chop cnt :", len(score_df.filter(pl.col("score") == 0)))
    print("bull cnt :", len(score_df.filter(pl.col("score") == 1)))
    print("bear cnt :", len(score_df.filter(pl.col("score") == 2)))

    pair_addrs = set([sf.stem for sf in score_files])
    data_files = [
        p for p in SAMPLES_DIR.glob("*.json.bz2") if p.name.split(".")[0] in pair_addrs
    ]
    data_df = load_json_parquets(data_files)

    joined_df = data_df.join(score_df, on=("pair_addr", "block_num"))
    joined_df = joined_df.drop_nulls()

    norm_df = get_normalizer(model_name, joined_df).preprocess(joined_df)

    resampled_df = (
        norm_df.clear()
        .vstack(norm_df.filter(pl.col("score") == 0).sample(fraction=0.5, seed=3333))
        .vstack(norm_df.filter(pl.col("score") == 1))
        .vstack(norm_df.filter(pl.col("score") == 2))
        .rechunk()
    )

    if mod_func:
        resampled_df = mod_func(resampled_df)

    pandas_df = resampled_df.to_pandas()
    validation_df = pandas_df.sample(frac=0.25, random_state=424242)
    training_df = pandas_df.drop(validation_df.index)

    print("training cnt   :", len(training_df))
    print("validation cnt :", len(validation_df))

    return LoadedData(
        training_df=training_df,
        validation_df=validation_df,
    )


@frozen
class TorchDatasets:
    training: DataLoader
    training_len: int
    validation: DataLoader


def prepare_torch_datasets(model_name: str) -> TorchDatasets:
    d = load_all_data(model_name)
    return TorchDatasets(
        training=df_to_dataloader(d.training_df),
        training_len=len(d.training_df),
        validation=df_to_dataloader(d.validation_df),
    )


def get_tf_model(model_name: str) -> KerasModelWrapper:
    model_file = MODEL_DIR / f"{model_name}.keras"
    norm_file = MODEL_DIR / f"{model_name}-norm.json"
    assert model_file.exists()
    assert norm_file.exists()
    model = load_model(model_file)
    assert isinstance(model, keras.Model)
    return KerasModelWrapper(
        model=model,
        normalizer=Normalizer.from_json_file(norm_file),
    )


def prepare_model(model_name: str, weights_file: Path | None = None) -> keras.Model:
    input_cnt = len(get_normalizer(model_name).map)
    leaky_relu = LeakyReLU(alpha=0.1)

    layers: list[Any] = []
    layers.append(Input(shape=(input_cnt,)))
    for _ in range(10):
        layers.append(
            Dense(
                units=380,
                activation=leaky_relu,
                kernel_regularizer=L2(1e-1),
                kernel_initializer="he_normal",
                bias_initializer="zeros",
            )
        )
        layers.append(BatchNormalization())
    layers.append(Dense(TOTAL_CLASSES, "softmax"))

    model = Sequential(layers)
    if weights_file:
        model.load_weights(weights_file)
    return model


def prepare_torch_model(
    model_name: str,
    weights_file: Path | None = None,
) -> nn.Sequential:
    input_cnt = len(get_normalizer(model_name).map)
    leaky_alpha = 0.1
    hidden_units = 380

    layers: list[Any] = []
    layers.append(nn.Linear(input_cnt, hidden_units))
    layers.append(nn.LeakyReLU(leaky_alpha))
    layers.append(nn.BatchNorm1d(hidden_units))
    for _ in range(10):
        layers.append(nn.Linear(input_cnt, hidden_units))
        layers.append(nn.LeakyReLU(leaky_alpha))
        layers.append(nn.BatchNorm1d(hidden_units))
    layers.append(nn.Softmax(TOTAL_CLASSES))

    model = nn.Sequential(*layers)
    if weights_file:
        model.load_state_dict(torch.load(weights_file))
    return model


def save_epoch():
    model_name = input("enter model name: ")

    model_file = MODEL_DIR / f"{model_name}.keras"
    if model_file.exists():
        print(f"model {model_name} already exists")
        return

    chkpt_dir = PROJECT_DIR / "model-chkpt" / model_name
    if not chkpt_dir.exists():
        print(f"no chkpts dir found for model {model_name}")
        return

    epoch_num = input("enter epoch number: ")
    chkpt_files = list(chkpt_dir.glob(f"{epoch_num}.hdf5"))
    if len(chkpt_files) == 0:
        print("no chkpt files found")
        return
    if len(chkpt_files) > 1:
        print("multiple chkpt files found")
        return
    chkpt_file = chkpt_files[0]

    model = prepare_model(model_name, chkpt_file)
    model.save(model_file, save_format="keras")
    print("done")


def train():
    model_name = input("enter model name: ")

    MODEL_DIR.mkdir(exist_ok=True)
    model_file = MODEL_DIR / f"{model_name}.keras"
    if model_file.exists():
        raise RuntimeError(f"model {model_name} already exists")

    data = prepare_torch_datasets(model_name)
    gc.collect()

    model = prepare_torch_model(model_name)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0,
        momentum=0.7,
    )

    def f1_metric(preds, labels):
        return f1_score(labels, preds, average="micro")

    def precision_metric(preds, labels):
        return precision_score(labels, preds, average="micro")

    def recall_metric(preds, labels):
        return recall_score(labels, preds, average="micro")

    steps_per_epoch = data.training_len // BATCH_SIZE
    scheduler = CyclicLR(
        optimizer,
        base_lr=5e-10,
        max_lr=2e-5,
        step_size_up=3 * steps_per_epoch,
        step_size_down=3 * steps_per_epoch,
    )

    model.compile(
        optimizer=optimizer,
        loss=nn.CrossEntropyLoss(),
        metrics=[f1_metric, precision_metric, recall_metric],
        cyclic_lr_scheduler=scheduler,
    )

    chkpt_dir = PROJECT_DIR / "model-chkpt"
    chkpt_dir.mkdir(exist_ok=True)
    model_chkpt_dir = chkpt_dir / model_name
    model_chkpt_dir.mkdir(exist_ok=True)
    chkpt_file_fmt = model_chkpt_dir / "{epoch:03d}.hdf5"

    model.fit(
        data.training,
        validation_data=data.validation,
        class_weight={0: 10.0, 1: 35.0, 2: 1.0},
        epochs=300,
        callbacks=[
            ModelCheckpoint(
                filepath=chkpt_file_fmt.as_posix(),
                save_weights_only=True,
            ),
            CSVLogger(
                filename=model_chkpt_dir / "log.csv",
                separator=",",
                append=True,
            ),
        ],
        verbose=1,  # type: ignore
        use_multiprocessing=True,
        workers=12,
    )

    print("done")
