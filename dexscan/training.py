import gc
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cattrs
import keras
import lightgbm as lgb
import numpy as np
import orjson
import pandas as pd
import polars as pl
import tensorflow as tf
from attrs import frozen
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Input, LeakyReLU
from keras.losses import CategoricalCrossentropy
from keras.metrics import F1Score, Precision, Recall
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.optimizers.schedules import LearningRateSchedule
from keras.regularizers import L2
from keras.utils import register_keras_serializable, to_categorical
from rich.progress import track

from dexscan.constants import PROJECT_DIR
from dexscan.sampling import (
    SAMPLES_DIR,
    TOTAL_CLASSES,
    InputData,
    get_data_polars_schema,
    get_score_polars_schema,
    get_scores_dir,
)
from dexscan.utils import json_to_df

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


def df_to_tf_ds(df: pd.DataFrame) -> tf.data.Dataset:
    df = df.copy()
    labels = df.pop("score")
    np_arr = df.to_numpy("float32")
    tensor_slices = (np_arr, to_categorical(labels))
    ds = tf.data.Dataset.from_tensor_slices(tensor_slices)
    ds = ds.shuffle(buffer_size=len(df))
    return ds


def df_to_lgbm_ds(df: pd.DataFrame) -> lgb.Dataset:
    cdf = df.copy()
    labels = cdf.pop("score")
    x_arr = cdf.to_numpy("float32")
    y_arr = labels.to_numpy("int32")
    return lgb.Dataset(x_arr, label=y_arr)


def load_parquets(files: Iterable[Path], schema: dict[str, Any]) -> pl.DataFrame:
    full_df: pl.DataFrame | None = None
    for json_file in track(files, "loading parquet files", transient=True):
        assert json_file.name.endswith(".json.bz2")
        partial_df = json_to_df(json_file.read_bytes(), schema)
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


@frozen
class LGBMModelWrapper:
    booster: lgb.Booster
    normalizer: Normalizer

    def predict(self, i: InputData) -> float:
        norm_df = self.normalizer.preprocess(i.to_df())
        input_arr = norm_df.to_numpy()
        res = self.booster.predict(input_arr)
        return float(res[0])  # type: ignore


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
    score_files = list(get_scores_dir(model_name).glob("*.json.bz2"))
    score_df = load_parquets(
        files=score_files,
        schema=get_score_polars_schema(),
    )

    print("chop cnt :", len(score_df.filter(pl.col("score") == 0)))
    print("bull cnt :", len(score_df.filter(pl.col("score") == 1)))
    print("bear cnt :", len(score_df.filter(pl.col("score") == 2)))

    pair_addrs = set([sf.name.split(".")[0] for sf in score_files])
    data_files = [
        p for p in SAMPLES_DIR.glob("*.json.bz2") if p.name.split(".")[0] in pair_addrs
    ]
    data_df = load_parquets(
        files=data_files,
        schema=get_data_polars_schema(),
    )

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
    validation_df = pandas_df.sample(frac=0.18, random_state=424242)
    training_df = pandas_df.drop(validation_df.index)

    print("training cnt   :", len(training_df))
    print("validation cnt :", len(validation_df))

    return LoadedData(
        training_df=training_df,
        validation_df=validation_df,
    )


@frozen
class LGBMDatasets:
    training: lgb.Dataset
    validation: lgb.Dataset


def prepare_lgbm_datasets(model_name: str) -> LGBMDatasets:
    def mod_func(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            score=pl.col("score").map_dict(
                remapping={2: 0},
                default=pl.col("score"),
            )
        )

    d = load_all_data(model_name, mod_func)
    return LGBMDatasets(
        training=df_to_lgbm_ds(d.training_df),
        validation=df_to_lgbm_ds(d.validation_df),
    )


@frozen
class TensorFlowDatasets:
    training: tf.data.Dataset
    training_len: int
    validation: tf.data.Dataset


def prepare_tf_datasets(model_name: str) -> TensorFlowDatasets:
    d = load_all_data(model_name)
    return TensorFlowDatasets(
        training=df_to_tf_ds(d.training_df).batch(BATCH_SIZE),
        training_len=len(d.training_df),
        validation=df_to_tf_ds(d.validation_df).batch(BATCH_SIZE),
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


def get_lgbm_model(model_name: str) -> LGBMModelWrapper:
    model_file = MODEL_DIR / f"{model_name}.txt"
    norm_file = MODEL_DIR / f"{model_name}-norm.json"
    assert model_file.exists()
    assert norm_file.exists()
    booster = lgb.Booster(model_file=model_file)
    return LGBMModelWrapper(
        booster=booster,
        normalizer=Normalizer.from_json_file(norm_file),
    )


def prepare_model(model_name: str, weights_file: Path | None = None) -> keras.Model:
    input_cnt = len(get_normalizer(model_name).map)
    leaky_relu = LeakyReLU(alpha=0.1)

    layers: list[Any] = []
    layers.append(Input(shape=(input_cnt,)))
    for _ in range(8):
        layers.append(
            Dense(
                units=300,
                activation=leaky_relu,
                kernel_regularizer=L2(5e-2),
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

    data = prepare_tf_datasets(model_name)
    gc.collect()

    model = prepare_model(model_name)

    steps_per_epoch = data.training_len // BATCH_SIZE
    cyclical_lr = CyclicalLearningRate(
        initial_learning_rate=5e-10,
        maximal_learning_rate=2e-5,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=3 * steps_per_epoch,
    )
    optimizer = SGD(
        learning_rate=cyclical_lr,  # type: ignore
        momentum=0.7,
    )

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[
            F1Score(name="f1", average="weighted", threshold=0.97),
            Precision(name="bull_p", class_id=1, thresholds=0.995),
            Recall(name="bull_r", class_id=1, thresholds=0.995),
            Precision(name="bear_p", class_id=2, thresholds=0.7),
            Recall(name="bear_r", class_id=2, thresholds=0.7),
        ],
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


def train_lgbm():
    model_name = input("enter model name: ")

    data = prepare_lgbm_datasets(model_name)

    nl = 3500
    bf = 0.6
    ff = 0.6

    booster = lgb.train(
        params={
            "objective": "binary",
            "num_leaves": nl,
            "learning_rate": 0.005,
            "bagging_fraction": bf,
            "feature_fraction": ff,
            "bagging_freq": 1,  # do not change
            "is_unbalance": True,
            "metric": ["auc", "binary_logloss"],
            "force_row_wise": True,
            "num_threads": 12,
        },
        num_boost_round=1_000_000,
        train_set=data.training,
        valid_sets=[data.validation],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
        ],
    )
    LGBM_MODEL_DIR.mkdir(exist_ok=True)
    model_file = LGBM_MODEL_DIR / f"{model_name}.txt"
    booster.save_model(model_file)

    print("done")
