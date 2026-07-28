import os
from mmengine.config import read_base
from torch.utils.data import default_collate

from compressor.datasets.evaluator import CompressorEvaluator
from compressor.datasets.pt_dataset import CsvPtDataset
from compressor.datasets.transforms import LoadNpzFile, PackCompressorInputs
from compressor.models.autoencoder import PairAutoencoder

with read_base():
    from .base import *  # noqa
    from .base import default_hooks

# ---- Model ----
model = dict(
    type=PairAutoencoder,
    in_channels=128,
    bottleneck_dim=16,

    encoder_channel_dims=[128, 64, 32, 16],
    decoder_channel_dims=[16, 32, 64, 128],
    
    num_transformer_blocks=4,
    num_attention_heads=4,
    spatial_encoding_dim=4,

    dropout=0.1,

    distogram_head_weight_path=os.environ["DISTOGRAM_HEAD_WEIGHT_PATH"],
    normalization_stats_path=os.environ["NORMALIZATION_STATS_PATH"],
    loss_weights=dict(mse=1.0, cosine=1.0, distogram=1.0),
)

# ---- Dataset ----
train_pipeline = [
    dict(type=LoadNpzFile),
    dict(type=PackCompressorInputs),
]
test_pipeline = [
    dict(type=LoadNpzFile),
    dict(type=PackCompressorInputs),
]

train_dataloader = dict(
    batch_size=1,     # per GPU batch size
    num_workers=4,
    dataset=dict(
        type=CsvPtDataset,
        ann_file=os.environ["TRAIN_ANN_FILE"], 
        path_prefix=os.environ.get("TRAIN_DATA_ROOT", ""),
        pipeline=train_pipeline,
    ),
    sampler=dict(type="InfiniteSampler", shuffle=True),
    collate_fn=dict(type=default_collate),
)

val_dataloader = dict(
    batch_size=1,     # per GPU batch size
    num_workers=4,
    dataset=dict(
        type=CsvPtDataset,
        ann_file=os.environ["VAL_ANN_FILE"],
        path_prefix=os.environ.get("VAL_DATA_ROOT", ""),
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type=default_collate),
)
test_dataloader = val_dataloader

# ---- Evaluator ----
val_evaluator = dict(type=CompressorEvaluator)
test_evaluator = dict(type=CompressorEvaluator)

# ---- Optimizer ----
lr = 3e-4
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=1e-5),
    clip_grad=dict(max_norm=1.0),
)

# ---- Scheduler ----
max_iters = 300_000
val_interval = 5_000
ckpt_interval = 5_000
warmup_iters = 5_000

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=warmup_iters,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_iters - warmup_iters,
        eta_min=lr * 0.01,
        begin=warmup_iters,
        end=max_iters,
        by_epoch=False,
    ),
]

# ---- Training ----
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval,
)

default_hooks.update(
    dict(
        logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False,),
        checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=ckpt_interval, max_keep_ckpts=3),
    )
)

log_processor = dict(by_epoch=False)

work_dir = "output/debug"
resume = False
randomness = dict(seed=0, deterministic=False)
