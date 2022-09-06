# %%
# credits:
# Flax code is adapted from https://github.com/huggingface/transformers/blob/main/examples/flax/vision/run_image_classification.py
# GAN related code are adapted from https://github.com/patil-suraj/vit-vqgan/
import inspect
import os
from functools import partial

# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from copy import deepcopy
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision.transforms as T
import wandb
from datasets import Dataset as HFDataset
from flax import jax_utils
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from lpips_j.lpips import LPIPS
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_vqgan import (
    StyleGANDiscriminator,
    StyleGANDiscriminatorConfig,
    ViTVQConfig,
    ViTVQModel,
)
from vqgan_jax.modeling_flax_vqgan import Decoder, VQModel


# %%
class EncoderImageDataset(torch.utils.data.Dataset):
    # this class was originally used to preprocess images into VQ indices
    # now we only use its load() method, and preprocess images into VQ indices on the fly
    def __init__(self, df, shape=(256, 256)):
        self.df = df
        self.shape = shape

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load(path):
        img = Image.open(path).convert("RGB").resize((256, 256))
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return img.permute(0, 2, 3, 1).numpy()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["resized_path"]
        return self.load(path)


class DecoderImageDataset(torch.utils.data.Dataset):
    def __init__(self, hfds, root=None):
        """hdfs: HFDataset"""
        self.hfds = hfds
        self.root = root

    def __len__(self):
        return len(self.hfds)

    def __getitem__(self, idx):
        example = self.hfds[idx]
        # indices = example["indices"]
        path = example["path"]
        if self.root is not None:
            path = os.path.join(self.root, path.lstrip("/"))
        orig_arr = EncoderImageDataset.load(path)
        return {
            # "indices": indices,
            "original": orig_arr,
            "name": Path(path).name,
        }

    @staticmethod
    def collate_fn(examples, return_names=False):
        res = {
            # "indices": [example["indices"] for example in examples],
            "original": np.concatenate(
                [example["original"] for example in examples], axis=0
            ),
        }
        if return_names:
            res["name"] = [example["name"] for example in examples]
        return res


def try_batch_size(fn, start_batch_size=1):
    # try batch size
    batch_size = start_batch_size
    while True:
        try:
            print(f"Trying batch size {batch_size}")
            fn(batch_size * 2)
            batch_size *= 2
        except Exception as e:
            return batch_size


# %%
def postpro(decoded_images):
    decoded_images = decoded_images.clip(0.0, 1.0)  # .reshape((-1, 256, 256, 3))
    return [
        Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        for decoded_img in decoded_images
    ]


def get_param_counts(params):
    param_counts = [k.size for k in jax.tree_util.tree_leaves(params)]
    param_counts = sum(param_counts)
    return param_counts


# %%
# paths and configs
learning_rate = 1e-4
gradient_accumulation_steps = 1
warmup_steps = 4000 * gradient_accumulation_steps
log_steps = 1 * gradient_accumulation_steps
eval_steps = 100 * gradient_accumulation_steps
log_steps = 10 * gradient_accumulation_steps
eval_steps = 100 * gradient_accumulation_steps
total_steps = 150_000 * gradient_accumulation_steps

model = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
hfds = HFDataset.from_json("/disks/datassd/arts/danbooru_image_paths_ds.json")
disc_config_path = "configs/vqgan/discriminator/config.json"
original_vqparams = deepcopy(model.params)
output_dir = Path("output-disc-warmup")
output_dir.mkdir(exist_ok=True)


def get_training_params():
    keys = ["decoder", "post_quant_conv", "quantize"]
    decoder_params = {k: v for k, v in original_vqparams.items() if k in keys}
    return deepcopy(decoder_params)


# %%
for k, v in original_vqparams.items():
    print(k, get_param_counts(v) / 1e6)
# %%
disc_config = StyleGANDiscriminatorConfig.from_pretrained(disc_config_path)
disc_model = StyleGANDiscriminator(
    disc_config,
    seed=42,
    _do_init=True,
)
lpips_fn = LPIPS()


def init_lpips(rng, image_size):
    x = jax.random.normal(rng, shape=(1, image_size, image_size, 3))
    return lpips_fn.init(rng, x, x)


# disc_params = disc_model.params
#
# %%


# encoder_params = {k: v for k, v in params.items() if k not in keys}
# lean_params = params
rng = jax.random.PRNGKey(0)
rng, dropout_rng = jax.random.split(rng)

lpips_params = init_lpips(rng, image_size=256)
params = get_training_params()

warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=learning_rate,
    transition_steps=warmup_steps + 1,  # ensure not 0
)
decay_fn = optax.linear_schedule(
    init_value=learning_rate,
    end_value=0,
    transition_steps=total_steps - warmup_steps,
)
schedule_fn = optax.join_schedules(
    schedules=[warmup_fn, decay_fn],
    boundaries=[warmup_steps],
)

optimizer = optax.adamw(learning_rate=schedule_fn)
# discriminator_optimizer
optimizer_disc = optax.adamw(learning_rate=learning_rate)
# gradient accumulation for main optimizer
optimizer = optax.MultiSteps(optimizer, gradient_accumulation_steps)

# Setup train state
class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )


state = TrainState.create(
    apply_fn=model.decode_code,
    params=jax.device_put(params),
    tx=optimizer,
    dropout_rng=dropout_rng,
)
state_disc = TrainState.create(
    apply_fn=disc_model,
    params=jax.device_put(disc_model.params),
    tx=optimizer_disc,
    dropout_rng=dropout_rng,
)


# we don't use this
def l1_loss(x, y):
    return jnp.abs(x - y).mean()


loss_fn = optax.l2_loss
# the empereically observed values from initial runs, we will scale them closer to the scale of l2 loss
scale_l2 = 0.001
scale_lpips = 0.25
# adjust scale to make the loss comparable to l2 loss
cost_l2 = 0.5
cost_lpips = scale_l2 / scale_lpips * 5
cost_gradient_penalty = 100000000  # this follows vit-vqgan repo
cost_disc = 0.01


# %%


def train_step(state, batch, state_disc):
    """Returns new_state, metrics, reconstruction"""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params, batch, dropout_rng, train=True):
        original = batch["original"]
        if "indices" in batch:
            indices = batch["indices"]
        else:
            quant_states, indices = model.encode(original)
        reconstruction = state.apply_fn(indices, params=params)
        loss_l2 = loss_fn(reconstruction, original).mean()
        disc_fake_scores = state_disc.apply_fn(
            reconstruction,
            params=state_disc.params,
            dropout_rng=dropout_rng,
            train=train,
        )
        loss_disc = jnp.mean(nn.softplus(-disc_fake_scores))
        loss_lpips = jnp.mean(lpips_fn.apply(lpips_params, original, reconstruction))

        loss = loss_l2 * cost_l2 + loss_lpips * cost_lpips + loss_disc * cost_disc
        loss_details = {
            "loss_l2": loss_l2 * cost_l2,
            "loss_lpips": loss_lpips * cost_lpips,
            "loss_disc": loss_disc * cost_disc,
        }
        return loss, (loss_details, reconstruction)

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, (loss_details, reconstruction)), grad = grad_fn(
        state.params, batch, dropout_rng, train=True
    )
    # legacy code, I didn't use multi gpu
    # grad = jax.lax.pmean(grad, "batch")

    new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

    metrics = loss_details | {"learning_rate": schedule_fn(state.step)}
    # metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics, reconstruction


# %%
def compute_stylegan_loss(
    disc_params, batch, fake_images, dropout_rng, disc_model_fn, train
):
    disc_fake_scores = disc_model_fn(
        fake_images, params=disc_params, dropout_rng=dropout_rng, train=train
    )
    disc_real_scores = disc_model_fn(
        batch, params=disc_params, dropout_rng=dropout_rng, train=train
    )
    # -log sigmoid(f(x)) = log (1 + exp(-f(x))) = softplus(-f(x))
    # -log(1-sigmoid(f(x))) = log (1 + exp(f(x))) = softplus(f(x))
    # https://github.com/pfnet-research/sngan_projection/issues/18#issuecomment-392683263
    loss_real = nn.softplus(-disc_real_scores)
    loss_fake = nn.softplus(disc_fake_scores)
    disc_loss_stylegan = jnp.mean(loss_real + loss_fake)

    # gradient penalty r1: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L63-L67
    r1_grads = jax.grad(
        lambda x: jnp.mean(
            disc_model_fn(x, params=disc_params, dropout_rng=dropout_rng, train=train)
        )
    )(batch)
    # get the squares of gradients
    r1_grads = jnp.mean(r1_grads**2)

    disc_loss = disc_loss_stylegan + cost_gradient_penalty * r1_grads
    disc_loss_details = {
        "pred_p_real": jnp.exp(-loss_real).mean(),  # p = 1 -> predict real is real
        "pred_p_fake": jnp.exp(-loss_fake).mean(),  # p = 1 -> predict fake is fake
        "loss_real": loss_real.mean(),
        "loss_fake": loss_fake.mean(),
        "loss_stylegan": disc_loss_stylegan,
        "loss_gradient_penalty": cost_gradient_penalty * r1_grads,
        "loss": disc_loss,
    }
    return disc_loss, disc_loss_details


train_compute_stylegan_loss = partial(compute_stylegan_loss, train=True)
grad_stylegan_fn = jax.value_and_grad(train_compute_stylegan_loss, has_aux=True)


def train_step_disc(state_disc, batch, fake_images):
    dropout_rng, new_dropout_rng = jax.random.split(state_disc.dropout_rng)
    # convert fake images to int then back to float, so discriminator can't cheat
    dtype = fake_images.dtype
    fake_images = (fake_images.clip(0, 1) * 255).astype(jnp.uint8).astype(dtype) / 255
    (disc_loss, disc_loss_details), disc_grads = grad_stylegan_fn(
        state_disc.params,
        batch,
        fake_images,
        dropout_rng,
        disc_model,
    )
    new_state = state_disc.apply_gradients(
        grads=disc_grads, dropout_rng=new_dropout_rng
    )
    metrics = disc_loss_details | {"learning_rate_disc": schedule_fn(state_disc.step)}
    # metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, metrics


# %%
# Take the first 100 images as validation set
train_ds = DecoderImageDataset(hfds.select(range(100, len(hfds))), root="/disks")
test_ds = DecoderImageDataset(hfds.select(range(100)), root="/disks")
# %%
jit_train_step = jax.jit(train_step)
jit_train_step_disc = jax.jit(train_step_disc)
# %%
def try_train_batch_size_fn(batch_size):
    example = train_ds[0]
    batch = train_ds.collate_fn([example] * batch_size)
    new_state, metrics, reconstruction = jit_train_step(state, batch, state_disc)
    new_state, metrics = jit_train_step_disc(
        state_disc, batch["original"], reconstruction
    )
    return


train_batch_size = 8
train_batch_size = try_batch_size(
    try_train_batch_size_fn, start_batch_size=train_batch_size
)
print(f"Training batch size: {train_batch_size}")
# %%
# try it again, make sure there is no error
try_train_batch_size_fn(train_batch_size)
print(f"Training batch size: {train_batch_size}")

# %%
def reconstruct_from_original(batch):
    original = batch["original"]
    if "indices" in batch:
        indices = batch["indices"]
    else:
        quant_states, indices = model.encode(original)
    reconstruction = state.apply_fn(indices)
    return reconstruction


# %%
wandb.init(project="vqgan")
wandb.log({"train_dataset_size": len(train_ds)})
# %%

dataloader = DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=False),
    num_workers=4,
    drop_last=True,
    prefetch_factor=4,
    persistent_workers=True,
)

# %%
state = TrainState.create(
    apply_fn=model.decode_code,
    params=jax.device_put(params),
    tx=optimizer,
    dropout_rng=dropout_rng,
)
state_disc = TrainState.create(
    apply_fn=disc_model,
    params=jax.device_put(disc_model.params),
    tx=optimizer_disc,
    dropout_rng=dropout_rng,
)
# %%
# data loader without shuffle, so we can see the progress on the same images
train_dl_eval = DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=False,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=True),
    num_workers=4,
    drop_last=True,
    prefetch_factor=4,
    persistent_workers=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=train_batch_size,
    shuffle=False,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=True),
    num_workers=4,
    drop_last=True,
    prefetch_factor=4,
    persistent_workers=True,
)
# %%
# evaluation functions
@jax.jit
def infer_fn(batch, state):
    original = batch["original"]
    if "indices" in batch:
        indices = batch["indices"]
    else:
        quant_states, indices = model.encode(original)
    reconstruction = state.apply_fn(indices, params=state.params)
    return reconstruction


def evaluate(use_tqdm=False, step=None):
    losses = []
    iterable = test_dl if not use_tqdm else tqdm(test_dl)
    for batch in iterable:
        name = batch.pop("name")
        reconstruction = infer_fn(batch, state)
        losses.append(loss_fn(reconstruction, batch["original"]).mean())
    loss = np.mean(jax.device_get(losses))
    wandb.log({"test_loss": loss, "step": step})


def log_images(dl, num_images=8, suffix="", step=None):
    logged_images = 0

    def batch_gen():
        while True:
            for batch in dl:
                yield batch

    batch_iter = batch_gen()
    while logged_images < num_images:
        batch = next(batch_iter)

        names = batch.pop("name")
        reconstruction = infer_fn(batch, state)
        left_right = np.concatenate([batch["original"], reconstruction], axis=2)

        images = postpro(left_right)
        for name, image in zip(names, images):
            wandb.log(
                {f"{name}{suffix}": wandb.Image(image, caption=name), "step": step}
            )
        logged_images += len(images)


def log_test_images(num_images=8, step=None):
    return log_images(dl=test_dl, num_images=num_images, step=step)


def log_train_images(num_images=8, step=None):
    return log_images(
        dl=train_dl_eval, num_images=num_images, suffix="|train", step=step
    )


def data_iter():
    while True:
        for batch in dataloader:
            yield batch


# %%
for steps, batch in zip(tqdm(range(total_steps)), data_iter()):
    state, metrics, reconstruction = jit_train_step(state, batch, state_disc)
    state_disc, metrics_disc = jit_train_step_disc(
        state_disc, batch["original"], reconstruction
    )
    # metrics = metrics | metrics_disc
    metrics["disc_step"] = metrics_disc
    metrics["step"] = steps
    if steps % log_steps == 1:
        wandb.log(metrics)
    if steps % eval_steps == 1:
        evaluate(step=steps)
        log_test_images(step=steps)
        log_train_images(step=steps)
        with Path(output_dir / "latest_state_disc.msgpack").open("wb") as f:
            f.write(to_bytes(jax.device_get(state_disc)))
        with Path(output_dir / "latest_state.msgpack").open("wb") as f:
            f.write(to_bytes(jax.device_get(state)))
