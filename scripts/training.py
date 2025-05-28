from dataclasses import dataclass
from functools import partial


import equinox as eqx
import jax
import jax.numpy as jnp


import dataset
import debugging
import model
import utils
from config import Config


import orbax
import optax


@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(model, x, y, state):
    model = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    pred, state = model(x, state)

    loss1 = jnp.mean((utils.batch_softargmax_heatmaps(pred) - y) ** 2)
    return loss1, (state, pred)


@eqx.filter_jit
def make_step(model, state, opt_state, x, y, optimizer):
    (loss, aux), grads = loss_fn(model, x, y, state)
    state, pred = aux
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, state, pred, grads


if __name__ == "__main__":

    config = Config()

    train_loader, test_loader = dataset.celeba_train_test_dataloaders(
        directory=config.dataset_directory,
        max_samples=config.max_samples,
        data_seed=config.data_seed,
        batch_size=config.batch_size,
    )

    batch_size = config.batch_size

    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.lr, decay_steps=config.lr_decay_steps, alpha=config.lr_alpha
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    key = jax.random.PRNGKey(config.nn_seed)

    model, state = eqx.nn.make_with_state(model.HourGlass)(
        key=key,
        block_expansion=config.block_expansion,
        in_features=config.input_channels,
        out_features=config.output_channels,
        num_blocks=config.num_blocks,
        max_features=config.max_features,
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    sample_index = 12

    for step, (batch_x, batch_y) in zip(range(config.steps), train_loader):

        model, opt_state, loss, state, pred, grads = make_step(
            model, state, opt_state, batch_x, batch_y, optimizer
        )

        # print(debugging.named_grad_norms(grads))

        print(f"> Log: step-count {step}  with loss = ", loss)

        if step % 10 == 0:
            debugging.visualize_training(
                batch_x[sample_index],
                batch_y[sample_index],
                pred[sample_index],
                directory="train",
                filename=f"train_{step}_{sample_index}",
            )

            print(
                f"Evaluation : accuracy ",
                debugging.evaluate_model(model, state, test_loader),
            )

    print("Saving model to ./models")
    utils.save_model(model, state, "keypoints.eqx", directory="./models")
