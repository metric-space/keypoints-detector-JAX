from utils import load_model

import jax
import equinox as eqx

if __name__ == '__main__':



    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    inference_model = jax.vmap(
        inference_model, axis_name="batch"
    )


