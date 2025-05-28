## What is this ?

While working on porting code [here](https://github.com/AliaksandrSiarohin/first-order-model) from pytorch to JAX & equinox 
I was curious to know whether the keypoint extractor as is could be trained on the [CelebA dataset]() to infer keypoints on human faces

Turns out the answer is yes

![eval](./resources/eval/eval_10.png)
![eval-10](./resources/eval/eval_23.png)
![eval-20](./resources/eval/eval_34.png)


## How to run

```bash

# virtual env is recommended

pip install -e .

python scripts/training.py

```
