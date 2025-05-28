## What is this ?

While working on porting code [here](https://github.com/AliaksandrSiarohin/first-order-model) from pytorch to JAX & equinox 
I was curious to know whether the keypoint extractor as is could be trained on the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to infer keypoints on human faces

Turns out the answer is yes (these are evals, red are true keypoints and green dots are predicted)


![eval](./resources/eval/eval_10.png)
![eval-10](./resources/eval/eval_23.png)
![eval-20](./resources/eval/eval_34.png)


Not bad for about 100 steps of training and 18000 split 8:2 between training and eval (takes about 10 minutes on a machine with a 3060 Nvidia GPU)

The interesting part of this is the [Hourglass architecture]() with a convolution head

## How to run

```bash

# virtual env is recommended

pip install -e .

python scripts/training.py

python scripts/inference.py some_human_face_picture.jpeg

```

