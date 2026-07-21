

### Installation ###

Follow these steps to install the `deep_learning_course` package and all required dependencies.

**1. Clone the repository**

```bash
git clone https://github.com/mehtimans/deep_learning_course.git
cd deep_learning_course
```

**2. Install PyTorch**

PyTorch and torchvision must be installed manually. 
Find the correct installation command for your OS, Python version, and CUDA setup here:

https://pytorch.org/get-started/locally/


**3. Install the package using setup.py**

Navigate to the `deep_learning_course` directory (the project root that contains `setup.py`) and run:

```bash
pip install -e .
```

This installs the `deep_learning_course` package and its required Python dependencies.

---

### Usage ###

After installing the package and PyTorch, you can run the homework scripts directly from the command line.

#### Running Autoencoder experiment

From the `experiments` directory:

```bash
python autoencoder_training.py \
    --device cuda \
    --epochs 1000 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --add_noise true \
    --val_split 0.2
```

This trains the **Autoencoder model** with:
- GPU (`cuda`)
- 1000 training epochs
- Batch size 128
- Learning rate 0.001
- Optional Gaussian noise added to inputs
- 20% validation split



#### Running MLP Classification 

Also from the `experiments` directory:

```bash
python mlp_training.py \
    --device cuda \
    --epochs 1000 \
    --learning_rate 0.001 \
    --add_noise true
```

This trains the **MLP classifier** on the provided dataset using:
- GPU (`cuda`)
- 1000 epochs
- Learning rate 0.001
- Optional input noise

---

### Repository Structure ###

The repository is organized to keep experiments, data, and outputs separated.

- **logs/** 
  When you run an experiment, the code automatically saves outputs such as training and validation plots, model checkpoints, configuration files, and TensorBoard logs in this directory. The save location can be changed in the experiment configuration if needed.

- **resources/** 
  This directory contains all datasets and external files required for the experiments. Each homework or experiment loads its required data from this folder.


