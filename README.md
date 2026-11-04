# A CNN from scratch on PatchCamelyon (PCam)

`pcam_cnn.ipynb` trains a small convolutional neural network, built from scratch in PyTorch, to find breast-cancer metastases in lymph-node histology patches.

## The task

[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) contains 96×96-pixel H&E-stained patches from sentinel lymph-node slides. Each patch is labelled:

- **1 (tumour):** the central 32×32 region contains at least one tumour pixel
- **0 (normal):** it doesn't

| Split | Patches |
|-------|--------:|
| train | 262,144 |
| valid |  32,768 |
| test  |  32,768 |

## Requirements

The notebook is written for **Google Colab** with a GPU.

1. Open the notebook in Colab.
2. Go to `Runtime → Change runtime type → T4 GPU`.
3. Run the cells from top to bottom.

You need:

- a Google account. The notebook mounts Google Drive to save the best model, because Colab's local disk is wiped when the session ends.
- about 7 GB of free space in Colab for the dataset download, which takes a few minutes.

The notebook installs the extra Python packages itself (`h5py`, `gdown`, `scikit-learn`, `torchinfo`, `torchview`, `graphviz`). PyTorch, torchvision, NumPy and Matplotlib come with Colab.

## What the notebook does

| Section | What happens |
|---|---|
| 1. Setup | Imports libraries, fixes random seeds, selects the GPU, defines the settings (`CFG`), mounts Google Drive and downloads PCam with `torchvision.datasets.PCAM` |
| 2. Exploratory data analysis | Shows 16 random patches with the labelled central region outlined, and checks the class balance in each split |
| 3. Augmentation & loaders | Training patches are randomly flipped, rotated by 90° and lightly colour-jittered. All patches are normalised. |
| 4. CNN architecture | `SimpleCNN` has four conv blocks (32 → 64 → 128 → 256 channels), global average pooling, dropout 0.5 and one output. A layer summary and model graph are shown. |
| 5. Training | AdamW, cosine learning-rate schedule and mixed precision. Validation AUC is checked every epoch, the best model is saved to Drive, and training stops early after 3 epochs without improvement. Loss and AUC curves are plotted. |
| 6. Test evaluation | Loads the best model and reports test AUC, accuracy, sensitivity and specificity, with an ROC curve and confusion matrix. Optional test-time augmentation averages predictions over 8 flips/rotations. |
| 7. Error analysis | Shows the 8 most confident false positives and false negatives |
| 8. Custom image | Upload your own H&E patch to get a tumour probability. A final cell downloads one tumour and one normal example patch from Hugging Face (`chehablab/MiniCAMELYON17`) to try it with. |

## Settings

All the main settings are in the `CFG` dictionary near the top:

| Setting | Default | Meaning |
|---|---|---|
| `train_fraction` | `1.0` | Fraction of the training set to use. Set it lower (e.g. `0.1`) for a quick test run. |
| `batch_size` | `256` | Patches per batch |
| `epochs` | `10` | Maximum number of training epochs |
| `lr` | `1e-3` | Learning rate |
| `weight_decay` | `1e-3` | AdamW regularisation |
| `num_workers` | `2` | Parallel data-loading workers |
| `patience` | `3` | Epochs without a validation-AUC improvement before training stops |

## Outputs

- `MyDrive/pcam_cnn/simplecnn_best.pt` in your Google Drive: the best model weights, with the epoch, validation AUC and settings
- Inline figures: sample patches, training curves, ROC curve, confusion matrix and misclassified patches

## Troubleshooting

- **The download fails with a Google Drive quota error.** This is a known problem with popular files. Wait a while and run the cell again, or use TensorFlow Datasets instead (`tfds.load("patch_camelyon", ...)`; see the end of the notebook). Its output needs a small wrapper to work with PyTorch.
- **The notebook says "no GPU".** Change the runtime type to a GPU before running. Training on CPU is very slow.
- **The validation loss rises while the training loss falls.** The model is overfitting. Use more augmentation, more dropout or more data.

## Next steps

1. Train on the full dataset for 15–20 epochs.
2. Change one thing at a time and compare validation AUC: remove augmentation, change the depth or width of the network, or change the learning rate.
3. Try transfer learning: replace `SimpleCNN` with `torchvision.models.resnet18(weights="IMAGENET1K_V1")` and change its last layer to `nn.Linear(512, 1)`.
