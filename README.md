# Refined Deep Learning for Weakly Supervised Microfluidic Cell Segmentation and Clustering with Conditional Variational Autoencoders (CVAEs)

This project is part of my internship within a collaboration between the hydrodynamics laboratory at École Polytechnique (LadHyX) and the biomedical imaging group at Télécom Paris. The goal is to explore the field of deep learning within the domain of cell biology through rigorous research and hands-on experimentation.

## Project Overview

As part of this research internship, our primary objective is to enhance Deep Learning tools meticulously trained on an internal microfluidic system. This system is designed to detect pathological conditions that provoke alterations in cell mechanics. Our research includes several challenging learning tasks, including weakly supervised cell segmentation and unsupervised cell clustering. To overcome these challenges, we rely on convolutional neural networks (CNNs) and variational autoencoders (VAEs), exploited to reveal hidden information in microfluidic imaging.

## Getting Started

### Prerequisites

Before you begin, ensure that you have the following prerequisites installed on your system:

- **Python**: You'll need Python 3.7 for running the project.

- **pip**: Make sure you have `pip`, the Python package manager, installed.

### Installation

1. **Clone the Repository**: Start by cloning this GitHub repository to your local machine. You can do this by running the following command in your terminal:

   ```bash
   git clone https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images.git
   ```

2. **Navigate to the Project Directory**: Change your current directory to the project's root directory:

   ```bash
   cd Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images
   ```

3. **Create a Virtual Environment (Optional)**: It's recommended to create a virtual environment to isolate your project dependencies. You can create one using `venv` or `virtualenv`. For example:

   ```bash
   python -m venv myenv
   ```

4. **Activate the Virtual Environment (Optional)**: If you created a virtual environment, activate it. On Windows, you can do this with:

   ```bash
   myenv\Scripts\activate
   ```

   On macOS and Linux:

   ```bash
   source myenv/bin/activate
   ```

5. **Install Dependencies**: Install the required libraries and dependencies using `pip` by running:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all the necessary packages and their specific versions listed in the `requirements.txt` file.

6. **Explore the Project**: You're now ready to explore the project. You can use the provided code examples and documentation to understand how the project functions.

#### Custom Dataset Training with cVAE

You can now use the Python script `train_cVAE.py` to train your cVAE model. Customize the following command to suit your dataset and training preferences:

```bash
sh script_run/cVAE.sh
```

- `--latent_dim`: Set the dimension of the latent space (default is 64).
- `--beta`: Adjust the Beta value for Beta-VAE (default is 1.0).
- `--epochs`: Specify the number of training epochs (default is 50).
- `--path`: Provide the path to your custom dataset.
- `--input_size`: Define the size in pixels of the input images (default is 192).
- `--saving_path`: Specify the path to save the trained model (default is './models/').

Example usage:

```bash
python train_cVAE.py --latent_dim 64 --beta 1.0 --epochs 50 --path ./custom_dataset/ --input_size 192 --saving_path ./trained_models/
```

Please ensure that your custom dataset adheres to the specified structure and preprocessing requirements described in the "Dataset Preparation" section below.

#### Dataset Preparation

To prepare your custom dataset for training a customized conditional VAE, ensure that it follows this structure:

- Place your images in a folder.
- Organize images into subfolders, each representing a category ( a mutation in our case).
- Resize images to the desired input size (e.g., 192x192 pixels in our case).

Example dataset structure:

```
custom_dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

When running the script, use the `--path` argument to specify the path to your custom dataset.

#### Results and Output

After training, the cVAE model will be saved in the directory specified by `--saving_path`. Depending on your use case, additional results or evaluation metrics may be provided.



## References
<a id="1">[1]</a> 
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). 
Cellpose: a generalist algorithm for cellular segmentation. 
Nature methods, 18(1), 100-106.

**Research Internship - 2023 at Télécom Paris**
