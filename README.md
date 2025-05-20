# Generating-Realistic-Images-using-DDPM

## Abstract

This project focuses on reproducing experiments from the Denoising Diffusion Probabilistic Models (DDPM) paper. The core objective is to train a DDPM model on the CIFAR-10 dataset and compare its performance against baseline generative models, including Variational Autoencoders (VAEs), Deep Convolutional Generative Adversarial Networks (DCGANs), and a VAE-GAN Hybrid. [cite: 4, 5] The analysis aims to determine if DDPMs offer significant advantages in terms of image quality, training stability, and computational efficiency over these conventional models. [cite: 6]

## Introduction

The field of generative modeling has seen remarkable advancements, with Denoising Diffusion Probabilistic Models (DDPMs) emerging as a powerful technique for generating high-fidelity images. This project delves into the practical implementation and evaluation of DDPMs. [cite: 3] By training a DDPM on the CIFAR-10 dataset, this work aims to replicate the findings of the original DDPM paper. [cite: 4] Furthermore, the project provides a comparative analysis by training and evaluating baseline models like VAEs, DCGANs, and VAE-GAN hybrids on the same dataset. [cite: 5, 6] This comparative study is crucial for understanding the trade-offs and potential benefits of using DDPMs for realistic image generation. [cite: 6]

## Technologies and Tools

The development and evaluation of the DDPM and baseline models involved the use of several key libraries and frameworks:

* **Primary Framework:**
    * **PyTorch (torch):** Used for building and training all neural network models. [cite: 20]
* **Neural Network Components (torch.nn):** Utilized for defining various neural network layers. [cite: 20]
* **Optimization (torch.optim):** The Adam optimizer was employed for model training. [cite: 20]
* **Dataset Handling:**
    * **torchvision.datasets:** For loading the CIFAR-10 dataset. [cite: 21]
    * **torchvision.transforms:** For preprocessing images, specifically normalization. [cite: 21]
    * **torch.utils.data:** For creating data loaders to efficiently feed data to the models. [cite: 21]
* **Progress Visualization:**
    * **tqdm:** Used to display progress bars during the training process. [cite: 22]
* **Dataset:**
    * **CIFAR-10:** A widely-used benchmark dataset consisting of 50,000 $32 \times 32$ color images across 10 classes. [cite: 9, 10] Its relatively small image size makes it computationally feasible for training diffusion models and it's a well-known benchmark for generative models. [cite: 11]

## Proposed System (DDPM Model Overview)

The core of this project is the Denoising Diffusion Probabilistic Model (DDPM). DDPMs consist of two main processes: a forward diffusion process and a reverse diffusion (denoising) process.

### Forward Diffusion Process
* Noise is progressively added to clean images over a set number of timesteps (e.g., 300 in this project). [cite: 16, 33]
* This process simulates a Markovian chain that gradually corrupts the data into pure Gaussian noise. [cite: 16, 34]
* Two types of noise schedules were explored:
    * **Linear Beta Schedule:** Noise is added according to a linear schedule for beta values. The noisy image $x_t$ is obtained from the clean image $x_0$ using the formula: $q(x_{t}|x_{0})=\mathcal{N}(x_{t};\sqrt{\alpha_{t}}x_{0},(1-\alpha_{t})I)$, where $\alpha_{t}=\prod_{s=1}^{t}(1-\beta_{s})$ and $\beta_s$ is the noise variance at step s. [cite: 16, 26, 27]
    * **Cosine Beta Schedule:** Noise is added using a cosine-based beta schedule, which can provide a smoother noise transition. [cite: 40] The term $\overline{\alpha}_{t}$ is defined by a cosine function: $\overline{\alpha}_{t}=cos^{2}(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2})$, with $s=0.008$ and T being total timesteps. [cite: 39, 40] The model can directly sample $x_t$ from $x_0$ using $q(x_{t}|x_{0})=\mathcal{N}(x_{t};\sqrt{\overline{\alpha_{t}}}x_{0},(1-\overline{\alpha}_{t})I)$ for efficient training. [cite: 42]

### Reverse Diffusion Process (Model Learning)
* A neural network, typically a U-Net architecture, is trained to predict the noise that was added at each timestep during the forward process. [cite: 17, 35]
* **Model Architecture (Simple U-Net):** [cite: 23]
    * **Input:** Noisy Image + Timestep. [cite: 23]
    * **Timestep Embedding:** The timestep $t$ is embedded using a small MLP and injected into intermediate convolution layers. [cite: 23, 24]
    * **Encoder (Downsampling path):** Consists of convolutional layers to reduce spatial dimensions. [cite: 24]
    * **Bottleneck:** Further convolution operations. [cite: 24]
    * **Decoder (Upsampling path):** Uses transposed convolution layers to increase spatial dimensions back to the original image size. [cite: 25]
    * For the cosine schedule model, key U-Net modules include Sinusoidal Time Embedding, Downsampling/upsampling layers, and Residual + Attention blocks. [cite: 38, 39]
* **Model Prediction & Loss Function:**
    * The model $\epsilon_{\theta}(x_{t},t)$ is trained to predict the added noise $\epsilon$. [cite: 28, 43]
    * The loss function is typically the Mean Squared Error (MSE) between the true noise and the predicted noise: $\mathcal{L}=\mathbb{E}_{x_{0},\epsilon,t}[||\epsilon-\epsilon_{\theta}(x_{t},t)||^{2}]$. [cite: 28, 44]
    * This loss can be interpreted as learning the score function (gradient of log probability) for the data distribution at each noise level. [cite: 46]

### Sampling (Image Generation)
* Image generation starts from random noise $x_T \sim \mathcal{N}(0,I)$. [cite: 29]
* The trained model iteratively denoises this random noise over the same number of timesteps, reversing the diffusion process to reconstruct a clean image. [cite: 18, 36]
* The formula for iterative denoising is: $x_{t-1}=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon_{\theta}(x_{t},t))+\sigma_{t}z$, where $z \sim \mathcal{N}(0,I)$ and $\sigma_t$ controls randomness. [cite: 29]
* For the cosine schedule, the reverse transition is $p_{\theta}(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t),\sigma_{t}^{2}I)$, with the mean $\mu_{\theta}(x_{t},t)=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}}}\epsilon_{\theta}(x_{t},t))$. [cite: 47, 48]

## Baseline Models

To provide a comparative context for the DDPM's performance, the following baseline models were also implemented and trained on the CIFAR-10 dataset: [cite: 5, 6, 19]

### 1. Deep Convolutional GAN (DCGAN) [cite: 50]
* **Architecture:**
    * **Generator:** Uses transposed convolutional layers to map a random noise vector (latent dimension: 100) to a $3 \times 32 \times 32$ image. [cite: 50, 53]
    * **Discriminator:** Employs convolutional layers to classify input images as real (from the dataset) or fake (from the generator). [cite: 51]
* **Training:**
    * **Loss Function:** Adversarial Loss (Minimax game, typically Binary Cross-Entropy). [cite: 52]
    * **Hyperparameters:** Latent Dimension = 100, Learning Rate = $2 \times 10^{-4}$ (Adam optimizers), Epochs = 100, Batch Size = 128. [cite: 53]
* **Evaluation:** Qualitative assessment of generated images and quantitative evaluation using Fréchet Inception Distance (FID) score. (Reported FID: ~47.25) [cite: 54]

### 2. Variational Autoencoder (VAE) [cite: 55]
* **Architecture:**
    * **Encoder:** Convolutional layers map input images ($3 \times 32 \times 32$) to a latent distribution (mean and variance). [cite: 55]
    * **Decoder:** Transposed convolutional layers map samples from the latent distribution back to image space ($3 \times 32 \times 32$). [cite: 56]
* **Training:**
    * **Loss Function:** Combined Reconstruction Loss (e.g., MSE or BCE) and KL Divergence (to regularize the latent space). [cite: 57]
    * **Hyperparameters:** Latent Dimension = 128, Learning Rate = $1 \times 10^{-3}$, Epochs = 100, Batch Size = 128. [cite: 58]
* **Evaluation:** Qualitative assessment and FID score. (Reported FID: ~189) [cite: 59]

### 3. VAE-GAN Hybrid [cite: 60]
* **Architecture:**
    * **Encoder:** Maps real images to a latent space. [cite: 60]
    * **Decoder/Generator:** Reconstructs images from the latent space and generates images from random noise. [cite: 61]
    * **Discriminator:** Distinguishes between real images, reconstructed images, and generated images. [cite: 62]
* **Training:**
    * **Loss Function:** Combination of VAE Reconstruction Loss, VAE KL Divergence Loss, and GAN Adversarial Loss. [cite: 63]
    * **Hyperparameters:** Latent Dimension = 256, Learning Rate = $2 \times 10^{-4}$, Epochs = 200, Batch Size = 128, Adversarial Weight = 0.1. [cite: 64]
* **Evaluation:** Qualitative assessment and FID score. (Reported FID: ~132) [cite: 65]

## Implementation

### Data
* **Dataset:** CIFAR-10, containing 50,000 $32 \times 32$ color images across 10 classes. [cite: 9, 10]
* **Preprocessing:** Images were normalized to the range [-1, 1] using torchvision transforms. [cite: 15, 31] This aligns with the input assumptions of diffusion models and ensures numerical stability. [cite: 32]

### Algorithmic Implementation (DDPM Training & Sampling Pipeline)

1.  **Data Preprocessing:** CIFAR-10 images are normalized. [cite: 15, 31]
2.  **Forward Diffusion Process:** Noise is added to clean images over 300 timesteps using either a linear or cosine beta schedule. [cite: 16, 33]
3.  **Reverse Diffusion Process (Model Training):** A U-Net model is trained to predict the added noise at each timestep. [cite: 17, 35]
4.  **Sampling (Image Generation):** Starting from pure noise, the trained U-Net model is used iteratively to remove noise and reconstruct clean images. [cite: 18, 36]
5.  **Baseline Model Training:** VAE, DCGAN, and VAE-GAN Hybrid models were trained on the same preprocessed CIFAR-10 dataset for comparison. [cite: 19, 37]

## Results [cite: 66]

### Hyperparameters (DDPM) [cite: 67]
Key hyperparameters for the DDPM models:
* **Batch Size:** 128 (Chosen for balance between training stability and GPU memory usage). [cite: 68]
* **Learning Rate:** $1 \times 10^{-4}$ (Common default for DDPMs, tuned manually based on literature). [cite: 68]
* **Timesteps (Diffusion Steps):** 300 (Reduced from 1000 to speed up training while aiming for decent performance). [cite: 68, 70]
* **Beta Range (Linear):** Linearly from $1 \times 10^{-4}$ to 0.02. [cite: 70]
* **Epochs (Cosine DDPM):** 1600 (Empirically chosen to ensure convergence without overfitting). [cite: 70] (Linear DDPM was trained for fewer epochs, ~100, due to resource limitations). [cite: 73]

### Discussion of Results [cite: 71]

**What Worked:**
* **Cosine DDPM Performance:** The Cosine DDPM model demonstrated superior performance, achieving the best results both visually (high-quality, diverse samples by Epoch 800) and in estimated quantitative terms (estimated FID: ~25-40). [cite: 71] This significantly outperformed all other models tested. [cite: 72]

**What Didn't Work (or Limitations):**
* **Linear DDPM Performance & Limitations:** The Linear DDPM achieved a high FID score of ~103. [cite: 72] This score reflects that its training was stopped early at 100 epochs due to limited computational resources. [cite: 73] Despite the poor quantitative score from incomplete training, visual inspection at 100 epochs showed promising initial results, suggesting potential for improvement with more training. [cite: 74]

**Quantitative Comparison (FID Scores):** [cite: 75]

| Model             | FID Score        | Training Stability |
| :---------------- | :--------------- | :----------------- |
| DDPM using cosine | Estimated ~25-40 | Very Stable        |
| DDPM using linear | ~103             | Very Stable        |
| DCGANs            | ~46              | Moderate Stability |
| VAE & GANs Hybrid | ~132             | Very Stable        |
| VAE               | ~189             | Very Stable        |
[cite: 76]

**Wall Clock Time for Training and Time for Sampling:** [cite: 77]

| Model             | Time for Training | Sampling Speed |
| :---------------- | :---------------- | :------------- |
| DDPM using cosine | 5 hours 30 mins   | Very Slow      |
| DDPM using linear | 4 hours           | Very Slow      |
| DCGANs            | 1 hour 20 mins    | Fast           |
| VAE & GANs Hybrid | 2 hours           | Fast           |
| VAE               | 2 hours           | Fast           |
[cite: 78]

### Visualization of Losses [cite: 79]

![image](https://github.com/user-attachments/assets/d2a5cfe5-1f38-43ab-becf-af302f61107e)

* **VAE Loss:** Steadily decreased and flattened, showing stable learning. [cite: 80]
* **DCGAN Discriminator Loss:** Fluctuated around a stable range, indicating active learning. [cite: 81]
* **DCGAN Generator Loss:** Showed fluctuations typical of GAN training. [cite: 82]
* **VAE-GAN Hybrid Loss:** Generally decreased despite some adversarial fluctuations, suggesting convergence. [cite: 83]
* **DDPM Loss (Noise Prediction):** Smoothly and consistently decreased towards convergence, highlighting very stable learning. [cite: 84]



### Visualization of Generated Samples [cite: 85]

![image](https://github.com/user-attachments/assets/af5820b2-3fc6-4593-9d2e-87430347413a)


**Qualitative Discussion of Generated Samples (based on Linear DDPM and baselines):** [cite: 86]
* **DCGAN:** Generated images exhibit recognizable shapes and textures with moderate clarity, though details can be coarse with noticeable noise. [cite: 86, 87]
* **VAE:** Samples are significantly blurred with minimal distinguishable structure, struggling to capture fine-grained details. [cite: 88, 89]
* **VAE-GAN Hybrid:** Results appear inconsistent; some have sharper contrasts, but most remain abstract and hard to interpret. [cite: 90, 91]
* **Linear DDPM (early training):** Shows promising results with clearer structure and textures, and more identifiable animal features compared to other methods, especially VAE, even at early stages of training. [cite: 92, 93, 94, 95]

**Visualization of DDPM Cosine (Progression over Epochs):** [cite: 96]

![image](https://github.com/user-attachments/assets/8799b9c8-7c9e-4006-afd9-214e8a8d985a)


* **Epoch 0:** Pure multicolored noise, as expected. [cite: 97, 98]
* **Epoch 100:** Emergence of color regions and shape blobs, indicating the model is learning low-level features. [cite: 100, 101, 102]
* **Epoch 500:** Partial object generation with some recognizable features, but many black/empty patches indicating incomplete generation. [cite: 103, 104, 105, 106]
* **Epoch 800:** Substantially clearer and more complete images with recognizable objects, backgrounds, and finer details. Far fewer black patches. [cite: 107, 108, 109]

## Pros and Cons of DDPM Approach [cite: 111, 114]

**Pros (Why this approach should be adopted):**
* **Stable and Scalable:** Diffusion models are generally easier to train than GANs, avoid issues like mode collapse, and can produce high-quality, controllable outputs with consistent convergence. [cite: 112]
* **Future-Ready and Extensible:** The DDPM approach can be improved further with techniques like DDIM sampling, Exponential Moving Average (EMA) of weights, and guidance methods, making it adaptable to more complex datasets and tasks. [cite: 113]

**Cons (Limitations/Problems with this approach):**
* **Slow Sampling Speed:** The iterative denoising process, often involving hundreds of steps, makes image generation significantly slower compared to single-pass models like GANs, especially for large-scale or real-time applications. [cite: 114]
* **Requires Long Training for Quality Convergence:** Achieving sharp, high-fidelity samples often demands a large number of training epochs, increasing computational cost and time, particularly on larger datasets. [cite: 115]

## Conclusion

This project successfully reproduced experiments for generating realistic images using Denoising Diffusion Probabilistic Models and compared their performance against VAE, DCGAN, and VAE-GAN hybrid models on the CIFAR-10 dataset. [cite: 4, 5, 6] The Cosine DDPM demonstrated superior image quality and training stability, achieving a significantly better estimated FID score compared to the baseline models and the Linear DDPM (which was limited by computational resources). [cite: 71, 72, 73] While DDPMs exhibit very stable training dynamics and can produce high-quality samples[cite: 84, 112], they suffer from slow sampling speeds and require extensive training time. [cite: 114, 115] The results confirm that DDPMs, particularly with improved schedules like the cosine annealing, are a powerful and promising approach for image generation, though practical application may require further research into accelerating sampling and reducing training overhead.

## Instructions to Run the Code [cite: 130]

Access the full project on GitHub:
[https://github.com/SUDHANSHUKADAM/Generating-Realistic-Images-using-DDPM](https://github.com/SUDHANSHUKADAM/Generating-Realistic-Images-using-DDPM) [cite: 130]

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SUDHANSHUKADAM/Generating-Realistic-Images-using-DDPM.git](https://github.com/SUDHANSHUKADAM/Generating-Realistic-Images-using-DDPM.git)
    cd Generating-Realistic-Images-using-DDPM
    ```
    [cite: 130]
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    [cite: 130]
3.  **Launch Jupyter Notebook:**
    Each notebook is standalone and includes both training and evaluation steps. A GPU-enabled environment is recommended for faster training. [cite: 131]


## References [cite: 132]
1.  Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239. [cite: 3, 134]
2.  Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. arXiv:2102.09672. [cite: 135]
3.  Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. arXiv preprint arXiv:2206.00364. [cite: 138]
4.  Jennewein, Douglas M. et al. "The Sol Supercomputer at Arizona State University." In Practice and Experience in Advanced Research Computing (pp. 296-301). Association for Computing Machinery, 2023. [cite: 132, 133]
5.  Shah, J., Gromis, M., & Pinto, R. (2024). Enhancing Diffusion Models for High-Quality Image Generation. arXiv preprint arXiv:2412.14422. (Note: The year 2412 seems to be a typo in the original document, likely intended to be a recent year). [cite: 136]
