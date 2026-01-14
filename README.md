# Under Construction

# AI-driven WHO 2021 classification of gliomas based only on H&E-stained slides

<img src="fig1a.jpg" width="1000px" align="center" />
<img src="fig1b.jpg" width="1000px" align="center" />

The WHO 2021 classification criteria for adult-type diffuse glioma integrate histology with molecular profiling for conclusive diagnosis. Since molecular profiling can be expensive and time-consuming, often necessitating outsourcing or leading to the "not otherwise specified (NOS) label," this study develops an AI-driven WHO 2021 classification of gliomas solely from H&E whole-slide images (WSIs).


## Environment
### Pre-requisites
* Linux (Tested on Ubuntu 22.04)
* NVIDIA GPU (Tested on Nvidia A6000/A100)

Next, use the environment configuration file to create a conda environment:
```bash
conda env create -n glioma_subtyping -f glioma_subtyping.yaml
```

Activate the environment:
```bash
conda activate glioma_subtyping
```

## WSI Patching and Curation

```bash
data/slides_20x/
	├── patient_1_slide_a.svs
	├── patient_1_slide_b.svs
	└── ...
data/slides_40x/
	├── patient_2_slide_a.svs
	├── patient_2_slide_b.svs
	└── ...
```


### 🛠 Workflow Logic

The pipeline automatically adjusts the extraction scale and file paths based on the input magnification. 
This ensures that the physical area covered by a patch remains consistent or follows your specific protocol.

| Input Argument | Target Data Directory | Output Directory | Patch Size | Down Sample Level |
| :--- | :--- | :--- | :--- | :--- |
| `40x` | `data/slides_40x/` | `data/slides_patches_40x/` | **512** | `2` |
| `20x` | `data/slides_20x/` | `data/slides_patches_20x/` | **256** | `1` | 


### Execution Examples
```shell
# Process 20x slides with 256px patches
./create_patches.sh 20x

# Process 40x slides with 512px patches
./create_patches.sh 40x
```
By setting `20x` to `256` and `40x` to `512`, you are effectively keeping the **field of view (FOV)** of each patch identical in terms of physical microns (assuming the 40x scan has twice the resolution of the 20x scan). This is a standard best practice in pathology machine learning to ensure the model sees the same amount of tissue per tile regardless of the scanner settings.

#### Output Directory Structure
```bash
data/slides_patches_20x/
	├── masks
    		├── patient_1_slide_a.png
    		├── patient_1_slide_b.png
    		└── ...
	├── patches
    		├── patient_1_slide_a.h5
    		├── patient_1_slide_b.h5
    		└── ...
	├── stitches
    		├── patient_1_slide_a.png
    		├── patient_1_slide_b.png
    		└── ...
	└── slides_processed.csv

data/slides_patches_40x/
├── masks
        ├── patient_1_slide_a.png
        ├── patient_1_slide_b.png
        └── ...
├── patches
        ├── patient_1_slide_a.h5
        ├── patient_1_slide_b.h5
        └── ...
├── stitches
        ├── patient_1_slide_a.png
        ├── patient_1_slide_b.png
        └── ...
└── slides_processed.csv
```

### 🧹 Patch Cleanup (Step 2)
After initial patching, the pipeline runs a **Cleanup Script** to filter out low-quality tiles.
#### Filtering Criteria:
1. **White Space**: Patches with >85% background are removed.
2. **Stain Detection**: Uses HED (Hematoxylin-Eosin-DAB) color deconvolution to ensure tissue is actually present.
3. **HSV Filtering**: Removes blurry or out-of-focus areas based on saturation and value thresholds.

#### Why this is necessary:
Whole Slide Images often contain artifacts, marker ink, or large empty regions. By cleaning the `.h5` files, you reduce the noise in your training set and significantly speed up the feature extraction (encoding) step.

## Creating Features
Run the extraction script by specifying magnification, batch size, and the desired model backbone. The script dynamically maps to the correct data and coordinate directories based on the magnification provided.

### Usage
```shell
./extract_features.sh <MAG> <BATCH_SIZE> <BACKBONE>
```
Example: 
```shell
./extract_features.sh 20x 256 uni
```

### Supported Backbones

We support several **state-of-the-art self-supervised and supervised models** for histopathology.  
For more details about each model, please refer to the original repositories to request access and follow their specific licensing terms.

- **UNI2** : [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)
- **CTransPath** : [https://github.com/Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath)
- **Lunit ViT** : [https://github.com/lunit-io/benchmark-ssl-pathology](https://github.com/lunit-io/benchmark-ssl-pathology)
- **ResNet-50** : ImageNet pretrained

## Training the models


### Usage Instructions
To run the training script, pass the **magnification level** and **backbone name** as arguments:
```bash
chmod +x train.sh
./train.sh <MAG> <BACKBONE>
```

Example: 
```bash
./train.sh 20x uni
```
### Methodology
This codebase is heavily based on [CLAM](https://github.com/mahmoodlab/CLAM/). However, unlike CLAM, instance-level clustering is not used.
The model is trained using a pure Attention-based Multiple Instance Learning (MIL) framework.

As we discussed in our paper, we have trained our model on three different groups of patients. To train the train model replace the **splits_dir** argument in the `main.py` file. 

## Evaluation 
To run the evaluation script, pass the **magnification level** and **backbone name** as arguments:
```bash
chmod +x eval.sh
./eval.sh <MAG> <BACKBONE>
```

Example: 
```bash
./eval.sh 20x uni
```

## References
1. Lu, Ming Y., et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." Nature biomedical engineering 5.6 (2021): 555-570.
2. Wang, Xiyue, et al. "RetCCL: Clustering-guided contrastive learning for whole-slide image retrieval." Medical image analysis 83 (2023): 102645.
3. Kang, Mingu, et al. "Benchmarking self-supervised learning on diverse pathology datasets." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
4. Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature medicine 30.3 (2024): 850-862.
5. Wang, Xiyue, et al. "Transformer-based unsupervised contrastive learning for histopathological image classification." Medical image analysis 81 (2022): 102559.
6. Srinidhi, Chetan L., and Anne L. Martel. "Improving self-supervised learning with hardness-aware dynamic curriculum learning: an application to digital pathology." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
7. Chen, Richard J., et al. "Scaling vision transformers to gigapixel images via hierarchical self-supervised learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.


## Citations

Shubham Innani, W Robert Bell, MacLean P Nasrallah, Bhakti Baheti, Spyridon Bakas, AI-driven WHO 2021 classification of gliomas based only on H&E-stained slides, Neuro-Oncology, 2025;, noaf189, https://doi.org/10.1093/neuonc/noaf189

```bash
@article{10.1093/neuonc/noaf189,
    author = {Innani, Shubham and Bell, W Robert and Nasrallah, MacLean P and Baheti, Bhakti and Bakas, Spyridon},
    title = {AI-driven WHO 2021 classification of gliomas based only on H\&amp;E-stained slides},
    journal = {Neuro-Oncology},
    pages = {noaf189},
    year = {2025},
    month = {08},
    abstract = {},
    issn = {1522-8517},
    doi = {10.1093/neuonc/noaf189},
    url = {https://doi.org/10.1093/neuonc/noaf189},
    eprint = {https://academic.oup.com/neuro-oncology/advance-article-pdf/doi/10.1093/neuonc/noaf189/64170409/noaf189.pdf},
}
```
