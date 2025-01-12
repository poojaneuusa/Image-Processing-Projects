# Unsupervised Image Clustering
This code performs unsupervised image clustering by encoding images into embeddings using a pre-trained model (`clip-ViT-B-32`) and then grouping similar images based on their cosine similarity scores. The clustering identifies groups of visually similar images without using labelled data, displaying representative samples from the largest clusters.
### Installation

Steps:

```bash
# Install the sentence-transformers library
pip install sentence_transformers

# Import required libraries
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import matplotlib.pyplot as plt
import glob
import torch
import os

# Download and unzip the image dataset
!wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/unsplash-25k-photos.zip
!unzip 'unsplash-25k-photos.zip' -d 'photos'
```

### How It Works

1. **Loading Images**: Load the dataset of images using the `glob` library.
   ```python
   img_names = list(glob.glob('photos/*.jpg'))[:2000]
   print("Images:", len(img_names))
   ```

2. **Model Initialization**: Use the `clip-ViT-B-32` model from the SentenceTransformers library to encode images into embeddings.
   ```python
   model = SentenceTransformer('clip-ViT-B-32')
   img_embed = model.encode([Image.open(img) for img in img_names], batch_size=32, convert_to_tensor=True, show_progress_bar=True)
   ```

3. **Cosine Similarity Calculation**: Compute cosine similarity between all image embeddings.
   ```python
   cos_scores = util.cos_sim(img_embed, img_embed)
   ```

4. **Community Extraction**: Identify clusters based on similarity scores exceeding a threshold.
   

5. **Unique Community Identification**: Remove overlapping clusters to extract unique communities.
   

6. **Visualizing Clusters**: Display sample images from each cluster.
  

### Input
- A folder of images (e.g., `photos/*.jpg`) containing the dataset to be clustered.

### Output
- **Cluster Details**: Displays the size of each identified cluster.
- **Visualizations**: Shows sample images from the largest clusters.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Zero Shot Image Classification
This implementation is efficient for zero-shot classification tasks where predefined textual labels are available. The model `clip-ViT-B-32` leverages a pre-trained vision-language model for embedding computation, ensuring high accuracy.
### Installation

Install the required libraries:

```bash
%pip install sentence_transformers
```

Clone the dataset:

```bash
!git clone https://github.com/laxmimerit/dog-cat-full-dataset.git
```


## How It Works

Zero-shot image classification allows the classification of images without training the model on labeled data. Instead, it uses textual labels to match embeddings of images and labels.

### Key Steps:
1. **Dataset Preparation**:
   - Load images of dogs and cats from the provided dataset.
   - Shuffle the dataset for randomness.

2. **Model Setup**:
   - Use the `clip-ViT-B-32` model from the Sentence Transformers library to generate embeddings for images and textual labels.

3. **Embedding Computation**:
   - Compute embeddings for the images.
   - Compute embeddings for textual labels (e.g., "dog" and "cat").

4. **Classification**:
   - Compute cosine similarity scores between image embeddings and label embeddings.
   - Assign labels based on the highest similarity score.

5. **Display Results**:
   - Show a subset of images along with their predicted labels.


### Input
- **Images**: 100 images of dogs and 100 images of cats from the dataset.
- **Labels**: Text labels `['dog', 'cat']`.

### Output
- Predicted label for each displayed image based on cosine similarity scores.
- Visualization of the images with their corresponding predicted labels.

### Example Output:
1. **Image Display**:
   - An image of a dog or cat is displayed.
2. **Predicted Label**:
   ```
   Predicted Label: dog
   -----
   ```


--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Duplicate Images Detecting

This script identifies duplicate or highly similar images in a dataset by using a pre-trained model(`clip-ViT-B-32`) to compute image embeddings and compare them with cosine similarity. It outputs the similarity scores and visually displays the duplicate image pairs for verification.
### Installation
Install the required libraries:
```bash
%pip install sentence_transformers
```

## How It Works
This script identifies duplicate images in a dataset by leveraging a pre-trained model (`clip-ViT-B-32`) from the `sentence_transformers` library. It computes image embeddings, compares them using cosine similarity, and displays pairs of images with high similarity scores.

### Steps:
1. **Download Dataset**: A dataset of images is downloaded and unzipped.
2. **Generate Embeddings**: The pre-trained model converts images into embeddings.
3. **Find Duplicates**: Using paraphrase mining, the embeddings are compared to find duplicate or highly similar images.
4. **Display Results**: The top duplicate image pairs are displayed with their similarity scores.

### Input
The input consists of:
- A dataset of images.
- The first 2000 images from the dataset are used for processing.

### Example Dataset
The script downloads a zip file containing 25,000 images:
```
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/unsplash-25k-photos.zip
```

### Output
The script outputs:
1. **Similarity Score**: A numerical score indicating the similarity between image pairs.
2. **Displayed Images**: The script visualizes the pairs of images identified as duplicates or highly similar.

### Example Output
- **Similarity Score**: `0.98`
- **Displayed Images**:
    - **Image 1**: First image in the duplicate pair.
    - **Image 2**: Second image in the duplicate pair.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Multilingual Image Search (Image Search from Hindi, Spanish and French Text)

This script enables semantic image search from text queries written in multiple languages (e.g., Hindi, French, Spanish) by leveraging a multilingual pre-trained model. It outputs the most relevant images matching the given query

### Installation
Install the required libraries:

```bash
%pip install sentence_transformers
```

### How It Works
This script performs multilingual image search by using a pre-trained multilingual model (`clip-ViT-B-32-multilingual-v1`). It encodes images and text in various languages (e.g., Hindi, French, Spanish) into embeddings, allowing semantic searches based on text queries to find matching images.

### Steps:
1. **Download Dataset**: A dataset of images is downloaded and unzipped.
2. **Generate Image Embeddings**: The pre-trained model converts images into embeddings.
3. **Process Query**: Text queries in multiple languages are encoded into embeddings.
4. **Perform Search**: Semantic similarity between the query and image embeddings is computed to retrieve the most relevant images.
5. **Display Results**: The top matching images for the query are displayed.

### Input
The input consists of:
- A dataset of images downloaded from a specified URL.
- A text query in Hindi, French, Spanish, or other supported languages.
- The number of top matches (`top_k`) to retrieve.

### Example Dataset
The script downloads a zip file containing 25,000 images:
```
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/unsplash-25k-photos.zip
```

### Example Query
Input Query: "Chat" (French for "cat")

### Output
The script outputs:
1. **Query**: The input query text.
2. **Image Paths**: File paths of the matching images.
3. **Displayed Images**: Visualizations of the top matching images for the query.

### Example Output
- **Query**: `Chat`
- **Image Path**: `photos/image123.jpg`
- **Displayed Images**:
    - Matching images are shown inline using `matplotlib`.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Text to Image Search
This project demonstrates using the `sentence-transformers` library to perform Text-to-Image Search. It leverages a pre-trained model to encode both text queries and image embeddings, and performs semantic search to find the most relevant images based on the input query.

### Installation

Install the following libraries:

```bash
pip install sentence_transformers
pip install Pillow
pip install matplotlib
pip install torch
```

### How It Works

- Model Loading: The code loads the `clip-ViT-B-32` model from the sentence-transformers library. This model is designed to compute embeddings for both text and images in a shared space.

- Image Embeddings: A dataset containing 25,000 images (downloaded via a wget command) is used for training. The code selects a subset of 2000 images and generates embeddings for each 
  image using the model.

- Text Query Encoding: When a user provides a text query, the system encodes it into a tensor.

- Semantic Search: The system then compares the text query's embedding with the image embeddings using cosine similarity. It retrieves the top-k most relevant images based on their 
  similarity to the query.

- Display Results: The code displays the relevant images to the user.

### Input

A text query (for example, "man on the mountain") is provided by the user.

### Output

The program outputs the top-k most relevant images based on the input query.

The output includes:
- The text query.
- The paths of the retrieved images.
- Visual representations of the most relevant images.
