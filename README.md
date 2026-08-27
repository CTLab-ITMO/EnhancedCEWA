# EnhancedCEWA: Unified Crowdsourced Label Aggregation for Classification and Detection

EnhancedCEWA is a Python framework for aggregating noisy crowdsourced labels by leveraging per‑class annotator expertise, inter‑annotator agreement, and model confidence. It works for both **image classification** and **object detection** tasks, and includes novel mechanisms for detection: spatial bias calibration, recall‑aware weighting, and entropy‑based model modulation.

---

## 📦 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/CTLab-ITMO/EnhancedCEWA.git
cd EnhancedCEWA
pip install -r requirements.txt
```
## 🚀 How to use EnhancedCEWA for Classification

To use EnhancedCEWA, you need two things:
1. **Raw annotations** from your crowd workers (with missing values for unlabeled samples).
2. **Model predictions** (probability distributions for classification, or bounding boxes + class probabilities for detection).

The framework will compute annotator expertise, inter-annotator agreement, and class reliability, then produce refined consensus labels with higher quality than simple majority voting.

---

**Goal**: Improve the quality of class labels.

**Step-by-step instructions**:

1. **Prepare your annotation matrix** (`annotations.csv`):
   - Format: CSV file with shape `(N, M)`.
   - `N` = number of examples (images).
   - `M` = number of annotators.
   - Values: integer class labels (0 to K-1) or `NaN` if the annotator did not label the example.

2. **Prepare your model predictions** (`model_probs.csv`):
   - Format: CSV file with shape `(N, K)`.
   - `K` = number of classes.
   - Each row must contain the softmax probabilities for the corresponding example (summing to 1.0).
   - The order of rows must match the annotation matrix.

3. **Set up the configuration** (`resources/config.yaml`):
   ```yaml
   task: classification
   classification:
     n_classes: 10                
     smoothing: 1e-5
     trust_threshold: 0.999        
     use_majority_vote: false      
     predictions_path: 'data/classification/model_probs.csv'
   annotations_path: 'data/classification/annotations.csv'
   output_dir: 'output/classification/'
   verbose: true
 
4. **Run the aggregation:**  
```bash
python run.py
```

5. **Check the output**
   - consensus.csv – contains the improved consensus labels (consensus_label) and their quality scores (quality).
   - annotator_scores.csv – reliability score for each annotator (higher is better).

## 🚀 How to Use EnhancedCEWA for Object Detection

**Goal**: Improve the quality of both class labels and bounding box coordinates in crowdsourced object detection annotations.

---

### Step-by-step instructions

#### 1. Prepare your annotation file (`annotations.csv`)

Format: CSV with the following **required** columns (each row = one object annotation):

| Column     | Type    | Description                                   |
|------------|---------|-----------------------------------------------|
| `image_id` | str     | Unique image identifier                       |
| `class_id` | int     | Object class (0..K-1)                         |
| `rad_id`   | int/str | Annotator ID                                  |
| `x_min`    | float   | Left coordinate of the bounding box (pixels)  |
| `y_min`    | float   | Top coordinate (pixels)                       |
| `x_max`    | float   | Right coordinate (pixels)                     |
| `y_max`    | float   | Bottom coordinate (pixels)                    |

**Important notes:**
- Images with **no objects** can be omitted from the CSV (the framework will handle them as empty).
- All coordinates must be in **absolute pixel values** (not normalized).
- Class IDs must be in the range `0` to `K-1` (where `K` = number of classes).

---

#### 2. Prepare your model predictions (`model_predictions.jsonl`)

Format: **JSONL** (JSON Lines) – one JSON object per line.

Each line must have the following structure:

```json
{
  "image_id": "image_001",
  "predictions": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": 0,
      "confidence": 0.92,
      "class_probs": [0.01, 0.02, 0.03, ..., 0.92]
    }
  ]
}
```

#### Model Predictions File (`model_predictions.jsonl`)

Format: **JSONL** (JSON Lines) – one JSON object per line.

**Root object fields:**

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | str | Image identifier. Must match `image_id` in annotation CSV. |
| `predictions` | list | List of prediction objects for this image. Empty list `[]` allowed. |
| `bbox` | list[float] | Bounding box coordinates `[x_min, y_min, x_max, y_max]` in **absolute pixels**. |
| `class` | int | Predicted class label (0 to `n_classes - 1`). |
| `confidence` | float | Detection confidence score (0.0 to 1.0). |
| `class_probs` | list[float] | Softmax probabilities for all `K` classes. Must be length `K` and sum to 1.0. |

3. **Set up the configuration** (`resources/config.yaml`):
   ```yaml
    task: detection
    detection:
      n_classes: 14                    
      iou_threshold: 0.5               
      trust_threshold: 0.999           
      use_majority_vote: false         
      use_recall_weight: true          
      use_entropy_modulation: true     
      use_spatial_bias: true           
      predictions_path: 'data/detection/model_predictions.jsonl'
    
    annotations_path: 'data/detection/annotations.csv'
    output_dir: 'output/detection/'
    verbose: true
    
4. **Run the aggregation:**  
```bash
python run.py
```

5. **Check the output**
   - consensus.csv – contains the improved bounding boxes and class labels (columns: image_id, class_id, x_min, y_min, x_max, y_max).

   - annotator_scores.csv – reliability score for each annotator (higher is better).

## ⚙️ Configuration Parameters

Below is the complete list of all configuration parameters available in the `resources/config.yaml` file.

| Parameter | Task | Type | Default | Description |
|-----------|------|------|---------|-------------|
| `task` | Both | str | `classification` | Type of task: `"classification"` or `"detection"` |
| `annotations_path` | Both | str | — | Path to the CSV file with raw annotations |
| `output_dir` | Both | str | `"output"` | Directory where results will be saved |
| `verbose` | Both | bool | `true` | Enable progress bars and detailed logging |
| `n_classes` | Both | int | `10` (classif.) / `14` (det.) | Number of classes in your dataset (K) |
| `smoothing` | Classification | float | `1e-5` | Smoothing factor to avoid division by zero |
| `trust_threshold` | Both | float | `0.999` | Confidence threshold for model override |
| `use_majority_vote` | Both | bool | `false` | If `true`, skip full aggregation and use simple majority vote (baseline) |
| `predictions_path` | Both | str | — | Path to model predictions file (CSV for classification, JSONL for detection) |
| `iou_threshold` | Detection | float | `0.5` | IoU threshold for clustering overlapping bounding boxes |
| `use_recall_weight` | Detection | bool | `true` | Enable recall-aware weighting: penalizes annotators who miss objects |
| `use_entropy_modulation` | Detection | bool | `true` | Enable spatial entropy modulation: reduces model influence when localization is unstable |
| `use_spatial_bias` | Detection | bool | `true` | Enable spatial bias calibration: corrects systematic box shifts per annotator |

# [See more examples in notebooks section](notebooks)