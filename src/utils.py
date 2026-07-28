import os
import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import json

import pandas as pd


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def build_iou_matrix(bboxes):
    n = len(bboxes)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = compute_iou(bboxes[i], bboxes[j])
    return mat


def greedy_cluster(iou_mat, thresh=0.5):
    n = len(iou_mat)
    used = [False] * n
    clusters = []

    pairs = [(iou_mat[i, j], i, j) for i in range(n) for j in range(i + 1, n) if iou_mat[i, j] >= thresh]
    pairs.sort(reverse=True, key=lambda x: x[0])

    for _, i, j in pairs:
        if used[i] or used[j]:
            continue
        cluster = [i, j]
        used[i] = used[j] = True
        changed = True
        while changed:
            changed = False
            for k in range(n):
                if not used[k] and any(iou_mat[k, m] >= thresh for m in cluster):
                    cluster.append(k)
                    used[k] = True
                    changed = True
        clusters.append(cluster)

    for i in range(n):
        if not used[i]:
            clusters.append([i])
    return clusters


def weighted_average_bbox(bboxes, weights):
    weights = np.array(weights)
    return np.average(bboxes, axis=0, weights=weights).tolist()


def weighted_vote_class(classes, weights):
    wc = {}
    for cls, w in zip(classes, weights):
        wc[cls] = wc.get(cls, 0.0) + w
    return max(wc, key=wc.get)


def log_combine(model_probs, ann_probs, alpha):
    eps = 1e-10
    log_comb = alpha * np.log(model_probs + eps) + (1 - alpha) * np.log(ann_probs + eps)
    log_comb = log_comb - np.max(log_comb)
    probs = np.exp(log_comb)
    return probs / np.sum(probs)


def entropy_from_probs(probs):
    return entropy(probs)


def clip_confidence(confidence, eps=1e-8):
    return np.clip(confidence, eps, 1.0)


def load_predictions(predictions_dir):
    all_preds = {}
    for fname in os.listdir(predictions_dir):
        if fname.endswith('.pkl'):
            image_id = fname.replace('.pkl', '')
            with open(os.path.join(predictions_dir, fname), 'rb') as f:
                all_preds[image_id] = pickle.load(f)
    return all_preds


def load_prediction(image_id, predictions_dir):
    file_path = os.path.join(predictions_dir, f"{image_id}.pkl")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_predictions_from_jsonl(input_path='model_predictions.jsonl'):
    model_predictions = {}
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Loading predictions"):
            data = json.loads(line)
            image_id = data['image_id']
            preds = data['predictions']

            for p in preds:
                if 'class_probs' in p:
                    p['class_probs'] = np.array(p['class_probs'], dtype=np.float32)
                if 'bbox' in p and not isinstance(p['bbox'], list):
                    p['bbox'] = p['bbox'].tolist()
                if 'confidence' in p:
                    p['confidence'] = float(p['confidence'])
                if 'class' in p:
                    p['class'] = int(p['class'])

            model_predictions[image_id] = preds

    return model_predictions


def save_predictions_to_jsonl(model_predictions, output_path='model_predictions.jsonl'):
    with open(output_path, 'w') as f:
        for image_id, preds in tqdm(model_predictions.items(), desc="Saving predictions"):
            preds_serializable = []
            for p in preds:
                p_copy = p.copy()
                if 'class_probs' in p_copy and isinstance(p_copy['class_probs'], np.ndarray):
                    p_copy['class_probs'] = p_copy['class_probs'].tolist()
                preds_serializable.append(p_copy)

            line = json.dumps({
                'image_id': image_id,
                'predictions': preds_serializable
            })
            f.write(line + '\n')

def load_predictions_from_csv(path):
    preds_csv = pd.read_csv(path)
    return preds_csv.to_numpy()