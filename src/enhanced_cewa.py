"""Core classes for EnhancedCEWA."""

import numpy as np
import pandas as pd
import logging
from scipy.stats import entropy
from tqdm import tqdm

from .utils import (
    compute_iou, build_iou_matrix, greedy_cluster,
    weighted_average_bbox, weighted_vote_class, log_combine,
    entropy_from_probs, clip_confidence
)

logger = logging.getLogger(__name__)


class EnhancedCEWA:
    def __init__(self, n_classes=10, trust_threshold=0.999):
        """
        Initialize base aggregator.

        Args:
            n_classes: int, number of classes
            trust_threshold: float, confidence threshold for model override
        """
        self.n_classes = n_classes
        self.trust_threshold = trust_threshold
        self.class_reliability = None

    def fit(self, annotations, model_probs=None, true_labels=None):
        """Compute annotator metrics."""
        raise NotImplementedError

    def predict_consensus(self, annotations, model_probs=None):
        """Return consensus labels."""
        raise NotImplementedError

    def score_annotators(self, annotations, model_probs=None):
        """Return annotator quality scores."""
        raise NotImplementedError


class ClassificationCEWA(EnhancedCEWA):
    def __init__(self, n_classes=10, smoothing=1e-5, trust_threshold=0.999,
                 use_majority_vote=False, model_type='cewa'):
        """
        Initialize ClassificationCEWA.

        Args:
            n_classes: int, number of classes
            smoothing: float, smoothing factor
            trust_threshold: float, confidence threshold
            use_majority_vote: bool, if True, skip aggregation and use majority vote
            model_type: str, 'cewa' or 'cewa2'
        """
        super().__init__(n_classes, trust_threshold)
        self.smoothing = smoothing
        self.model_type = model_type
        self.use_majority_vote = use_majority_vote
        self.annotator_expertise = None
        self.annotator_agreement = None

    def fit(self, annotations, model_probs, true_labels=None):
        """
        Compute annotator metrics.

        Args:
            annotations: pandas DataFrame or numpy array of shape (n_examples, n_annotators)
            model_probs: numpy array of shape (n_examples, n_classes)
            true_labels: optional numpy array of shape (n_examples,)
        Returns:
            self
        """
        if self.use_majority_vote:
            logger.info("Using majority vote, skipping metric computation")
            return self

        ann_np = self._to_numpy(annotations)
        n_ex, n_ann = ann_np.shape
        K = self.n_classes

        cons_labels = self._get_consensus_labels(ann_np, model_probs, true_labels)
        self._compute_class_reliability(model_probs, cons_labels)

        self._compute_metrics_global(ann_np, model_probs, cons_labels, n_ann, K)


        logger.info("ClassificationCEWA metrics computed")
        return self

    def predict_consensus(self, annotations, model_probs):
        """
        Predict consensus labels.

        Args:
            annotations: pandas DataFrame or numpy array of shape (n_examples, n_annotators)
            model_probs: numpy array of shape (n_examples, n_classes)
        Returns:
            consensus_labels: numpy array of shape (n_examples,)
            quality_scores: numpy array of shape (n_examples,)
            consensus_probs: numpy array of shape (n_examples, n_classes)
        """
        if self.use_majority_vote:
            return self._predict_majority_vote(annotations)

        ann_np = self._to_numpy(annotations)
        n_ex, n_ann = ann_np.shape
        K = self.n_classes
        cons_probs = np.zeros((n_ex, K))

        model_entropy = entropy(model_probs.T)
        scaling = (1 - (model_entropy / np.log(K)))[:, np.newaxis]
        model_conf = scaling * self.class_reliability

        for i in tqdm(range(n_ex), desc="calculating better labels"):
            annotators = [j for j in range(n_ann) if not np.isnan(ann_np[i, j])]
            if annotators:
                cons_probs[i] = self._aggregate_single_example(
                    i, ann_np, model_probs, model_conf, annotators
                )
            else:
                cons_probs[i] = model_probs[i]

        cons_labels = np.argmax(cons_probs, axis=1)
        quality = cons_probs[np.arange(n_ex), cons_labels]
        return cons_labels, quality, cons_probs

    def score_annotators(self, annotations, model_probs=None):
        """
        Compute annotator quality scores.

        Args:
            annotations: pandas DataFrame or numpy array of shape (n_examples, n_annotators)
            model_probs: not used (kept for API consistency)
        Returns:
            numpy array of shape (n_annotators,)
        """
        if self.use_majority_vote:
            return np.zeros(self._to_numpy(annotations).shape[1])

        ann_np = self._to_numpy(annotations)
        n_ann = ann_np.shape[1]
        scores = np.zeros(n_ann)

        for j in range(n_ann):
            valid = ~np.isnan(ann_np[:, j])
            if not np.any(valid):
                continue
            labels = ann_np[valid, j].astype(int)
            expertise = np.mean([self.annotator_expertise[j, c] for c in labels])
            agreement = np.mean([self.annotator_agreement[j, c] for c in labels])
            scores[j] = 0.3 * expertise + 0.7 * agreement

        return scores

    def _to_numpy(self, data):
        if hasattr(data, 'to_numpy'):
            return data.to_numpy(dtype=float)
        return np.array(data, dtype=float)

    def _get_consensus_labels(self, ann_np, model_probs, true_labels):
        if true_labels is not None:
            return true_labels
        n_ex = ann_np.shape[0]
        model_cons = np.argmax(model_probs, axis=1)
        cons_labels = np.where(
            model_probs[np.arange(n_ex), model_cons] > self.trust_threshold,
            model_cons,
            np.array([
                np.argmax(np.bincount(ann_np[i][~np.isnan(ann_np[i])].astype(int),
                                      minlength=self.n_classes))
                for i in range(n_ex)
            ])
        )
        return cons_labels

    def _compute_class_reliability(self, model_probs, cons_labels):
        K = self.n_classes
        self.class_reliability = np.zeros(K)
        for c in range(K):
            mask = (cons_labels == c)
            if np.any(mask):
                self.class_reliability[c] = np.mean(model_probs[mask, c])

    def _compute_metrics_global(self, ann_np, model_probs, cons_labels, n_ann, K):
        self.annotator_expertise = np.zeros((n_ann, K))
        self.annotator_agreement = np.zeros((n_ann, K))

        for j in tqdm(range(n_ann), desc="computing annotator matrices"):
            valid = ~np.isnan(ann_np[:, j])
            if not np.any(valid):
                continue
            labels = ann_np[valid, j].astype(int)
            cons_sub = cons_labels[valid]
            model_sub = model_probs[valid]

            for c in range(K):
                mask = (labels == c)
                if not np.any(mask):
                    continue
                conf = model_sub[mask, c]
                correct = (cons_sub[mask] == c)
                w_correct = np.sum(conf * correct)
                w_total = np.sum(conf)
                self.annotator_expertise[j, c] = (w_correct + self.smoothing) / (w_total + self.smoothing)

                # Agreement
                other = []
                for i in np.where(valid)[0][mask]:
                    others = ann_np[i][~np.isnan(ann_np[i]) & (np.arange(n_ann) != j)]
                    other.extend(others)
                if other:
                    self.annotator_agreement[j, c] = np.mean(np.array(other) == c)

        # Smoothing
        global_agr = np.mean(self.annotator_agreement[self.annotator_agreement > 0])
        self.annotator_agreement = np.where(self.annotator_agreement == 0, global_agr, self.annotator_agreement)

    def _aggregate_single_example(self, i, ann_np, model_probs, model_conf, annotators):
        K = self.n_classes
        ann_contrib = np.zeros(K)
        total_w = 0
        for j in annotators:
            label = int(ann_np[i, j])

            w = (self.annotator_expertise[j, label] *
                 self.annotator_agreement[j, label] *
                 self.class_reliability[label])

            ann_contrib[label] += w
            total_w += w

        if total_w > 0:
            ann_contrib /= total_w

        model_w = np.mean(model_conf[i] / self.trust_threshold)
        model_w = np.clip(model_w, 0, 1)
        ann_w = 1 - model_w

        log_comb = (model_w * np.log(model_probs[i] + 1e-10) +
                    ann_w * np.log(ann_contrib + 1e-10))
        probs = np.exp(log_comb - np.max(log_comb))
        return probs / np.sum(probs)

    def _predict_majority_vote(self, annotations):
        ann_np = self._to_numpy(annotations)
        n_ex = ann_np.shape[0]
        cons_labels = np.array([
            np.argmax(np.bincount(ann_np[i][~np.isnan(ann_np[i])].astype(int),
                                  minlength=self.n_classes))
            for i in range(n_ex)
        ])
        cons_probs = np.eye(self.n_classes)[cons_labels]
        quality = np.ones(n_ex)
        return cons_labels, quality, cons_probs


class DetectionCEWA(EnhancedCEWA):
    def __init__(self, n_classes=14, iou_threshold=0.5, trust_threshold=0.999,
                 use_majority_vote=False, use_recall_weight=True,
                 use_entropy_modulation=True, use_spatial_bias=True):
        """
        Initialize DetectionCEWA.

        Args:
            n_classes: int, number of classes
            iou_threshold: float, IoU threshold for clustering
            trust_threshold: float, confidence threshold
            use_majority_vote: bool, if True, use majority vote
            use_recall_weight: bool, enable recall-aware weighting
            use_entropy_modulation: bool, enable spatial entropy modulation
            use_spatial_bias: bool, enable spatial bias calibration
        """
        super().__init__(n_classes, trust_threshold)
        self.iou_threshold = iou_threshold
        self.use_majority_vote = use_majority_vote
        self.use_recall_weight = use_recall_weight
        self.use_entropy_modulation = use_entropy_modulation
        self.use_spatial_bias = use_spatial_bias

        self.annotator_ids = None
        self.annotator_index = None

        self.class_expertise = None
        self.loc_expertise = None
        self.class_agreement = None
        self.loc_agreement = None
        self.recall = None
        self.f1_weight = None

        self._total_objects = None
        self._correct_class = None
        self._sum_iou = None
        self._recall_hits = None
        self._recall_total = None
        self._agreement_class_count = None
        self._agreement_loc_sum = None

    def fit(self, annotations_df, model_predictions=None, verbose=False):
        """
        Compute annotator metrics from DataFrame.

        Args:
            annotations_df: pandas DataFrame with columns:
                image_id, class_id, rad_id, x_min, y_min, x_max, y_max
            model_predictions: dict (optional) not used for metric computation
            verbose: bool, enable progress bar
        Returns:
            self
        """
        if self.use_majority_vote:
            logger.info("Using majority vote, skipping metric computation")
            return self

        df = self._clean_data(annotations_df)
        self._init_annotator_mapping(df)
        self._init_metrics()

        grouped = df.groupby('image_id')

        for _, group in tqdm(grouped, "Collecting statistics for annotator metrics..."):
            ann_items = self._extract_ann_items(group)
            if not ann_items:
                continue
            clusters = self._match_clusters(ann_items)
            for cluster_idx in clusters:
                cluster_anns = [ann_items[i] for i in cluster_idx]
                cons_class, cons_bbox = self._cluster_to_consensus(cluster_anns, weights=None)
                cons_obj = {'class': cons_class, 'bbox': cons_bbox}
                ann_by_rad = self._build_ann_by_rad(ann_items)
                self._update_metrics(cons_obj, ann_by_rad)

        self._finalize_metrics()
        self.class_reliability = self._compute_class_reliability(model_predictions)

        logger.info("DetectionCEWA metrics computed")
        return self

    def predict_consensus(self, annotations_df, model_predictions=None, verbose=False):
        """
        Aggregate annotations for all images.

        Args:
            annotations_df: pandas DataFrame with columns:
                image_id, class_id, rad_id, x_min, y_min, x_max, y_max
            model_predictions: dict {image_id: list of pred dicts with keys:
                'bbox', 'class', 'confidence', 'class_probs'}
            verbose: bool, enable progress bar
        Returns:
            pandas DataFrame with columns: image_id, class_id, x_min, y_min, x_max, y_max
        """
        if self.use_majority_vote:
            return self._predict_majority_vote(annotations_df)

        df = self._clean_data(annotations_df)
        grouped = df.groupby('image_id')
        all_consensus = []

        for image_id, group in tqdm(grouped, desc="Aggregating images..."):
            model_preds = model_predictions.get(image_id, []) if model_predictions else []
            objs = self._aggregate_image(image_id, group, model_preds)
            all_consensus.extend(objs)

        return pd.DataFrame(all_consensus)

    def score_annotators(self, annotations_df=None, model_probs=None):
        """
        Compute annotator quality scores.

        Args:
            annotations_df: not used (kept for API consistency)
            model_probs: not used
        Returns:
            numpy array of shape (n_annotators,)
        """
        if self.use_majority_vote:
            return np.zeros(len(self.annotator_ids or []))

        n_ann = len(self.annotator_ids)
        scores = np.zeros(n_ann)
        for idx in range(n_ann):
            avg_exp = np.mean(self.class_expertise[idx, :] * self.loc_expertise[idx, :])
            avg_agr = np.mean(self.class_agreement[idx, :] * self.loc_agreement[idx, :])
            avg_rec = np.mean(self.recall[idx, :])
            scores[idx] = 0.4 * avg_exp + 0.3 * avg_agr + 0.3 * avg_rec
        return scores

    def _clean_data(self, df):
        return df[df['class_id'] < self.n_classes].copy()

    def _init_annotator_mapping(self, df):
        self.annotator_ids = sorted(df['rad_id'].unique())
        self.annotator_index = {aid: i for i, aid in enumerate(self.annotator_ids)}

    def _init_metrics(self):
        n_ann = len(self.annotator_ids)
        K = self.n_classes
        zeros = lambda: np.zeros((n_ann, K))
        self.class_expertise = zeros()
        self.loc_expertise = zeros()
        self.class_agreement = zeros()
        self.loc_agreement = zeros()
        self.recall = zeros()
        self._total_objects = zeros()
        self._correct_class = zeros()
        self._sum_iou = zeros()
        self._recall_hits = zeros()
        self._recall_total = zeros()
        self._agreement_class_count = zeros()
        self._agreement_loc_sum = zeros()

    def _extract_ann_items(self, group):
        items = []
        for _, row in group.iterrows():
            items.append({
                'rad_id': row['rad_id'],
                'class': row['class_id'],
                'bbox': [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
            })
        return items

    def _match_clusters(self, ann_items):
        if not ann_items:
            return []
        bboxes = [a['bbox'] for a in ann_items]
        iou_mat = build_iou_matrix(bboxes)
        return greedy_cluster(iou_mat, thresh=self.iou_threshold)

    def _cluster_to_consensus(self, cluster_anns, weights=None):
        classes = [a['class'] for a in cluster_anns]
        bboxes = [a['bbox'] for a in cluster_anns]
        if weights is None:
            weights = [1.0] * len(cluster_anns)
        cons_class = weighted_vote_class(classes, weights)
        cons_bbox = weighted_average_bbox(bboxes, weights)
        return cons_class, cons_bbox

    def _build_ann_by_rad(self, ann_items):
        ann_by_rad = {}
        for ann in ann_items:
            ann_by_rad.setdefault(ann['rad_id'], []).append(ann)
        return ann_by_rad

    def _update_metrics(self, cons_obj, ann_by_rad):
        cons_class = cons_obj['class']
        cons_bbox = cons_obj['bbox']
        idx_map = self.annotator_index
        rads = list(ann_by_rad.keys())

        for rad in rads:
            idx = idx_map[rad]
            self._recall_total[idx, cons_class] += 1
            best_iou, best_ann = 0, None
            for ann in ann_by_rad[rad]:
                iou = compute_iou(ann['bbox'], cons_bbox)
                if iou > best_iou:
                    best_iou, best_ann = iou, ann
            if best_iou >= self.iou_threshold and best_ann:
                self._recall_hits[idx, cons_class] += 1
                if best_ann['class'] == cons_class:
                    self._correct_class[idx, cons_class] += 1
                self._sum_iou[idx, cons_class] += best_iou
                self._total_objects[idx, cons_class] += 1

        for i_rad in rads:
            i_idx = idx_map[i_rad]
            for j_rad in rads:
                if i_rad == j_rad:
                    continue
                j_idx = idx_map[j_rad]
                for ann_i in ann_by_rad[i_rad]:
                    best_iou, ann_j_match = 0, None
                    for ann_j in ann_by_rad[j_rad]:
                        iou = compute_iou(ann_i['bbox'], ann_j['bbox'])
                        if iou > best_iou:
                            best_iou, ann_j_match = iou, ann_j
                    if best_iou >= self.iou_threshold and ann_j_match:
                        if ann_i['class'] == ann_j_match['class']:
                            self._agreement_class_count[i_idx, ann_i['class']] += 1
                        self._agreement_loc_sum[i_idx, ann_i['class']] += best_iou

    def _finalize_metrics(self):
        n_ann = len(self.annotator_ids)
        eps = 1e-8
        for idx in range(n_ann):
            for c in range(self.n_classes):
                total = self._total_objects[idx, c]
                if total > 0:
                    self.class_expertise[idx, c] = self._correct_class[idx, c] / total
                    self.loc_expertise[idx, c] = self._sum_iou[idx, c] / total
                    self.class_agreement[idx, c] = self._agreement_class_count[idx, c] / total
                    self.loc_agreement[idx, c] = self._agreement_loc_sum[idx, c] / total
                else:
                    self.class_expertise[idx, c] = eps
                    self.loc_expertise[idx, c] = eps
                    self.class_agreement[idx, c] = eps
                    self.loc_agreement[idx, c] = eps
                if self._recall_total[idx, c] > 0:
                    self.recall[idx, c] = self._recall_hits[idx, c] / self._recall_total[idx, c]
                else:
                    self.recall[idx, c] = eps
        precision = self.class_expertise * self.loc_expertise
        f1_num = 2 * precision * self.recall
        f1_den = precision + self.recall + eps
        self.f1_weight = f1_num / f1_den
        self.f1_weight = np.clip(self.f1_weight, eps, 1.0)

    def _compute_class_reliability(self, model_predictions):
        if model_predictions:
            return np.ones(self.n_classes) * 0.8
        return np.ones(self.n_classes) * 0.8

    def _aggregate_image(self, image_id, group, model_preds):
        ann_items = self._extract_ann_items(group)
        if not ann_items:
            return []
        clusters = self._match_clusters(ann_items)
        results = []
        for cluster_idx in clusters:
            cluster_anns = [ann_items[i] for i in cluster_idx]
            cons_class, cons_bbox = self._aggregate_cluster(cluster_anns)
            ann_probs = self._get_ann_probs(cluster_anns)
            if model_preds and self.use_entropy_modulation:
                combined = self._combine_with_model(ann_probs, model_preds, cons_bbox)
                cons_class = np.argmax(combined)
            results.append({
                'image_id': image_id,
                'class_id': cons_class,
                'x_min': cons_bbox[0],
                'y_min': cons_bbox[1],
                'x_max': cons_bbox[2],
                'y_max': cons_bbox[3]
            })
        return results

    def _aggregate_cluster(self, cluster_anns):
        classes = [a['class'] for a in cluster_anns]
        bboxes = [a['bbox'] for a in cluster_anns]
        weights = []
        for ann in cluster_anns:
            idx = self.annotator_index[ann['rad_id']]
            if self.use_recall_weight:
                w = self.f1_weight[idx, ann['class']]
            else:
                w = (self.class_expertise[idx, ann['class']] *
                     self.loc_expertise[idx, ann['class']] *
                     self.class_agreement[idx, ann['class']] *
                     self.loc_agreement[idx, ann['class']])
            w *= self.class_reliability[ann['class']]
            weights.append(max(w, 1e-8))
        cons_class = weighted_vote_class(classes, weights)
        cons_bbox = weighted_average_bbox(bboxes, weights)
        return cons_class, cons_bbox

    def _get_ann_probs(self, cluster_anns):
        probs = np.zeros(self.n_classes)
        for ann in cluster_anns:
            idx = self.annotator_index[ann['rad_id']]
            w = self.f1_weight[idx, ann['class']] if self.use_recall_weight else 1.0
            probs[ann['class']] += w
        if np.sum(probs) > 0:
            probs /= np.sum(probs)
        else:
            probs = np.ones(self.n_classes) / self.n_classes
        return probs

    def _combine_with_model(self, ann_probs, model_preds, cons_bbox):
        if not model_preds:
            return ann_probs
        alpha = self._compute_model_alpha(model_preds, cons_bbox)
        model_probs = np.mean([np.array(p['class_probs']) for p in model_preds], axis=0)
        return log_combine(model_probs, ann_probs, alpha)

    def _compute_model_alpha(self, model_preds, cons_bbox):
        if not self.use_entropy_modulation:
            return clip_confidence(max(p['confidence'] for p in model_preds))
        overlaps = []
        for pred in model_preds:
            iou = compute_iou(pred['bbox'], cons_bbox)
            if iou >= self.iou_threshold:
                probs = np.array(pred['class_probs'])
                ent = entropy_from_probs(probs)
                overlaps.append(ent)
        if overlaps:
            spatial_conf = np.exp(-np.mean(overlaps))
        else:
            spatial_conf = 0.0
        class_conf = np.mean([p['confidence'] for p in model_preds])
        return clip_confidence(spatial_conf * class_conf)

    def _predict_majority_vote(self, annotations_df):
        df = self._clean_data(annotations_df)
        grouped = df.groupby('image_id')
        results = []
        for image_id, group in grouped:
            ann_items = self._extract_ann_items(group)
            if not ann_items:
                continue
            clusters = self._match_clusters(ann_items)
            for cluster_idx in clusters:
                cluster_anns = [ann_items[i] for i in cluster_idx]
                cons_class, cons_bbox = self._cluster_to_consensus(cluster_anns, weights=None)
                results.append({
                    'image_id': image_id,
                    'class_id': cons_class,
                    'x_min': cons_bbox[0],
                    'y_min': cons_bbox[1],
                    'x_max': cons_bbox[2],
                    'y_max': cons_bbox[3]
                })
        return pd.DataFrame(results)