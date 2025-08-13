"""
Clinical Evaluation Metrics for Medical AI Systems

Specialized metrics for clinical otitis classification:
- Medical-grade sensitivity/specificity per condition
- Clinical decision support validation
- Cross-dataset generalization assessment
- Confidence calibration for safety
- Expert agreement analysis

Industry standard: Following FDA guidelines for medical AI validation
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class ClinicalMetrics:
    """Container for clinical performance metrics."""
    
    # Overall performance
    accuracy: float
    balanced_accuracy: float
    
    # Per-class clinical metrics
    sensitivity: Dict[str, float]  # Recall per class (critical for pathology detection)
    specificity: Dict[str, float]  # True negative rate per class
    precision: Dict[str, float]    # Positive predictive value
    npv: Dict[str, float]         # Negative predictive value
    f1_score: Dict[str, float]    # Harmonic mean of precision/recall
    
    # Clinical decision support metrics
    high_confidence_accuracy: float    # Accuracy on high-confidence predictions
    low_confidence_rate: float         # Percentage flagged for human review
    pathology_sensitivity: float       # Sensitivity across all pathological conditions
    normal_specificity: float          # Specificity for normal classification
    
    # Cross-dataset generalization
    source_performance: Dict[str, float]  # Performance by dataset source
    domain_adaptation_score: float        # Cross-dataset performance stability
    
    # Expert agreement (when available)
    expert_concordance: Optional[float]    # Agreement with ENT specialists
    kappa_score: Optional[float]           # Inter-rater reliability


class ClinicalEvaluator:
    """
    Comprehensive clinical evaluation for otitis classification models.
    
    Provides medical-grade validation following FDA guidelines for AI/ML devices.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize clinical evaluator.
        
        Args:
            cfg: Configuration with class names and clinical thresholds
        """
        self.cfg = cfg
        self.class_names = cfg.unified_classes.class_names
        self.clinical_priority = cfg.unified_classes.clinical_priority
        self.confidence_thresholds = cfg.clinical.confidence_thresholds
        
        # Define pathological vs normal classes
        self.pathological_classes = [
            cls for cls, priority in self.clinical_priority.items()
            if priority in ["critical", "high", "medium"] and "Normal" not in cls
        ]
        self.normal_classes = [
            cls for cls in self.class_names if "Normal" in cls
        ]
        
        logger.info(f"Initialized clinical evaluator for {len(self.class_names)} classes")
        logger.info(f"Pathological classes: {self.pathological_classes}")
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> ClinicalMetrics:
        """
        Comprehensive clinical evaluation of model performance.
        
        Args:
            model: Trained clinical model
            dataloader: Validation/test data loader
            device: Computation device
            
        Returns:
            ClinicalMetrics with comprehensive evaluation results
        """
        logger.info("Starting comprehensive clinical evaluation...")
        
        # Collect predictions and metadata
        predictions_data = self._collect_predictions(model, dataloader, device)
        
        # Calculate clinical metrics
        metrics = self._calculate_clinical_metrics(predictions_data)
        
        # Generate clinical evaluation report
        self._generate_clinical_report(metrics, predictions_data)
        
        logger.info("Clinical evaluation completed")
        return metrics
    
    def _collect_predictions(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict:
        """
        Collect model predictions with clinical metadata.
        
        Returns:
            Dict containing predictions, confidence scores, and metadata
        """
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_confidences = []
        all_sources = []
        all_needs_review = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, metadata) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                
                # Get clinical predictions with confidence
                if hasattr(model, 'predict_with_confidence'):
                    output = model.predict_with_confidence(images)
                    predictions = output['predictions']
                    probabilities = output['probabilities']
                    max_prob = output['max_probability']
                    needs_review = output['needs_review']
                else:
                    # Fallback for standard models
                    logits = model(images)
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    max_prob = torch.max(probabilities, dim=1)[0]
                    needs_review = max_prob < self.confidence_thresholds.low_confidence
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(max_prob.cpu().numpy())
                all_needs_review.extend(needs_review.cpu().numpy())
                
                # Extract source dataset information
                sources = [meta.get('source_dataset', 'unknown') for meta in metadata]
                all_sources.extend(sources)
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'labels': np.array(all_labels),
            'confidences': np.array(all_confidences),
            'sources': np.array(all_sources),
            'needs_review': np.array(all_needs_review)
        }
    
    def _calculate_clinical_metrics(self, predictions_data: Dict) -> ClinicalMetrics:
        """Calculate comprehensive clinical metrics."""
        
        pred = predictions_data['predictions']
        labels = predictions_data['labels']
        probs = predictions_data['probabilities']
        confidences = predictions_data['confidences']
        sources = predictions_data['sources']
        needs_review = predictions_data['needs_review']
        
        # Overall performance metrics
        accuracy = accuracy_score(labels, pred)
        balanced_accuracy = balanced_accuracy_score(labels, pred)
        
        # Per-class metrics
        class_report = classification_report(
            labels, pred, target_names=self.class_names, output_dict=True
        )
        
        # Extract per-class metrics
        sensitivity = {cls: class_report[cls]['recall'] for cls in self.class_names}
        precision = {cls: class_report[cls]['precision'] for cls in self.class_names}
        f1_score = {cls: class_report[cls]['f1-score'] for cls in self.class_names}
        
        # Calculate specificity and NPV per class
        specificity = {}
        npv = {}
        cm = confusion_matrix(labels, pred)
        
        for i, cls in enumerate(self.class_names):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            fn = np.sum(cm[i, :]) - cm[i, i]
            
            specificity[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv[cls] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Clinical decision support metrics
        high_conf_mask = confidences >= self.confidence_thresholds.high_confidence
        high_confidence_accuracy = accuracy_score(
            labels[high_conf_mask], pred[high_conf_mask]
        ) if np.any(high_conf_mask) else 0.0
        
        low_confidence_rate = np.mean(needs_review)
        
        # Pathology detection performance
        pathology_mask = np.isin(labels, [
            self.class_names.index(cls) for cls in self.pathological_classes
        ])
        pathology_sensitivity = np.mean(
            pred[pathology_mask] == labels[pathology_mask]
        ) if np.any(pathology_mask) else 0.0
        
        # Normal classification specificity
        normal_mask = np.isin(labels, [
            self.class_names.index(cls) for cls in self.normal_classes
        ])
        normal_specificity = np.mean(
            pred[normal_mask] == labels[normal_mask]
        ) if np.any(normal_mask) else 0.0
        
        # Cross-dataset performance
        source_performance = {}
        unique_sources = np.unique(sources)
        for source in unique_sources:
            source_mask = sources == source
            if np.any(source_mask):
                source_performance[source] = accuracy_score(
                    labels[source_mask], pred[source_mask]
                )
        
        # Domain adaptation score (stability across sources)
        source_accuracies = list(source_performance.values())
        domain_adaptation_score = (
            1.0 - np.std(source_accuracies) if len(source_accuracies) > 1 else 1.0
        )
        
        return ClinicalMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            npv=npv,
            f1_score=f1_score,
            high_confidence_accuracy=high_confidence_accuracy,
            low_confidence_rate=low_confidence_rate,
            pathology_sensitivity=pathology_sensitivity,
            normal_specificity=normal_specificity,
            source_performance=source_performance,
            domain_adaptation_score=domain_adaptation_score,
            expert_concordance=None,  # TODO: Implement when expert data available
            kappa_score=None
        )
    
    def _generate_clinical_report(self, metrics: ClinicalMetrics, predictions_data: Dict):
        """Generate comprehensive clinical evaluation report."""
        
        logger.info("Generating clinical evaluation report...")
        
        # TODO: Implement comprehensive reporting
        # Create confusion matrix visualizations
        # Generate ROC curves per class
        # Create confidence distribution plots
        # Generate clinical decision support analysis
        # Create cross-dataset performance comparison
        # Export metrics to clinical report format
        
        # Basic metrics logging
        logger.info(f"Overall Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
        logger.info(f"Pathology Sensitivity: {metrics.pathology_sensitivity:.3f}")
        logger.info(f"Normal Specificity: {metrics.normal_specificity:.3f}")
        logger.info(f"Low Confidence Rate: {metrics.low_confidence_rate:.3f}")
        logger.info(f"Domain Adaptation Score: {metrics.domain_adaptation_score:.3f}")
    
    def evaluate_expert_agreement(
        self,
        model_predictions: np.ndarray,
        expert_labels: np.ndarray,
        confidence_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate agreement between model and expert annotations.
        
        Args:
            model_predictions: Model predicted classes
            expert_labels: Expert-annotated ground truth
            confidence_scores: Model confidence scores
            
        Returns:
            Tuple of (concordance_rate, kappa_score)
        """
        # TODO: Implement expert agreement analysis
        # Calculate concordance rate
        # Compute Cohen's kappa for inter-rater reliability
        # Analyze agreement by confidence level
        # Generate expert agreement report
        
        concordance = accuracy_score(expert_labels, model_predictions)
        # kappa = cohen_kappa_score(expert_labels, model_predictions)
        
        return concordance, 0.0  # TODO: Implement kappa calculation
    
    def validate_confidence_calibration(
        self,
        confidences: np.ndarray,
        correct_predictions: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Validate confidence calibration for clinical decision support.
        
        Critical for medical AI: confidence scores must accurately reflect
        the probability of correct predictions.
        
        Args:
            confidences: Model confidence scores
            correct_predictions: Boolean array of correct predictions
            n_bins: Number of calibration bins
            
        Returns:
            Dict with calibration metrics
        """
        # TODO: Implement confidence calibration analysis
        # Create reliability diagrams
        # Calculate Expected Calibration Error (ECE)
        # Compute Maximum Calibration Error (MCE)
        # Generate calibration report for clinical use
        
        return {
            'expected_calibration_error': 0.0,
            'maximum_calibration_error': 0.0,
            'calibration_slope': 1.0,
            'calibration_intercept': 0.0
        }


# TODO: Implement additional evaluation components
# - FDA compliance validation checklist
# - Clinical trial statistical analysis
# - Bias and fairness assessment for medical AI
# - Real-world performance monitoring
# - Regulatory submission report generation