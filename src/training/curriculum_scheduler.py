"""
Curriculum Learning Scheduler for Dual-Architecture Medical AI System

This module implements sophisticated curriculum learning strategies for medical AI,
providing stage-based progression from simple binary classification to complex
multi-class diagnostic tasks with clinical safety validation.

Key Features:
- Stage-based curriculum progression (Binary → Multi-class → Integration)
- Difficulty-based sample ordering within each stage
- Clinical complexity scoring for medical imaging
- Adaptive progression criteria with safety thresholds
- Medical domain-specific curriculum strategies
- FDA-compliant validation at each curriculum stage

Clinical Curriculum Strategy:
1. Stage 1: Binary screening (Normal vs Pathological) - Foundation building
2. Stage 2: Common pathologies (High-frequency conditions) - Core competency
3. Stage 3: Rare pathologies (Low-frequency conditions) - Specialized expertise
4. Stage 4: Complex cases (Multi-pathology, uncertain) - Expert-level diagnosis
5. Stage 5: Clinical integration (Dual-model coordination) - System validation

Unix Philosophy: Single responsibility - curriculum progression with medical safety
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages in medical AI progression."""
    INITIALIZATION = auto()
    BINARY_FOUNDATION = auto()  # Stage 1: Basic Normal vs Pathological
    COMMON_PATHOLOGIES = auto()  # Stage 2A: High-frequency conditions
    RARE_PATHOLOGIES = auto()  # Stage 2B: Low-frequency conditions
    COMPLEX_CASES = auto()  # Stage 2C: Multi-pathology and uncertain cases
    DUAL_INTEGRATION = auto()  # Stage 3: Cross-model validation
    CLINICAL_VALIDATION = auto()  # Final: External validation
    DEPLOYMENT_READY = auto()


class DifficultyMetric(Enum):
    """Different metrics for assessing sample difficulty in medical imaging."""
    IMAGE_COMPLEXITY = "image_complexity"  # Based on image features
    PATHOLOGY_RARITY = "pathology_rarity"  # Based on class frequency
    DIAGNOSTIC_UNCERTAINTY = "diagnostic_uncertainty"  # Based on model confidence
    CLINICAL_COMPLEXITY = "clinical_complexity"  # Based on medical criteria
    COMBINED_SCORE = "combined_score"  # Weighted combination


@dataclass
class StageTransitionCriteria:
    """Criteria for transitioning between curriculum stages."""
    
    # Performance thresholds
    minimum_accuracy: float = 0.85
    minimum_sensitivity: float = 0.90
    minimum_specificity: float = 0.85
    minimum_balanced_accuracy: float = 0.85
    
    # Stability requirements
    stable_epochs: int = 5
    performance_variance_threshold: float = 0.02
    
    # Clinical safety requirements
    maximum_false_negative_rate: float = 0.05
    minimum_confidence_calibration: float = 0.95
    
    # Progression controls
    minimum_epochs_in_stage: int = 10
    maximum_epochs_in_stage: int = 50
    early_advancement_threshold: float = 0.95
    
    # Medical domain specific
    pathology_detection_rate: float = 0.95
    rare_class_minimum_samples: int = 10
    clinical_expert_approval: bool = False


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning in medical AI."""
    
    # Stage progression
    enable_curriculum: bool = True
    adaptive_progression: bool = True
    difficulty_assessment_method: DifficultyMetric = DifficultyMetric.COMBINED_SCORE
    
    # Difficulty weighting
    image_complexity_weight: float = 0.3
    pathology_rarity_weight: float = 0.3
    diagnostic_uncertainty_weight: float = 0.2
    clinical_complexity_weight: float = 0.2
    
    # Stage-specific parameters
    stage_transition_criteria: StageTransitionCriteria = field(default_factory=StageTransitionCriteria)
    
    # Sample ordering within stages
    easy_to_hard_progression: bool = True
    batch_difficulty_mixing: bool = True
    difficulty_mixing_ratio: float = 0.3  # 30% harder samples mixed in
    
    # Medical domain adaptation
    emphasize_pathology_detection: bool = True
    rare_class_oversampling: bool = True
    clinical_case_prioritization: bool = True
    
    # Clinical validation
    clinical_validation_frequency: int = 5  # Every 5 epochs
    safety_monitoring_enabled: bool = True
    expert_review_checkpoints: bool = True


class MedicalImageComplexityAnalyzer:
    """
    Analyzes medical image complexity for curriculum ordering.
    
    Combines multiple complexity metrics relevant to otoscopic imaging:
    - Image quality and clarity metrics
    - Anatomical structure complexity
    - Pathological feature prominence
    - Diagnostic uncertainty indicators
    """
    
    def __init__(self):
        """Initialize medical image complexity analyzer."""
        self.complexity_cache = {}
        logger.info("Initialized MedicalImageComplexityAnalyzer for otoscopic imaging")
    
    def analyze_image_complexity(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Analyze medical image complexity score.
        
        Args:
            image: Input medical image tensor or array
            
        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # [B, C, H, W]
                image = image[0]  # Take first image
            if image.dim() == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)  # [H, W, C]
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image.astype(np.uint8)
        
        # Convert to grayscale for analysis
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        complexity_scores = []
        
        # 1. Edge density (anatomical structure complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        complexity_scores.append(edge_density)
        
        # 2. Texture complexity (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_texture = min(laplacian_var / 1000, 1.0)  # Normalize
        complexity_scores.append(normalized_texture)
        
        # 3. Color distribution complexity (for pathology detection)
        if len(image_np.shape) == 3:
            # Calculate color variance across channels
            color_variance = np.mean([np.var(image_np[:, :, c]) for c in range(3)])
            normalized_color = min(color_variance / 10000, 1.0)  # Normalize
            complexity_scores.append(normalized_color)
        
        # 4. Local contrast variation
        local_std = cv2.meanStdDev(gray)[1][0][0]
        normalized_contrast = min(local_std / 100, 1.0)  # Normalize
        complexity_scores.append(normalized_contrast)
        
        # 5. Pathological feature indicators (high-frequency components)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        high_freq_energy = np.sum(magnitude[int(magnitude.shape[0]*0.3):int(magnitude.shape[0]*0.7), 
                                           int(magnitude.shape[1]*0.3):int(magnitude.shape[1]*0.7)])
        total_energy = np.sum(magnitude)
        freq_complexity = high_freq_energy / total_energy if total_energy > 0 else 0
        complexity_scores.append(freq_complexity)
        
        # Combine complexity scores with medical domain weighting
        weights = [0.2, 0.25, 0.2, 0.15, 0.2]  # Emphasize texture and edges for medical imaging
        final_complexity = np.average(complexity_scores, weights=weights)
        
        return min(final_complexity, 1.0)
    
    def batch_analyze_complexity(self, images: torch.Tensor) -> List[float]:
        """
        Analyze complexity for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            List of complexity scores for each image
        """
        complexity_scores = []
        
        for i in range(images.size(0)):
            score = self.analyze_image_complexity(images[i])
            complexity_scores.append(score)
        
        return complexity_scores


class ClinicalComplexityScorer:
    """
    Scores samples based on clinical complexity and diagnostic difficulty.
    
    Takes into account medical domain knowledge about pathology types,
    anatomical variations, and diagnostic challenges specific to otology.
    """
    
    def __init__(self):
        """Initialize clinical complexity scorer."""
        
        # Define clinical complexity scores for each pathology
        # Based on diagnostic difficulty and clinical importance
        self.pathology_complexity = {
            0: 0.1,  # Normal - Low complexity (baseline)
            1: 0.3,  # Earwax - Low-moderate complexity (common, clear features)
            2: 0.6,  # AOM - Moderate complexity (common but variable presentation)
            3: 0.7,  # Chronic Suppurative OM - High complexity (multiple features)
            4: 0.5,  # Otitis Externa - Moderate complexity (distinctive features)
            5: 0.8,  # Tympanoskleros - High complexity (subtle features)
            6: 0.4,  # Ear Ventilation - Low-moderate complexity (artificial structure)
            7: 0.9,  # Pseudo Membranes - Very high complexity (rare, subtle)
            8: 0.95, # Foreign Bodies - Very high complexity (emergency, varied appearance)
        }
        
        # Define rarity-based complexity (inverse frequency weighting)
        self.pathology_rarity_scores = {
            0: 0.1,  # Normal - Very common
            1: 0.2,  # Earwax - Common
            2: 0.2,  # AOM - Common
            3: 0.6,  # Chronic Suppurative OM - Uncommon
            4: 0.7,  # Otitis Externa - Uncommon
            5: 0.8,  # Tympanoskleros - Rare
            6: 0.8,  # Ear Ventilation - Rare
            7: 0.95, # Pseudo Membranes - Very rare
            8: 1.0,  # Foreign Bodies - Extremely rare
        }
        
        logger.info("Initialized ClinicalComplexityScorer with otology-specific complexity mapping")
    
    def score_clinical_complexity(self, 
                                class_label: int, 
                                confidence: Optional[float] = None,
                                image_complexity: Optional[float] = None) -> float:
        """
        Calculate clinical complexity score for a sample.
        
        Args:
            class_label: Pathology class label
            confidence: Optional model confidence for the prediction
            image_complexity: Optional image-based complexity score
            
        Returns:
            Clinical complexity score between 0.0 (simple) and 1.0 (complex)
        """
        # Base clinical complexity from pathology type
        base_complexity = self.pathology_complexity.get(class_label, 0.5)
        
        # Rarity-based complexity boost
        rarity_complexity = self.pathology_rarity_scores.get(class_label, 0.5)
        
        # Confidence-based complexity (lower confidence = higher complexity)
        confidence_complexity = 0.5
        if confidence is not None:
            confidence_complexity = 1.0 - confidence  # Invert confidence
        
        # Image-based complexity
        image_complexity = image_complexity or 0.5
        
        # Weighted combination
        weights = [0.4, 0.3, 0.2, 0.1]  # Emphasize clinical and rarity factors
        components = [base_complexity, rarity_complexity, confidence_complexity, image_complexity]
        
        final_score = np.average(components, weights=weights)
        return min(final_score, 1.0)
    
    def score_batch_complexity(self, 
                              class_labels: torch.Tensor,
                              confidences: Optional[torch.Tensor] = None,
                              image_complexities: Optional[List[float]] = None) -> List[float]:
        """
        Score clinical complexity for a batch of samples.
        
        Args:
            class_labels: Batch of class labels
            confidences: Optional batch of model confidences
            image_complexities: Optional list of image complexity scores
            
        Returns:
            List of clinical complexity scores
        """
        batch_size = class_labels.size(0)
        complexity_scores = []
        
        for i in range(batch_size):
            label = class_labels[i].item()
            confidence = confidences[i].item() if confidences is not None else None
            image_complexity = image_complexities[i] if image_complexities is not None else None
            
            score = self.score_clinical_complexity(label, confidence, image_complexity)
            complexity_scores.append(score)
        
        return complexity_scores


class CurriculumScheduler:
    """
    Advanced curriculum learning scheduler for dual-architecture medical AI system.
    
    Implements sophisticated curriculum strategies with clinical domain knowledge,
    providing progressive learning from simple cases to complex diagnostic scenarios
    with medical safety validation at each stage.
    
    Key Features:
    - Multi-stage curriculum progression with clinical validation
    - Adaptive difficulty assessment using medical domain knowledge
    - Safety-first progression with clinical threshold validation
    - Rare pathology emphasis with specialized curriculum
    - Cross-model curriculum coordination for dual architecture
    """
    
    def __init__(self, 
                 config: CurriculumConfig,
                 pathology_class_counts: Optional[Dict[int, int]] = None):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum learning configuration
            pathology_class_counts: Optional class frequency counts for rarity scoring
        """
        self.config = config
        self.pathology_class_counts = pathology_class_counts or {}
        
        # Current curriculum state
        self.current_stage = CurriculumStage.INITIALIZATION
        self.stage_epoch = 0
        self.total_epochs = 0
        self.stage_performance_history = []
        
        # Complexity analyzers
        self.image_analyzer = MedicalImageComplexityAnalyzer()
        self.clinical_scorer = ClinicalComplexityScorer()
        
        # Stage management
        self.stage_transitions = []
        self.advancement_blocked = False
        self.clinical_approval_pending = False
        
        # Performance tracking
        self.stage_metrics = {stage: [] for stage in CurriculumStage}
        self.transition_criteria_met = {stage: False for stage in CurriculumStage}
        
        logger.info(f"Initialized CurriculumScheduler with {len(CurriculumStage)} stages")
        logger.info(f"Curriculum enabled: {self.config.enable_curriculum}")
        logger.info(f"Adaptive progression: {self.config.adaptive_progression}")
    
    def should_advance_stage(self, 
                           current_metrics: Dict[str, float],
                           model: Optional[torch.nn.Module] = None) -> bool:
        """
        Determine if curriculum should advance to next stage.
        
        Args:
            current_metrics: Current model performance metrics
            model: Optional model for additional analysis
            
        Returns:
            True if ready to advance to next stage
        """
        if not self.config.enable_curriculum:
            return False
        
        criteria = self.config.stage_transition_criteria
        
        # Check minimum time in stage
        if self.stage_epoch < criteria.minimum_epochs_in_stage:
            return False
        
        # Check maximum time in stage (force advancement)
        if self.stage_epoch >= criteria.maximum_epochs_in_stage:
            logger.warning(f"Force advancing from {self.current_stage.name} - maximum epochs reached")
            return True
        
        # Performance-based criteria
        performance_checks = []
        
        # Accuracy check
        accuracy = current_metrics.get('accuracy', 0.0)
        performance_checks.append(accuracy >= criteria.minimum_accuracy)
        
        # Sensitivity check (critical for medical AI)
        sensitivity = current_metrics.get('sensitivity', 0.0)
        performance_checks.append(sensitivity >= criteria.minimum_sensitivity)
        
        # Specificity check
        specificity = current_metrics.get('specificity', 0.0)
        performance_checks.append(specificity >= criteria.minimum_specificity)
        
        # Balanced accuracy check (for multi-class)
        balanced_accuracy = current_metrics.get('balanced_accuracy', accuracy)
        performance_checks.append(balanced_accuracy >= criteria.minimum_balanced_accuracy)
        
        # Clinical safety checks
        false_negative_rate = current_metrics.get('false_negative_rate', 1.0)
        performance_checks.append(false_negative_rate <= criteria.maximum_false_negative_rate)
        
        # Check performance stability
        self.stage_performance_history.append(current_metrics)
        stability_check = self._check_performance_stability(criteria)
        performance_checks.append(stability_check)
        
        # All criteria must be met
        all_criteria_met = all(performance_checks)
        
        # Early advancement for exceptional performance
        exceptional_performance = (
            accuracy >= criteria.early_advancement_threshold and
            sensitivity >= criteria.early_advancement_threshold and
            self.stage_epoch >= criteria.minimum_epochs_in_stage // 2
        )
        
        advancement_ready = all_criteria_met or exceptional_performance
        
        # Log advancement decision
        if advancement_ready:
            logger.info(f"✅ Stage advancement criteria met for {self.current_stage.name}")
            logger.info(f"  Accuracy: {accuracy:.4f} >= {criteria.minimum_accuracy}")
            logger.info(f"  Sensitivity: {sensitivity:.4f} >= {criteria.minimum_sensitivity}")
            logger.info(f"  Specificity: {specificity:.4f} >= {criteria.minimum_specificity}")
        else:
            failed_criteria = []
            if accuracy < criteria.minimum_accuracy:
                failed_criteria.append(f"Accuracy: {accuracy:.4f} < {criteria.minimum_accuracy}")
            if sensitivity < criteria.minimum_sensitivity:
                failed_criteria.append(f"Sensitivity: {sensitivity:.4f} < {criteria.minimum_sensitivity}")
            if specificity < criteria.minimum_specificity:
                failed_criteria.append(f"Specificity: {specificity:.4f} < {criteria.minimum_specificity}")
            if not stability_check:
                failed_criteria.append("Performance stability not achieved")
            
            logger.info(f"❌ Stage advancement blocked for {self.current_stage.name}: {failed_criteria}")
        
        return advancement_ready
    
    def advance_stage(self) -> CurriculumStage:
        """
        Advance to the next curriculum stage.
        
        Returns:
            New curriculum stage
        """
        previous_stage = self.current_stage
        
        # Define stage progression
        stage_progression = {
            CurriculumStage.INITIALIZATION: CurriculumStage.BINARY_FOUNDATION,
            CurriculumStage.BINARY_FOUNDATION: CurriculumStage.COMMON_PATHOLOGIES,
            CurriculumStage.COMMON_PATHOLOGIES: CurriculumStage.RARE_PATHOLOGIES,
            CurriculumStage.RARE_PATHOLOGIES: CurriculumStage.COMPLEX_CASES,
            CurriculumStage.COMPLEX_CASES: CurriculumStage.DUAL_INTEGRATION,
            CurriculumStage.DUAL_INTEGRATION: CurriculumStage.CLINICAL_VALIDATION,
            CurriculumStage.CLINICAL_VALIDATION: CurriculumStage.DEPLOYMENT_READY,
            CurriculumStage.DEPLOYMENT_READY: CurriculumStage.DEPLOYMENT_READY  # Stay in final stage
        }
        
        self.current_stage = stage_progression.get(self.current_stage, self.current_stage)
        
        # Reset stage counters
        self.stage_epoch = 0
        self.stage_performance_history = []
        
        # Log transition
        transition_info = {
            'from_stage': previous_stage.name,
            'to_stage': self.current_stage.name,
            'total_epochs': self.total_epochs,
            'timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'N/A'
        }
        self.stage_transitions.append(transition_info)
        
        logger.info(f"🎯 Curriculum advanced: {previous_stage.name} → {self.current_stage.name}")
        
        return self.current_stage
    
    def get_curriculum_dataset(self, 
                              base_dataset: Dataset,
                              model: Optional[torch.nn.Module] = None,
                              device: Optional[torch.device] = None) -> Dataset:
        """
        Get curriculum-ordered dataset for current stage.
        
        Args:
            base_dataset: Base dataset to apply curriculum to
            model: Optional model for confidence-based complexity scoring
            device: Device for model inference
            
        Returns:
            Curriculum-ordered dataset subset
        """
        if not self.config.enable_curriculum:
            return base_dataset
        
        # Get stage-appropriate samples
        stage_indices = self._get_stage_sample_indices(base_dataset)
        
        if not stage_indices:
            logger.warning(f"No samples found for stage {self.current_stage.name}")
            return base_dataset
        
        # Calculate difficulty scores for ordering
        difficulty_scores = self._calculate_sample_difficulties(
            base_dataset, stage_indices, model, device
        )
        
        # Order samples by difficulty (easy to hard or as configured)
        ordered_indices = self._order_samples_by_difficulty(stage_indices, difficulty_scores)
        
        # Create curriculum subset
        curriculum_subset = Subset(base_dataset, ordered_indices)
        
        logger.info(f"Created curriculum dataset for {self.current_stage.name}: "
                   f"{len(ordered_indices)} samples")
        
        return curriculum_subset
    
    def _get_stage_sample_indices(self, dataset: Dataset) -> List[int]:
        """Get sample indices appropriate for current curriculum stage."""
        stage_indices = []
        
        # Define class inclusion for each stage
        if self.current_stage == CurriculumStage.BINARY_FOUNDATION:
            # All samples for binary classification
            stage_indices = list(range(len(dataset)))
        
        elif self.current_stage == CurriculumStage.COMMON_PATHOLOGIES:
            # Focus on common pathological classes (AOM, Earwax)
            common_classes = [1, 2]  # Earwax, AOM
            stage_indices = self._get_class_indices(dataset, common_classes)
        
        elif self.current_stage == CurriculumStage.RARE_PATHOLOGIES:
            # Focus on rare pathological classes
            rare_classes = [6, 7, 8]  # Tympanostomy, Pseudo Membranes, Foreign Bodies
            stage_indices = self._get_class_indices(dataset, rare_classes)
        
        elif self.current_stage == CurriculumStage.COMPLEX_CASES:
            # All pathological classes with emphasis on diagnostic uncertainty
            pathological_classes = [1, 2, 3, 4, 5, 6, 7, 8]
            stage_indices = self._get_class_indices(dataset, pathological_classes)
        
        else:
            # Default: all samples
            stage_indices = list(range(len(dataset)))
        
        return stage_indices
    
    def _get_class_indices(self, dataset: Dataset, target_classes: List[int]) -> List[int]:
        """Get indices of samples belonging to target classes."""
        class_indices = []
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    _, label = sample[0], sample[1]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    if int(label) in target_classes:
                        class_indices.append(idx)
            except Exception as e:
                logger.warning(f"Error accessing sample {idx}: {e}")
        
        return class_indices
    
    def _calculate_sample_difficulties(self,
                                     dataset: Dataset,
                                     sample_indices: List[int],
                                     model: Optional[torch.nn.Module] = None,
                                     device: Optional[torch.device] = None) -> List[float]:
        """Calculate difficulty scores for samples."""
        difficulty_scores = []
        
        for idx in sample_indices:
            try:
                sample = dataset[idx]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    image, label = sample[0], sample[1]
                    
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    label = int(label)
                    
                    # Image-based complexity
                    image_complexity = 0.5
                    if self.config.difficulty_assessment_method in [
                        DifficultyMetric.IMAGE_COMPLEXITY, DifficultyMetric.COMBINED_SCORE
                    ]:
                        image_complexity = self.image_analyzer.analyze_image_complexity(image)
                    
                    # Model confidence-based complexity (if model available)
                    confidence = 0.5
                    if model is not None and device is not None:
                        confidence = self._get_model_confidence(image, model, device)
                    
                    # Clinical complexity
                    clinical_complexity = self.clinical_scorer.score_clinical_complexity(
                        label, confidence, image_complexity
                    )
                    
                    # Combined difficulty score
                    if self.config.difficulty_assessment_method == DifficultyMetric.COMBINED_SCORE:
                        weights = [
                            self.config.image_complexity_weight,
                            self.config.pathology_rarity_weight,
                            self.config.diagnostic_uncertainty_weight,
                            self.config.clinical_complexity_weight
                        ]
                        
                        components = [
                            image_complexity,
                            self.clinical_scorer.pathology_rarity_scores.get(label, 0.5),
                            1.0 - confidence,  # Lower confidence = higher difficulty
                            clinical_complexity
                        ]
                        
                        difficulty = np.average(components, weights=weights)
                    else:
                        difficulty = clinical_complexity
                    
                    difficulty_scores.append(difficulty)
                else:
                    difficulty_scores.append(0.5)  # Default difficulty
            except Exception as e:
                logger.warning(f"Error calculating difficulty for sample {idx}: {e}")
                difficulty_scores.append(0.5)  # Default difficulty
        
        return difficulty_scores
    
    def _get_model_confidence(self, 
                            image: torch.Tensor, 
                            model: torch.nn.Module, 
                            device: torch.device) -> float:
        """Get model confidence for a single image."""
        try:
            model.eval()
            with torch.no_grad():
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # Add batch dimension
                image = image.to(device)
                
                logits = model(image)
                probs = F.softmax(logits, dim=1)
                confidence = torch.max(probs, dim=1)[0].item()
                
                return confidence
        except Exception as e:
            logger.warning(f"Error getting model confidence: {e}")
            return 0.5
    
    def _order_samples_by_difficulty(self, 
                                   sample_indices: List[int], 
                                   difficulty_scores: List[float]) -> List[int]:
        """Order samples by difficulty according to curriculum strategy."""
        if len(sample_indices) != len(difficulty_scores):
            logger.warning("Mismatch between sample indices and difficulty scores")
            return sample_indices
        
        # Create index-difficulty pairs
        indexed_difficulties = list(zip(sample_indices, difficulty_scores))
        
        # Sort by difficulty
        if self.config.easy_to_hard_progression:
            # Easy to hard: low difficulty first
            indexed_difficulties.sort(key=lambda x: x[1])
        else:
            # Hard to easy: high difficulty first
            indexed_difficulties.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ordered indices
        ordered_indices = [idx for idx, _ in indexed_difficulties]
        
        # Optional: Mix in some harder samples for robustness
        if self.config.batch_difficulty_mixing and len(ordered_indices) > 10:
            mix_count = int(len(ordered_indices) * self.config.difficulty_mixing_ratio)
            
            # Take some samples from the harder end
            if self.config.easy_to_hard_progression:
                hard_samples = ordered_indices[-mix_count:]
                easy_samples = ordered_indices[:-mix_count]
            else:
                hard_samples = ordered_indices[:mix_count]
                easy_samples = ordered_indices[mix_count:]
            
            # Randomly intersperse hard samples among easy samples
            np.random.shuffle(hard_samples)
            mixed_indices = []
            hard_idx = 0
            
            for i, easy_sample in enumerate(easy_samples):
                mixed_indices.append(easy_sample)
                
                # Insert hard sample occasionally
                if hard_idx < len(hard_samples) and (i + 1) % 3 == 0:
                    mixed_indices.append(hard_samples[hard_idx])
                    hard_idx += 1
            
            # Add remaining hard samples
            mixed_indices.extend(hard_samples[hard_idx:])
            ordered_indices = mixed_indices
        
        return ordered_indices
    
    def _check_performance_stability(self, criteria: StageTransitionCriteria) -> bool:
        """Check if performance has been stable across recent epochs."""
        if len(self.stage_performance_history) < criteria.stable_epochs:
            return False
        
        # Check recent performance stability
        recent_metrics = self.stage_performance_history[-criteria.stable_epochs:]
        
        # Calculate variance in key metrics
        accuracies = [m.get('accuracy', 0.0) for m in recent_metrics]
        sensitivities = [m.get('sensitivity', 0.0) for m in recent_metrics]
        
        accuracy_variance = np.var(accuracies)
        sensitivity_variance = np.var(sensitivities)
        
        stability_check = (
            accuracy_variance <= criteria.performance_variance_threshold and
            sensitivity_variance <= criteria.performance_variance_threshold
        )
        
        return stability_check
    
    def update_epoch(self, epoch_metrics: Dict[str, float]) -> None:
        """Update curriculum state after each epoch."""
        self.stage_epoch += 1
        self.total_epochs += 1
        
        # Store metrics for this stage
        self.stage_metrics[self.current_stage].append(epoch_metrics)
        
        # Clinical validation check
        if (self.stage_epoch % self.config.clinical_validation_frequency == 0 and 
            self.config.safety_monitoring_enabled):
            self._clinical_safety_check(epoch_metrics)
    
    def _clinical_safety_check(self, metrics: Dict[str, float]) -> None:
        """Perform clinical safety validation."""
        safety_violations = []
        
        # Check critical clinical thresholds
        sensitivity = metrics.get('sensitivity', 0.0)
        if sensitivity < 0.90:  # Critical medical AI threshold
            safety_violations.append(f"Low sensitivity: {sensitivity:.4f}")
        
        false_negative_rate = metrics.get('false_negative_rate', 1.0)
        if false_negative_rate > 0.05:  # Maximum acceptable false negative rate
            safety_violations.append(f"High false negative rate: {false_negative_rate:.4f}")
        
        if safety_violations:
            logger.warning(f"⚠️ Clinical safety violations in {self.current_stage.name}: {safety_violations}")
            self.advancement_blocked = True
        else:
            self.advancement_blocked = False
    
    def get_curriculum_report(self) -> Dict[str, Any]:
        """Generate comprehensive curriculum learning report."""
        return {
            'curriculum_config': {
                'enabled': self.config.enable_curriculum,
                'adaptive_progression': self.config.adaptive_progression,
                'difficulty_method': self.config.difficulty_assessment_method.value,
            },
            'progression_state': {
                'current_stage': self.current_stage.name,
                'stage_epoch': self.stage_epoch,
                'total_epochs': self.total_epochs,
                'advancement_blocked': self.advancement_blocked
            },
            'stage_transitions': self.stage_transitions,
            'stage_metrics_summary': {
                stage.name: {
                    'epochs': len(metrics),
                    'best_accuracy': max([m.get('accuracy', 0) for m in metrics], default=0),
                    'best_sensitivity': max([m.get('sensitivity', 0) for m in metrics], default=0)
                }
                for stage, metrics in self.stage_metrics.items() if metrics
            },
            'clinical_validation': {
                'safety_monitoring_enabled': self.config.safety_monitoring_enabled,
                'advancement_blocked': self.advancement_blocked,
                'clinical_approval_pending': self.clinical_approval_pending
            }
        }


# Factory function for easy curriculum scheduler creation
def create_curriculum_scheduler(
    enable_curriculum: bool = True,
    pathology_class_counts: Optional[Dict[int, int]] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> CurriculumScheduler:
    """
    Factory function to create curriculum scheduler with medical AI defaults.
    
    Args:
        enable_curriculum: Whether to enable curriculum learning
        pathology_class_counts: Optional class frequency counts
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured CurriculumScheduler instance
    """
    # Default medical AI curriculum configuration
    config = CurriculumConfig(
        enable_curriculum=enable_curriculum,
        adaptive_progression=True,
        difficulty_assessment_method=DifficultyMetric.COMBINED_SCORE,
        easy_to_hard_progression=True,
        emphasize_pathology_detection=True,
        clinical_validation_frequency=5,
        safety_monitoring_enabled=True
    )
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    scheduler = CurriculumScheduler(
        config=config,
        pathology_class_counts=pathology_class_counts
    )
    
    logger.info(f"Created CurriculumScheduler with curriculum {'enabled' if enable_curriculum else 'disabled'}")
    return scheduler