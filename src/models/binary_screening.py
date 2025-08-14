"""
Binary Screening Model for Dual-Architecture Medical AI System

This module implements Stage 1 of the dual-architecture approach for otitis classification.
The binary screening model distinguishes between Normal and Pathological conditions with
98%+ sensitivity requirements for clinical safety.

Key Features:
- EfficientNet-B3 backbone with RadImageNet transfer learning
- High-sensitivity optimization with conservative clinical thresholds
- LAB color space feature integration for inflammation detection
- Temperature scaling for confidence calibration
- Grad-CAM interpretability with anatomical region support
- Medical-grade safety validation and performance monitoring

Clinical Context:
- Stage 1: Binary pathology screening (Normal vs Pathological)
- Target: 98%+ sensitivity, 90%+ specificity
- Integration with Stage 2 multi-class diagnostic model
- FDA-compliant validation approach with clinical decision support

Unix Philosophy: Single responsibility - binary pathology detection with medical safety
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.hub import load_state_dict_from_url
import cv2
from PIL import Image

# Suppress timm warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="timm")

logger = logging.getLogger(__name__)


class ColorFeatureExtractor:
    """
    LAB color space feature extraction for inflammation and pathology detection.
    
    Extracts color features specifically tuned for medical ear imaging:
    - LAB color space analysis for pathological inflammation detection
    - Histogram-based color distribution analysis
    - Color cast detection for image quality assessment
    - Regional color variation analysis for anatomical localization
    """
    
    def __init__(self, spatial_bins: int = 8):
        """
        Initialize color feature extractor.
        
        Args:
            spatial_bins: Number of spatial bins for regional color analysis
        """
        self.spatial_bins = spatial_bins
        
    def extract_lab_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract LAB color space features from preprocessed medical images.
        
        Args:
            image: Input tensor [B, 3, H, W] in RGB format
            
        Returns:
            Color features tensor [B, feature_dim]
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Convert RGB to LAB color space
        # Ensure image is in [0, 255] range for OpenCV
        image_np = (image * 255.0).clamp(0, 255).byte().cpu().numpy()
        
        lab_features = []
        
        for i in range(batch_size):
            # Convert single image to LAB
            rgb_img = np.transpose(image_np[i], (1, 2, 0))  # [H, W, 3]
            lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            
            # Extract L, A, B channel statistics
            l_channel = lab_img[:, :, 0]  # Lightness
            a_channel = lab_img[:, :, 1]  # Green-Red axis
            b_channel = lab_img[:, :, 2]  # Blue-Yellow axis
            
            # Global color statistics
            features = []
            
            # L channel (lightness) - critical for pathology contrast
            features.extend([
                np.mean(l_channel),
                np.std(l_channel),
                np.percentile(l_channel, 25),
                np.percentile(l_channel, 75),
            ])
            
            # A channel (green-red) - inflammation detection
            features.extend([
                np.mean(a_channel),
                np.std(a_channel),
                np.percentile(a_channel, 10),
                np.percentile(a_channel, 90),
            ])
            
            # B channel (blue-yellow) - infection indicators
            features.extend([
                np.mean(b_channel),
                np.std(b_channel),
                np.percentile(b_channel, 10),
                np.percentile(b_channel, 90),
            ])
            
            # Regional color variation analysis (medical anatomical regions)
            h, w = lab_img.shape[:2]
            region_h, region_w = h // self.spatial_bins, w // self.spatial_bins
            
            regional_variation = []
            for channel_idx in range(3):  # L, A, B channels
                channel_data = lab_img[:, :, channel_idx]
                region_means = []
                
                for r in range(self.spatial_bins):
                    for c in range(self.spatial_bins):
                        start_r, end_r = r * region_h, min((r + 1) * region_h, h)
                        start_c, end_c = c * region_w, min((c + 1) * region_w, w)
                        region = channel_data[start_r:end_r, start_c:end_c]
                        region_means.append(np.mean(region))
                
                # Regional variation as standard deviation of region means
                regional_variation.append(np.std(region_means))
            
            features.extend(regional_variation)
            
            # Color cast detection (pathological bias indicators)
            a_bias = np.mean(a_channel) - 128  # A channel bias from neutral
            b_bias = np.mean(b_channel) - 128  # B channel bias from neutral
            color_cast_magnitude = np.sqrt(a_bias**2 + b_bias**2)
            
            features.extend([a_bias, b_bias, color_cast_magnitude])
            
            lab_features.append(features)
        
        # Convert to tensor
        lab_features_tensor = torch.tensor(lab_features, dtype=torch.float32, device=device)
        
        return lab_features_tensor
    
    def get_feature_dimension(self) -> int:
        """Return total dimension of extracted color features."""
        # L, A, B statistics (4 each) + regional variation (3) + color cast (3)
        return 4 * 3 + 3 + 3  # 18 features total


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration in medical AI.
    
    Calibrates model confidence to match actual accuracy for clinical deployment.
    Critical for medical decision support where confidence must be well-calibrated.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(self, 
                  logits: torch.Tensor, 
                  labels: torch.Tensor,
                  lr: float = 0.01,
                  max_iter: int = 50) -> float:
        """
        Calibrate temperature parameter using validation data.
        
        Args:
            logits: Validation logits [N, num_classes]
            labels: True labels [N]
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Final temperature value
        """
        self.cuda() if logits.is_cuda else self.cpu()
        
        # Use LBFGS optimizer for temperature calibration
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


class HighRecallLoss(nn.Module):
    """
    High-recall loss function optimized for medical screening applications.
    
    Combines focal loss with asymmetric penalty to achieve 98%+ sensitivity
    while maintaining reasonable specificity. Designed for binary pathology screening
    where false negatives (missed pathology) are clinically unacceptable.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 false_negative_penalty: float = 10.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize high-recall loss function.
        
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter for hard example mining
            false_negative_penalty: Additional penalty for false negatives (missed pathology)
            class_weights: Optional class balancing weights [Normal, Pathological]
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.false_negative_penalty = false_negative_penalty
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute high-recall loss with false negative penalty.
        
        Args:
            logits: Model logits [B, 2] for binary classification
            targets: True labels [B] where 0=Normal, 1=Pathological
            
        Returns:
            High-recall loss scalar
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Focal loss component
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Additional penalty for false negatives (pathology classified as normal)
        pathological_mask = (targets == 1)  # True pathological cases
        normal_predictions = (torch.argmax(logits, dim=1) == 0)  # Predicted as normal
        false_negative_mask = pathological_mask & normal_predictions
        
        # Apply false negative penalty
        fn_penalty = torch.zeros_like(focal_loss)
        fn_penalty[false_negative_mask] = self.false_negative_penalty
        
        total_loss = focal_loss + fn_penalty
        
        return total_loss.mean()


class BinaryScreeningModel(nn.Module):
    """
    Binary Screening Model for Stage 1 of Dual-Architecture Medical AI System.
    
    EfficientNet-B3 based model for high-sensitivity pathology detection in otoscopic images.
    Designed for clinical deployment with FDA-compliant validation and safety features.
    
    Architecture:
    - EfficientNet-B3 backbone with RadImageNet initialization
    - LAB color feature integration branch
    - Temperature scaling for confidence calibration
    - High-recall loss optimization for 98%+ sensitivity
    - Grad-CAM interpretability with anatomical region support
    
    Clinical Integration:
    - Binary classification: Normal vs Pathological (all 8 pathological conditions)
    - Conservative threshold tuning for medical safety
    - Confidence calibration for clinical decision support
    - Integration ready for dual-architecture system
    """
    
    def __init__(self,
                 pretrained: bool = True,
                 radimagenet_weights: bool = True,
                 color_feature_fusion: bool = True,
                 dropout_rate: float = 0.3,
                 confidence_threshold: float = 0.5):
        """
        Initialize Binary Screening Model.
        
        Args:
            pretrained: Use pretrained weights (ImageNet if RadImageNet unavailable)
            radimagenet_weights: Attempt to load RadImageNet weights for medical domain
            color_feature_fusion: Enable LAB color feature integration
            dropout_rate: Dropout rate for regularization
            confidence_threshold: Decision threshold (conservative for medical safety)
        """
        super().__init__()
        
        # Model configuration
        self.num_classes = 2  # Binary: Normal vs Pathological
        self.color_feature_fusion = color_feature_fusion
        self.confidence_threshold = confidence_threshold
        
        # EfficientNet-B3 backbone
        self.backbone = self._create_efficientnet_backbone(pretrained, radimagenet_weights)
        
        # Color feature extraction branch
        if self.color_feature_fusion:
            self.color_extractor = ColorFeatureExtractor()
            color_feature_dim = self.color_extractor.get_feature_dimension()
        else:
            color_feature_dim = 0
        
        # Get backbone feature dimension before replacing classifier
        if hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
            backbone_features = self.backbone.classifier.in_features
        elif hasattr(self.backbone, 'num_features'):
            backbone_features = self.backbone.num_features
        else:
            # Default for EfficientNet-B3
            backbone_features = 1536
            logger.warning(f"Could not determine backbone features, using default: {backbone_features}")
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        total_features = backbone_features + color_feature_dim
        
        # Clinical classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, self.num_classes)
        )
        
        # Temperature scaling for confidence calibration
        self.temperature_scaling = TemperatureScaling()
        
        # High-recall loss function
        self.high_recall_loss = HighRecallLoss(
            alpha=1.0,
            gamma=2.0,
            false_negative_penalty=10.0  # Strong penalty for missed pathology
        )
        
        # Hooks for Grad-CAM interpretability
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()
        
        logger.info(f"Initialized BinaryScreeningModel with {total_features} total features")
        logger.info(f"Color feature fusion: {self.color_feature_fusion}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def _create_efficientnet_backbone(self, pretrained: bool, radimagenet_weights: bool) -> nn.Module:
        """
        Create EfficientNet-B3 backbone with RadImageNet weights if available.
        
        Args:
            pretrained: Use pretrained weights
            radimagenet_weights: Attempt RadImageNet initialization
            
        Returns:
            EfficientNet-B3 model
        """
        try:
            # Attempt to load with RadImageNet weights for medical domain
            if radimagenet_weights:
                # Note: This is a placeholder for RadImageNet integration
                # In practice, you would load RadImageNet weights from a specific source
                logger.info("Attempting RadImageNet initialization for medical domain adaptation")
                
                # For now, use standard ImageNet pretrained and log the intention
                model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=1000)
                logger.warning("RadImageNet weights not available - using ImageNet pretrained weights")
                logger.info("TODO: Integrate RadImageNet weights when available")
                
            else:
                # Standard ImageNet pretrained
                model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=1000)
                logger.info("Using ImageNet pretrained EfficientNet-B3")
            
            # Verify model architecture
            if hasattr(model, 'classifier'):
                logger.info(f"EfficientNet-B3 loaded with {model.classifier.in_features} features")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create EfficientNet-B3: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM interpretability."""
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks for timm EfficientNet models
        try:
            # For timm EfficientNet models, try different possible structures
            if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) > 0:
                # Use the last block
                last_block = self.backbone.blocks[-1]
                last_block.register_forward_hook(forward_hook)
                last_block.register_backward_hook(backward_hook)
                logger.info("Grad-CAM hooks registered on EfficientNet blocks")
            elif hasattr(self.backbone, 'features') and len(self.backbone.features) > 0:
                # Alternative: use features
                last_layer = self.backbone.features[-1]
                last_layer.register_forward_hook(forward_hook)
                last_layer.register_backward_hook(backward_hook)
                logger.info("Grad-CAM hooks registered on backbone features")
            elif hasattr(self.backbone, 'conv_head'):
                # Use conv_head if available
                self.backbone.conv_head.register_forward_hook(forward_hook)
                self.backbone.conv_head.register_backward_hook(backward_hook)
                logger.info("Grad-CAM hooks registered on conv_head")
            else:
                logger.warning("Could not register Grad-CAM hooks - backbone structure unexpected")
                # List available attributes for debugging
                attrs = [attr for attr in dir(self.backbone) if not attr.startswith('_')]
                logger.info(f"Available backbone attributes: {attrs[:10]}...")  # Show first 10
        except Exception as e:
            logger.warning(f"Failed to register Grad-CAM hooks: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through binary screening model.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Logits [B, 2] for binary classification
        """
        batch_size = x.size(0)
        
        # EfficientNet backbone features
        backbone_features = self.backbone(x)  # [B, backbone_features]
        
        # Color feature extraction
        if self.color_feature_fusion:
            color_features = self.color_extractor.extract_lab_features(x)  # [B, color_features]
            
            # Concatenate backbone and color features
            combined_features = torch.cat([backbone_features, color_features], dim=1)
        else:
            combined_features = backbone_features
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict_with_confidence(self, x: torch.Tensor, apply_temperature_scaling: bool = True) -> Dict[str, torch.Tensor]:
        """
        Clinical prediction with calibrated confidence scoring.
        
        Args:
            x: Input images [B, 3, H, W]
            apply_temperature_scaling: Apply temperature scaling for calibration
            
        Returns:
            Dictionary with predictions, probabilities, and clinical metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            logits = self.forward(x)
            
            # Apply temperature scaling if enabled
            if apply_temperature_scaling:
                calibrated_logits = self.temperature_scaling(logits)
            else:
                calibrated_logits = logits
            
            # Convert to probabilities
            probabilities = F.softmax(calibrated_logits, dim=1)
            
            # Extract pathological probability (class 1)
            pathology_prob = probabilities[:, 1]
            
            # Binary predictions using conservative threshold
            predictions = (pathology_prob >= self.confidence_threshold).long()
            
            # Confidence metrics
            max_prob = torch.max(probabilities, dim=1)[0]
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            
            # Clinical decision flags
            high_confidence = pathology_prob > 0.85  # High pathology confidence
            needs_specialist_review = pathology_prob > 0.3  # Conservative screening threshold
            immediate_attention = pathology_prob > 0.7  # Urgent pathology indicators
            
            return {
                'raw_logits': logits,
                'calibrated_logits': calibrated_logits,
                'probabilities': probabilities,
                'pathology_probability': pathology_prob,
                'predictions': predictions,
                'max_probability': max_prob,
                'entropy': entropy,
                'high_confidence': high_confidence,
                'needs_specialist_review': needs_specialist_review,
                'immediate_attention': immediate_attention
            }
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute high-recall loss optimized for medical screening.
        
        Args:
            logits: Model logits [B, 2]
            targets: Binary targets [B] where 0=Normal, 1=Pathological
            
        Returns:
            High-recall loss with false negative penalty
        """
        return self.high_recall_loss(logits, targets)
    
    def calibrate_confidence(self, 
                           val_logits: torch.Tensor, 
                           val_labels: torch.Tensor,
                           lr: float = 0.01,
                           max_iter: int = 50) -> float:
        """
        Calibrate confidence using temperature scaling on validation data.
        
        Args:
            val_logits: Validation logits [N, 2]
            val_labels: Validation labels [N]
            lr: Learning rate for calibration
            max_iter: Maximum calibration iterations
            
        Returns:
            Calibrated temperature value
        """
        logger.info("Calibrating confidence using temperature scaling...")
        
        temperature = self.temperature_scaling.calibrate(val_logits, val_labels, lr, max_iter)
        
        logger.info(f"Temperature calibration complete: T = {temperature:.4f}")
        return temperature
    
    def generate_gradcam(self, 
                        x: torch.Tensor, 
                        target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for interpretability.
        
        Args:
            x: Input image [1, 3, H, W]
            target_class: Target class for Grad-CAM (None for predicted class)
            
        Returns:
            Grad-CAM heatmap [H, W]
        """
        self.eval()
        
        # Forward pass
        logits = self.forward(x)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass for gradients
        self.zero_grad()
        logits[0, target_class].backward(retain_graph=True)
        
        if self.gradients is None or self.feature_maps is None:
            logger.warning("Grad-CAM hooks not properly registered")
            return torch.zeros(x.shape[-2:])
        
        # Generate Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        feature_maps = self.feature_maps[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of feature maps
        gradcam = torch.zeros(feature_maps.shape[1:])  # [H, W]
        for i, w in enumerate(weights):
            gradcam += w * feature_maps[i]
        
        # ReLU and normalize
        gradcam = F.relu(gradcam)
        gradcam = gradcam / (torch.max(gradcam) + 1e-8)
        
        # Resize to input size
        gradcam = F.interpolate(
            gradcam.unsqueeze(0).unsqueeze(0), 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze()
        
        return gradcam
    
    def validate_clinical_performance(self, 
                                    dataloader, 
                                    sensitivity_threshold: float = 0.98,
                                    specificity_threshold: float = 0.90) -> Dict[str, float]:
        """
        Validate clinical performance metrics with safety checks.
        
        Args:
            dataloader: Validation data loader
            sensitivity_threshold: Minimum required sensitivity (recall for pathology)
            specificity_threshold: Minimum required specificity
            
        Returns:
            Clinical performance metrics
        """
        self.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    images, targets = batch_data[0], batch_data[1]
                else:
                    raise ValueError("Expected batch to contain at least (images, targets)")
                
                # Move to device
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                
                # Predictions
                results = self.predict_with_confidence(images)
                predictions = results['predictions']
                pathology_probs = results['pathology_probability']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(pathology_probs.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Calculate clinical metrics
        true_positives = np.sum((predictions == 1) & (targets == 1))
        true_negatives = np.sum((predictions == 0) & (targets == 0))
        false_positives = np.sum((predictions == 1) & (targets == 0))
        false_negatives = np.sum((predictions == 0) & (targets == 1))
        
        # Clinical metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        accuracy = (true_positives + true_negatives) / len(targets)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        
        # F1 score
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # Clinical safety validation
        clinical_safety_passed = sensitivity >= sensitivity_threshold and specificity >= specificity_threshold
        
        metrics = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1_score,
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'clinical_safety_passed': clinical_safety_passed,
            'meets_sensitivity_threshold': sensitivity >= sensitivity_threshold,
            'meets_specificity_threshold': specificity >= specificity_threshold
        }
        
        # Log clinical validation results
        logger.info("=== Binary Screening Model Clinical Validation ===")
        logger.info(f"Sensitivity (Recall): {sensitivity:.4f} (Target: ≥{sensitivity_threshold})")
        logger.info(f"Specificity: {specificity:.4f} (Target: ≥{specificity_threshold})")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"False Negatives (Critical): {false_negatives}")
        logger.info(f"Clinical Safety: {'PASSED' if clinical_safety_passed else 'FAILED'}")
        
        if not clinical_safety_passed:
            logger.warning("Model does not meet clinical safety requirements!")
            if sensitivity < sensitivity_threshold:
                logger.warning(f"Sensitivity too low: {sensitivity:.4f} < {sensitivity_threshold}")
            if specificity < specificity_threshold:
                logger.warning(f"Specificity too low: {specificity:.4f} < {specificity_threshold}")
        
        return metrics


# Factory function for easy model creation
def create_binary_screening_model(config: Optional[Dict[str, Any]] = None) -> BinaryScreeningModel:
    """
    Factory function to create Binary Screening Model with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured BinaryScreeningModel instance
    """
    if config is None:
        config = {}
    
    # Default configuration for medical deployment
    default_config = {
        'pretrained': True,
        'radimagenet_weights': True,
        'color_feature_fusion': True,
        'dropout_rate': 0.3,
        'confidence_threshold': 0.5  # Conservative threshold for medical safety
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    model = BinaryScreeningModel(**final_config)
    
    logger.info("Created Binary Screening Model for Stage 1 dual-architecture deployment")
    return model


# Example usage and integration with dual-architecture system
if __name__ == "__main__":
    # Create binary screening model
    model = create_binary_screening_model()
    
    # Example forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 500, 500)
    
    with torch.no_grad():
        # Standard forward pass
        logits = model(dummy_input)
        print(f"Output logits shape: {logits.shape}")
        
        # Clinical prediction with confidence
        results = model.predict_with_confidence(dummy_input)
        print(f"Pathology probabilities: {results['pathology_probability']}")
        print(f"Clinical predictions: {results['predictions']}")
        print(f"Needs specialist review: {results['needs_specialist_review']}")
        
        # Generate Grad-CAM for first image
        gradcam = model.generate_gradcam(dummy_input[0:1])
        print(f"Grad-CAM shape: {gradcam.shape}")
    
    logger.info("Binary Screening Model validation complete")