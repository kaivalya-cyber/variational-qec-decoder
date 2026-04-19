"""Central export module for all advanced research features.

Novel Contribution: Integrated suite of hybrid classical-quantum tools 
for next-generation adaptive QEC decoding.
"""

from .belief_propagation import BeliefPropagator
from .bp_enhanced_decoder import BPEnhancedVariationalDecoder
from .noise_fingerprinter import HardwareNoiseFingerprinter
from .personalized_decoder import PersonalizedDecoder, SyntheticHardwareSimulator
from .block_decoder import LogicalBlockDecoder
from .cross_qubit_correlations import CrossQubitCorrelationAnalyzer
from .confidence_calibrator import DecoderConfidenceCalibrator, AdaptiveFallbackDecoder
from .syndrome_autoencoder import SyndromeAutoencoder
from .compressed_decoder import CompressedVariationalDecoder

__all__ = [
    "BeliefPropagator",
    "BPEnhancedVariationalDecoder",
    "HardwareNoiseFingerprinter",
    "PersonalizedDecoder",
    "SyntheticHardwareSimulator",
    "LogicalBlockDecoder",
    "CrossQubitCorrelationAnalyzer",
    "DecoderConfidenceCalibrator",
    "AdaptiveFallbackDecoder",
    "SyndromeAutoencoder",
    "CompressedVariationalDecoder",
]
