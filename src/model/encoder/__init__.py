from typing import Optional

from .encoder import Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg
from .encoder_test_unet3d_online import EmbodiedSplatEncoderTestUnet3d_Online

from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

ENCODERS = {
    "encoder_test_unet3d_online": (EmbodiedSplatEncoderTestUnet3d_Online, EncoderVisualizerEpipolar)
}

EncoderCfg = EncoderEpipolarCfg

def get_encoder(cfg: EncoderCfg, depth_range=[0.5,15.0]) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg, depth_range=depth_range)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
