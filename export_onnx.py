import argparse
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.jit import TracerWarning

from model import build_toy_camera_matrices
from train_toy import SyntheticConfig, build_model


EXPORT_BATCH_SIZE = 1
EXPORT_OPSET = 18


class ToyBEVFormerOnnxWrapper(nn.Module):
    """Wrap the toy model so ONNX export returns a fixed tuple."""

    def __init__(self, model: nn.Module, image_size: tuple[int, int]):
        super().__init__()
        self.model = model
        self.image_size = image_size

    def forward(
        self,
        image_feats: torch.Tensor,
        lidar2img: torch.Tensor,
        prev_bev: torch.Tensor,
        ego_shift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            image_feats=image_feats,
            lidar2img=lidar2img,
            image_size=self.image_size,
            prev_bev=prev_bev,
            ego_shift=ego_shift,
            return_debug=False,
        )
        return (
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["bev_feature"],
            outputs["decoder_reference_points"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the BEVFormer toy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint saved by train_toy.py.")
    parser.add_argument("--onnx-path", type=str, default="", help="Output ONNX path. Defaults to <checkpoint>.onnx")
    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, SyntheticConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Expected a checkpoint saved by train_toy.py containing both "
            "'model_state_dict' and 'config'."
        )

    config = SyntheticConfig(**checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def make_dummy_inputs(config: SyntheticConfig, device: torch.device) -> tuple[torch.Tensor, ...]:
    image_feats = torch.randn(
        EXPORT_BATCH_SIZE,
        config.num_cams,
        config.embed_dims,
        config.feat_size[0],
        config.feat_size[1],
        device=device,
    )
    lidar2img = build_toy_camera_matrices(
        num_cams=config.num_cams,
        image_size=config.image_size,
        device=device,
        dtype=image_feats.dtype,
    ).unsqueeze(0)
    prev_bev = torch.zeros(
        EXPORT_BATCH_SIZE,
        config.bev_h * config.bev_w,
        config.embed_dims,
        device=device,
        dtype=image_feats.dtype,
    )
    ego_shift = torch.zeros(EXPORT_BATCH_SIZE, 2, device=device, dtype=image_feats.dtype)
    return image_feats, lidar2img, prev_bev, ego_shift


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx_path) if args.onnx_path else checkpoint_path.with_suffix(".onnx")

    model, config = load_model_and_config(str(checkpoint_path), device)
    wrapper = ToyBEVFormerOnnxWrapper(model, config.image_size).to(device)
    dummy_inputs = make_dummy_inputs(config, device)
    mha_fastpath_enabled = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="You are using the legacy TorchScript-based ONNX export.*",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings("ignore", category=TracerWarning)
                torch.onnx.export(
                    wrapper,
                    dummy_inputs,
                    str(onnx_path),
                    dynamo=False,
                    export_params=True,
                    opset_version=EXPORT_OPSET,
                    do_constant_folding=True,
                    external_data=False,
                    input_names=["image_feats", "lidar2img", "prev_bev", "ego_shift"],
                    output_names=["pred_logits", "pred_boxes", "bev_feature", "decoder_reference_points"],
                )
    finally:
        if mha_fastpath_enabled is not None:
            torch.backends.mha.set_fastpath_enabled(mha_fastpath_enabled)

    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"exported onnx: {onnx_path}")


if __name__ == "__main__":
    main()
