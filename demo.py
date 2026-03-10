import torch

from model import ToyBEVFormer, build_toy_camera_matrices


def print_tensor_summary(name: str, tensor: torch.Tensor) -> None:
    print(f"{name:<24} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (64, 64)

    model = ToyBEVFormer(
        embed_dims=128,
        bev_h=20,
        bev_w=20,
        num_cams=6,
        num_points_in_pillar=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        num_temporal_points=4,
        num_spatial_points=4,
        num_object_queries=32,
        num_classes=3,
        ffn_hidden_dims=256,
    ).to(device)

    batch_size = 2
    image_feats = torch.randn(batch_size, 6, 128, 16, 16, device=device)
    lidar2img = build_toy_camera_matrices(
        num_cams=6,
        image_size=image_size,
        device=device,
        dtype=image_feats.dtype,
    ).unsqueeze(0).expand(batch_size, -1, -1, -1)

    print("=== Frame 1: no prev_bev ===")
    outputs_t0 = model(
        image_feats=image_feats,
        lidar2img=lidar2img,
        image_size=image_size,
        prev_bev=None,
        ego_shift=None,
        return_debug=True,
    )

    print_tensor_summary("bev_feature", outputs_t0["bev_feature"])
    print_tensor_summary("pred_logits", outputs_t0["pred_logits"])
    print_tensor_summary("pred_boxes", outputs_t0["pred_boxes"])
    print_tensor_summary("reference_points_cam", outputs_t0["reference_points_cam"])
    print_tensor_summary("bev_mask", outputs_t0["bev_mask"])

    visible_ratio = outputs_t0["bev_mask"].float().mean().item()
    print(f"visible anchor ratio       {visible_ratio:.4f}")

    temporal_debug = outputs_t0["encoder_debug"][0]["temporal"]
    spatial_debug = outputs_t0["encoder_debug"][0]["spatial"]
    print_tensor_summary("temporal locations", temporal_debug["sampling_locations"])
    print_tensor_summary("spatial locations", spatial_debug["sampling_locations"])

    print("\n=== Frame 2: with prev_bev and ego shift ===")
    next_image_feats = torch.randn(batch_size, 6, 128, 16, 16, device=device)
    ego_shift = torch.tensor(
        [
            [0.03, -0.01],
            [0.01, 0.02],
        ],
        device=device,
        dtype=image_feats.dtype,
    )

    outputs_t1 = model(
        image_feats=next_image_feats,
        lidar2img=lidar2img,
        image_size=image_size,
        prev_bev=outputs_t0["bev_feature"].detach(),
        ego_shift=ego_shift,
        return_debug=True,
    )

    print_tensor_summary("bev_feature", outputs_t1["bev_feature"])
    print_tensor_summary("pred_logits", outputs_t1["pred_logits"])
    print_tensor_summary("pred_boxes", outputs_t1["pred_boxes"])

    temporal_locations = outputs_t1["encoder_debug"][0]["temporal"]["sampling_locations"]
    prev_frame_refs = temporal_locations[:, :, 0]
    curr_frame_refs = temporal_locations[:, :, 1]
    mean_shift = (prev_frame_refs - curr_frame_refs).abs().mean().item()
    print(f"temporal location delta    {mean_shift:.4f}")

    sample_boxes = outputs_t1["pred_boxes"][0, :3].detach().cpu()
    print("\nfirst 3 predicted boxes")
    print(sample_boxes)


if __name__ == "__main__":
    main()
