import argparse
from pathlib import Path

import torch

from model import ToyBEVFormer, build_toy_camera_matrices


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "visualize_toy.py requires matplotlib. Please run it in an environment "
            "that already has matplotlib installed."
        ) from exc
    return plt


def plot_camera_projection(
    plt,
    save_path: Path,
    image_size,
    anchor_points: torch.Tensor,
    sample_points: torch.Tensor,
    visible_mask: torch.Tensor,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    img_h, img_w = image_size
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.grid(True, alpha=0.25)

    visible_idx = visible_mask.nonzero(as_tuple=False).flatten()
    hidden_idx = (~visible_mask).nonzero(as_tuple=False).flatten()

    if len(hidden_idx) > 0:
        hidden_pts = anchor_points[hidden_idx]
        ax.scatter(
            hidden_pts[:, 0] * img_w,
            hidden_pts[:, 1] * img_h,
            s=35,
            c="lightgray",
            marker="x",
            label="hidden anchors",
        )

    if len(visible_idx) > 0:
        visible_pts = anchor_points[visible_idx]
        ax.scatter(
            visible_pts[:, 0] * img_w,
            visible_pts[:, 1] * img_h,
            s=50,
            c="tab:red",
            label="pillar anchors",
        )

        for idx, pt in enumerate(visible_pts):
            ax.text(pt[0].item() * img_w + 1.5, pt[1].item() * img_h + 1.5, f"a{idx}", fontsize=8)

    sample_xy = sample_points.reshape(-1, 2)
    ax.scatter(
        sample_xy[:, 0] * img_w,
        sample_xy[:, 1] * img_h,
        s=8,
        c="tab:blue",
        alpha=0.45,
        label="sampling points",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_bev_visibility(
    plt,
    save_path: Path,
    bev_h: int,
    bev_w: int,
    visible_camera_count: torch.Tensor,
    query_idx: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    heatmap = visible_camera_count.reshape(bev_h, bev_w).cpu()
    image = ax.imshow(heatmap, cmap="viridis")
    qy = query_idx // bev_w
    qx = query_idx % bev_w
    ax.scatter([qx], [qy], c="red", s=60, marker="o", label="selected query")
    ax.set_title("Visible Cameras Per BEV Query")
    ax.set_xlabel("bev x")
    ax.set_ylabel("bev y")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="# visible cameras")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BEVFormer toy attention geometry.")
    parser.add_argument("--query-idx", type=int, default=210, help="Which BEV query index to visualize.")
    parser.add_argument("--camera-idx", type=int, default=0, help="Which camera index to visualize.")
    parser.add_argument("--layer-idx", type=int, default=0, help="Which encoder layer's debug tensors to inspect.")
    parser.add_argument("--save-dir", type=str, default="toy_visualizations", help="Directory for output images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt = require_matplotlib()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (64, 64)
    feat_size = (16, 16)

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
        num_decoder_points=4,
        num_object_queries=32,
        num_classes=3,
        ffn_hidden_dims=256,
    ).to(device)

    image_feats = torch.randn(1, 6, 128, feat_size[0], feat_size[1], device=device)
    lidar2img = build_toy_camera_matrices(
        num_cams=6,
        image_size=image_size,
        device=device,
        dtype=image_feats.dtype,
    ).unsqueeze(0)
    ego_shift = torch.tensor([[0.03, -0.01]], device=device, dtype=image_feats.dtype)

    outputs = model(
        image_feats=image_feats,
        lidar2img=lidar2img,
        image_size=image_size,
        prev_bev=None,
        ego_shift=ego_shift,
        return_debug=True,
    )

    query_idx = max(0, min(args.query_idx, model.bev_h * model.bev_w - 1))
    camera_idx = max(0, min(args.camera_idx, model.num_cams - 1))
    layer_idx = max(0, min(args.layer_idx, len(outputs["encoder_debug"]) - 1))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    spatial_debug = outputs["encoder_debug"][layer_idx]["spatial"]
    reference_points_cam = outputs["reference_points_cam"][0, camera_idx].detach().cpu()
    bev_mask = outputs["bev_mask"][0, camera_idx].detach().cpu()
    sample_points = spatial_debug["sampling_locations"][0, camera_idx, query_idx].detach().cpu()
    visible_counts = outputs["bev_mask"][0].any(dim=-1).sum(dim=0)

    projection_path = save_dir / f"query_{query_idx:03d}_camera_{camera_idx}_projection.png"
    plot_camera_projection(
        plt=plt,
        save_path=projection_path,
        image_size=image_size,
        anchor_points=reference_points_cam[query_idx],
        sample_points=sample_points,
        visible_mask=bev_mask[query_idx],
        title=f"Query {query_idx} in Camera {camera_idx}",
    )

    visibility_path = save_dir / "bev_visibility_map.png"
    plot_bev_visibility(
        plt=plt,
        save_path=visibility_path,
        bev_h=model.bev_h,
        bev_w=model.bev_w,
        visible_camera_count=visible_counts,
        query_idx=query_idx,
    )

    print(f"saved: {projection_path}")
    print(f"saved: {visibility_path}")
    print(f"selected query: {query_idx}")
    print(f"selected camera: {camera_idx}")
    print(f"visible anchors in selected camera: {int(bev_mask[query_idx].sum().item())}")


if __name__ == "__main__":
    main()
