import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import ToyBEVFormer, build_toy_camera_matrices


@dataclass
class SyntheticConfig:
    embed_dims: int = 128
    bev_h: int = 20
    bev_w: int = 20
    num_cams: int = 6
    image_size: Tuple[int, int] = (64, 64)
    feat_size: Tuple[int, int] = (16, 16)
    num_object_queries: int = 16
    num_foreground_classes: int = 3
    num_points_in_pillar: int = 4
    num_decoder_points: int = 4
    pc_range: Tuple[float, float, float, float, float, float] = (-10.0, -10.0, -2.0, 10.0, 10.0, 4.0)
    active_prob: float = 0.7
    noise_std: float = 0.01
    gaussian_sigma: float = 1.2
    ego_shift_range: float = 1.2

    @property
    def num_classes(self) -> int:
        return self.num_foreground_classes + 1

    @property
    def background_class_id(self) -> int:
        return self.num_classes - 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def rand_uniform(
    generator: torch.Generator,
    low: float,
    high: float,
    size: Tuple[int, ...],
) -> torch.Tensor:
    return torch.rand(size, generator=generator) * (high - low) + low


def encode_object_feature(
    class_id: int,
    slot_idx: int,
    norm_box: torch.Tensor,
    embed_dims: int,
) -> torch.Tensor:
    """Deterministic object descriptor used to render synthetic features."""
    class_value = float(class_id + 1) / 8.0
    slot_value = float(slot_idx + 1) / 32.0
    raw = torch.cat(
        [
            norm_box,
            torch.tensor([class_value, slot_value], dtype=norm_box.dtype),
        ]
    )

    banks = [raw]
    for freq in [1.0, 2.0, 4.0, 8.0]:
        banks.append(torch.sin(raw * math.pi * freq))
        banks.append(torch.cos(raw * math.pi * freq))

    feature = torch.cat(banks, dim=0)
    repeat = math.ceil(embed_dims / feature.numel())
    feature = feature.repeat(repeat)[:embed_dims]
    return feature


def box_to_normalized(box: torch.Tensor, pc_range: Tuple[float, float, float, float, float, float]) -> torch.Tensor:
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    norm = torch.zeros(7, dtype=box.dtype)
    norm[0] = (box[0] - x_min) / (x_max - x_min)
    norm[1] = (box[1] - y_min) / (y_max - y_min)
    norm[2] = (box[2] - z_min) / (z_max - z_min)
    norm[3] = box[3] / (x_max - x_min)
    norm[4] = box[4] / (y_max - y_min)
    norm[5] = box[5] / (z_max - z_min)
    norm[6] = (box[6] + math.pi) / (2.0 * math.pi)
    return norm.clamp(0.0, 1.0)


def center_to_bev_index(
    x: float,
    y: float,
    pc_range: Tuple[float, float, float, float, float, float],
    bev_h: int,
    bev_w: int,
) -> Tuple[float, float]:
    x_min, y_min, _, x_max, y_max, _ = pc_range
    u = (x - x_min) / (x_max - x_min) * (bev_w - 1)
    v = (y - y_min) / (y_max - y_min) * (bev_h - 1)
    return u, v


def add_gaussian_blob(
    feature_map: torch.Tensor,
    center_x: float,
    center_y: float,
    feature_vec: torch.Tensor,
    sigma: float,
) -> None:
    """Add a small Gaussian-weighted feature blob into a 2D feature map."""
    _, height, width = feature_map.shape
    radius = max(1, int(math.ceil(2.0 * sigma)))

    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(width - 1, int(math.ceil(center_x + radius)))
    y0 = max(0, int(math.floor(center_y - radius)))
    y1 = min(height - 1, int(math.ceil(center_y + radius)))
    if x1 < x0 or y1 < y0:
        return

    ys = torch.arange(y0, y1 + 1, dtype=feature_map.dtype)
    xs = torch.arange(x0, x1 + 1, dtype=feature_map.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    weight = torch.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2.0 * sigma * sigma))
    feature_map[:, y0 : y1 + 1, x0 : x1 + 1] += feature_vec[:, None, None] * weight[None]


def render_bev_feature_map(
    objects: List[Dict[str, torch.Tensor]],
    config: SyntheticConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    feature_map = torch.randn(
        config.embed_dims,
        config.bev_h,
        config.bev_w,
        generator=generator,
    ) * config.noise_std

    for obj in objects:
        if not obj["active"]:
            continue
        cx, cy = center_to_bev_index(
            x=float(obj["box"][0]),
            y=float(obj["box"][1]),
            pc_range=config.pc_range,
            bev_h=config.bev_h,
            bev_w=config.bev_w,
        )
        add_gaussian_blob(
            feature_map=feature_map,
            center_x=cx,
            center_y=cy,
            feature_vec=obj["feature"],
            sigma=config.gaussian_sigma,
        )

    return feature_map


def render_image_features(
    objects: List[Dict[str, torch.Tensor]],
    lidar2img: torch.Tensor,
    config: SyntheticConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    feat_h, feat_w = config.feat_size
    img_h, img_w = config.image_size
    image_feats = torch.randn(
        config.num_cams,
        config.embed_dims,
        feat_h,
        feat_w,
        generator=generator,
    ) * config.noise_std

    pillar_offsets = torch.linspace(-0.5, 0.5, config.num_points_in_pillar, dtype=torch.float32)
    for obj in objects:
        if not obj["active"]:
            continue

        box = obj["box"]
        height = box[5].item()
        for cam_idx in range(config.num_cams):
            for anchor in pillar_offsets:
                point = torch.tensor(
                    [
                        box[0].item(),
                        box[1].item(),
                        box[2].item() + anchor.item() * height,
                        1.0,
                    ],
                    dtype=torch.float32,
                )
                cam_point = lidar2img[cam_idx] @ point
                depth = cam_point[2].item()
                if depth <= 1e-4:
                    continue

                u = cam_point[0].item() / depth / img_w
                v = cam_point[1].item() / depth / img_h
                if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                    continue

                feat_x = u * (feat_w - 1)
                feat_y = v * (feat_h - 1)
                add_gaussian_blob(
                    feature_map=image_feats[cam_idx],
                    center_x=feat_x,
                    center_y=feat_y,
                    feature_vec=obj["feature"],
                    sigma=config.gaussian_sigma,
                )

    return image_feats


def make_sample(seed: int, config: SyntheticConfig, lidar2img: torch.Tensor) -> Dict[str, torch.Tensor]:
    generator = make_generator(seed)
    x_min, y_min, z_min, x_max, y_max, z_max = config.pc_range

    ego_dx = rand_uniform(generator, -config.ego_shift_range, config.ego_shift_range, (1,)).item()
    ego_dy = rand_uniform(generator, -config.ego_shift_range, config.ego_shift_range, (1,)).item()
    world_dx = ego_dx
    world_dy = ego_dy

    prev_objects: List[Dict[str, torch.Tensor]] = []
    curr_objects: List[Dict[str, torch.Tensor]] = []
    target_classes = torch.full((config.num_object_queries,), config.background_class_id, dtype=torch.long)
    target_boxes = torch.zeros(config.num_object_queries, 7, dtype=torch.float32)

    for slot_idx in range(config.num_object_queries):
        active = torch.rand((), generator=generator).item() < config.active_prob
        if not active:
            dummy_box = torch.zeros(7, dtype=torch.float32)
            dummy_feature = torch.zeros(config.embed_dims, dtype=torch.float32)
            prev_objects.append({"active": False, "box": dummy_box, "feature": dummy_feature})
            curr_objects.append({"active": False, "box": dummy_box, "feature": dummy_feature})
            continue

        class_id = int(torch.randint(0, config.num_foreground_classes, (1,), generator=generator).item())
        prev_x = rand_uniform(generator, x_min + 1.5, x_max - 1.5, (1,)).item()
        prev_y = rand_uniform(generator, y_min + 1.5, y_max - 1.5, (1,)).item()
        prev_z = rand_uniform(generator, -0.5, 1.0, (1,)).item()
        size_x = rand_uniform(generator, 1.2, 3.2, (1,)).item()
        size_y = rand_uniform(generator, 1.0, 2.8, (1,)).item()
        size_z = rand_uniform(generator, 1.2, 2.2, (1,)).item()
        yaw = rand_uniform(generator, -math.pi, math.pi, (1,)).item()

        curr_x = float(max(x_min + 1.0, min(x_max - 1.0, prev_x + world_dx)))
        curr_y = float(max(y_min + 1.0, min(y_max - 1.0, prev_y + world_dy)))
        curr_z = float(max(z_min + 0.5, min(z_max - 0.5, prev_z)))

        prev_box = torch.tensor([prev_x, prev_y, prev_z, size_x, size_y, size_z, yaw], dtype=torch.float32)
        curr_box = torch.tensor([curr_x, curr_y, curr_z, size_x, size_y, size_z, yaw], dtype=torch.float32)
        curr_norm_box = box_to_normalized(curr_box, config.pc_range)
        feature = encode_object_feature(
            class_id=class_id,
            slot_idx=slot_idx,
            norm_box=curr_norm_box,
            embed_dims=config.embed_dims,
        )

        prev_objects.append({"active": True, "box": prev_box, "feature": feature})
        curr_objects.append({"active": True, "box": curr_box, "feature": feature})
        target_classes[slot_idx] = class_id
        target_boxes[slot_idx] = curr_norm_box

    prev_bev = render_bev_feature_map(prev_objects, config, generator).permute(1, 2, 0).reshape(-1, config.embed_dims)
    image_feats = render_image_features(curr_objects, lidar2img, config, generator)
    ego_shift = torch.tensor(
        [
            -world_dx / (x_max - x_min),
            -world_dy / (y_max - y_min),
        ],
        dtype=torch.float32,
    )

    return {
        "image_feats": image_feats,
        "prev_bev": prev_bev,
        "ego_shift": ego_shift,
        "target_classes": target_classes,
        "target_boxes": target_boxes,
    }


class SyntheticBEVFormerDataset(Dataset):
    def __init__(self, num_samples: int, seed: int, config: SyntheticConfig):
        self.num_samples = num_samples
        self.seed = seed
        self.config = config
        self.lidar2img = build_toy_camera_matrices(
            num_cams=config.num_cams,
            image_size=config.image_size,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = make_sample(self.seed + idx, self.config, self.lidar2img)
        sample["lidar2img"] = self.lidar2img.clone()
        return sample


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    output = {}
    for key in batch[0]:
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    return output


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    background_class_id: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"].clone()
    # Decoder refinement already writes normalized x/y/z into pred_boxes.
    # The remaining size/yaw channels still come from raw regression logits.
    pred_boxes[..., 3:] = pred_boxes[..., 3:].sigmoid()
    target_classes = batch["target_classes"]
    target_boxes = batch["target_boxes"]

    cls_loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), target_classes.reshape(-1))

    active_mask = target_classes != background_class_id
    if active_mask.any():
        box_loss = F.l1_loss(pred_boxes[active_mask], target_boxes[active_mask])
    else:
        box_loss = pred_boxes.sum() * 0.0

    total_loss = cls_loss + 5.0 * box_loss

    with torch.no_grad():
        pred_classes = pred_logits.argmax(dim=-1)
        cls_acc = (pred_classes == target_classes).float().mean().item()
        if active_mask.any():
            box_mae = (pred_boxes[active_mask] - target_boxes[active_mask]).abs().mean().item()
        else:
            box_mae = 0.0

    metrics = {
        "loss": total_loss.item(),
        "cls_loss": cls_loss.item(),
        "box_loss": box_loss.item(),
        "cls_acc": cls_acc,
        "box_mae": box_mae,
    }
    return total_loss, metrics


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: SyntheticConfig,
    epoch: int,
) -> None:
    model.train()
    running = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "cls_acc": 0.0, "box_mae": 0.0}

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            image_feats=batch["image_feats"],
            lidar2img=batch["lidar2img"],
            image_size=config.image_size,
            prev_bev=batch["prev_bev"],
            ego_shift=batch["ego_shift"],
            return_debug=False,
        )

        loss, metrics = compute_losses(outputs, batch, config.background_class_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += metrics[key]

        if step == 1 or step % 10 == 0 or step == len(dataloader):
            avg = {key: value / step for key, value in running.items()}
            print(
                f"epoch {epoch:02d} step {step:03d}/{len(dataloader):03d} "
                f"loss={avg['loss']:.4f} cls={avg['cls_loss']:.4f} box={avg['box_loss']:.4f} "
                f"acc={avg['cls_acc']:.4f} box_mae={avg['box_mae']:.4f}"
            )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: SyntheticConfig,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "cls_acc": 0.0, "box_mae": 0.0}
    count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            image_feats=batch["image_feats"],
            lidar2img=batch["lidar2img"],
            image_size=config.image_size,
            prev_bev=batch["prev_bev"],
            ego_shift=batch["ego_shift"],
            return_debug=False,
        )
        _, metrics = compute_losses(outputs, batch, config.background_class_id)
        for key in totals:
            totals[key] += metrics[key]
        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


def build_model(config: SyntheticConfig) -> ToyBEVFormer:
    return ToyBEVFormer(
        embed_dims=config.embed_dims,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        num_cams=config.num_cams,
        num_points_in_pillar=config.num_points_in_pillar,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        num_temporal_points=4,
        num_spatial_points=4,
        num_decoder_points=config.num_decoder_points,
        num_object_queries=config.num_object_queries,
        num_classes=config.num_classes,
        ffn_hidden_dims=256,
        pc_range=config.pc_range,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BEVFormer toy model on synthetic data.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="toy_bevformer.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SyntheticConfig()

    train_dataset = SyntheticBEVFormerDataset(
        num_samples=args.train_samples,
        seed=args.seed,
        config=config,
    )
    val_dataset = SyntheticBEVFormerDataset(
        num_samples=args.val_samples,
        seed=args.seed + 100_000,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, config, epoch)
        val_metrics = evaluate(model, val_loader, device, config)
        print(
            f"epoch {epoch:02d} val "
            f"loss={val_metrics['loss']:.4f} cls={val_metrics['cls_loss']:.4f} box={val_metrics['box_loss']:.4f} "
            f"acc={val_metrics['cls_acc']:.4f} box_mae={val_metrics['box_mae']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            print(f"saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
