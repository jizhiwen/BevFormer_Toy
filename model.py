import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_reference_points_2d(
    bev_h: int,
    bev_w: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return BEV center points normalized to [0, 1]."""
    ys = torch.linspace(0.5 / bev_h, 1.0 - 0.5 / bev_h, bev_h, device=device, dtype=dtype)
    xs = torch.linspace(0.5 / bev_w, 1.0 - 0.5 / bev_w, bev_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ref_2d = torch.stack([grid_x, grid_y], dim=-1).reshape(1, bev_h * bev_w, 2)
    return ref_2d.expand(batch_size, -1, -1)


def make_reference_points_3d(
    bev_h: int,
    bev_w: int,
    num_points_in_pillar: int,
    pc_range: Tuple[float, float, float, float, float, float],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return 3D pillar points in the LiDAR / ego coordinate frame."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    ys = torch.linspace(y_min, y_max, bev_h + 1, device=device, dtype=dtype)
    xs = torch.linspace(x_min, x_max, bev_w + 1, device=device, dtype=dtype)
    zs = torch.linspace(z_min, z_max, num_points_in_pillar + 1, device=device, dtype=dtype)

    xs = 0.5 * (xs[:-1] + xs[1:])
    ys = 0.5 * (ys[:-1] + ys[1:])
    zs = 0.5 * (zs[:-1] + zs[1:])

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([grid_x, grid_y], dim=-1).reshape(bev_h * bev_w, 2)
    z = zs.view(1, num_points_in_pillar, 1).expand(bev_h * bev_w, -1, -1)
    xy = xy.unsqueeze(1).expand(-1, num_points_in_pillar, -1)
    ref_3d = torch.cat([xy, z], dim=-1).unsqueeze(0)
    return ref_3d.expand(batch_size, -1, -1, -1)


def project_points_to_cameras(
    reference_points_3d: torch.Tensor,
    lidar2img: torch.Tensor,
    image_size: Tuple[int, int],
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D pillar points to every camera plane."""
    batch_size, num_query, num_points_in_pillar, _ = reference_points_3d.shape
    _, num_cams, _, _ = lidar2img.shape
    img_h, img_w = image_size

    ones = torch.ones(
        batch_size,
        num_query,
        num_points_in_pillar,
        1,
        device=reference_points_3d.device,
        dtype=reference_points_3d.dtype,
    )
    hom_points = torch.cat([reference_points_3d, ones], dim=-1)

    # Each BEV pillar point is projected to every camera independently.
    cam_points = torch.einsum("bcij,bndj->bcndi", lidar2img, hom_points)
    depths = cam_points[..., 2]
    uv = cam_points[..., :2] / depths.clamp_min(eps).unsqueeze(-1)
    uv[..., 0] = uv[..., 0] / img_w
    uv[..., 1] = uv[..., 1] / img_h

    # bev_mask says whether a pillar anchor is visible in a camera.
    bev_mask = (
        (depths > eps)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= 1.0)
    )
    return uv, bev_mask


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that gracefully handles fully masked rows."""
    mask = mask.to(dtype=logits.dtype)
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(masked_logits, dim=dim) * mask
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return weights / denom


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid used for reference-point refinement."""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def sample_from_feature_map(
    feature_map: torch.Tensor,
    sample_points: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Sample features with grid_sample.

    feature_map: [B, C, H, W]
    sample_points: [B, N, num_heads, P, 2], normalized to [0, 1]
    return: [B, N, num_heads, P, head_dim]
    """
    batch_size, channels, height, width = feature_map.shape
    _, num_query, _, num_points, _ = sample_points.shape
    assert channels % num_heads == 0
    head_dim = channels // num_heads

    feat = feature_map.view(batch_size, num_heads, head_dim, height, width)
    feat = feat.reshape(batch_size * num_heads, head_dim, height, width)

    grid = sample_points.permute(0, 2, 1, 3, 4).reshape(batch_size * num_heads, num_query, num_points, 2)
    grid = grid * 2.0 - 1.0
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.view(batch_size, num_heads, head_dim, num_query, num_points)
    sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()
    return sampled


def build_toy_camera_matrices(
    num_cams: int,
    image_size: Tuple[int, int],
    radius: float = 14.0,
    height: float = 6.0,
    focal_length: float = 48.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a ring of simple pinhole cameras looking at the origin."""
    if device is None:
        device = torch.device("cpu")

    img_h, img_w = image_size
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

    matrices = []
    for cam_idx in range(num_cams):
        angle = 2.0 * math.pi * cam_idx / num_cams
        eye = torch.tensor(
            [radius * math.cos(angle), radius * math.sin(angle), height],
            device=device,
            dtype=dtype,
        )
        forward = F.normalize(target - eye, dim=0)
        right = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
        down = F.normalize(torch.cross(forward, right, dim=0), dim=0)

        extrinsic = torch.eye(4, device=device, dtype=dtype)
        extrinsic[:3, :3] = torch.stack([right, down, forward], dim=0)
        extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye

        intrinsic = torch.eye(4, device=device, dtype=dtype)
        intrinsic[0, 0] = focal_length
        intrinsic[1, 1] = focal_length
        intrinsic[0, 2] = img_w / 2.0
        intrinsic[1, 2] = img_h / 2.0

        matrices.append(intrinsic @ extrinsic)

    return torch.stack(matrices, dim=0)


class FeedForward(nn.Module):
    def __init__(self, embed_dims: int, hidden_dims: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSelfAttention(nn.Module):
    """
    Teaching version of BEVFormer temporal attention.

    Each BEV query predicts offsets around:
    - a shifted previous-BEV reference point
    - the current-BEV reference point
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        num_points: int = 4,
        sampling_radius: float = 0.15,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_radius = sampling_radius

        self.offset_proj = nn.Linear(embed_dims * 2, 2 * num_heads * num_points * 2)
        self.weight_proj = nn.Linear(embed_dims * 2, 2 * num_heads * num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        reference_points_2d: torch.Tensor,
        bev_h: int,
        bev_w: int,
        ego_shift: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_query, _ = query.shape
        if ego_shift is None:
            ego_shift = torch.zeros(batch_size, 2, device=query.device, dtype=query.dtype)
        if prev_bev is None:
            prev_bev = query

        # Turn token sequences back into BEV feature maps so we can sample them
        # around learned 2D locations with grid_sample.
        current_map = query.transpose(1, 2).reshape(batch_size, self.embed_dims, bev_h, bev_w)
        prev_map = prev_bev.transpose(1, 2).reshape(batch_size, self.embed_dims, bev_h, bev_w)

        attn_input = torch.cat([prev_bev, query], dim=-1)
        offsets = torch.tanh(self.offset_proj(attn_input))
        offsets = offsets.view(batch_size, num_query, 2, self.num_heads, self.num_points, 2)
        offsets = offsets * self.sampling_radius

        weight_logits = self.weight_proj(attn_input)
        weight_logits = weight_logits.view(batch_size, num_query, 2, self.num_heads, self.num_points)

        # We sample around two centers:
        # 1. shifted previous-BEV center
        # 2. current-BEV center
        base_refs = torch.stack(
            [
                reference_points_2d + ego_shift[:, None, :],
                reference_points_2d,
            ],
            dim=2,
        )

        outputs = []
        debug_locations = []
        debug_weights = []
        for src_idx, source_map in enumerate([prev_map, current_map]):
            # Learned offsets move the sampler away from the default reference point.
            locations = base_refs[:, :, src_idx, None, None, :] + offsets[:, :, src_idx]
            weights = torch.softmax(weight_logits[:, :, src_idx], dim=-1)
            sampled = sample_from_feature_map(source_map, locations, self.num_heads)
            src_output = (sampled * weights.unsqueeze(-1)).sum(dim=3)
            outputs.append(src_output)

            if return_debug:
                debug_locations.append(locations.detach())
                debug_weights.append(weights.detach())

        fused = torch.stack(outputs, dim=2).mean(dim=2).reshape(batch_size, num_query, self.embed_dims)
        fused = self.output_proj(fused)

        debug = {}
        if return_debug:
            debug = {
                "sampling_locations": torch.stack(debug_locations, dim=2),
                "attention_weights": torch.stack(debug_weights, dim=2),
            }
        return fused, debug


class SpatialCrossAttention(nn.Module):
    """
    Teaching version of BEVFormer spatial cross attention.

    Each BEV query projects a small 3D pillar into every camera, then samples
    image features around those projected points.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        num_points: int = 4,
        num_points_in_pillar: int = 4,
        sampling_radius: float = 0.12,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_points_in_pillar = num_points_in_pillar
        self.sampling_radius = sampling_radius

        self.offset_proj = nn.Linear(
            embed_dims,
            num_heads * num_points_in_pillar * num_points * 2,
        )
        self.weight_proj = nn.Linear(
            embed_dims,
            num_heads * num_points_in_pillar * num_points,
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        image_feats: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_query, _ = query.shape
        _, num_cams, channels, _, _ = image_feats.shape
        assert channels == self.embed_dims
        assert reference_points_cam.size(3) == self.num_points_in_pillar

        offsets = torch.tanh(self.offset_proj(query))
        offsets = offsets.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_points_in_pillar,
            self.num_points,
            2,
        )
        offsets = offsets * self.sampling_radius

        weight_logits = self.weight_proj(query)
        weight_logits = weight_logits.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_points_in_pillar * self.num_points,
        )

        cam_outputs = []
        cam_valid = []
        debug_locations: List[torch.Tensor] = []
        debug_weights: List[torch.Tensor] = []

        for cam_idx in range(num_cams):
            ref = reference_points_cam[:, cam_idx]
            visible = bev_mask[:, cam_idx]

            # For one camera, each BEV query owns several pillar anchors, and each
            # anchor expands into several learned sampling points around it.
            locations = ref[:, :, None, :, None, :] + offsets
            locations = locations.view(
                batch_size,
                num_query,
                self.num_heads,
                self.num_points_in_pillar * self.num_points,
                2,
            )

            # Invisible anchors should not receive attention weight.
            visible_mask = visible[:, :, None, :, None].expand(
                -1, -1, self.num_heads, -1, self.num_points
            )
            visible_mask = visible_mask.reshape(
                batch_size,
                num_query,
                self.num_heads,
                self.num_points_in_pillar * self.num_points,
            )
            weights = masked_softmax(weight_logits, visible_mask, dim=-1)

            sampled = sample_from_feature_map(image_feats[:, cam_idx], locations, self.num_heads)
            cam_output = (sampled * weights.unsqueeze(-1)).sum(dim=3).reshape(batch_size, num_query, self.embed_dims)

            has_visible_anchor = visible.any(dim=-1)
            cam_output = cam_output * has_visible_anchor.unsqueeze(-1).to(cam_output.dtype)
            cam_outputs.append(cam_output)
            cam_valid.append(has_visible_anchor)

            if return_debug:
                debug_locations.append(locations.detach())
                debug_weights.append(weights.detach())

        # Aggregate all visible cameras back into the original BEV query slots.
        fused = torch.stack(cam_outputs, dim=0).sum(dim=0)
        counts = torch.stack(cam_valid, dim=0).sum(dim=0).clamp_min(1)
        fused = fused / counts.unsqueeze(-1).to(fused.dtype)
        fused = self.output_proj(fused)

        debug = {}
        if return_debug:
            debug = {
                "sampling_locations": torch.stack(debug_locations, dim=1),
                "attention_weights": torch.stack(debug_weights, dim=1),
                "visible_cameras_per_query": torch.stack(cam_valid, dim=1),
            }
        return fused, debug


class BEVFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_temporal_points: int,
        num_spatial_points: int,
        num_points_in_pillar: int,
        ffn_hidden_dims: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_temporal_points,
        )
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_spatial_points,
            num_points_in_pillar=num_points_in_pillar,
        )
        self.ffn = FeedForward(embed_dims, ffn_hidden_dims, dropout)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        image_feats: torch.Tensor,
        reference_points_2d: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        bev_h: int,
        bev_w: int,
        ego_shift: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        temporal_out, temporal_debug = self.temporal_attn(
            query=self.norm1(query),
            prev_bev=None if prev_bev is None else self.norm1(prev_bev),
            reference_points_2d=reference_points_2d,
            bev_h=bev_h,
            bev_w=bev_w,
            ego_shift=ego_shift,
            return_debug=return_debug,
        )
        query = query + self.dropout1(temporal_out)

        spatial_out, spatial_debug = self.spatial_attn(
            query=self.norm2(query),
            image_feats=image_feats,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            return_debug=return_debug,
        )
        query = query + self.dropout2(spatial_out)

        query = query + self.dropout3(self.ffn(self.norm3(query)))

        debug = {}
        if return_debug:
            debug = {
                "temporal": temporal_debug,
                "spatial": spatial_debug,
            }
        return query, debug


class DeformableDecoderCrossAttention(nn.Module):
    """
    Deformable cross-attention over BEV memory.

    Each object query uses a learned 2D reference point on the BEV plane and
    predicts a small set of offsets around that center.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        num_points: int = 4,
        sampling_radius: float = 0.2,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_radius = sampling_radius

        self.offset_proj = nn.Linear(embed_dims, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(embed_dims, num_heads * num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        bev_h: int,
        bev_w: int,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_query, _ = query.shape
        memory_map = memory.transpose(1, 2).reshape(batch_size, self.embed_dims, bev_h, bev_w)

        offsets = torch.tanh(self.offset_proj(query))
        offsets = offsets.view(batch_size, num_query, self.num_heads, self.num_points, 2)
        offsets = offsets * self.sampling_radius

        weights = self.weight_proj(query).view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_points,
        )
        weights = torch.softmax(weights, dim=-1)

        base_refs = reference_points[..., :2].unsqueeze(2).unsqueeze(3)
        locations = base_refs + offsets
        sampled = sample_from_feature_map(memory_map, locations, self.num_heads)
        fused = (sampled * weights.unsqueeze(-1)).sum(dim=3).reshape(batch_size, num_query, self.embed_dims)
        fused = self.output_proj(fused)

        debug = {}
        if return_debug:
            debug = {
                "sampling_locations": locations.detach(),
                "attention_weights": weights.detach(),
            }
        return fused, debug


class DetectionDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_points: int,
        ffn_hidden_dims: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableDecoderCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.ffn = FeedForward(embed_dims, ffn_hidden_dims, dropout)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        bev_h: int,
        bev_w: int,
        query_pos: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q = self.norm1(query + (0 if query_pos is None else query_pos))
        self_out, self_weights = self.self_attn(
            q,
            q,
            q,
            need_weights=return_debug,
            average_attn_weights=False,
        )
        query = query + self.dropout1(self_out)

        q = self.norm2(query + (0 if query_pos is None else query_pos))
        cross_out, cross_debug = self.cross_attn(
            q,
            memory,
            reference_points=reference_points,
            bev_h=bev_h,
            bev_w=bev_w,
            return_debug=return_debug,
        )
        query = query + self.dropout2(cross_out)
        query = query + self.dropout3(self.ffn(self.norm3(query)))

        debug = {}
        if return_debug:
            debug = {
                "self_attention_weights": self_weights.detach(),
                "cross_attention_weights": cross_debug["attention_weights"],
                "cross_sampling_locations": cross_debug["sampling_locations"],
                "reference_points": reference_points.detach(),
            }
        return query, debug


class ToyBEVFormer(nn.Module):
    """
    Minimal BEVFormer-like model for study purposes.

    Simplifications compared with the official implementation:
    - pure PyTorch, no MMCV custom CUDA ops
    - single-scale image features
    - no can_bus MLP
    - decoder uses deformable cross-attention over BEV memory
    - reference-point refinement is kept, but box parameterization is simplified
    """

    def __init__(
        self,
        embed_dims: int = 128,
        bev_h: int = 20,
        bev_w: int = 20,
        num_cams: int = 6,
        num_points_in_pillar: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        num_temporal_points: int = 4,
        num_spatial_points: int = 4,
        num_decoder_points: int = 4,
        num_object_queries: int = 100,
        num_classes: int = 3,
        ffn_hidden_dims: int = 256,
        dropout: float = 0.1,
        pc_range: Tuple[float, float, float, float, float, float] = (-10.0, -10.0, -2.0, 10.0, 10.0, 4.0),
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.num_points_in_pillar = num_points_in_pillar
        self.num_object_queries = num_object_queries
        self.pc_range = pc_range

        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims) * 0.02)
        self.object_queries = nn.Parameter(torch.randn(num_object_queries, embed_dims) * 0.02)
        self.object_query_pos = nn.Parameter(torch.randn(num_object_queries, embed_dims) * 0.02)
        self.bev_pos_mlp = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.encoder_layers = nn.ModuleList(
            [
                BEVFormerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_temporal_points=num_temporal_points,
                    num_spatial_points=num_spatial_points,
                    num_points_in_pillar=num_points_in_pillar,
                    ffn_hidden_dims=ffn_hidden_dims,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DetectionDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_points=num_decoder_points,
                    ffn_hidden_dims=ffn_hidden_dims,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.reference_point_head = nn.Linear(embed_dims, 3)
        self.decoder_reg_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, 7),
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.class_head = nn.Linear(embed_dims, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 7),
        )

    def forward(
        self,
        image_feats: torch.Tensor,
        lidar2img: torch.Tensor,
        image_size: Tuple[int, int],
        prev_bev: Optional[torch.Tensor] = None,
        ego_shift: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        image_feats: [B, num_cams, C, H, W]
        lidar2img: [B, num_cams, 4, 4]
        prev_bev: [B, bev_h * bev_w, C]
        ego_shift: [B, 2], normalized shift on the BEV plane
        """
        batch_size, num_cams, channels, _, _ = image_feats.shape
        assert num_cams == self.num_cams
        assert channels == self.embed_dims

        device = image_feats.device
        dtype = image_feats.dtype

        # ref_2d answers: where is the default center of each BEV grid cell?
        reference_points_2d = make_reference_points_2d(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        # ref_3d answers: which 3D pillar anchors belong to this BEV grid cell?
        reference_points_3d = make_reference_points_3d(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            num_points_in_pillar=self.num_points_in_pillar,
            pc_range=self.pc_range,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        # Project the 3D anchors into every camera to obtain spatial attention centers.
        reference_points_cam, bev_mask = project_points_to_cameras(
            reference_points_3d=reference_points_3d,
            lidar2img=lidar2img,
            image_size=image_size,
        )

        bev_query = self.bev_queries.unsqueeze(0).expand(batch_size, -1, -1)
        bev_query = bev_query + self.bev_pos_mlp(reference_points_2d)

        encoder_debug = []
        memory = bev_query
        for layer in self.encoder_layers:
            memory, layer_debug = layer(
                query=memory,
                prev_bev=prev_bev,
                image_feats=image_feats,
                reference_points_2d=reference_points_2d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                ego_shift=ego_shift,
                return_debug=return_debug,
            )
            encoder_debug.append(layer_debug)

        # Decoder reads the fused BEV memory with a fixed set of object queries.
        obj_query = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        obj_query_pos = self.object_query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_point_head(obj_query_pos).sigmoid()

        decoder_debug = []
        decoded = obj_query
        box_deltas = None
        for lid, layer in enumerate(self.decoder_layers):
            decoded, layer_debug = layer(
                query=decoded,
                memory=memory,
                reference_points=reference_points,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                query_pos=obj_query_pos,
                return_debug=return_debug,
            )

            box_deltas = self.decoder_reg_branches[lid](decoded)
            new_reference_points = torch.zeros_like(reference_points)
            new_reference_points[..., :2] = (
                box_deltas[..., :2] + inverse_sigmoid(reference_points[..., :2])
            )
            new_reference_points[..., 2:3] = (
                box_deltas[..., 2:3] + inverse_sigmoid(reference_points[..., 2:3])
            )
            new_reference_points = new_reference_points.sigmoid()

            if return_debug:
                layer_debug["box_deltas"] = box_deltas.detach()
                layer_debug["updated_reference_points"] = new_reference_points.detach()
            decoder_debug.append(layer_debug)
            reference_points = new_reference_points.detach()

        pred_logits = self.class_head(decoded)
        pred_boxes = self.box_head(decoded)
        if box_deltas is not None:
            pred_boxes[..., :2] = reference_points[..., :2]
            pred_boxes[..., 2:3] = reference_points[..., 2:3]

        outputs = {
            "bev_feature": memory,
            "decoder_queries": decoded,
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "decoder_reference_points": reference_points,
            "reference_points_cam": reference_points_cam,
            "bev_mask": bev_mask,
        }
        if return_debug:
            outputs["encoder_debug"] = encoder_debug
            outputs["decoder_debug"] = decoder_debug
        return outputs
