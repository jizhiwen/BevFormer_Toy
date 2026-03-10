# BEVFormer Toy

这是一个从零写的教学版 `BEVFormer`，目标不是复现论文精度，而是把下面这些核心概念落成**可运行、可读懂**的 PyTorch 代码：

- `BEV query`
- `Temporal Self-Attention`
- `Spatial Cross-Attention`
- `3D pillar -> multi-camera projection`
- `object query -> decoder -> boxes`

## 文件说明

- `model.py`
  - 主模型实现
  - 包含参考点生成、3D 投影、Temporal / Spatial attention、decoder
- `demo.py`
  - 随机生成多相机特征
  - 构造一组 toy 相机矩阵
  - 演示第一帧和第二帧带 `prev_bev` 的前向过程
- `train_toy.py`
  - 合成一个最小但有规律的 toy 数据集
  - 训练分类头和 3D 框回归头
  - 演示 `prev_bev + image_feats + ego_shift` 如何形成可学习闭环
- `visualize_toy.py`
  - 把某个 `BEV query` 在相机中的投影点和采样点画出来
  - 生成 `bev_mask` 可见性热力图
- `toy_model_guide_zh.md`
  - 详细解释 toy 模型结构、训练脚本和源码阅读顺序

## 运行方式

```bash
python demo.py
```

如果你想直接跑一个最小训练闭环：

```bash
python train_toy.py --epochs 1 --batch-size 4 --train-samples 16 --val-samples 8
```

如果你想看 attention 的几何可视化：

```bash
python visualize_toy.py --query-idx 210 --camera-idx 0
```

运行后你会看到：

- `bev_feature` 的 shape
- `reference_points_cam` 的 shape
- `bev_mask` 的 shape
- temporal / spatial sampling locations 的 shape
- toy decoder 输出的 `pred_boxes`

训练时你会看到：

- 分类损失 `cls_loss`
- 框回归损失 `box_loss`
- query slot 分类准确率 `cls_acc`
- 激活目标上的框平均误差 `box_mae`

可视化脚本会输出：

- 某个 query 在指定相机里的 pillar 投影点
- 该 query 在该相机中的 spatial sampling points
- 全局 `bev_mask` 可见相机数热力图

## 这个 toy 版保留了什么

- 每个 `BEV query` 都对应一个固定 BEV 网格格子
- temporal attention 会同时看：
  - 当前 BEV
  - 历史 `prev_bev`
- spatial attention 会：
  - 在每个 BEV 格子上沿高度方向采样多个 3D 点
  - 投影到多个相机
  - 只在投影位置附近做稀疏采样
- decoder 会用 object query 从 BEV memory 中读出目标信息

## 这个 toy 版省略了什么

- MMCV / CUDA 的 `MSDeformableAttention`
- 多尺度 FPN 特征
- 公版中的 `can_bus` MLP、旋转补偿、box refine cascade
- 完整的检测训练目标和数据集管线

所以它更适合回答：

**“BEVFormer 的 attention 到底是怎么流动的？”**

而不是回答：

**“怎么训练到论文里的检测精度？”**

## 推荐阅读顺序

1. 先看 `demo.py`
2. 再看 `model.py` 里的这些函数：
   - `make_reference_points_2d`
   - `make_reference_points_3d`
   - `project_points_to_cameras`
3. 然后看这两个模块：
   - `TemporalSelfAttention`
   - `SpatialCrossAttention`
4. 再看 `visualize_toy.py`，把投影点和采样点对上
5. 再看 `train_toy.py` 里 synthetic dataset 的生成逻辑
6. 最后看 `ToyBEVFormer.forward()`

如果你已经读过 `bevformer_tutorial_zh.md`，可以把这个 toy 实现当成那篇文档的配套代码版本。

如果你想看更完整的中文说明，请直接打开 `toy_model_guide_zh.md`。
