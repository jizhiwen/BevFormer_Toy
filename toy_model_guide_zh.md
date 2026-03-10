# BEVFormer Toy 详细说明

这份文档专门解释 `BevFormer_Toy` 里的 toy 模型。  
它不是论文复现，也不是工程部署版本，而是一个为了**理解 BEVFormer Attention 与数据流**而写的教学版实现。

如果你现在的真实感受是：

```text
源码里每一行都认识，但连起来不知道它到底在干什么
```

那么这份文档就是写给你的。

---

## 0. 先别看源码，先抓住主线

如果你现在只想先懂 20%，那请先记这一句话：

**BEVFormer 就是在给地面上的每个格子派一个小侦探，这个侦探先去上一帧地图里找历史线索，再去多个相机里找当前证据，最后把这些证据写回这个格子。**

后面的所有函数，其实都只是这句话的展开。

### 0.1 先把术语翻译成人话

- `BEV query`
  - 鸟瞰图上一个格子的 token
  - 你可以把它想成“负责这个格子的小侦探”
- `prev_bev`
  - 上一帧已经整理好的 BEV 地图
  - 你可以把它想成“上一帧留下的地图笔记”
- `reference_points_2d`
  - 这个格子在 BEV 平面上的中心点
  - 你可以把它想成“这个侦探默认站的位置”
- `reference_points_3d`
  - 这个格子在 3D 世界里沿高度方向采样出来的一组点
  - 你可以把它想成“这个格子竖起来的一根柱子上的探针”
- `reference_points_cam`
  - 这些 3D 探针投影到相机图像上的点
  - 你可以把它想成“这个格子在相机里应该先看哪里”
- `bev_mask`
  - 哪个相机真的看得到这个格子
  - 你可以把它想成“可见性开关”
- `Temporal Self-Attention`
  - 当前格子去历史 BEV 上找过去的信息
- `Spatial Cross-Attention`
  - 当前格子去多相机图像上找当前证据
- `object query`
  - 最后负责“从整张 BEV 地图里把目标读出来”的检测 query

### 0.2 一个 BEV 格子的一生

不要一上来就想“整个模型怎么跑”。  
先只看**一个 BEV 格子**会发生什么。

一个 BEV 格子 `q` 在 toy 版里会经历：

1. 先确定自己在 BEV 地图上的中心
   - 这就是 `reference_points_2d`
2. 再在 3D 世界里把自己扩成一根柱子
   - 这就是 `reference_points_3d`
3. 先去历史地图 `prev_bev` 看看
   - “上一帧我附近是什么样？”
4. 再把这根柱子投影到各个相机
   - 这就是 `reference_points_cam`
5. 用 `bev_mask` 判断哪些相机真的看得到自己
6. 在可见相机的投影附近采样图像特征
7. 把历史信息和多相机信息融合
8. 得到这个格子当前时刻的新表示

最后，所有格子一起构成一张新的 `BEV feature map`，再交给 decoder 去读目标。

### 0.3 和标准 Transformer 最大的不同

标准 Transformer 更像：

```text
token -> token
q 和所有 k 做相似度
再加权所有 v
```

BEVFormer 更像：

```text
BEV grid cell -> 历史 BEV / 多相机图像
先拿到一个几何参考点
再只在参考点附近采样少量位置
最后加权融合
```

所以更适合把它理解成：

**带几何先验的稀疏查询系统。**

### 0.4 第一次看源码时，最重要的 3 个问题

第一次读 `model.py` 时，不要试图把每一行都吃掉。  
你只需要反复问自己 3 个问题：

1. 这个 query 现在在哪个空间里？
   - BEV 空间，还是图像空间？
2. 这个 query 现在在向谁取信息？
   - 历史 BEV，还是多相机图像，还是 decoder memory？
3. 它是“全量匹配”，还是“围绕参考点做少量采样”？

如果这 3 个问题你能一直答出来，那你其实已经在理解 BEVFormer 了。

---

## 1. 这个 toy 模型想解决什么问题

原版 `BEVFormer` 很难直接从零读懂，主要有 3 个原因：

1. 代码依赖 `MMCV / custom CUDA / deformable attention`
2. 同时牵涉到多相机、时序、BEV 网格、3D box、decoder
3. 工程实现里有很多为了训练和部署加入的细节

所以 `BevFormer_Toy` 的目标不是复现公版全部功能，而是把核心逻辑拆成一个可读、可跑、可训练的小模型。

这个 toy 版保留了 4 个最关键的思想：

1. 用固定 `BEV query` 表示地面上的 BEV 网格
2. 用 `Temporal Self-Attention` 融合历史 `prev_bev`
3. 用 `Spatial Cross-Attention` 从多相机特征中取证据
4. 用 `object query` 从 BEV memory 里解码目标

---

## 2. 目录结构

`BevFormer_Toy` 里的核心文件是：

- `model.py`
  - toy 模型主体
- `demo.py`
  - 前向演示脚本
- `train_toy.py`
  - 合成数据训练脚本
- `visualize_toy.py`
  - attention 几何可视化脚本
- `toy_model_guide_zh.md`
  - 当前这份说明文档
- `bevformer_tutorial_zh.md`
  - 更偏概念解释的教程

推荐阅读顺序：

1. 先看 `demo.py`
2. 再看 `model.py` 里的主干函数
3. 再看 `visualize_toy.py`
4. 最后看 `train_toy.py`

---

## 3. `model.py` 里的整体结构

可以把 `ToyBEVFormer` 想成下面这条主干：

```text
多相机特征 + prev_bev
    -> Encoder
        -> Temporal Self-Attention
        -> Spatial Cross-Attention
        -> FFN
    -> 得到当前帧 BEV feature
    -> Decoder
        -> object queries 读取 BEV memory
    -> 分类头 + 框回归头
```

对应到 `ToyBEVFormer.forward()`，主要步骤是：

1. 生成 `reference_points_2d`
2. 生成 `reference_points_3d`
3. 把 `reference_points_3d` 投影到各相机，得到：
   - `reference_points_cam`
   - `bev_mask`
4. 初始化 `bev_query`
5. 经过多层 encoder
6. 经过多层 decoder
7. 输出：
   - `bev_feature`
   - `pred_logits`
   - `pred_boxes`

---

## 4. 先理解 4 个最重要的中间量

### 4.1 `reference_points_2d`

由 `make_reference_points_2d()` 生成。

它表示：

**每个 BEV 格子在 BEV 平面上的中心点。**

shape：

```text
[B, num_bev_queries, 2]
```

它主要给 `TemporalSelfAttention` 用。

### 4.2 `reference_points_3d`

由 `make_reference_points_3d()` 生成。

它表示：

**每个 BEV 格子沿高度方向采样出来的一组 3D pillar 点。**

shape：

```text
[B, num_bev_queries, num_points_in_pillar, 3]
```

它主要给 `SpatialCrossAttention` 用。

### 4.3 `reference_points_cam`

由 `project_points_to_cameras()` 生成。

它表示：

**每个 3D pillar 点投影到每个相机图像平面后的 2D 坐标。**

shape：

```text
[B, num_cams, num_bev_queries, num_points_in_pillar, 2]
```

这就是空间 attention 真正拿来采样图像特征的“默认中心点”。

### 4.4 `bev_mask`

它表示：

**某个 pillar anchor 在某个 camera 里是不是可见。**

shape：

```text
[B, num_cams, num_bev_queries, num_points_in_pillar]
```

它不是可有可无的小细节，而是整个 `SpatialCrossAttention` 能成立的重要前提。

---

## 5. `TemporalSelfAttention` 到底做了什么

toy 版里的 `TemporalSelfAttention` 保留了原版最重要的思想：

- 当前 `BEV query` 不直接和全部历史 token 两两相乘
- 而是在历史 `prev_bev` 和当前 `query` 上各自围绕参考点做少量采样
- 再把两边结果融合

可以拆成下面几步：

1. 输入当前 `query`
2. 如果有 `prev_bev`，就把它也拿进来
3. 组合 `prev_bev + query` 预测：
   - `offsets`
   - `weights`
4. 在两个源上采样：
   - 历史 BEV
   - 当前 BEV
5. 把采样结果做平均融合
6. 再过线性层输出

### 5.1 为什么这里是两份 source

在 toy 版中，temporal attention 的采样源就是：

- `prev_map`
- `current_map`

也就是说，对一个当前 BEV 格子，它不是只看历史，也不是只看自己，而是同时看两边，再融合。

### 5.2 `ego_shift` 在 toy 版里做了什么

真实 BEVFormer 里，自车运动会让“上一帧同一世界位置”对应到当前 BEV 的不同位置。

toy 版里没有完整 `can_bus` 与旋转补偿，只保留了最核心的一点：

**对历史 reference point 做一个平移 `ego_shift`。**

所以 `TemporalSelfAttention` 里会构造两个 base reference：

- 历史参考点：`reference_points_2d + ego_shift`
- 当前参考点：`reference_points_2d`

---

## 6. `SpatialCrossAttention` 到底做了什么

这是 toy 模型里最能体现 BEVFormer 思想的部分。

核心问题是：

**BEV 上某个格子在真实 3D 空间对应的位置，投影到多相机后应该看图像上的哪里？**

toy 版的回答步骤是：

1. 每个 BEV 格子沿高度方向取多个 3D 点
2. 把这些 3D 点投影到各相机
3. 用 `bev_mask` 过滤掉不可见的投影点
4. 以投影点为中心预测 `offsets`
5. 在投影点附近采样图像特征
6. 对每个相机分别得到一个结果
7. 最后把所有可见相机结果平均融合

### 6.1 为什么有 `bev_mask`

不是每个 BEV 格子在每个相机里都可见。

所以 `bev_mask` 负责告诉模型：

- 哪些 camera / anchor 是可见的
- 哪些位置完全不该参与 attention

### 6.2 toy 版和原版最大的区别

原版空间 attention 是高效 CUDA 版的 `MSDeformableAttention3D`。  
toy 版为了可读性，改成了纯 PyTorch 的：

- `grid_sample`
- 少量采样点
- 显式 offsets / weights

所以思想是一样的，但实现更直观。

---

## 7. `sample_from_feature_map()` 到底在干什么

这是 toy 版里最像“deformable sampling 核心”的函数。

它解决的问题是：

**如果我已经知道要看哪些点，怎么把这些点上的特征取出来？**

输入 shape：

- `feature_map`: `[B, C, H, W]`
- `sample_points`: `[B, N, num_heads, P, 2]`

输出 shape：

```text
[B, N, num_heads, P, head_dim]
```

### 7.1 这里取出来的是“像素值”吗

不是。

这里取出来的不是原始图像的 RGB 像素值，而是：

**特征图上的特征向量。**

你可以这样区分：

- 原图上的值
  - 更像 `[R, G, B]`
- feature map 上的值
  - 更像 `[0.12, -0.33, 1.07, ..., 0.58]`

也就是说：

```text
原图上采样 -> 像素值
feature map 上采样 -> 特征值
BEVFormer 实际采样 -> 特征值
```

在这个 toy 项目里更明显，因为 `image_feats` 本身就不是从真实图片 backbone 提出来的，而是合成出来的 feature map。

### 7.2 为什么先把通道拆成 `num_heads`

源码里先做：

```text
[B, C, H, W]
    -> [B, num_heads, head_dim, H, W]
```

因为后面 attention 的权重是按 head 分开的，所以每个 head 应该只负责自己的特征子空间。

### 7.3 为什么要把 `[0, 1]` 坐标变成 `[-1, 1]`

因为 `F.grid_sample()` 的采样坐标规范就是 `[-1, 1]`。

所以这里并不是在改几何意义，而是在适配 PyTorch API。

### 7.4 它到底怎么提取特征

如果一个采样点正好落在整数格点中心，比如：

```text
(x=10, y=7)
```

那直接取这个位置上的特征向量就可以。

但真实情况通常不是这样。  
attention 预测出来的采样点往往是浮点数，比如：

```text
(x=10.3, y=7.6)
```

这个点并不正好落在某一个离散格点上，所以不能直接“索引一个像素”。

这时候 `grid_sample()` 会做**双线性插值**。

### 7.5 双线性插值是什么意思

假设采样点落在 4 个相邻格点之间：

```text
左上: (10, 7)
右上: (11, 7)
左下: (10, 8)
右下: (11, 8)
```

那么 `grid_sample()` 不会只取其中一个点，而是：

1. 先取这 4 个位置各自的特征向量
2. 根据采样点离它们有多近，算 4 个权重
3. 用这 4 个权重做加权平均

于是最后拿到的不是“某一个格点的特征”，而是：

```text
一个由附近 4 个格点平滑插值得到的特征向量
```

### 7.6 这个函数和 attention weights 的关系

`sample_from_feature_map()` 本身**不负责决定哪个点更重要**。

它只负责：

```text
“你给我采样坐标，我就把这些坐标上的特征取出来。”
```

而“哪些点更重要”是外面的 attention 模块决定的：

- `TemporalSelfAttention` 里的 `weight_logits`
- `SpatialCrossAttention` 里的 `weight_logits`

也就是说，整个流程分成两步：

1. `sample_from_feature_map()`
   - 负责取值
2. attention weights
   - 负责加权

### 7.7 一句最直白的总结

`sample_from_feature_map()` 的本质就是：

**根据给定的连续坐标，在特征图上用双线性插值读出对应位置的特征向量。**

---

## 8. `visualize_toy.py` 能帮助你看到什么

如果你觉得仅靠打印 shape 还是太抽象，那么 `visualize_toy.py` 就是专门为这个问题准备的。

它会生成两类图：

### 8.1 单个 query 的相机投影图

这张图会展示：

- 某个 `BEV query` 在指定相机中的 pillar anchor 投影点
- 哪些 anchor 是可见的
- 这个 query 最终在该相机附近采样的 spatial sampling points

### 8.2 全局 `bev_mask` 可见性热力图

这张图会展示：

- 每个 BEV query 被多少个相机看见
- 你当前选中的 query 在 BEV 网格上的位置

---

## 9. `train_toy.py` 为什么不是纯随机训练

如果只是随便生成随机图像特征和随机标签，那么模型学不到稳定规律，loss 下降也没有解释意义。

所以 `train_toy.py` 用的是一个**合成但有规律的数据集**。

### 9.1 合成数据集的设计目标

目标是让模型真正需要用到这几个输入：

- `prev_bev`
- `image_feats`
- `ego_shift`

并且让监督信号和输入之间存在真实关联。

### 9.2 每个训练样本怎么生成

一个样本大致按下面流程生成：

1. 随机决定哪些 object slot 是激活的
2. 对每个激活目标，随机生成：
   - 类别
   - 3D 中心
   - 3D 尺寸
   - yaw
3. 生成上一帧目标位置 `prev_box`
4. 再根据一个 world motion 生成当前帧目标位置 `curr_box`
5. 把当前框编码成一个 deterministic feature vector
6. 用这些 feature vector 去渲染：
   - `prev_bev`
   - 当前多相机特征 `image_feats`
7. 把当前帧类别和 3D box 当成监督标签

### 9.3 为什么这样设计

这样设计后：

- `prev_bev` 里有上一帧的场景痕迹
- `ego_shift` 告诉模型如何对齐历史位置
- 多相机特征里有当前帧投影后的视觉证据
- 目标标签与这些证据是匹配的

所以 toy 模型不是在瞎猜，而是在学一个简化版的时空融合问题。

---

## 10. `train_toy.py` 的损失怎么定义

toy 版训练目标很简单：

### 10.1 分类损失

对每个 object query / slot，做一个多分类：

- 前景类别 0
- 前景类别 1
- 前景类别 2
- 背景类

这里用的是标准 `cross_entropy`。

### 10.2 框回归损失

对激活目标，用 `L1 loss` 回归归一化后的 7 维 box：

- `x`
- `y`
- `z`
- `dx`
- `dy`
- `dz`
- `yaw`

训练时会对 `pred_boxes` 先做 `sigmoid()`，然后和归一化后的 GT box 做 L1。

### 10.3 总损失

```text
total_loss = cls_loss + 5 * box_loss
```

这个比重只是 toy 版经验设置，不代表论文配置。

---

## 11. toy 版和真实 BEVFormer 的对应关系

你可以这样对照：

- 公版 `bev_embedding`
  - toy 版里的 `self.bev_queries`
- 公版 `prev_bev`
  - toy 版里的 `prev_bev`
- 公版 `ref_2d`
  - toy 版里的 `reference_points_2d`
- 公版 `ref_3d`
  - toy 版里的 `reference_points_3d`
- 公版 `reference_points_cam`
  - toy 版里的 `reference_points_cam`
- 公版 `bev_mask`
  - toy 版里的 `bev_mask`
- 公版 `TemporalSelfAttention`
  - toy 版里的同名模块
- 公版 `SpatialCrossAttention`
  - toy 版里的同名模块
- 公版 detection decoder
  - toy 版里的 `DetectionDecoderLayer`

---

## 12. `model.py` 源码最短阅读路线

第一次读源码时，建议只看这 6 段：

1. `make_reference_points_2d()`
2. `make_reference_points_3d()`
3. `project_points_to_cameras()`
4. `TemporalSelfAttention.forward()`
5. `SpatialCrossAttention.forward()`
6. `ToyBEVFormer.forward()`

第一次最重要的不是把所有 `reshape / permute` 都看懂，而是下面这 4 个问题：

1. 当前张量是在 `BEV` 空间，还是在 `image` 空间？
2. 当前 query 正在向谁取信息？
3. 当前 reference point 是哪一个？
4. 最后采样结果被写回到哪里？

只要这 4 个问题你答得出来，你就已经在真正理解源码了。

---

## 13. `model.py` 源码逐段解释

这一章只讲最关键的部分，不追求面面俱到，而追求把主线讲顺。

### 13.1 `make_reference_points_2d()`

作用：

**给每个 BEV 网格格子生成一个平面中心点。**

输入：

- `bev_h`
- `bev_w`
- `batch_size`

输出：

```text
[B, bev_h * bev_w, 2]
```

核心逻辑：

- 用 `torch.linspace()` 生成 x / y 方向均匀网格
- 用 `torch.meshgrid()` 拼成二维平面
- 再 `reshape` 成 token 序列

为什么用 `0.5 / bev_h` 和 `0.5 / bev_w`？

因为我们想取的是**每个格子的中心**，而不是格子边界。

### 13.2 `make_reference_points_3d()`

作用：

**把每个 BEV 格子从一个 2D 点扩展成一根 3D pillar。**

输出：

```text
[B, bev_h * bev_w, num_points_in_pillar, 3]
```

源码思路：

1. 在 `pc_range` 内建立 `(x, y)` 网格中心
2. 在高度方向 `z` 上均匀采样多个点
3. 把 `(x, y)` 和多个 `z` 拼起来

为什么一定要这样做？

因为 BEV query 只天然对应地面上的 `(x, y)`，但真实物体有高度。

### 13.3 `project_points_to_cameras()`

作用：

**把每个 3D pillar anchor 投影到每个 camera。**

输入：

- `reference_points_3d`: `[B, N, D, 3]`
- `lidar2img`: `[B, num_cams, 4, 4]`

输出：

- `uv`: `[B, num_cams, N, D, 2]`
- `bev_mask`: `[B, num_cams, N, D]`

核心逻辑：

1. `(x, y, z)` 先补成齐次坐标 `(x, y, z, 1)`
2. 用 `lidar2img` 做矩阵乘法
3. 用深度做透视除法
4. 把坐标归一化到 `[0, 1]`
5. 判断点是否在图像范围内，得到 `bev_mask`

这个函数的意义可以总结成一句：

**把 3D 世界里的 pillar anchors 变成图像空间里的默认 attention 中心。**

### 13.4 `masked_softmax()`

作用：

**让不可见位置不参与 softmax 权重分配。**

它做的事情是：

1. 把 mask 为 0 的位置用很小的负数替换
2. softmax
3. 再乘一遍 mask
4. 最后重新归一化

这样做的好处是：

- 不可见点不会被分到权重
- 即使一整行都无效，也不会产生 `nan`

### 13.5 `sample_from_feature_map()`

这部分你可以直接对照第 `7` 节去理解。

一句话总结：

**给我一张特征图和一堆连续坐标，我用双线性插值把这些坐标上的特征向量取出来。**

### 13.6 `TemporalSelfAttention.forward()`

输入：

- `query`: 当前 BEV tokens，`[B, N, C]`
- `prev_bev`: 历史 BEV tokens，`[B, N, C]`
- `reference_points_2d`
- `ego_shift`

输出：

```text
[B, N, C]
```

核心步骤：

1. 把 `query` 和 `prev_bev` 还原成二维 BEV feature map
2. 把二者拼起来，预测 `offsets` 和 `weights`
3. 构造两个 base reference：
   - 历史参考点：`reference_points_2d + ego_shift`
   - 当前参考点：`reference_points_2d`
4. 分别在 `prev_map` 和 `current_map` 上采样
5. 把两边结果做平均融合

这说明 temporal attention 的本质不是：

```text
所有 BEV token 两两做相似度
```

而是：

```text
围绕历史和当前参考点做局部稀疏采样
```

### 13.7 `SpatialCrossAttention.forward()`

输入：

- `query`: `[B, N, C]`
- `image_feats`: `[B, num_cams, C, H, W]`
- `reference_points_cam`: `[B, num_cams, N, D, 2]`
- `bev_mask`: `[B, num_cams, N, D]`

输出：

```text
[B, N, C]
```

核心步骤：

1. 用 `query` 预测 `offsets` 和 `weights`
2. 对每个 camera：
   - 取该 camera 的 `reference_points_cam`
   - 用 `bev_mask` 过滤不可见 anchor
   - 在投影点附近采样图像特征
   - 得到该 camera 对这个 query 的贡献
3. 把所有可见 camera 的结果平均融合

空间 attention 的本质可以总结成：

**先用几何告诉模型“该看图里的哪里”，再让模型在那附近做少量细调采样。**

### 13.8 `BEVFormerEncoderLayer`

这个类只是把前面的几个模块串起来：

1. `TemporalSelfAttention`
2. `SpatialCrossAttention`
3. `FeedForward`

再加上：

- `LayerNorm`
- residual connection
- dropout

执行顺序是：

```text
temporal -> residual
spatial -> residual
ffn -> residual
```

### 13.9 `DetectionDecoderLayer`

toy 版里 decoder 故意简化成标准 `nn.MultiheadAttention`。

作用是：

**用 object queries 去读取 encoder 输出的 BEV memory。**

forward 分成 3 步：

1. object queries 自己做 self-attention
2. object queries 对 `memory` 做 cross-attention
3. FFN

这里的 `memory` 就是 encoder 输出的 `bev_feature`。

### 13.10 `ToyBEVFormer.__init__()`

这个构造函数主要在定义三类可学习参数：

1. BEV 相关参数
   - `self.bev_queries`
   - `self.bev_pos_mlp`
2. decoder query 相关参数
   - `self.object_queries`
   - `self.object_query_pos`
3. 预测头
   - `class_head`
   - `box_head`

### 13.11 `ToyBEVFormer.forward()`

这是整个 toy 模型最重要的总控函数。

建议你第一次读时只抓主干：

1. 检查输入 shape
2. 生成 `reference_points_2d`
3. 生成 `reference_points_3d`
4. 投影得到 `reference_points_cam + bev_mask`
5. 初始化 `bev_query`
6. 走 encoder，得到 `memory`
7. 准备 object queries
8. 走 decoder
9. 输出：
   - `pred_logits`
   - `pred_boxes`

如果必须用一句话概括这个 `forward()`，那就是：

**先用显式几何生成 reference points，再让 query 围绕这些 reference 做稀疏采样，最后把融合后的 BEV memory 交给 decoder 读出目标。**

---

## 14. 你真的看懂没有？用这 5 个问题检查

读完以后，试着不用看文档回答下面 5 个问题：

1. `BEV query` 到底是什么？
2. 为什么要同时有 `reference_points_2d` 和 `reference_points_3d`？
3. 为什么 `SpatialCrossAttention` 不是看整张图，而是看投影点附近？
4. `sample_from_feature_map()` 取出来的是像素值还是特征值？
5. 为什么最后还需要 `object query`？

如果这 5 个问题你都能答出来，那你对 toy 版的主线就已经真正掌握了。

---

## 15. 怎么运行

### 15.1 看前向

```bash
python demo.py
```

### 15.2 跑一个最小训练

```bash
python train_toy.py --epochs 1 --batch-size 4 --train-samples 16 --val-samples 8
```

### 15.3 看几何可视化

```bash
python visualize_toy.py --query-idx 210 --camera-idx 0
```

### 15.4 跑稍微完整一点的 toy 训练

```bash
python train_toy.py --epochs 5 --batch-size 8 --train-samples 256 --val-samples 64
```

---

## 16. 最后用一句话收束

如果你想用一句最不容易忘的话来记住这个 toy 模型，那就是：

**BEVFormer = 让地面上的每个格子，先去历史地图和多相机图像里搜集证据，再把这些证据合成为当前时刻的俯视图表示。**
