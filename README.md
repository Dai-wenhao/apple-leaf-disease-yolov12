# 苹果叶片病害 YOLOv12 改进模型

基于 YOLOv12-turbo 的苹果叶片病害检测改进,面向整叶级四分类任务(Healthy / Scab / Black_rot / Cedar_rust)。

## 改进机制

| 模块 | 作用 | 接入位置 |
|------|------|----------|
| DilatedContext | 骨架多尺度膨胀上下文增强 | 骨干 P3 |
| GSConv | 颈部轻量化(标准卷积 + 深度可分离混合) | 颈部下采样 |
| MambaoutLayer | 门控卷积建模 | 骨干末端 |

## 文件结构

```
ultralytics/nn/modules/conv.py            # DilatedContext / GAM / GSConv
ultralytics/nn/modules/block.py           # MambaBlock / MambaoutLayer
ultralytics/nn/modules/custom_modules.py  # GSConv / MambaoutLayer(独立文件)
ultralytics/nn/modules/__init__.py        # 模块导出
ultralytics/nn/tasks.py                   # parse_model 注册与参数解析

YOLOv12-turbo.yaml                                 # 基线
YOLOv12-turbo-DilatedContext.yaml                  # +DilatedContext
YOLOv12-turbo-DilatedContext-GSConv.yaml           # +GSConv
YOLOv12-turbo-DilatedContext-GSConv-Mambaout.yaml  # 完整模型

train_dc.py / train_dc_gsconv.py / train_dc_gsconv_mambaout.py   # 训练脚本
val_test_all.py / val_test_full.py / val_test_gsconv.py          # 测试集评估脚本
```

## 数据集

本仓库不含数据,以下两个开源数据集需自行下载:

| 数据集 | 类别 | 用途 | 放置目录 |
|--------|------|------|----------|
| Plant Pathology Apple Dataset | 4 类(Healthy / Scab / Black_rot / Cedar_rust) | 主实验(整叶级) | `dataset/dataset_new_apple_split/` |
| ATLDSD | 5 类(Alternaria leaf spot / Brown spot / Gray spot / Healthy leaf / Rust) | 迁移/对比实验 | `dataset/ATLDSD/` |

## 使用说明

本仓库只包含**相对官方 Ultralytics 的改动文件**,不含完整 ultralytics 源码、数据集与权重。运行需:

1. 安装 Ultralytics 8.3.63(PyTorch 2.6 / CUDA 12.4)
2. 将 `ultralytics/nn/` 下 5 个改动文件覆盖到安装目录对应位置
3. 下载 COCO 预训练权重 `yolov12n.pt` 放到项目根目录
4. 数据集:按上表下载并组织为 `images/{train, val, test}` 结构,并修改各 `data.yaml` 中的 `path`

## 训练

三个消融模型对应三个训练脚本,训练配置一致:epochs=50、batch=4、imgsz=640、Adam(lr0=0.001, lrf=0.01)、cos 退火、warmup=3、COCO 预训练、hsv_h=0.01 等(详见各脚本)。

```bash
python train_dc_gsconv_mambaout.py   # 完整模型
```

## 评估

测试集口径评估,结果用于论文消融表(mAP50-95):

```bash
python val_test_all.py   # 基线 + 三个消融模型
```

## 实验结果

见论文第 4 章消融实验。
