# BDCI-TPU-2023

赛题：[基于TPU平台实现超分辨率重建模型部署 Competitions - DataFountain](https://www.datafountain.cn/competitions/972/datasets)

环境搭建：[TPU-Coder-Cup/CCF2023 at main · sophgo/TPU-Coder-Cup (github.com)](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023)

## 文件结构

```shell
.
├── readme.md		                AI大作业代码说明
├── bmodel/                         整理的 bmodel
├── results/                        整理的部分结果
├── src-train/						用于模型训练
├── upscale-multithread.py          用于多线程分瓦片推理
├── upscale-notile-batch.py         用于多批不分瓦片推理
├── upscale-notile.py               用于不分瓦片推理
├── upscale.py                      竞赛包：用于分瓦片推理，baseline
├── 算能-超分辨率重建模型迁移DEMO.md    竞赛包：TPU竞赛说明
├── doc/                            竞赛包：TPU竞赛辅助资料
├── dataset/						竞赛包：TPU竞赛数据（因文件过大，细节见百度网盘）
├── fix.py                          竞赛包：用于图片整合
├── metrics/                        竞赛包：用于计算NIQE
├── npuengine.py                    竞赛包：用于创建 model on TPU
├── sophon-0.4.6-py3-none-any.whl   竞赛包：TPU竞赛平台构建
└── sophon-0.4.8-py3-none-any.whl   竞赛包：TPU竞赛平台构建

```

代码管理：[Maxpicca-Li/BDCI-TPU-2023: for CCF BDCI 2023 contest (github.com)](https://github.com/Maxpicca-Li/BDCI-TPU-2023)

其中，注有“竞赛包”的，为赛方提供的竞赛资料。

## dataset

链接: https://pan.baidu.com/s/1LjD-6GZ3vw7rLRyBbwMM5g?pwd=3y3f 提取码: 3y3f

```shell
├── CCF_train.zip								总训练数据
├── testA.zip									A榜测试数据
├── testB.zip									B榜测试数据
├── train-173107.zip							A榜训练数据
└── tpu-mlir_v1.2.8-g32d7b3ec-20230802.tar.gz	tpu-milr数据
```



## 命令说明

```shell
# 1. 模型训练
python [脚本名称].py --config ./configs/ninasr.yaml --cuda

# 2. 模型转换 pt -> milr，算能 docker 环境平台
model_transform.py \
 --model_name [模型名] \
 --input_shape [[1,3,200,200]] \
 --model_def [模型].pt \
 --mlir [模型].mlir \

# 3. 模型部署 milr -> bmodel，算能 docker 环境平台
model_deploy.py \
 --mlir [模型].mlir \
 --quantize F16 \
 --chip bm1684x \
 --model [模型].bmodel

# 4. 模型推理，算能 TPU 平台
python3 [upscale-xxx].py \
--model_path models/[模型名称].bmodel \
--input dataset/TestB \
--output results/TestB \
--report results/[结果名称].json
```
