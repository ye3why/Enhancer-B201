# Enhancer-B201
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center"><a href="https://github.com/ye3why/Enhancer-B201"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a></div>

<div align="left"> </div>

## 介绍
易用、高效、可扩展的视频增强工具。


## 使用方法
### 安装依赖
- Pytorch == 1.10 (建议)
- ffmpeg

使用anaconda配置环境
```bash
sudo apt install ffmpeg
conda create --name enhancer-b201 python=3.7
conda activate enhancer-b201
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

或使用docker配置环境，详见 [**使用Docker环境**](#使用Docker环境)

### 查看目前支持的模型
```bash
python code/enhance.py --list
python code/enhance_img.py --list
```
依赖的pytorch版本取决于各个模型


### 输入视频，输出视频
```bash
CUDA_VISIBLE_DEVICES=0 python code/enhance.py --input_path ./demo/demo_video.mp4 --output_dir demo/result --models EDSRx2
```

### 输入图片数据集或帧序列, 输出图片结果
```bash
CUDA_VISIBLE_DEVICES=0 python code/enhance_img.py --input_path path/to/img_datasets --output_dir path/to/save/results --models MODEL1 MODEL2 MODEL3
```

### 计算指标
```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluate.py --restored path/to/restored_results --gt path/to/ground_truth --metric_list psnr ssim lpips niqe
```

## 其他
### 使用yml文件提供参数
支持直接使用命令行进行增强或使用yml文件提供参数


- 参考options/template.yml中的设置进行配置

- 增强
```bash
python code/enhance.py --opt options/xxxxxxx.yml
```


- 部分参数也可通过命令行提供，参见
```bash
python code/enhance.py --help
python code/enhance_img.py --help
```


- 视频增强选项说明, options/template.yml
```yaml
input_path:
    - ~/demo_video/test1.mp4 # 视频
    - ~/demo_video/  # 该文件夹下所有视频
output_dir: 

batch_size_per_gpu: 1

chop:  # 将输入切块 (仅对chopable选项为true的模型生效)
chop_threshold: # 对w*h大于此数值的输入切块

tile: false # 将输入切成固定大小的块 (仅对chopable选项为true的模型生效)
tile_size: 512
tile_pad: 32

# 用于处理输入的模型
# 将会按照以下出现的顺序串行执行, 同一模型可多次使用
# 如，以下设置将对输入去压缩失真，再进行两次二倍超分辨率
models:
    - CAR
    - SRx2
    - SRx2

# 视频输出参数，供ffmpeg使用
output_ext: .mov
video_spec:
    video_bitrate: 50M
    vcodec: h264
    colorspace: bt709
    pix_fmt: yuv420p

# ffmpeg中的-vf
# 将会按照出现的顺序执行
vfs:
    scale: '3840x2160'  # 对输出视频缩放
    fps: *fps   # 按给定帧率插帧

```

### ~~对增强后的视频添加film grain~~
Out of Date
```bash
python util/film_grain.py --opt options/demo.yml
```

### 使用Docker环境
- build Dockerfile
```bash
sh docker/getzip.sh
docker build -t enhancer-b201 docker/
```

- **使用docker运行**

```bash
docker run --runtime nvidia -it --rm --ipc=host -v /本地数据目录:/data enhancer-b201 bash
```

- 将预训练模型放入docker中/Enhancer-B201/weights


## 扩展
### 注册新模型
- 将模型放入code/models，并在类前添加@MODEL_REGISTRY.register()

- 将对应预训练参数放入weights

- 在code/settings/中配置新模型


### 模型声明参数说明
```yaml
模型名称: # 模型调用名称
  class: # 模型类名
  modelargs:
    # 模型的定义参数
  pretrain: # 模型的预训练文件路径
  dataset_class: # 需要的Dataset, 参见code/datasets.py
                       # VideoDataset # 输入多帧，输出一帧
                       # ImageDataset # 输入一帧，输出一帧
                       # VideoMinMoutDataset # 输入多帧，输出相同数量的帧
                       # FrameInterpDataset # 输入多帧，输出两帧
  nframes: # 模型输入帧数
  chopable: # 能否对输入切块
  need_pad: # 输入需要pad到的整数倍，比如4，8
  saveimg_function: # 输出图片的保存方法， sdr，hdr, 参见code/save_func.py
  model_scale: # 输出分辨率放大倍数
  frame_scale: # 输出帧率放大倍数
```


### 添加数据读取方式
在code/dataset.py中添加新的Dataset类
注意以下规则：
- Dataset类输出的filename数量需和模型输出帧数对应, 比如插帧模型，输出两帧，那么需要提供两个文件名


### 添加图片存储方式
在code/save_func.py中添加新的图片存储函数


## 常见问题
- out of memory

    使用--chop或--tile，并根据显存大小适当减小--chop_threshold或--tile_size

- pytorch版本问题

    请根据模型需要的pytorch版本配置环境

- 配置docker时，提示不是git repo

    请将程序目录打包为enhancer-b201.tar.gz并放入docker/文件夹中

- ffmpeg报错
    
    请确认使用的ffmpeg版本是否支持对应功能
    
    可在选项中设置debug: true输出详细的ffmpeg信息

    设置--ffmpeg_cmd可更换ffmpeg


- 由于文件名包含空格导致ffmpeg报错
    
    替换文件名中的空格
    
    可以使用utils/renamefiles.py替换目录下所有文件名

