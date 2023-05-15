# Enhancer-B201
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center"><a href="https://github.com/ye3why/Enhancer-B201"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"></a></div>

<div align="left"> </div>

## 介绍
易用、高效、可扩展的视频增强工具。


## Get Started
### 依赖
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


### 输入视频，输出视频
```bash
CUDA_VISIBLE_DEVICES=0,1 python enhance.py --input_path ./demo/demo_video.mp4 --output_dir demo/result --models EDSRx2
```

### 输入图片数据集
```bash
python enhance.py --list
CUDA_VISIBLE_DEVICES=0,1 python enhance_img.py --input_path path/to/img_datasets --output_dir path/to/save/results --models MODEL1 MODEL2 MODEL3
```

## 使用方法
### 对视频进行增强
- 复制options/template.yml,如：
```bash
cp options/template.yml options/test.yml
```
- 修改options/test.yml,

   1、设置输入输出路径, 支持单个视频和视频文件夹输入
   
   2、取消注释models:下需要使用的模型, 程序将根据这里的模型顺序依次执行，同一模型可多次使用

   3、设置输出视频的编码参数


- 增强
```bash
python enhance.py --opt options/test.yml
```

### 视频增强选项说明
参见options/template.yml
```yaml
video_path: 视频文件夹的list，可以是文件夹或单个视频，支持多个条目
    - ~/demo_video/test1.mp4 视频
    - ~/demo_video/ 该文件夹下所有视频
output_dir: 

batch_size_per_gpu: 1

chop_forward: 一些大的模型需要切块输入，仅对models.yml中的chopable选项为true的模型生效
chop_threshold: 对w*h大于此数值的patch进行切块

# 用于处理输入的模型(模型需要在models.yml中声明)
# 将会按照以下出现的顺序串行执行
# 同一模型可多次使用
# 如，以下设置将对输入去压缩失真，再进行两次二倍超分辨率
models:
    - CAR
    - SRx2
    - SRx2

# 视频输出参数
# 提供三个模板，普通视频，hlg视频， prores格式，请根据实际需要进行修改
video_spec:
    ext: .mov   # 输出格式
    fps: &fps 50 # 输出视频帧率，若模型输出帧率不足，请将vfs中fps取消备注进行插帧
    bitrate: 50M # 输出码率
    output_kwargs:
        vcodec: libx265 # 编解码器
        colorspace: bt709 # 色彩空间
        pix_fmt: yuv420p # 像素格式

# ffmpeg 中的 video filters
# 可选，取消注释进行使用，也可添加其他ffmpeg支持的filter
# 将会按照出现的顺序执行
vfs:
    scale: '3840x2160'  # 对输出视频缩放
    hqdn3d: # 去噪
    unsharp: '3:3:1' # 锐化
    fps: *fps   # 按给定帧率插帧
    sar_dar: 'setsar=16/15,setdar=4/3' # 单个像素长宽比

```

## 其他功能

### 对增强后的视频添加film grain
```bash
python util/film_grain.py --opt options/demo.yml
```

### 对图片数据集进行测试

```bash
python enhance_img.py -i 图片文件夹或单张图片路径 -o 数出文件夹 -m 测试用的模型(models.yml中的名称)
```

### 使用Docker环境
**build Dockerfile 或加载docker镜像**
- build Dockerfile
```bash
sh docker/getzip.sh  # 如提示不是git repo, 请将目录打包为enhancer-b201.tar.gz并放入docker/文件夹中
docker build -t enhancer-b201 docker/
```
若修改了代码，请先执行git commit或手动打包程序目录

- 加载docker镜像

下载镜像 [百度云](https://pan.baidu.com/s/1R7n6PrInY8rUHBml5Eamvg) 提取码：tmlw

加载镜像
```bash
docker load -i enhancer-b201_docker.tar
```

**使用docker运行**

```bash
docker run --runtime nvidia -it --rm --ipc=host -v /本地数据目录:/data enhancer-b201 bash
```

程序位于docker中/Enhancer-B201

将预训练模型放入docker中/Enhancer-B201/weights


## 扩展
### 注册新模型
- 将模型文件放入models文件夹，并在模型类前添加@MODEL_REGISTRY.register()

- 将对应预训练参数放入weights文件夹

- 在models.yml中声明新模型

### 模型声明参数说明
```yaml
模型名称: 在配置文件中调用的名称
  class: 模型的类名
  modelargs:
    模型类的参数
  pretrain: 模型的预训练文件路径
  dataset_class: 使用的Dataset的类名，Dataset类输出的filename数量需和模型输出帧数对应,
                 比如插帧模型，输出两帧，那么需要提供两个文件名
                目前可用VideoDataset: 输入多帧，输出一帧
                        ImageDataset: 输入一帧，输出一帧
                        VideoMinMoutDataset: 输入多帧，输出相同数量的帧
                        FrameInterpDataset: 输入多帧，输出两帧
  nframes: 模型输入帧数
  chopable: 模型是否支持视频增强选项中的chop(一些大模型需要)
  need_pad: 输入需要pad到的整数倍，比如4，8
  multi_output: 若模型一次输出多帧需设为true, 如序列输入序列输出或插帧
  saveimg_function: 模型输出图片的保存方法， 比如sdr，hdr
  model_scale: 模型的超分倍数，其他模型设置为1
```

### 添加新的数据读取方式
在dataset.py中添加新的Dataset类

通过@DATASET_REGISTRY.register()注册

在models.yml中供需要的模型使用, 通过类名调用

注意以下规则：
- 视频解为帧的文件名是帧序号数字，从1开始递增, 1, 2, 3, 4, 5 ,6...
- Dataset类输出的filename数量需和模型输出帧数对应, 比如插帧模型，输出两帧，那么需要提供两个文件名, 且也是帧序号的数字

### 添加新的图片存储方式
添加新的图片存储方式，如hdr视频需要16bit png存储

在utils.py中添加新的图片存储函数

通过@SAVEIMG_REGISTRY.register('name')注册

在models.yml中供需要的模型使用, 通过name调用

## 目前支持的模型
依赖的pytorch版本取决于各个模型的pytorch版本
### 超分辨率
| 模型            | 功能      | 备注                                                 |
| :---:           | :---:     | :---:                                                |
| **视频**        |           |                                                      |
| RealBasicVSR_x4 | 4倍, 视频 | [code](https://github.com/ckkelvinchan/RealBasicVSR) |
| SRx2            | 2倍, 视频 | 依赖pytorch1.10，需编译DCN                           |
| SRx3            | 3倍, 视频 | 依赖pytorch1.10，需编译DCN                           |
| SRx4            | 4倍, 视频 | 依赖pytorch1.10，需编译DCN                           |
| BasicVSR_ds_x2  | 2倍, 视频 |                                                      |
| VRTx4_bd        | 4倍, 视频 | [code](https://github.com/JingyunLiang/VRT)          |
| **图像**        |           |                                                      |
| EDSRx2          | 2倍, 单图 | [code](https://github.com/sanghyun-son/EDSR-PyTorch) |
| RealESRGANx2    | 2倍, 单图 | [code](https://github.com/xinntao/Real-ESRGAN)       |
| RealESRGANx4    | 4倍, 单图 | [code](https://github.com/xinntao/Real-ESRGAN)       |

### 去降质
| 模型             | 功能                       | 备注                                    |
| :---:            | :---:                      | :---:                                   |
| **去噪**         |                            |                                         |
| Denoise_low      | 通用去噪，轻度             |                                         |
| Denoise_medium   | 通用去噪，中度             |                                         |
| Denoise_high     | 通用去噪，重度             |                                         |
| **去压缩失真**   |                            |                                         |
| CAR              | 去压缩失真                 |                                         |
| **去划痕**       |                            |                                         |
| Descratch_bw     | 去黑色及白色的划痕         |                                         |
| Descratch_white  | 去白色的划痕               |                                         |
| **去交错**       |                            |                                         |
| DeInterlace      | 仅去交错                   |                                         |
| MFDIN_2X         | 去交错同时2倍超分          |                                         |
| MFDIN_2P         | 去交错同时2倍插帧          |                                         |
| MFDIN_2X2P       | 去交错同时2倍超分和2倍插帧 | [code](https://github.com/anymyb/MFDIN) |
| **色像较正**     |                            |                                         |
| ChromaticCorrect | 色像差矫正                 |                                         |

### HDR
| 模型  | 功能         | 备注  |
| :---: | :---:        | :---: |
| HLG   | SDR转HDR HLG |       |

## 常见问题
- out of memory

    设置chop_forward为true，并根据显存大小适当减小chop_threshold

- 环境配置困难

    若配置环境遇到困难，请使用docker环境

- pytorch版本问题

    请根据超分辨率模型需要的pytorch, 选择pytorch版本

- 配置docker时，提示不是git repo

    请将程序目录打包为enhancer-b201.tar.gz并放入docker/文件夹中

- ffmpeg报错
    
    请确认使用的ffmpeg版本是否支持对应功能

    可在选项中设置debug: true输出详细的ffmpeg信息


- 由于文件名包含空格导致ffmpeg报错
    
    替换文件名中的空格
    
    可以使用utils/renamefiles.py替换目录下所有文件名

