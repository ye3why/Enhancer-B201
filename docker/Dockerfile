ARG PYTORCH="1.0.1"
ARG CUDA="10.0"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

MAINTAINER yeyy

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y nano vim python-software-properties software-properties-common\
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && add-apt-repository ppa:jonathonf/ffmpeg-4 \
    && apt-get update \
    && apt-get upgrade -y libstdc++6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "channels:" >> /root/.condarc \
    && echo "  - defaults" >> /root/.condarc \
    && echo "show_channel_urls: true" >> /root/.condarc \
    && echo "default_channels:" >> /root/.condarc \
    && echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> /root/.condarc \
    && echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> /root/.condarc \
    && echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> /root/.condarc \
    && echo "custom_channels:" >> /root/.condarc \
    && echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc \
    && echo "  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc \
    && echo "  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc \
    && echo "  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc \
    && echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc \
    && echo "  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> /root/.condarc

#RUN conda init bash \
    #&& . ~/.bashrc \
    #&& conda clean -i \
    #&& conda create -n ve \
    #&& conda activate ve \
    #&& conda install _libgcc_mutex=0.1=main asn1crypto=0.24.0=py36_0 backcall=0.1.0=py36_0 beautifulsoup4=4.7.1=py36_1 blas=1.0=mkl bzip2=1.0.8=h7b6447c_0 ca-certificates=2021.7.5=h06a4308_1 cairo=1.16.0=hf32fb01_1 certifi=2021.5.30=py36h06a4308_0  chardet=3.0.4=py36_1 cryptography=2.5=py36h1ba5d50_0 decorator=4.3.2=py36_0 ffmpeg=4.0=hcdf2ecd_0 filelock=3.0.10=py36_0 fontconfig=2.13.1=h6c09931_0 freeglut=3.0.0=hf484d3e_5 freetype=2.10.4=h5ab3b9f_0  glob2=0.6=py36_1 graphite2=1.3.14=h23475e2_0 harfbuzz=1.8.8=hffaf4a1_0 hdf5=1.10.2=hba1933b_1 icu=58.2=he6710b0_3 idna=2.8=py36_0 intel-openmp=2019.1=144 ipython=7.2.0=py36h39e3cac_0 ipython_genutils=0.2.0=py36_0 jasper=2.0.14=h07fcdf6_1 jedi=0.13.2=py36_0 jinja2=2.10=py36_0 jpeg=9b=h024ee3a_2 ld_impl_linux-64=2.35.1=h7274673_9  libedit=3.1.20170329=h6b74fdf_2 libffi=3.3=he6710b0_2 libgcc-ng=9.1.0=hdf63c60_0 libgfortran-ng=7.3.0=hdf63c60_0 libglu=9.0.0=hf484d3e_1 liblief=0.9.0=h7725739_2 libopencv=3.4.2=hb342d67_1 libopus=1.3.1=h7b6447c_0 libpng=1.6.37=hbc83047_0 libstdcxx-ng=8.2.0=hdf63c60_1 libtiff=4.2.0=h85742a9_0 libuuid=1.0.3=h1bed415_2 libvpx=1.7.0=h439df22_0 libwebp-base=1.2.0=h27cfd23_0 libxcb=1.14=h7b6447c_0 libxml2=2.9.10=hb55368b_3  lzo=2.10=h49e0be7_2 markupsafe=1.1.0=py36h7b6447c_0 mkl=2019.1=144 mkl_fft=1.0.10=py36ha843d7b_0 mkl_random=1.0.2=py36hd81dba3_0 ncurses=6.2=he6710b0_1 numpy=1.16.2=py36h7e9f1db_0 numpy-base=1.16.2=py36hde5b4d6_0 opencv=3.4.2=py36h6fd60c2_1 openssl=1.1.1k=h27cfd23_0 parso=0.3.2=py36_0 patchelf=0.9=he6710b0_3 pcre=8.45=h295c915_0 pexpect=4.6.0=py36_0 pickleshare=0.7.5=py36_0 pip=21.2.2=py36h06a4308_0 pixman=0.40.0=h7b6447c_0 pkginfo=1.5.0.1=py36_0 prompt_toolkit=2.0.8=py_0 psutil=5.5.0=py36h7b6447c_0 ptyprocess=0.6.0=py36_0 py-lief=0.9.0=py36h7725739_2 py-opencv=3.4.2=py36hb342d67_1 pycosat=0.6.3=py36h14c3975_0 pycparser=2.19=py36_0 pygments=2.3.1=py36_0 pyopenssl=19.0.0=py36_0 pysocks=1.6.8=py36_0 python=3.6.13=h12debd9_1 python-libarchive-c=2.8=py36_6 pytz=2018.9=py36_0 pyyaml=3.13=py36h14c3975_0 readline=8.1=h27cfd23_0 requests=2.21.0=py36_0 ruamel_yaml=0.15.46=py36h14c3975_0 setuptools=52.0.0=py36h06a4308_0 six=1.12.0=py36_0 soupsieve=1.7.1=py36_0 sqlite=3.36.0=hc218d9a_0 tk=8.6.10=hbc83047_0 tqdm=4.29.1=py_0 traitlets=4.3.2=py36_0 urllib3=1.24.1=py36_0 wcwidth=0.1.7=py36_0 wheel=0.37.0=pyhd3eb1b0_0 xz=5.2.5=h7b6447c_0 yaml=0.1.7=had09818_2 zlib=1.2.11=h7b6447c_3 zstd=1.4.9=haebb681_0 \
    #&& conda clean --all \
    #&& pip install --no-cache-dir ffmpeg-python torch==1.0 torchvision==0.2.1 scipy==1.1.0
ADD enhancer-b201.tar.gz /Enhancer-B201

#RUN conda init bash \
    #&& . ~/.bashrc \
    #&& conda clean -i \
    #&& conda env create -f /Enhancer-B201/docker/pytorch1.0_docker.yml \
    #&& conda clean --all

#RUN ln -sf /usr/bin/ffmpeg /opt/conda/envs/ve/bin/ffmpeg

RUN conda init bash \
    && . ~/.bashrc \
    && conda clean -i \
    && conda env create -f /Enhancer-B201/docker/pytorch1.10_docker.yml \
    && conda clean --all

RUN ln -sf /usr/bin/ffmpeg /opt/conda/envs/ve10/bin/ffmpeg

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "export LC_ALL=\"C.UTF-8\"" >> ~/.bashrc \
    && echo "conda activate ve10" >> ~/.bashrc

WORKDIR /Enhancer-B201/models/DCNv2/

RUN conda init bash \
    && . ~/.bashrc \
    && conda activate ve10 \
    && python setup_indocker.py build develop

WORKDIR /Enhancer-B201
