# Caution: do not push this image publicly since it contains a ssh key

# our local base image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

LABEL description="Container for Engorgio (Pytorch)"
SHELL ["/bin/bash", "-c"]

# setting
WORKDIR /workspace/research

# install system packages 
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        git mercurial subversion \
        g++ gdb make ninja-build rsync zip \
        ffmpeg tmux vim curl 
RUN apt-get install -y htop tmux nasm yasm

# install git-lfs
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
RUN apt-get install -y git-lfs

# install python packages
RUN pip install tensorboard
RUN pip install xlrd

# install awesomevim
RUN git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime && \
    sh ~/.vim_runtime/install_awesome_vimrc.sh && \
    echo "stty -ixon" >> ~/.bashrc

# upgrade pillow
RUN pip install --upgrade Pillow

# install ffmpeg (with av1 support)
RUN apt-get -y update
RUN apt-get -y install software-properties-common 
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-4 && \ 
    apt-get -y update && \
    apt-get install -y ffmpeg

# Authorize SSH Host
ARG ssh_prv_key
ARG ssh_pub_key
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# Set environment variables
ENV ENGORGIO_CODE_ROOT /workspace/research/engorgio
ENV ENGORGIO_DATA_ROOT /workspace/research/engorgio/dataset
ENV ENGORGIO_RESULT_ROOT /workspace/research/engorgio/result
ENV PYTHONPATH  $ENGORGIO_CODE_ROOT:$PYTHONPATH
ENV J2K_LIBRARY_PATH /workspace/research/engorgio/third_party/libj2k-kakadu
ENV LD_LIBRARY_PATH $J2K_LIBRARY_PATH:$LD_LIBRARY_PATH
ENV PATH $J2K_LIBRARY_PATH/bin:$PATH
