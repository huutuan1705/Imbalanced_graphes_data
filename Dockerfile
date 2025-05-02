FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Thiết lập biến môi trường
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài đặt các gói cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo các thư mục cần thiết
WORKDIR /app
COPY requirements.txt .

# Cài đặt các thư viện Python cần thiết
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
# COPY datasets/ /app/datasets/
# COPY models/ /app/models/
# COPY utils/ /app/utils/
# COPY train.py eval.py /app/

# Tạo thư mục cho dữ liệu, kết quả và log
RUN mkdir -p /app/data /app/logs /app/results

# Thiết lập biến môi trường Python
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Tạo script khởi động
# RUN echo '#!/bin/bash\n\
# if [ "$1" = "train" ]; then\n\
#   echo "Starting training..."\n\
#   python3 /app/train.py $@\n\
# elif [ "$1" = "eval" ]; then\n\
#   echo "Starting evaluation..."\n\
#   python3 /app/eval.py $@\n\
# else\n\
#   echo "===== MAML Few-Shot Learning Docker Container ====="\n\
#   echo "Usage:"\n\
#   echo "  train [args] - Run training process"\n\
#   echo "  eval [args] - Run evaluation process"\n\
#   echo "Example:"\n\
#   echo "  train --data_path /app/data --n_way 5 --k_shot 1"\n\
#   echo "  eval --data_path /app/data --model_path /app/logs/5way_1shot_timestamp/best_model.pth"\n\
# fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Setup volume mount points
VOLUME ["/app/data", "/app/logs", "/app/results"]

# Thiết lập entrypoint
# ENTRYPOINT ["/app/entrypoint.sh"]

# Lệnh mặc định khi không có tham số (hiển thị hướng dẫn)
# CMD ["help"]