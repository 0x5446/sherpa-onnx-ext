#!/bin/sh
WORKDIR=$1
if [ -z "$WORKDIR" ]; then
    echo "Usage: $0 <WORKDIR>"
    exit 1
fi

if [ ! -f onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz ] && [ ! -d onnxruntime-linux-x64-gpu-1.18.0 ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz
fi
if [ ! -d onnxruntime-linux-x64-gpu-1.18.0 ] && [ -f onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz ]; then
    tar xvf onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz
    rm onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz
fi
ln -sf /$WORKDIR/onnxruntime-linux-x64-gpu-1.18.0/lib/libonnxruntime*.so* /usr/local/lib/ && \
ldconfig
    
cd $WORKDIR/sherpa-onnx-ext && \
    export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON" && \
    export SHERPA_ONNXRUNTIME_LIB_DIR=/$WORKDIR/onnxruntime-linux-x64-gpu-1.18.0/lib && \
    export SHERPA_ONNXRUNTIME_INCLUDE_DIR=/$WORKDIR/onnxruntime-linux-x64-gpu-1.18.0/include && \
    python setup.py install && \
    python -c "import sherpa_onnx; print('安装的sherpa_onnx版本:', sherpa_onnx.__version__)"
