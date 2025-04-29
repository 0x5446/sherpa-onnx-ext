#!/bin/sh
WORKDIR=$1
if [ -z "$WORKDIR" ]; then
    echo "Usage: $0 <WORKDIR>"
    exit 1
fi
cd sherpa-onnx-ext && \
    export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON" && \
    export SHERPA_ONNXRUNTIME_LIB_DIR=/$WORKDIR/onnxruntime-linux-x64-gpu-1.18.0/lib && \
    export SHERPA_ONNXRUNTIME_INCLUDE_DIR=/$WORKDIR/onnxruntime-linux-x64-gpu-1.18.0/include && \
    python setup.py install && \
    python -c "import sherpa_onnx; print('安装的sherpa_onnx版本:', sherpa_onnx.__version__)" && \
    find /opt/asr -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true