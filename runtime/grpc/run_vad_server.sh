#!/bin/bash

./build/bin/funasr-vad-server \
  --port-id 10101 \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --vad-quant true \
  2>&1