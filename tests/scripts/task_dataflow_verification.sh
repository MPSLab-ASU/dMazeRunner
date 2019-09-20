#!/bin/bash

export PYTHONPATH=python:tvm/python:tvm/topi/python:tvm/nnvm/python

rm -rf tvm/python/tvm/*.pyc tvm/python/tvm/*/*.pyc tvm/python/tvm/*/*/*.pyc

TVM_FFI=ctypes NOSE_NOCAPTURE=1 python -m nose -v tests/verification/convolution || exit -1
TVM_FFI=ctypes NOSE_NOCAPTURE=1 python -m nose -v tests/verification/gemm || exit -1
