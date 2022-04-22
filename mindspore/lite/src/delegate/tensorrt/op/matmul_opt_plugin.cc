/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/delegate/tensorrt/op/matmul_opt_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/cuda_impl/cuda_helper.h"
#include "NvInferRuntimeCommon.h"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(MatmulOptPluginCreater);
template class TensorRTPluginCreater<MatmulOptPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

// MatmulOptPlugin
int MatmulOptPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                             const void *const *inputs, void *const *outputs, void *workspace,
                             cudaStream_t stream) noexcept {
  CHECK_NULL_RETURN(cublas_handle_);
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
  const nvinfer1::PluginTensorDesc desc_a = inputDesc[0];
  const nvinfer1::PluginTensorDesc desc_b = inputDesc[1];
  const nvinfer1::PluginTensorDesc desc_c = outputDesc[0];

  if (desc_a.dims.nbDims == DIMENSION_2D) {
    // a: m * k, b: k * n, c: m * n
    int m = desc_c.dims.d[0];
    int n = desc_c.dims.d[1];
    int k = b_trans_ ? desc_b.dims.d[1] : desc_b.dims.d[0];
    const int mm_params[]{m, n, k};
    CublasMM1Batch(inputs[0], inputs[1], outputs[0], mm_params, operations_, data_types_, cublas_handle_);
  } else if (desc_a.dims.nbDims == DIMENSION_3D) {
    return RunBatchedMatmul(inputDesc, outputDesc, inputs, outputs, workspace, stream);
  } else {
    MS_LOG(ERROR) << layer_name_ << " input dims needs check a: " << desc_a.dims.nbDims;
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulOptPlugin::RunBatchedMatmul(const nvinfer1::PluginTensorDesc *inputDesc,
                                      const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                      void *const *outputs, void *workspace, cudaStream_t stream) {
  const nvinfer1::PluginTensorDesc desc_b = inputDesc[1];
  const nvinfer1::PluginTensorDesc desc_c = outputDesc[0];
  int batch = desc_c.dims.d[0];
  int m = desc_c.dims.d[1];
  int n = desc_c.dims.d[DIMENSION_2D];
  int k = b_trans_ ? desc_b.dims.d[DIMENSION_2D] : desc_b.dims.d[1];
  const int mm_params[]{m, n, k, batch};
  for (int i = 0; i < batch; i++) {
    a_addrs_[i] = inputs[0] + i * m * k * sizeof(float);
    b_addrs_[i] = inputs[1] + i * k * n * sizeof(float);
    c_addrs_[i] = outputs[0] + i * m * n * sizeof(float);
  }
  int data_size = batch * sizeof(void *);
  int max_batchsize = a_addrs_.size();
  if (a_device_addrs_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&a_device_addrs_, sizeof(void *) * max_batchsize));
  }
  if (b_device_addrs_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&b_device_addrs_, sizeof(void *) * max_batchsize));
  }
  if (c_device_addrs_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&c_device_addrs_, sizeof(void *) * max_batchsize));
  }
  CUDA_CHECK(cudaMemcpy(a_device_addrs_, a_addrs_.data(), data_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_device_addrs_, b_addrs_.data(), data_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c_device_addrs_, c_addrs_.data(), data_size, cudaMemcpyHostToDevice));

  CublasMMBatched(a_device_addrs_, b_device_addrs_, c_device_addrs_, mm_params, operations_, data_types_,
                  cublas_handle_);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *MatmulOptPlugin::clone() const noexcept {
  auto *plugin = new MatmulOptPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs MatmulOptPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                         int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  if (nbInputs != INPUT_SIZE2 && nbInputs != INPUT_SIZE3) {
    MS_LOG(ERROR) << "invalid input size " << nbInputs << " of " << layer_name_;
    return out_dims;
  }
  out_dims.nbDims = inputs[0].nbDims;
  if (out_dims.nbDims == DIMENSION_2D) {
    out_dims.d[0] = a_trans_ ? inputs[0].d[1] : inputs[0].d[0];
    out_dims.d[1] = b_trans_ ? inputs[1].d[0] : inputs[1].d[1];
    return out_dims;
  } else if (out_dims.nbDims == DIMENSION_3D) {
    out_dims.d[0] = inputs[0].d[0];
    out_dims.d[1] = a_trans_ ? inputs[0].d[DIMENSION_2D] : inputs[0].d[1];
    out_dims.d[DIMENSION_2D] = b_trans_ ? inputs[1].d[1] : inputs[1].d[DIMENSION_2D];
    return out_dims;
  }
  MS_LOG(ERROR) << "invalid input dims " << out_dims.nbDims << " of " << layer_name_;
  return out_dims;
}

void MatmulOptPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                      const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  operations_[0] = a_trans_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  operations_[1] = b_trans_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  data_types_[0] = ConvertDataType(in[0].desc.type);             // input a
  data_types_[1] = ConvertDataType(in[1].desc.type);             // input b
  data_types_[THIRD_INPUT] = ConvertDataType(out[0].desc.type);  // output c
  data_types_[FOURTH_INPUT] =
    (in[0].desc.type == nvinfer1::DataType::kHALF || in[1].desc.type == nvinfer1::DataType::kHALF)
      ? CUDA_R_16F
      : CUDA_R_32F;  // compute type
  if (in[0].max.nbDims == DIMENSION_3D) {
    int max_batchsize = in[0].max.d[0];
    a_addrs_.resize(max_batchsize);
    b_addrs_.resize(max_batchsize);
    c_addrs_.resize(max_batchsize);
    if (a_device_addrs_ == nullptr) {
      CUDA_CHECK_VOID(cudaMalloc(&a_device_addrs_, sizeof(void *) * max_batchsize));
    }
    if (b_device_addrs_ == nullptr) {
      CUDA_CHECK_VOID(cudaMalloc(&b_device_addrs_, sizeof(void *) * max_batchsize));
    }
    if (c_device_addrs_ == nullptr) {
      CUDA_CHECK_VOID(cudaMalloc(&c_device_addrs_, sizeof(void *) * max_batchsize));
    }
  }
}

int MatmulOptPlugin::initialize() noexcept {
  if (cublas_handle_ == nullptr) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  }
  for (int i = 0; i < DIMENSION_4D; i++) {
    if (data_types_[i] != CUDA_R_32F) {
      MS_LOG(ERROR) << layer_name_ << " only support fp32";
      return RET_ERROR;
    }
  }
}

void MatmulOptPlugin::terminate() noexcept {
  if (cublas_handle_ != nullptr) {
    auto cublas_ret = cublasDestroy(cublas_handle_);
    if (cublas_ret != CUBLAS_STATUS_SUCCESS) {
      MS_LOG(ERROR) << "cublasDestroy failed: " << cublas_ret;
    } else {
      cublas_handle_ = nullptr;
    }
  }
  cudaError_t err;
  if (a_device_addrs_ != nullptr) {
    err = cudaFree(a_device_addrs_);
    if (err != cudaSuccess) {
      MS_LOG(ERROR) << layer_name_ << " free cuda device mem failed " << err;
    }
    a_device_addrs_ = nullptr;
  }
  if (b_device_addrs_ != nullptr) {
    err = cudaFree(b_device_addrs_);
    if (err != cudaSuccess) {
      MS_LOG(ERROR) << layer_name_ << " free cuda device mem failed " << err;
    }
    b_device_addrs_ = nullptr;
  }
  if (c_device_addrs_ != nullptr) {
    err = cudaFree(c_device_addrs_);
    if (err != cudaSuccess) {
      MS_LOG(ERROR) << layer_name_ << " free cuda device mem failed " << err;
    }
    c_device_addrs_ = nullptr;
  }
}

size_t MatmulOptPlugin::getSerializationSize() const noexcept { return 2 * sizeof(bool); }

void MatmulOptPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &a_trans_, sizeof(bool));
  SerializeValue(&buffer, &b_trans_, sizeof(bool));
}
}  // namespace mindspore::lite
