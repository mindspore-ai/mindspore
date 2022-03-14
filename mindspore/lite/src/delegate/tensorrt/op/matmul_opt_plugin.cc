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
const char *MATMUL_OPT_PLUGIN_VERSION{"1"};
const char *MATMUL_OPT_PLUGIN_NAME{"MatmulOptPluginCreater"};
nvinfer1::PluginFieldCollection MatmulOptPluginCreater::field_collection_{};
std::vector<nvinfer1::PluginField> MatmulOptPluginCreater::fields_;
REGISTER_TENSORRT_PLUGIN(MatmulOptPluginCreater);

// MatmulOptPlugin
int MatmulOptPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                             const void *const *inputs, void *const *outputs, void *workspace,
                             cudaStream_t stream) noexcept {
  CHECK_NULL_RETURN(cublas_handle_);
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
  const nvinfer1::PluginTensorDesc desc_a = inputDesc[0];
  const nvinfer1::PluginTensorDesc desc_b = inputDesc[1];
  // a: m * k, b: k * n, c: m * n
  int m = a_trans_ ? desc_a.dims.d[1] : desc_a.dims.d[0];
  int k = a_trans_ ? desc_a.dims.d[0] : desc_a.dims.d[1];
  int n = b_trans_ ? desc_b.dims.d[0] : desc_b.dims.d[1];
  const int mm_params[]{m, n, k};
  const int trans_params[]{n, m};
  if (desc_a.type == nvinfer1::DataType::kFLOAT && desc_b.type == nvinfer1::DataType::kFLOAT) {
    CublasMM1Batch(inputs[0], inputs[1], c_addr_trans_, mm_params, operations_, data_types_, type_compute_,
                   cublas_handle_);
    Cublas2DTranspose(static_cast<const float *>(c_addr_trans_), static_cast<float *>(outputs[0]), trans_params,
                      cublas_handle_);
  } else {
    MS_LOG(ERROR) << layer_name_ << " input datatype needs check a: " << static_cast<int>(desc_a.type)
                  << ", b: " << static_cast<int>(desc_a.type);
    return RET_ERROR;
  }
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
  out_dims.d[0] = a_trans_ ? inputs[0].d[1] : inputs[0].d[0];
  out_dims.d[1] = b_trans_ ? inputs[1].d[0] : inputs[1].d[1];
  return out_dims;
}

bool MatmulOptPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                                int nbOutputs) noexcept {
  return true;
}

void MatmulOptPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                      const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  bias_index_ = (nbInputs == INPUT_SIZE3) ? kBiasIndex : -1;
  if (a_addr_trans_ == nullptr) {
    CUDA_CHECK_VOID(cudaMalloc(&a_addr_trans_, GetDimsVolume(in[0].max) * sizeof(float)));
  }
  if (b_addr_trans_ == nullptr) {
    CUDA_CHECK_VOID(cudaMalloc(&b_addr_trans_, GetDimsVolume(in[1].max) * sizeof(float)));
  }
  if (c_addr_trans_ == nullptr) {
    CUDA_CHECK_VOID(cudaMalloc(&c_addr_trans_, GetDimsVolume(out[0].max) * sizeof(float)));
  }
  operations_[0] = a_trans_ ? CUBLAS_OP_N : CUBLAS_OP_T;
  operations_[1] = b_trans_ ? CUBLAS_OP_N : CUBLAS_OP_T;
  data_types_[0] = ConvertDataType(in[0].desc.type);            // input a
  data_types_[1] = ConvertDataType(in[1].desc.type);            // input b
  data_types_[kBiasIndex] = ConvertDataType(out[0].desc.type);  // output c
  type_compute_ = (in[0].desc.type == nvinfer1::DataType::kHALF || in[1].desc.type == nvinfer1::DataType::kHALF)
                    ? CUBLAS_COMPUTE_32F_FAST_16BF
                    : CUBLAS_COMPUTE_32F;
}

size_t MatmulOptPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                         const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

nvinfer1::DataType MatmulOptPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                      int nbInputs) const noexcept {
  return inputTypes[0];
}

const char *MatmulOptPlugin::getPluginType() const noexcept { return MATMUL_OPT_PLUGIN_NAME; }

const char *MatmulOptPlugin::getPluginVersion() const noexcept { return MATMUL_OPT_PLUGIN_VERSION; }

int MatmulOptPlugin::getNbOutputs() const noexcept { return 1; }

int MatmulOptPlugin::initialize() noexcept {
  if (cublas_handle_ == nullptr) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  }
}

void MatmulOptPlugin::FreeCudaDeviceMemory(void **addr) {
  if (addr != nullptr) {
    CUDA_CHECK_VOID(cudaFree(*addr));
    addr = nullptr;
  }
}

void MatmulOptPlugin::terminate() noexcept {
  FreeCudaDeviceMemory(&a_addr_trans_);
  FreeCudaDeviceMemory(&b_addr_trans_);
  FreeCudaDeviceMemory(&c_addr_trans_);
  if (cublas_handle_ != nullptr) {
    auto cublas_ret = cublasDestroy(cublas_handle_);
    if (cublas_ret != CUBLAS_STATUS_SUCCESS) {
      MS_LOG(ERROR) << "cublasDestroy failed: " << cublas_ret;
    } else {
      cublas_handle_ = nullptr;
    }
  }
}

size_t MatmulOptPlugin::getSerializationSize() const noexcept { return 2 * sizeof(bool); }

void MatmulOptPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &a_trans_, sizeof(bool));
  SerializeValue(&buffer, &b_trans_, sizeof(bool));
}

void MatmulOptPlugin::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void MatmulOptPlugin::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *MatmulOptPlugin::getPluginNamespace() const noexcept { return name_space_.c_str(); }

// MatmulOptPluginCreater
MatmulOptPluginCreater::MatmulOptPluginCreater() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  field_collection_.nbFields = fields_.size();
  field_collection_.fields = fields_.data();
}

const char *MatmulOptPluginCreater::getPluginName() const noexcept { return MATMUL_OPT_PLUGIN_NAME; }

const char *MatmulOptPluginCreater::getPluginVersion() const noexcept { return MATMUL_OPT_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection *MatmulOptPluginCreater::getFieldNames() noexcept { return &field_collection_; }

nvinfer1::IPluginV2 *MatmulOptPluginCreater::createPlugin(const char *name,
                                                          const nvinfer1::PluginFieldCollection *fc) noexcept {
  const nvinfer1::PluginField *fields = fc->fields;
  bool a_trans = static_cast<const bool *>(fields[0].data)[0];
  bool b_trans = static_cast<const bool *>(fields[1].data)[0];
  return new (std::nothrow) MatmulOptPlugin(name, a_trans, b_trans);
}

nvinfer1::IPluginV2 *MatmulOptPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                               size_t serialLength) noexcept {
  bool a_trans;
  DeserializeValue(&serialData, &serialLength, &a_trans, sizeof(bool));
  bool b_trans;
  DeserializeValue(&serialData, &serialLength, &b_trans, sizeof(bool));
  MS_LOG(DEBUG) << name << " is deserialize as a_trans: " << a_trans << ", b_trans: " << b_trans;
  return new (std::nothrow) MatmulOptPlugin(name, a_trans, b_trans);
}

void MatmulOptPluginCreater::setPluginNamespace(const char *libNamespace) noexcept { name_space_ = libNamespace; }

const char *MatmulOptPluginCreater::getPluginNamespace() const noexcept { return name_space_.c_str(); }
}  // namespace mindspore::lite
