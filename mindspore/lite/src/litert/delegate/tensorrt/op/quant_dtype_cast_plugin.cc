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

#include "src/litert/delegate/tensorrt/op/quant_dtype_cast_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/quant_impl.cuh"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(QuantDTypeCastPluginCreater);
template class TensorRTPluginCreater<QuantDTypeCastPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int QuantDTypeCastPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                  const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                  void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  int element_cnt = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
  auto input = static_cast<const int8_t *>(inputs[0]);
  auto output = static_cast<float *>(outputs[0]);
  if (param_num_ == 1) {
    DeQuantWithPerLayer(input, output, element_cnt, scale_[0], zp_[0], stream, device_id_);
  } else {
    auto bucket_count = shape_[preferred_dim_];
    size_t stride = 1;
    for (size_t i = preferred_dim_ + 1; i < shape_.size(); i++) {
      stride *= shape_[i];
    }
    DeQuantWithPerChannel(input, output, element_cnt, scale_, zp_, stride, bucket_count, stream, device_id_);
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *QuantDTypeCastPlugin::clone() const noexcept {
  auto *plugin = new QuantDTypeCastPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t QuantDTypeCastPlugin::getSerializationSize() const noexcept {
  return sizeof(size_t) + param_num_ * sizeof(float) + param_num_ * sizeof(int) + shape_.size() * sizeof(int);
}

void QuantDTypeCastPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &preferred_dim_, sizeof(size_t));
  SerializeValue(&buffer, scale_, param_num_ * sizeof(float));
  SerializeValue(&buffer, zp_, param_num_ * sizeof(int));
  SerializeValue(&buffer, shape_.data(), shape_.size() * sizeof(int64_t));
}

nvinfer1::DimsExprs QuantDTypeCastPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                              int nbInputs,
                                                              nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = shape_.size();
  for (int i = 0; i < shape_.size(); i++) {
    out_dims.d[i] = exprBuilder.constant(shape_.at(i));
  }
  return out_dims;
}
}  // namespace mindspore::lite
