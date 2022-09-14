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

#include "src/litert/delegate/tensorrt/op/normalize_opt_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/normalize_impl.cuh"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(NormalizeOptPluginCreater);
template class TensorRTPluginCreater<NormalizeOptPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int NormalizeOptPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  auto input = static_cast<const float *>(inputs[0]);
  auto gamma = static_cast<const float *>(inputs[1]);
  auto beta = static_cast<const float *>(inputs[2]);
  auto output = static_cast<float *>(outputs[0]);
  auto input_dims = inputDesc[0].dims;
  size_t dim_at_axis = input_dims.d[axis_];
  int element_cnt = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());
  Normalize(input, gamma, beta, output, dim_at_axis, epsilion_, element_cnt, stream, device_id_);
}

nvinfer1::IPluginV2DynamicExt *NormalizeOptPlugin::clone() const noexcept {
  auto *plugin = new NormalizeOptPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t NormalizeOptPlugin::getSerializationSize() const noexcept { return sizeof(size_t) + sizeof(float); }

void NormalizeOptPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &axis_, sizeof(size_t));
  SerializeValue(&buffer, &epsilion_, sizeof(float));
}
}  // namespace mindspore::lite
