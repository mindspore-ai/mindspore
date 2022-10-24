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

#include "src/litert/delegate/tensorrt/op/fse_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/common/log_util.h"

namespace mindspore::lite {
namespace {
constexpr size_t kTableExtend = 3;
constexpr size_t kAlignOffset = 7;
}  // namespace
REGISTER_TENSORRT_PLUGIN(FSEPluginCreater);
template class TensorRTPluginCreater<FSEPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int FSEPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                       const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  cudaMalloc(&bs_device_, sizeof(BitStreamState));
  cudaMemcpy(bs_device_, &bs_, sizeof(BitStreamState), cudaMemcpyHostToDevice);
  int element_cnt = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
  auto input = static_cast<const uint64_t *>(inputs[0]);
  auto output = static_cast<float *>(outputs[0]);
  FSEDeCompressed(bs_device_, input, output, element_cnt, states_table_device_, bit_count_table_device_,
                  symbol_table_device_, static_cast<float *>(centroids_device_), table_log_, stream, device_id_);
  cudaFree(bs_device_);
  bs_device_ = nullptr;
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *FSEPlugin::clone() const noexcept {
  auto *plugin = new FSEPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs FSEPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                   nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = shape_.size();
  for (size_t i = 0; i < shape_.size(); i++) {
    out_dims.d[i] = exprBuilder.constant(shape_.at(i));
  }
  return out_dims;
}
}  // namespace mindspore::lite
