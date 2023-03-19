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

#include "src/extendrt/delegate/tensorrt/op/vsl_compress_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"

namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
}  // namespace

REGISTER_TENSORRT_PLUGIN(VslCompressPluginCreater);
template class TensorRTPluginCreater<VslCompressPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int VslCompressPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                               const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                               void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  fastertransformer::invokeBuildSequenceLength(static_cast<const int *>(inputs[0]), batch_size_,
                                               static_cast<int *>(outputs[0]), seq_len_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *VslCompressPlugin::clone() const noexcept {
  auto *plugin = new VslCompressPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs VslCompressPlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs,
                                                           int nbInputDims,
                                                           nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = 1;
  dims.d[0] = exprBuilder.constant(static_cast<const int>(inputs[0].d[0]->getConstantValue()));
  return dims;
}

void VslCompressPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  batch_size_ = static_cast<const int>(in[0].desc.dims.d[0]);
  seq_len_ = static_cast<const int>(in[0].desc.dims.d[1]);
}

size_t VslCompressPlugin::getSerializationSize() const noexcept { return 2 * sizeof(int); }

void VslCompressPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &batch_size_, sizeof(int));
  SerializeValue(&buffer, &seq_len_, sizeof(int));
}
}  // namespace mindspore::lite
