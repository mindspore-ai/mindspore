/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/use_past_embedding_tensorrt.h"
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
#include "ops/use_past_embedding.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/use_past_embedding.cuh"
namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
}  // namespace
int UsePastEmbeddingTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                        const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != C3NUM) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int UsePastEmbeddingTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto plugin = std::make_shared<UsePastEmbeddingPlugin>(input_tensor->getName(), device_id_);
  const int input_number = inputs().size();
  nvinfer1::ITensor *inputTensors[input_number];
  for (int i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }
  nvinfer1::IPluginV2Layer *layer = ctx->network()->addPluginV2(inputTensors, input_number, *plugin);
  if (layer == nullptr) {
    MS_LOG(ERROR) << "add op failed for TensorRT.";
    return RET_ERROR;
  }
  layer->setName((op_name_ + "plugin_use_past_embedding").c_str());
  nvinfer1::ITensor *tensor = layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{tensor, Format::NCHW, true}, out_tensors_[0].Name());
  std::cout << out_tensors_[0].Name() << " Registered" << std::endl;
  this->layer_ = layer;
  return RET_OK;
}
REGISTER_TENSORRT_PLUGIN(UsePastEmbeddingPluginCreater);
template class TensorRTPluginCreater<UsePastEmbeddingPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;
int UsePastEmbeddingPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                    void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  const int *input_position = static_cast<const int *>(inputs[0]);
  void *init_reset_cast = const_cast<void *>(inputs[C1NUM]);
  int *init_reset = static_cast<int *>(init_reset_cast);
  void *batch_valid_length_cast = const_cast<void *>(inputs[C2NUM]);
  int *batch_valid_length = static_cast<int *>(batch_valid_length_cast);
  InvokeUsePastEmbedding(input_position, static_cast<int *>(outputs[0]), init_reset, batch_valid_length, size_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *UsePastEmbeddingPlugin::clone() const noexcept {
  auto *plugin = new UsePastEmbeddingPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t UsePastEmbeddingPlugin::getSerializationSize() const noexcept {
  size_t size = sizeof(size_);
  return size;
}
void UsePastEmbeddingPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  size_ = static_cast<int>(out[0].desc.dims.d[1]);
}

void UsePastEmbeddingPlugin::serialize(void *buffer) const noexcept { SerializeValue(&buffer, &size_, sizeof(size_)); }

bool UsePastEmbeddingPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc,
                                                       int nbInputs, int nbOutputs) noexcept {
  bool res = (tensorsDesc[pos].type == nvinfer1::DataType::kINT32) ? true : false;
  return res;
}

nvinfer1::DimsExprs UsePastEmbeddingPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                                int nbInputDims,
                                                                nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  if (index == 0) {
    int num_dims = inputs[0].nbDims;
    dims.nbDims = num_dims;
    for (int i = 0; i < num_dims; i++) {
      dims.d[i] = exprBuilder.constant(inputs[0].d[i]->getConstantValue());
    }
  }
  return dims;
}

REGISTER_TENSORRT_CREATOR(ops::kNameUsePastEmbedding, UsePastEmbeddingTensorRT)
}  // namespace mindspore::lite
