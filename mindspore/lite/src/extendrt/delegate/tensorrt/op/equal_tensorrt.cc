/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/equal_tensorrt.h"
#include <numeric>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "ops/equal.h"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(EqualPluginCreater);
template class TensorRTPluginCreater<EqualPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int EqualTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int EqualTensorRT::AddInnerOp(TensorRTContext *ctx) {
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_};
  auto plugin = std::make_shared<EqualPlugin>(op_name_, device_id_);
  nvinfer1::IPluginV2Layer *equal_layer = ctx->network()->addPluginV2(inputTensors, INPUT_SIZE2, *plugin);
  if (equal_layer == nullptr) {
    MS_LOG(ERROR) << "create equal layer failed for: " << op_name_;
    return RET_ERROR;
  }
  layer_ = equal_layer;
  nvinfer1::ITensor *equal_out = equal_layer->getOutput(0);
  equal_layer->setName(op_name_.c_str());
  ctx->RegisterTensor(ITensorHelper{equal_out, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

int EqualPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                         const void *const *inputs, void *const *outputs, void *workspace,
                         cudaStream_t stream) noexcept {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int element_cnt = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());

  if (inputDesc->type == nvinfer1::DataType::kINT32) {
    const int *input1 = static_cast<const int *>(inputs[0]);
    const int *input2 = static_cast<const int *>(inputs[1]);
    int *output = static_cast<int *>(outputs[0]);
    Equal(input1, input2, output, element_cnt, stream);
  } else if (inputDesc->type == nvinfer1::DataType::kFLOAT) {
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);
    float *output = static_cast<float *>(outputs[0]);
    Equal(input1, input2, output, element_cnt, stream);
  } else {
    MS_LOG(ERROR) << "unsupported equal data type";
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *EqualPlugin::clone() const noexcept {
  auto *plugin = new EqualPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}
#if TRT_VERSION_LS(7, 2)
REGISTER_TENSORRT_CREATOR(ops::kNameEqual, EqualTensorRT)
#endif
}  // namespace mindspore::lite
