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

#include "src/extendrt/delegate/tensorrt/op/logical_not_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/logical.cuh"
#include "ops/logical_not.h"

namespace mindspore::lite {
int LogicalNotTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                  const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  return RET_OK;
}

int LogicalNotTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr || this->in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  ITensorHelper input_helper = input(ctx, 0);
  if (input_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    auto cast_layer = ctx->network()->addIdentity(*input_helper.trt_tensor_);
    if (cast_layer == nullptr) {
      MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
      return RET_ERROR;
    }
    cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
    input_helper.trt_tensor_ = cast_layer->getOutput(0);
  }
  auto plugin = std::make_shared<LogicalNotPlugin>(op_name_, schema::PrimitiveType_LogicalNot);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create ActivationOptPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input_helper.trt_tensor_};
  nvinfer1::IPluginV2Layer *logical_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  this->layer_ = logical_layer;
  nvinfer1::ITensor *op_out_tensor = logical_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "addElementWise out tensor is nullptr.";
    return RET_ERROR;
  }
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input_helper.format_, input_helper.same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(LogicalNotPluginCreater);
template class TensorRTPluginCreater<LogicalNotPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int LogicalNotPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) noexcept {
  return RunCudaLogical(inputDesc, inputs, outputs, stream);
}

int LogicalNotPlugin::RunCudaLogical(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                     void *const *outputs, cudaStream_t stream) {
  switch (primitive_type_) {
    case (schema::PrimitiveType_LogicalNot): {
      LogicalNot(static_cast<const int *>(inputs[0]), static_cast<int *>(outputs[0]), GetDimsVolume(inputDesc[0].dims),
                 stream);
      break;
    }
    default: {
      MS_LOG(ERROR) << "invalid logical type: " << static_cast<int>(primitive_type_);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *LogicalNotPlugin::clone() const noexcept {
  auto *plugin = new LogicalNotPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

bool LogicalNotPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                                 int nbOutputs) noexcept {
  return tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         (tensorsDesc[pos].type == nvinfer1::DataType::kFLOAT || tensorsDesc[pos].type == nvinfer1::DataType::kINT32);
}

size_t LogicalNotPlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType); }

void LogicalNotPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &primitive_type_, sizeof(schema::PrimitiveType));
}
REGISTER_TENSORRT_CREATOR(ops::kNameLogicalNot, LogicalNotTensorRT)
}  // namespace mindspore::lite
