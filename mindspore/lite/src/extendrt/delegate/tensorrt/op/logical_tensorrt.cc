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

#include "src/extendrt/delegate/tensorrt/op/logical_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/logical.cuh"
#include "ops/logical_and.h"
#include "ops/logical_or.h"

namespace mindspore::lite {
int LogicalTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int LogicalTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_};
  for (size_t i = 0; i != in_tensors_.size(); ++i) {
    ITensorHelper input_helper = input(ctx, i);
    if (input_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
      auto cast_layer = ctx->network()->addIdentity(*input_helper.trt_tensor_);
      if (cast_layer == nullptr) {
        MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
        return RET_ERROR;
      }
      cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
      inputTensors[i] = cast_layer->getOutput(0);
    }
  }
  schema::PrimitiveType schema_type = schema::PrimitiveType_NONE;
  if (type_ == ops::kNameLogicalAnd) {
    schema_type = schema::PrimitiveType_LogicalAnd;
  } else if (type_ == ops::kNameLogicalOr) {
    schema_type = schema::PrimitiveType_LogicalOr;
  }

  auto plugin = std::make_shared<LogicalPlugin>(op_name_, schema_type);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create ActivationOptPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::IPluginV2Layer *logical_layer = ctx->network()->addPluginV2(inputTensors, 2, *plugin);
  this->layer_ = logical_layer;
  nvinfer1::ITensor *op_out_tensor = logical_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "addElementWise out tensor is nullptr.";
    return RET_ERROR;
  }
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(LogicalPluginCreater);
template class TensorRTPluginCreater<LogicalPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int LogicalPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                           const void *const *inputs, void *const *outputs, void *workspace,
                           cudaStream_t stream) noexcept {
  return RunCudaLogical(inputDesc, inputs, outputs, stream);
}

int LogicalPlugin::RunCudaLogical(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                  void *const *outputs, cudaStream_t stream) {
  switch (primitive_type_) {
    case (schema::PrimitiveType_LogicalAnd): {
      LogicalAnd(static_cast<const int *>(inputs[0]), static_cast<const int *>(inputs[1]),
                 static_cast<int *>(outputs[0]), GetDimsVolume(inputDesc[0].dims), stream);
      break;
    }
    case (schema::PrimitiveType_LogicalOr): {
      LogicalOr(static_cast<const int *>(inputs[0]), static_cast<const int *>(inputs[1]),
                static_cast<int *>(outputs[0]), GetDimsVolume(inputDesc[0].dims), stream);
      break;
    }
    default: {
      MS_LOG(ERROR) << "invalid logical type: " << static_cast<int>(primitive_type_);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *LogicalPlugin::clone() const noexcept {
  auto *plugin = new LogicalPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

bool LogicalPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                              int nbOutputs) noexcept {
  if (tensorsDesc[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {
    return tensorsDesc[pos].type == nvinfer1::DataType::kINT32;
  }
  if (pos < nbInputs) {
    return tensorsDesc[pos].type == tensorsDesc[pos - 1].type;
  }
  if (pos < nbInputs + nbOutputs) {
    return tensorsDesc[pos].type == nvinfer1::DataType::kINT32;
  }
  return false;
}

size_t LogicalPlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType); }

void LogicalPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &primitive_type_, sizeof(schema::PrimitiveType));
}

REGISTER_TENSORRT_CREATOR(ops::kNameLogicalOr, LogicalTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameLogicalAnd, LogicalTensorRT)
}  // namespace mindspore::lite
