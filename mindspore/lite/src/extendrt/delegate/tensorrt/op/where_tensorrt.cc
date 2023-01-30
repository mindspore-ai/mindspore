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

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/where_tensorrt.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/where_impl.cuh"
#include "ops/select.h"
#include "ops/where.h"

namespace mindspore::lite {
constexpr int INPUT_X_INDEX = 1;
constexpr int INPUT_Y_INDEX = 2;

int WhereTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::ITensor *WhereTensorRT::GetBroadcastTensor(TensorRTContext *ctx, nvinfer1::ITensor *input_tensor) {
  auto input_cond_dims = input(ctx, 0).trt_tensor_->getDimensions();
  nvinfer1::Dims in_tensor_dims = input_tensor->getDimensions();
  while (in_tensor_dims.nbDims < input_cond_dims.nbDims) {
    input_tensor = ExpandDim(ctx, input_tensor, 0);
    if (input_tensor->getDimensions().nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    }
    nvinfer1::IShuffleLayer *shuffle_layer = ctx->network()->addShuffle(*input_tensor);
    shuffle_layer->setReshapeDimensions(input_tensor->getDimensions());
    input_tensor = shuffle_layer->getOutput(0);
    in_tensor_dims = input_tensor->getDimensions();
  }
  return input_tensor;
}

int WhereTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  auto input_x = input(ctx, INPUT_X_INDEX).trt_tensor_;
  auto input_y = input(ctx, INPUT_Y_INDEX).trt_tensor_;
  if (in_tensors_[INPUT_X_INDEX].DataType() != in_tensors_[INPUT_Y_INDEX].DataType()) {
    auto target_type_index =
      INPUT_X_INDEX + (in_tensors_[INPUT_X_INDEX].DataType() < in_tensors_[INPUT_Y_INDEX].DataType());
    if (INPUT_X_INDEX != target_type_index) {
      input_x = TRTTensorCast(ctx, input_x, ConvertDataType(in_tensors_[INPUT_X_INDEX].DataType()), op_name_ + "_cast");
    }
    if (INPUT_Y_INDEX != target_type_index) {
      input_y = TRTTensorCast(ctx, input_y, ConvertDataType(in_tensors_[INPUT_Y_INDEX].DataType()), op_name_ + "_cast");
    }
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input_x, input_y};
  ITensorHelper cond_helper = input(ctx, 0);
  if (cond_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    inputTensors[0] = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kINT32, op_name_ + "_cast_in");
  }
  auto input_x_dims = input(ctx, INPUT_X_INDEX).trt_tensor_->getDimensions();
  auto input_y_dims = input(ctx, INPUT_Y_INDEX).trt_tensor_->getDimensions();
  // broadcast to same shape
  if (input_x_dims.nbDims != input_y_dims.nbDims) {
    if (input_x_dims.nbDims > input_y_dims.nbDims) {
      auto input_shape_tensor = ctx->network()->addShape(*input(ctx, INPUT_X_INDEX).trt_tensor_)->getOutput(0);
      auto inputy = GetBroadcastTensor(ctx, input_y);
      auto size_tensor = ctx->network()->addShape(*inputy)->getOutput(0);
      size_tensor = ctx->network()
                      ->addElementWise(*input_shape_tensor, *size_tensor, nvinfer1::ElementWiseOperation::kMAX)
                      ->getOutput(0);
      inputTensors[INPUT_Y_INDEX] = Broadcast(ctx, inputy, size_tensor);
    } else {
      auto input_shape_tensor = ctx->network()->addShape(*input(ctx, INPUT_Y_INDEX).trt_tensor_)->getOutput(0);
      auto inputx = GetBroadcastTensor(ctx, input_x);
      auto size_tensor = ctx->network()->addShape(*inputx)->getOutput(0);
      size_tensor = ctx->network()
                      ->addElementWise(*input_shape_tensor, *size_tensor, nvinfer1::ElementWiseOperation::kMAX)
                      ->getOutput(0);
      inputTensors[INPUT_X_INDEX] = Broadcast(ctx, inputx, size_tensor);
    }
  }

  auto plugin = std::make_shared<WherePlugin>(op_name_);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create WherePlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::IPluginV2Layer *where_layer = ctx->network()->addPluginV2(inputTensors, 3, *plugin);
  this->layer_ = where_layer;
  nvinfer1::ITensor *op_out_tensor = where_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "where out tensor is nullptr.";
    return RET_ERROR;
  }
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());

  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(WherePluginCreater);
template class TensorRTPluginCreater<WherePlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int WherePlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                         const void *const *inputs, void *const *outputs, void *workspace,
                         cudaStream_t stream) noexcept {
  return RunCudaWhere(inputDesc, inputs, outputs, stream);
}

int WherePlugin::RunCudaWhere(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                              void *const *outputs, cudaStream_t stream) {
  if (inputDesc[0].type == nvinfer1::DataType::kINT32 && inputDesc[INPUT_Y_INDEX].type == nvinfer1::DataType::kINT32) {
    Where(static_cast<const int *>(inputs[0]), static_cast<const int *>(inputs[INPUT_X_INDEX]),
          static_cast<const int *>(inputs[INPUT_Y_INDEX]), static_cast<int *>(outputs[0]),
          GetDimsVolume(inputDesc[0].dims), device_id_, stream);
  } else if (inputDesc[0].type == nvinfer1::DataType::kINT32 &&
             inputDesc[INPUT_Y_INDEX].type == nvinfer1::DataType::kFLOAT) {
    Where(static_cast<const int *>(inputs[0]), static_cast<const float *>(inputs[INPUT_X_INDEX]),
          static_cast<const float *>(inputs[INPUT_Y_INDEX]), static_cast<float *>(outputs[0]),
          GetDimsVolume(inputDesc[0].dims), device_id_, stream);
  } else if (inputDesc[0].type == nvinfer1::DataType::kINT32 &&
             inputDesc[INPUT_Y_INDEX].type == nvinfer1::DataType::kHALF) {
    Where(static_cast<const int *>(inputs[0]), static_cast<const half *>(inputs[INPUT_X_INDEX]),
          static_cast<const half *>(inputs[INPUT_Y_INDEX]), static_cast<half *>(outputs[0]),
          GetDimsVolume(inputDesc[0].dims), device_id_, stream);
  } else {
    MS_LOG(ERROR) << "invalid where type";
    return RET_ERROR;
  }
  return RET_OK;
}  // namespace mindspore::lite

nvinfer1::IPluginV2DynamicExt *WherePlugin::clone() const noexcept {
  auto *plugin = new WherePlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DataType WherePlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  return inputTypes[1];
}

bool WherePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                            int nbOutputs) noexcept {
  if (tensorsDesc[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {
    return tensorsDesc[pos].type == nvinfer1::DataType::kINT32;
  } else if (pos == 1) {
    return tensorsDesc[pos].type == nvinfer1::DataType::kFLOAT || tensorsDesc[pos].type == nvinfer1::DataType::kINT32;
  } else {
    return tensorsDesc[pos].type == tensorsDesc[pos - 1].type;
  }
  MS_LOG(ERROR) << tensorsDesc[pos].type << " " << tensorsDesc[pos].format;
  return false;
}

size_t WherePlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType); }

void WherePlugin::serialize(void *buffer) const noexcept {}

REGISTER_TENSORRT_CREATOR(ops::kNameWhere, WhereTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameSelect, WhereTensorRT)
}  // namespace mindspore::lite
