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
#include "src/extendrt/delegate/tensorrt/cuda_impl/where.cuh"

namespace mindspore::lite {
constexpr int INPUT_X_INDEX = 1;
constexpr int INPUT_Y_INDEX = 2;

int WhereTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
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

int WhereTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }

  if (input(ctx, INPUT_X_INDEX).trt_tensor_->getType() != input(ctx, INPUT_Y_INDEX).trt_tensor_->getType()) {
    MS_LOG(ERROR) << "input tensors must in same type";
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, INPUT_X_INDEX).trt_tensor_,
                                       input(ctx, INPUT_Y_INDEX).trt_tensor_};
  ITensorHelper cond_helper = input(ctx, 0);
  if (cond_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    inputTensors[0] = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kINT32, op_name_ + "_cast_in");
  }
  auto input_x_dims = input(ctx, INPUT_X_INDEX).trt_tensor_->getDimensions();
  auto input_y_dims = input(ctx, INPUT_Y_INDEX).trt_tensor_->getDimensions();
  // broadcast to same shape
  if (input_x_dims.nbDims != input_y_dims.nbDims) {
    if (input_x_dims.nbDims > input_y_dims.nbDims) {
      auto expect_shape = ConvertMSShape(input(ctx, INPUT_X_INDEX).trt_tensor_->getDimensions());
      inputTensors[INPUT_Y_INDEX] =
        ConvertConstantTensorWithDims(ctx, in_tensors_[INPUT_Y_INDEX], expect_shape, op_name_ + "_broadcast_inputy");
    } else {
      auto expect_shape = ConvertMSShape(input(ctx, INPUT_Y_INDEX).trt_tensor_->getDimensions());
      inputTensors[INPUT_X_INDEX] =
        ConvertConstantTensorWithDims(ctx, in_tensors_[INPUT_X_INDEX], expect_shape, op_name_ + "_broadcast_inputx");
    }
  }

  auto plugin = std::make_shared<WherePlugin>(op_name_, op_primitive_->value_type());
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
          GetDimsVolume(inputDesc[0].dims), stream);
  } else if (inputDesc[0].type == nvinfer1::DataType::kINT32 &&
             inputDesc[INPUT_Y_INDEX].type == nvinfer1::DataType::kFLOAT) {
    Where(static_cast<const int *>(inputs[0]), static_cast<const float *>(inputs[INPUT_X_INDEX]),
          static_cast<const float *>(inputs[INPUT_Y_INDEX]), static_cast<float *>(outputs[0]),
          GetDimsVolume(inputDesc[0].dims), stream);
  } else {
    MS_LOG(ERROR) << "invalid where type: " << static_cast<int>(primitive_type_);
    return RET_ERROR;
  }
  return RET_OK;
}  // namespace mindspore::lite

nvinfer1::IPluginV2DynamicExt *WherePlugin::clone() const noexcept {
  auto *plugin = new WherePlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t WherePlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType); }

void WherePlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &primitive_type_, sizeof(schema::PrimitiveType));
}

REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Where, WhereTensorRT)
}  // namespace mindspore::lite
