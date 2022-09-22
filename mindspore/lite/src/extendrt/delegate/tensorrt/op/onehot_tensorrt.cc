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
#include "src/extendrt/delegate/tensorrt/op/onehot_tensorrt.h"
#include "one_hot_impl.cuh"

namespace mindspore::lite {
constexpr int INDICES_INDEX = 0;
constexpr int DEPTH_INDEX = 1;
constexpr int ON_VALUE_INDEX = 2;
constexpr int OFF_VALUE_INDEX = 3;

int OnehotTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE4 && in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int OnehotTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, DEPTH_INDEX).trt_tensor_,
                                       input(ctx, ON_VALUE_INDEX).trt_tensor_, input(ctx, OFF_VALUE_INDEX).trt_tensor_};
  ITensorHelper indice_helper = input(ctx, 0);
  if (indice_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    inputTensors[0] = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }
  ITensorHelper depth_helper = input(ctx, DEPTH_INDEX);
  if (depth_helper.trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    inputTensors[DEPTH_INDEX] =
      TRTTensorCast(ctx, input(ctx, DEPTH_INDEX).trt_tensor_, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }
  mindspore::MSTensor &depth_tensor = in_tensors_[DEPTH_INDEX];
  if (depth_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "get depth input tensor null for " << op_name_;
    return RET_ERROR;
  }
  const int *depth_ptr = reinterpret_cast<const int *>(depth_tensor.Data().get());
  int depth = *depth_ptr;
  auto plugin = std::make_shared<OnehotPlugin>(op_name_, depth, op_primitive_->value_type());
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create OnehotPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::IPluginV2Layer *onehot_layer = ctx->network()->addPluginV2(inputTensors, 4, *plugin);
  this->layer_ = onehot_layer;
  nvinfer1::ITensor *op_out_tensor = onehot_layer->getOutput(0);
  if (op_out_tensor == nullptr) {
    MS_LOG(ERROR) << "onehot out tensor is nullptr.";
    return RET_ERROR;
  }
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());

  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(OnehotPluginCreater);
template class TensorRTPluginCreater<OnehotPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int OnehotPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace,
                          cudaStream_t stream) noexcept {
  return RunCudaOneHot(inputDesc, inputs, outputs, stream);
}

int OnehotPlugin::RunCudaOneHot(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                void *const *outputs, cudaStream_t stream) {
  for (int i = 0; i != inputDesc[0].dims.nbDims; ++i) {
    feature_dims_ *= inputDesc[0].dims.d[i];
  }
  if (inputDesc[0].type == nvinfer1::DataType::kINT32 && inputDesc[ON_VALUE_INDEX].type == nvinfer1::DataType::kFLOAT) {
    OneHot<float, int>(static_cast<const int *>(inputs[0]), depth_, static_cast<const float *>(inputs[ON_VALUE_INDEX]),
                       static_cast<const float *>(inputs[OFF_VALUE_INDEX]), batch_dims_, feature_dims_,
                       static_cast<float *>(outputs[0]), device_id_, stream);
  } else {
    MS_LOG(ERROR) << "invalid onehot type: " << static_cast<int>(primitive_type_);
    return RET_ERROR;
  }

  return RET_OK;
}
nvinfer1::DimsExprs OnehotPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                                      nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = inputs[0].nbDims + 1;
  auto indice_dims = inputs[0].nbDims;
  if (indice_dims == 1) {
    dims.d[0] = inputs[0].d[0];
    auto depth = exprBuilder.constant(depth_);
    dims.d[1] = depth;
  } else {
    for (int i = 0; i != inputs[0].nbDims; ++i) {
      dims.d[i] = inputs[0].d[i];
    }
    auto depth = exprBuilder.constant(depth_);
    dims.d[inputs[0].nbDims] = depth;
  }
  return dims;
}

nvinfer1::IPluginV2DynamicExt *OnehotPlugin::clone() const noexcept {
  auto *plugin = new OnehotPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t OnehotPlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType) + 3 * sizeof(int); }

void OnehotPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &batch_dims_, sizeof(int));
  SerializeValue(&buffer, &feature_dims_, sizeof(int));
  SerializeValue(&buffer, &depth_, sizeof(int));
  SerializeValue(&buffer, &primitive_type_, sizeof(schema::PrimitiveType));
}

REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_OneHot, OnehotTensorRT)
}  // namespace mindspore::lite
