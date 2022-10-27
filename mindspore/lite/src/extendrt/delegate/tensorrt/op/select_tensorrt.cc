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
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/select_tensorrt.h"
#include "select_impl.cuh"

namespace mindspore::lite {
int SelectTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int SelectTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_, input(ctx, 2).trt_tensor_};
  // input tensors must have the same type and size
  auto input1_dims = inputTensors[1]->getDimensions().nbDims;
  auto input2_dims = inputTensors[2]->getDimensions().nbDims;
  if (input1_dims != input2_dims) {
    MS_LOG(ERROR) << "select tensors must have same size";
    return RET_ERROR;
  }
  bool same_shape = SameDims(inputTensors[0]->getDimensions(), ConvertMSShape(inputTensors[1]->getDimensions())) &&
                    SameDims(inputTensors[1]->getDimensions(), ConvertMSShape(inputTensors[2]->getDimensions()));
  if (!same_shape) {
    MS_LOG(ERROR) << "select tensors must have same size";
    return RET_ERROR;
  }
  for (int i = 0; i != in_tensors_.size(); ++i) {
    if (inputTensors[i]->getDimensions().nbDims == DIMENSION_4D && input(ctx, i).format_ == Format::NHWC) {
      nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(ctx, *input(ctx, i).trt_tensor_);
      CHECK_NULL_RETURN(transpose_layer_in);
      transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
      this->transpose_layer_ = transpose_layer_in;
      inputTensors[i] = transpose_layer_in->getOutput(0);
    }
  }

  if (input(ctx, 0).trt_tensor_->getType() != nvinfer1::DataType::kINT32) {
    inputTensors[0] = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }

  auto plugin = std::make_shared<SelectPlugin>(op_name_, op_primitive_->value_type());
  CHECK_NULL_RETURN(plugin);
  nvinfer1::IPluginV2Layer *select_layer = ctx->network()->addPluginV2(inputTensors, 3, *plugin);
  CHECK_NULL_RETURN(select_layer);
  this->layer_ = select_layer;
  nvinfer1::ITensor *op_out_tensor = select_layer->getOutput(0);
  CHECK_NULL_RETURN(op_out_tensor);
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(SelectPluginCreater);
template class TensorRTPluginCreater<SelectPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int SelectPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace,
                          cudaStream_t stream) noexcept {
  return RunCudaSelect(inputDesc, inputs, outputs, stream);
}

int SelectPlugin::RunCudaSelect(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                void *const *outputs, cudaStream_t stream) {
  if (inputDesc[1].type == nvinfer1::DataType::kINT32) {
    CalSelect(GetDimsVolume(inputDesc[1].dims), static_cast<const int *>(inputs[0]),
              static_cast<const int *>(inputs[1]), static_cast<const int *>(inputs[2]), static_cast<int *>(outputs[0]),
              device_id_, stream);
  } else if (inputDesc[1].type == nvinfer1::DataType::kFLOAT) {
    CalSelect(GetDimsVolume(inputDesc[1].dims), static_cast<const int *>(inputs[0]),
              static_cast<const float *>(inputs[1]), static_cast<const float *>(inputs[2]),
              static_cast<float *>(outputs[0]), device_id_, stream);
  } else {
    MS_LOG(ERROR) << "unsupported Select data type";
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *SelectPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) SelectPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "malloc select plugin failed";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DataType SelectPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  return inputTypes[1];
}

size_t SelectPlugin::getSerializationSize() const noexcept { return sizeof(schema::PrimitiveType); }

void SelectPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &primitive_type_, sizeof(schema::PrimitiveType));
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Select, SelectTensorRT)
}  // namespace mindspore::lite
