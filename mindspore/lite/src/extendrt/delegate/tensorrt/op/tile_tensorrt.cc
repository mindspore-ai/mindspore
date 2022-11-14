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

#include "src/extendrt/delegate/tensorrt/op/tile_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/tile_fusion.h"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(TilePluginCreater);
template class TensorRTPluginCreater<TilePlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int TileTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
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

int TileTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto repeats_tensor = in_tensors_[1];
  CHECK_NULL_RETURN(repeats_tensor.Data());
  if (repeats_tensor.ElementNum() != input(ctx, 0).trt_tensor_->getDimensions().nbDims) {
    MS_LOG(ERROR) << op_name_ << " has input dims: " << input(ctx, 0).trt_tensor_->getDimensions().nbDims
                  << ", and invalid repeats cnt: " << repeats_tensor.ElementNum();
    return RET_ERROR;
  }
  int ret = ParseData2Vector(in_tensors_[1], &repeats_);
  if (ret != RET_OK || repeats_.size() == 0) {
    MS_LOG(ERROR) << op_name_ << " has invalid repeats tensor";
    return ret;
  }
  ITensorHelper tile_input;

  ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &tile_input);
  if (ret != RET_OK || tile_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << op_name_ << " preprocess tensor failed.";
    return RET_ERROR;
  }

  return RunAsConcat(ctx, tile_input);
}
int TileTensorRT::RunAsConcat(TensorRTContext *ctx, const ITensorHelper &tile_input) {
  std::vector<int> repeats(repeats_.size());
  for (size_t i = 0; i != repeats_.size(); ++i) {
    repeats[i] = static_cast<int>(repeats_[i]) * tile_input.trt_tensor_->getDimensions().d[i];
  }
  auto shape = ctx->ConvertTo1DTensor(repeats);
  auto output = Broadcast(ctx, tile_input.trt_tensor_, shape);
  auto layer = ctx->network()->addIdentity(*output);
  layer_ = layer;
  auto tile_out = layer->getOutput(0);

  ctx->RegisterTensor(ITensorHelper{tile_out, tile_input.format_, true}, out_tensors_[0].Name());
  return RET_OK;
}
int TileTensorRT::RunAsPlugin(TensorRTContext *ctx, const ITensorHelper &tile_input) {
  // Floating point Exception
  nvinfer1::ITensor *inputTensors[] = {tile_input.trt_tensor_};
  auto plugin = std::make_shared<TilePlugin>(op_name_, repeats_, device_id_);
  nvinfer1::IPluginV2Layer *tile_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  CHECK_NULL_RETURN(tile_layer);
  nvinfer1::ITensor *tile_out = tile_layer->getOutput(0);
  tile_layer->setName(op_name_.c_str());
  ctx->RegisterTensor(ITensorHelper{tile_out, tile_input.format_, true}, out_tensors_[0].Name());
  this->layer_ = tile_layer;
  return RET_OK;
}
// plugin

int TilePlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                        const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) noexcept {
  nvinfer1::Dims output_dims = outputDesc[0].dims;
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  if (device_input_shape_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&device_input_shape_, input_dims.nbDims * sizeof(size_t)));
    CHECK_NULL_RETURN(device_input_shape_);
  }
  if (device_output_shape_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&device_output_shape_, output_dims.nbDims * sizeof(size_t)));
    CHECK_NULL_RETURN(device_output_shape_);
  }
  size_t input_shape[nvinfer1::Dims::MAX_DIMS];
  size_t output_shape[nvinfer1::Dims::MAX_DIMS];
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape[i] = static_cast<size_t>(input_dims.d[i]);
    output_shape[i] = static_cast<size_t>(output_dims.d[i]);
  }
  CUDA_CHECK(cudaMemcpy(device_input_shape_, input_shape, input_dims.nbDims * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(
    cudaMemcpy(device_output_shape_, output_shape, output_dims.nbDims * sizeof(size_t), cudaMemcpyHostToDevice));
  MS_LOG(ERROR) << layer_name_ << " has more axis to concat: " << repeats_.size();
  return RET_ERROR;
}

nvinfer1::DimsExprs TilePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                                    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = inputs[0].nbDims;
  for (int i = 0; i < out_dims.nbDims; i++) {
    auto repeat = exprBuilder.constant(repeats_[i]);
    out_dims.d[i] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[i], *repeat);
  }
  return out_dims;
}

nvinfer1::IPluginV2DynamicExt *TilePlugin::clone() const noexcept {
  auto *plugin = new TilePlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

void TilePlugin::terminate() noexcept {
  if (device_input_shape_ != nullptr) {
    auto cuda_ret = cudaFree(device_input_shape_);
    if (cuda_ret != cudaSuccess) {
      MS_LOG(ERROR) << "free cuda memory failed for " << layer_name_;
    }
  }
  if (device_output_shape_ != nullptr) {
    auto cuda_ret = cudaFree(device_output_shape_);
    if (cuda_ret != cudaSuccess) {
      MS_LOG(ERROR) << "free cuda memory failed for " << layer_name_;
    }
  }
}

size_t TilePlugin::getSerializationSize() const noexcept { return sizeof(float) * repeats_.size() + sizeof(size_t); }

void TilePlugin::serialize(void *buffer) const noexcept {
  size_t dims = repeats_.size();
  SerializeValue(&buffer, &dims, sizeof(size_t));
  for (float one_repeat : repeats_) {
    SerializeValue(&buffer, &one_repeat, sizeof(float));
  }
}
REGISTER_TENSORRT_CREATOR(ops::kNameTileFusion, TileTensorRT)
}  // namespace mindspore::lite
