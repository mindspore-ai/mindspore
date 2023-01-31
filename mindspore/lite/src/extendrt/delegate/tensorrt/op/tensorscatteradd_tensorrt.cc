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

#include "src/extendrt/delegate/tensorrt/op/tensorscatteradd_tensorrt.h"
#include <numeric>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/tensor_scatter_add.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cuh"

namespace mindspore::lite {
int TensorScatterAddTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                        const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorScatterAddTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (in_tensors_[0].IsConst()) {
    ITensorHelper scatter_input;
    scatter_input.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[0], op_name_);
    scatter_input.format_ = Format::NCHW;
    ctx->RegisterTensor(scatter_input, in_tensors_[0].Name());
  }

  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_, input(ctx, 1).trt_tensor_,
                                       input(ctx, INPUT_SIZE2).trt_tensor_};
  auto plugin = std::make_shared<TensorScatterAddPlugin>(input(ctx, 0).trt_tensor_->getName(), device_id_);
  nvinfer1::IPluginV2Layer *scatter_layer = ctx->network()->addPluginV2(inputTensors, 3, *plugin);
  if (scatter_layer == nullptr) {
    MS_LOG(ERROR) << "addScatter failed for TensorRT.";
    return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = scatter_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = scatter_layer;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(TensorScatterAddPluginCreater);
template class TensorRTPluginCreater<TensorScatterAddPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int TensorScatterAddPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                    void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaTensorScatterAdd(inputDesc, inputs, outputs, stream);
}

int TensorScatterAddPlugin::RunCudaTensorScatterAdd(const nvinfer1::PluginTensorDesc *inputDesc,
                                                    const void *const *inputs, void *const *outputs,
                                                    cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  size_t input_num = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>());
  nvinfer1::Dims update_dims = inputDesc[INPUT_SIZE2].dims;
  size_t update_num = std::accumulate(update_dims.d, update_dims.d + update_dims.nbDims, 1, std::multiplies<int>());

  size_t indice_dim_0 = inputDesc[1].dims.d[0];
  size_t indice_dim_1 = inputDesc[1].dims.d[1];
  int block_size = 1;
  for (int i = indice_dim_1; i != input_dims.nbDims; ++i) {
    block_size *= input_dims.d[i];
  }
  std::vector<int> indice_stride(indice_dim_1, 0);
  indice_stride[indice_stride.size() - 1] = block_size;
  for (int i = indice_dim_1 - 1; i > 0; --i) {
    indice_stride[i - 1] = indice_stride[i] * input_dims.d[i];
  }

  int *indice_stride_dptr{nullptr};
  cudaMalloc(&indice_stride_dptr, indice_stride.size() * sizeof(int));
  cudaMemcpy(indice_stride_dptr, indice_stride.data(), indice_stride.size() * sizeof(int), cudaMemcpyHostToDevice);

  int *input_shape_dptr{nullptr};
  cudaMalloc(&input_shape_dptr, input_dims.nbDims * sizeof(int));
  cudaMemcpy(input_shape_dptr, input_dims.d, input_dims.nbDims * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(outputs[0], inputs[0], input_num * sizeof(float), cudaMemcpyDeviceToDevice);
  TensorScatterArithmetic(TensorScatterArithmeticFunctionType::TENSOR_SCATTER_FUNC_ADD,
                          static_cast<const float *>(inputs[0]), static_cast<const int *>(inputs[1]),
                          static_cast<const float *>(inputs[INPUT_SIZE2]), static_cast<float *>(outputs[0]), block_size,
                          update_num, input_num, indice_dim_0, indice_dim_1, indice_stride_dptr, input_shape_dptr,
                          device_id_, stream);
  cudaFree(indice_stride_dptr);
  cudaFree(input_shape_dptr);

  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *TensorScatterAddPlugin::clone() const noexcept {
  auto *plugin = new TensorScatterAddPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

bool TensorScatterAddPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc,
                                                       int nbInputs, int nbOutputs) noexcept {
  if (tensorsDesc[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  return true;
}

size_t TensorScatterAddPlugin::getSerializationSize() const noexcept { return 0; }

void TensorScatterAddPlugin::serialize(void *buffer) const noexcept {}

REGISTER_TENSORRT_CREATOR(ops::kNameTensorScatterAdd, TensorScatterAddTensorRT)
}  // namespace mindspore::lite
