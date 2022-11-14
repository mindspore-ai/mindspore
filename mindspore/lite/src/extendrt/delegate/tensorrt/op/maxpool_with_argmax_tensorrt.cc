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

#include "src/extendrt/delegate/tensorrt/op/maxpool_with_argmax_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_with_argmax_impl.cuh"
#include "ops/max_pool_with_argmax.h"

namespace mindspore::lite {
int MaxPoolWithArgMaxTensorRT::IsSupport(const BaseOperatorPtr &base_operator,
                                         const std::vector<TensorInfo> &in_tensors,
                                         const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int MaxPoolWithArgMaxTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto op = AsOps<ops::MaxPoolWithArgmax>();
  CHECK_NULL_RETURN(op);
  auto pad_mode = op->get_pad_mode();
  auto stride = op->get_strides();
  auto kernel_size = op->get_kernel_size();

  auto plugin = std::make_shared<MaxPoolWithArgMaxPlugin>(op_name_, kernel_size, stride, pad_mode);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create ActivationOptPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_};
  nvinfer1::IPluginV2Layer *layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  this->layer_ = layer;
  nvinfer1::ITensor *op_out_tensor = layer->getOutput(0);
  CHECK_NULL_RETURN(op_out_tensor);
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  op_out_tensor = layer->getOutput(1);
  CHECK_NULL_RETURN(op_out_tensor);
  ctx->RegisterTensor(ITensorHelper{op_out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[1].Name());
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(MaxPoolWithArgMaxPluginCreater);
template class TensorRTPluginCreater<MaxPoolWithArgMaxPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int MaxPoolWithArgMaxPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                     const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                     void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaMaxPoolWithArgmax(inputDesc, inputs, outputs, stream);
}

int MaxPoolWithArgMaxPlugin::RunCudaMaxPoolWithArgmax(const nvinfer1::PluginTensorDesc *inputDesc,
                                                      const void *const *inputs, void *const *outputs,
                                                      cudaStream_t stream) {
  auto dims = inputDesc[0].dims;
  int n = dims.d[0];
  int c = dims.d[1];
  int h = dims.d[INPUT_SIZE2];
  int w = dims.d[INPUT_SIZE3];
  int th = h / strides_[1] + (h % strides_[1] == 0);
  int ph = std::max<int>(0, (th - 1) * strides_[1] + kernel_size_[1] - h) / INPUT_SIZE2;
  int tw = w / strides_[INPUT_SIZE2] + (w % strides_[INPUT_SIZE2] == 0);
  int pw = std::max<int>(0, (tw - 1) * strides_[INPUT_SIZE2] + kernel_size_[INPUT_SIZE2] - w) / INPUT_SIZE2;
  int out_h = 0;
  int out_w = 0;
  if (pad_mode_ == PadMode::VALID) {
    out_h = std::ceil((h - (kernel_size_[1] - 1)) / strides_[1]);
    out_w = std::ceil((w - (kernel_size_[INPUT_SIZE2] - 1)) / strides_[INPUT_SIZE2]);
  } else {
    out_h = std::ceil(h / strides_[1]);
    out_w = std::ceil(w / strides_[INPUT_SIZE2]);
  }
  CalMaxPoolWithArgmax<float, int>(static_cast<const float *>(inputs[0]), n, c, h, w, kernel_size_[1],
                                   kernel_size_[INPUT_SIZE2], strides_[1], strides_[INPUT_SIZE2], ph, pw, out_h, out_w,
                                   static_cast<float *>(outputs[0]), static_cast<int *>(outputs[1]), device_id_,
                                   stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *MaxPoolWithArgMaxPlugin::clone() const noexcept {
  auto *plugin = new MaxPoolWithArgMaxPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t MaxPoolWithArgMaxPlugin::getSerializationSize() const noexcept {
  return sizeof(float) * (INPUT_SIZE4 + INPUT_SIZE4) + sizeof(PadMode);
}

void MaxPoolWithArgMaxPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &kernel_size_[0], sizeof(float) * INPUT_SIZE4);
  SerializeValue(&buffer, &strides_[0], sizeof(float) * INPUT_SIZE4);
  SerializeValue(&buffer, &pad_mode_, sizeof(PadMode));
}

REGISTER_TENSORRT_CREATOR(ops::kNameMaxPoolWithArgmax, MaxPoolWithArgMaxTensorRT)
}  // namespace mindspore::lite
