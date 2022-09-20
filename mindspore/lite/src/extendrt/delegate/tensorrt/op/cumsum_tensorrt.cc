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

#include "src/extendrt/delegate/tensorrt/op/cumsum_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cumsum_impl.cuh"
#include "ops/cumsum.h"

namespace mindspore::lite {
int CumsumTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                              const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }

  if (out_tensors.size() < 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int CumsumTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper input_helper;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &input_helper);
  if (ret != RET_OK || input_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return ret;
  }
  auto axis_vec = ConvertTensorAsIntVector(in_tensors_[1]);
  if (axis_vec.size() != 1) {
    MS_LOG(ERROR) << "Failed to get axis input, axis size " << axis_vec.size() << ", node: " << op_name_;
    return RET_ERROR;
  }
  int axis = axis_vec[0];
  auto cumsum_op = AsOps<ops::CumSum>();
  bool exclusive = cumsum_op->get_exclusive();
  bool reverse = cumsum_op->get_reverse();
  auto plugin =
    std::make_shared<CumsumPlugin>(input_helper.trt_tensor_->getName(), axis, exclusive, reverse, device_id_);
  nvinfer1::ITensor *inputTensors[] = {input_helper.trt_tensor_};
  nvinfer1::IPluginV2Layer *cumsum_opt_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  if (cumsum_opt_layer == nullptr) {
    MS_LOG(ERROR) << "add cumsum op failed for TensorRT.";
    return RET_ERROR;
  }
  cumsum_opt_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = cumsum_opt_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, input_helper.format_, input_helper.same_format_},
                      out_tensors_[0].Name());
  this->layer_ = cumsum_opt_layer;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(CumsumPluginCreater);
template class TensorRTPluginCreater<CumsumPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int CumsumPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace,
                          cudaStream_t stream) noexcept {
  return RunCudaCumsum(inputDesc, inputs, outputs, stream);
}

int CumsumPlugin::RunCudaCumsum(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                void *const *outputs, cudaStream_t stream) {
  auto &dims = inputDesc[0].dims;
  size_t out_dim = 1;
  for (int i = 0; i < axis_; ++i) {
    out_dim *= dims.d[i];
  }
  size_t in_dim = 1;
  for (int i = axis_ + 1; i < dims.nbDims; ++i) {
    in_dim *= dims.d[i];
  }
  size_t axis_dim = dims.d[axis_];
  size_t stride = axis_dim * in_dim;
  size_t stride2 = in_dim;
  CumSum<int32_t>(static_cast<const int32_t *>(inputs[0]), static_cast<int32_t *>(outputs[0]), nullptr, out_dim,
                  axis_dim, in_dim, stride, stride2, exclusive_, reverse_, device_id_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *CumsumPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) CumsumPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "new plugin failed!";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t CumsumPlugin::getSerializationSize() const noexcept { return sizeof(int) + 2 * sizeof(bool); }

void CumsumPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &axis_, sizeof(int));
  SerializeValue(&buffer, &exclusive_, sizeof(bool));
  SerializeValue(&buffer, &reverse_, sizeof(bool));
}
REGISTER_TENSORRT_CREATOR(ops::kNameCumSum, CumsumTensorRT)
}  // namespace mindspore::lite
