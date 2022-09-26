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

#include "src/extendrt/delegate/tensorrt/op/spacetobatch_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/spacetobatch_impl.cuh"
#include "ops/space_to_batch_nd.h"

namespace mindspore::lite {
int SpaceToBatchTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                    const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }

  if (out_tensors.size() < 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int SpaceToBatchTensorRT::AddInnerOp(TensorRTContext *ctx) {
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto block_size_vec = ConvertTensorAsIntVector(in_tensors_[1]);
  constexpr size_t block_input_elem_count = 2;
  if (block_size_vec.size() != block_input_elem_count) {
    MS_LOG(ERROR) << "Failed to get block input, block size " << block_size_vec.size() << ", node: " << op_name_;
    return RET_ERROR;
  }
  int bh = block_size_vec[0];
  int bw = block_size_vec[1];
  if (bh != bw) {
    MS_LOG(ERROR) << "block_h not equal block_w " << op_name_;
    return RET_ERROR;
  }
  auto pad_vec = ConvertTensorAsIntVector(in_tensors_[INPUT_SIZE2]);
  constexpr size_t pad_input_elem_count = 4;
  if (pad_vec.size() != pad_input_elem_count) {
    MS_LOG(ERROR) << "Failed to get pad input, pad size " << pad_vec.size() << ", node: " << op_name_;
    return RET_ERROR;
  }
  int ph0 = pad_vec[0];
  int ph1 = pad_vec[1];
  int pw0 = pad_vec[INPUT_SIZE2];
  int pw1 = pad_vec[INPUT_SIZE3];

  auto plugin = std::make_shared<SpaceToBatchPlugin>(input_tensor->getName(), bh, ph0, ph1, pw0, pw1, device_id_);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "add spacetobatch plugin failed for" << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {input_tensor};
  nvinfer1::IPluginV2Layer *space2batch_opt_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  if (space2batch_opt_layer == nullptr) {
    MS_LOG(ERROR) << "add spacetobatch op failed for TensorRT.";
    return RET_ERROR;
  }
  space2batch_opt_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = space2batch_opt_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = space2batch_opt_layer;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(SpaceToBatchPluginCreater);
template class TensorRTPluginCreater<SpaceToBatchPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int SpaceToBatchPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaSpaceToBatch(inputDesc, inputs, outputs, stream);
}

int SpaceToBatchPlugin::RunCudaSpaceToBatch(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                            void *const *outputs, cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int in = input_dims.d[0];
  int ic = input_dims.d[1];
  int ih = input_dims.d[2];
  int iw = input_dims.d[3];
  int on = in * bh_ * bh_;
  int oc = ic;
  int oh = (ih + ph0_ + ph1_) / bh_;
  int ow = (iw + pw0_ + pw1_) / bh_;

  int size = in * ic * ih * iw;

  CalSpaceToBatch<float>(size, static_cast<const float *>(inputs[0]), in, ih, iw, ic, on, oh, ow, oc, ph0_, ph1_, pw0_,
                         pw1_, bh_, static_cast<float *>(outputs[0]), device_id_, stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *SpaceToBatchPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) SpaceToBatchPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "new plugin failed!";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t SpaceToBatchPlugin::getSerializationSize() const noexcept { return sizeof(int) * 5; }

nvinfer1::DimsExprs SpaceToBatchPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                            int nbInputDims,
                                                            nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  dims.nbDims = inputs[0].nbDims;
  auto bh_mul_bh = exprBuilder.constant(bh_ * bh_);
  dims.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[0], *bh_mul_bh);
  dims.d[1] = inputs[0].d[1];
  auto bh = exprBuilder.constant(bh_);
  auto ph_sum = exprBuilder.constant(ph0_ + ph1_);
  auto sum0 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *inputs[0].d[INPUT_SIZE2], *ph_sum);
  dims.d[INPUT_SIZE2] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *sum0, *bh);
  auto pw_sum = exprBuilder.constant(pw0_ + pw1_);
  auto sum1 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *inputs[0].d[INPUT_SIZE3], *pw_sum);
  dims.d[INPUT_SIZE3] = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *sum1, *bh);
  return dims;
}

void SpaceToBatchPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &bh_, sizeof(int));
  SerializeValue(&buffer, &ph0_, sizeof(int));
  SerializeValue(&buffer, &ph1_, sizeof(int));
  SerializeValue(&buffer, &pw0_, sizeof(int));
  SerializeValue(&buffer, &pw1_, sizeof(int));
}
REGISTER_TENSORRT_CREATOR(ops::kNameSpaceToBatchND, SpaceToBatchTensorRT)
}  // namespace mindspore::lite
