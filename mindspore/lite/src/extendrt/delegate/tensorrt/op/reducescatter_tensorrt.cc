/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/reducescatter_tensorrt.h"
#include <numeric>
#include <thread>
#include "NvInferRuntimeCommon.h"
#include "ops/reduce_scatter.h"

namespace mindspore::lite {
REGISTER_TENSORRT_PLUGIN(ReduceScatterPluginCreater);
template class TensorRTPluginCreater<ReduceScatterPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int ReduceScatterTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                     const std::vector<TensorInfo> &out_tensors) {
#ifndef LITE_CUDA_DISTRIBUTION
  MS_LOG(ERROR)
    << "Unsupported package for gpu distribution feature, please recompile with MS_ENABLE_CUDA_DISTRIBUTION set to on.";
  return RET_ERROR;
#else
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
#endif
}

int ReduceScatterTensorRT::AddInnerOp(TensorRTContext *ctx) {
  nvinfer1::ITensor *inputTensors[] = {input(ctx, 0).trt_tensor_};
  auto reduce_op = AsOps<ops::ReduceScatter>();
  if (reduce_op == nullptr) {
    MS_LOG(ERROR) << "convert failed for " << op_name_;
    return RET_ERROR;
  }
  auto reduce_mode = reduce_op->get_mode();
  auto rank = GetGPUGroupSize();
  auto plugin = std::make_shared<ReduceScatterPlugin>(op_name_, reduce_mode, rank, device_id_);
  MS_LOG(INFO) << op_name_ << " group size: " << rank << ", rank id: " << GetRankID();
  nvinfer1::IPluginV2Layer *reduce_scatter_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
  if (reduce_scatter_layer == nullptr) {
    MS_LOG(ERROR) << "create ReduceScatter layer failed for: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *reduce_scatter_out = reduce_scatter_layer->getOutput(0);
  reduce_scatter_layer->setName(op_name_.c_str());
  ctx->RegisterTensor(ITensorHelper{reduce_scatter_out, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  this->layer_ = reduce_scatter_layer;
  return RET_OK;
}

// ReduceScatterPlugin
int ReduceScatterPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                 const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                 void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  MS_LOG(INFO) << "ReduceScatter run at rank id: " << GetRankID() << " stream: " << stream;
  nvinfer1::Dims output_dims = outputDesc[0].dims;
  int recieve_element_cnt =
    std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>());
  const void *input = inputs[0];
  void *output = outputs[0];
  auto data_type = inputDesc->type;
  auto ret = DistributionCollective::instance().ReduceScatterWrapper(input, output, recieve_element_cnt, data_type,
                                                                     red_mode_, stream, NCCL_WORLD_GROUP);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReduceScatter nccl run failed for " << layer_name_;
    return ret;
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *ReduceScatterPlugin::clone() const noexcept {
  auto *plugin = new ReduceScatterPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs ReduceScatterPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                             int nbInputs,
                                                             nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = inputs->nbDims;
  auto rank_dim = exprBuilder.constant(rank_);
  out_dims.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *inputs->d[0], *rank_dim);
  for (int i = 1; i < inputs->nbDims; i++) {
    out_dims.d[i] = inputs->d[i];
  }
  return out_dims;
}

size_t ReduceScatterPlugin::getSerializationSize() const noexcept { return sizeof(ReduceMode); }

void ReduceScatterPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &red_mode_, sizeof(ReduceMode));
}

REGISTER_TENSORRT_CREATOR(ops::kNameReduceScatter, ReduceScatterTensorRT)
}  // namespace mindspore::lite
