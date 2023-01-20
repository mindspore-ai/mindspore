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

#include "src/extendrt/delegate/tensorrt/op/mha_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "ops/attention.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/allocator.h"

namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
constexpr std::size_t kThree = 3;
std::ostream &operator<<(std::ostream &s, const nvinfer1::ITensor &t) {
  const auto &dims = t.getDimensions();
  s << "ndims=" << dims.nbDims << " [";
  for (int i = 0; i < dims.nbDims; ++i) {
    s << dims.d[i] << " ";
  }
  s << "]";
  return s;
}
}  // namespace

// Multi Head Attention TensorRT op
int MhaTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() < C7NUM || in_tensors.size() > C9NUM) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int MhaTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto mha_op = AsOps<ops::Attention>();
  if (mha_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  int head_number = mha_op->get_head_num();
  int head_size = mha_op->get_head_size();
  auto compute_type = runtime_->GetRuntimePrecisionMode();
  bool is_cross = mha_op->get_cross();
  bool is_position_bias = mha_op->get_position_bias();
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  fastertransformer::encoderParamT params;
  memset_s(&params, sizeof(params), 0, sizeof(params));
  params.head_num = head_number;
  params.head_size = head_size;
  params.hidden_size = head_number * head_size;
  params.cublas_handle = GetCublasHandle();
  params.qkv_bias = !is_position_bias;
  params.projection_bias = !is_position_bias;
  params.is_cross = is_cross;
  params.position_bias = is_position_bias;
  auto plugin =
    std::make_shared<MhaPlugin>(input_tensor->getName(), compute_type, params, GetCublasLtHandle(), device_id_);
  const int input_number = inputs().size();
  nvinfer1::ITensor *inputTensors[input_number];
  for (int i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }
  nvinfer1::IPluginV2Layer *mha_layer = ctx->network()->addPluginV2(inputTensors, input_number, *plugin);
  if (mha_layer == nullptr) {
    MS_LOG(ERROR) << "add mha op failed for TensorRT.";
    return RET_ERROR;
  }
  mha_layer->setName((op_name_ + "plugin_attention").c_str());
  nvinfer1::ITensor *attn_tensor = mha_layer->getOutput(0);
#ifndef TEST_
  ctx->RegisterTensor(ITensorHelper{attn_tensor, Format::NCHW, true}, out_tensors_[0].Name());
#else  /* TEST_ */
  ctx->RegisterTensor(ITensorHelper{attn_tensor, Format::NCHW, true}, out_tensors_[0].Name() + "attn");
#endif /* TEST_ */
  this->layer_ = mha_layer;
#ifdef TEST_
  auto weight_projection = input(ctx, 4).trt_tensor_;
  auto bias_projection = input(ctx, 6).trt_tensor_;
#endif /* TEST_ */

#ifdef TEST_
  auto matmul_layer = ctx->network()->addMatrixMultiply(*attn_tensor, nvinfer1::MatrixOperation::kNONE,
                                                        *weight_projection, nvinfer1::MatrixOperation::kNONE);
  if (matmul_layer == nullptr) {
    MS_LOG(ERROR) << "failed to add matmul layer";
    return RET_ERROR;
  }
  matmul_layer->setName((op_name_ + "_matmul").c_str());
  auto matmul_tensor = matmul_layer->getOutput(0);
  auto shuffle_layer = ctx->network()->addShuffle(*bias_projection);
  const auto size = bias_projection->getDimensions().d[0];
  shuffle_layer->setReshapeDimensions(nvinfer1::Dims{2, {1, size}});
  auto shuffle_tensor = shuffle_layer->getOutput(0);
  auto addbias = ctx->network()->addElementWise(*matmul_tensor, *shuffle_tensor, nvinfer1::ElementWiseOperation::kSUM);
  if (addbias == nullptr) {
    MS_LOG(ERROR) << "failed to add bias layer";
    return RET_ERROR;
  }
  addbias->setName((op_name_ + "_bias").c_str());
  auto bias_out = addbias->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{bias_out, Format::NCHW, true}, out_tensors_[0].Name());
#endif /* TEST_ */
  return RET_OK;
}

//  PLUGIN of Multi Head Attention Layer
REGISTER_TENSORRT_PLUGIN(MhaPluginCreater);
template class TensorRTPluginCreater<MhaPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int MhaPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                       const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    return RunCudaMha<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    return RunCudaMha<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

template <typename T>
int MhaPlugin::RunCudaMha(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream,
                          cublasGemmAlgo_t algoId) {
  int cross_tensor_offset = (params_.is_cross) ? 1 : 0;
  const int weight_projection_tensor_idx = 4 + cross_tensor_offset;
  const int bias_projection_tensor_idx = 6 + cross_tensor_offset;
  const int attn_mask_tensor_idx = 7 + cross_tensor_offset;
  const int bias_qkv_tensor_idx = 5 + cross_tensor_offset;
  const int weight_qkv_tensor_idx = 3;
  const int position_bias_tensor_idx = 6 + cross_tensor_offset;
  params_.stream = stream;
  params_.algo = algoId;
  void *inputs_attn[num_of_inputs_];
  int index = 0;
  inputs_attn[index++] = const_cast<void *>(inputs[0]);
  if (params_.is_cross) {
    inputs_attn[index++] = const_cast<void *>(inputs[1]);
    inputs_attn[index++] = const_cast<void *>(inputs[weight_qkv_tensor_idx]);
    inputs_attn[index++] = const_cast<void *>(inputs[weight_qkv_tensor_idx + 1]);
  } else {
    inputs_attn[index++] = const_cast<void *>(inputs[weight_qkv_tensor_idx]);
  }
  if (params_.qkv_bias) {
    inputs_attn[index++] = const_cast<void *>(inputs[bias_qkv_tensor_idx]);
  }
  if (params_.position_bias) {
    inputs_attn[index++] = const_cast<void *>(inputs[position_bias_tensor_idx]);
    inputs_attn[index++] = const_cast<void *>(inputs[attn_mask_tensor_idx - C2NUM]);
  } else {
    inputs_attn[index++] = const_cast<void *>(inputs[attn_mask_tensor_idx]);
  }
  inputs_attn[index++] = const_cast<void *>(inputs[weight_projection_tensor_idx]);
  if (params_.projection_bias) {
    inputs_attn[index++] = const_cast<void *>(inputs[bias_projection_tensor_idx]);
  }
  void *outputs_attn[] = {outputs[0]};
  fastertransformer::forward_attn<T>(reinterpret_cast<T **>(inputs_attn), num_of_inputs_,
                                     reinterpret_cast<T **>(outputs_attn), num_of_outputs_, &params_, workspace);
  return RET_OK;
}

bool MhaPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                          int nbOutputs) noexcept {
  auto type = (compute_type_ == RuntimePrecisionMode_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  for (int i = 0; i < pos; i++) {
    if (tensorsDesc[pos].type != tensorsDesc[i].type) return false;
  }
  bool res = (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && (tensorsDesc[pos].type == type);
  return res;
}

void MhaPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  int cross_tensor_offset = 0;
  int position_bias_tensor_offsets = 0;
  if (params_.is_cross) cross_tensor_offset = 1;
  if (params_.position_bias) position_bias_tensor_offsets = 1;
  const int attn_mask_tensor_idx = 7 + cross_tensor_offset - position_bias_tensor_offsets;
  const int request_batch_size = static_cast<const int>(in[attn_mask_tensor_idx].desc.dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(in[attn_mask_tensor_idx].desc.dims.d[1]);
  const int request_tgt_seq_len = static_cast<const int>(in[attn_mask_tensor_idx].desc.dims.d[2]);
  params_.batch_size = request_batch_size;
  params_.src_seq_len = request_src_seq_len;
  params_.tgt_seq_len = request_tgt_seq_len;
  num_of_inputs_ = nbInputs;
  num_of_outputs_ = nbOutputs;
}

size_t MhaPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    return fastertransformer::GetAttnWorkspaceSize<half>(&params_);
  } else {
    return fastertransformer::GetAttnWorkspaceSize<float>(&params_);
  }
}

nvinfer1::DimsExprs MhaPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                                   nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  if (index == 0) {
#ifndef TEST_
    int num_dims = inputs[0].nbDims;
    dims.nbDims = num_dims;
    if (num_dims == INPUT_SIZE2) {
      dims.d[0] = exprBuilder.constant(inputs[nbInputDims - 1].d[0]->getConstantValue() *
                                       inputs[nbInputDims - 1].d[1]->getConstantValue());
      auto hidden_size = exprBuilder.constant(params_.head_size * params_.head_num);
      dims.d[1] = hidden_size;
    } else if (num_dims == INPUT_SIZE3) {
      dims.d[0] = inputs[nbInputDims - 1].d[0];  // batch
      dims.d[1] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
      auto hidden_size = exprBuilder.constant(params_.head_size * params_.head_num);
      dims.d[kTwo] = hidden_size;
    }
  } else {
    dims.nbDims = INPUT_SIZE4;
    dims.d[0] = inputs[nbInputDims - 1].d[0];  // batch
    dims.d[1] = exprBuilder.constant(params_.head_num);
    dims.d[kTwo] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
    dims.d[kThree] = exprBuilder.constant(params_.head_size);
  }
#else
    dims.nbDims = C2NUM;
    dims.d[0] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
    auto hidden_size = exprBuilder.constant(head_size_ * head_number_);
    dims.d[1] = hidden_size;
  }
#endif
  return dims;
}

nvinfer1::IPluginV2DynamicExt *MhaPlugin::clone() const noexcept {
  auto *plugin = new MhaPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

int MhaPlugin::initialize() noexcept { return 0; }

void MhaPlugin::terminate() noexcept {}

size_t MhaPlugin::getSerializationSize() const noexcept {
  return sizeof(int) + sizeof(fastertransformer::encoderParamT);
}

void MhaPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &compute_type_, sizeof(int));
  SerializeValue(&buffer, &params_, sizeof(fastertransformer::encoderParamT));
}
REGISTER_TENSORRT_CREATOR(ops::kNameAttention, MhaTensorRT)
}  // namespace mindspore::lite
