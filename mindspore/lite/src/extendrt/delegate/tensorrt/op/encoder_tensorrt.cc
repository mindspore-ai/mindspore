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

#include "src/extendrt/delegate/tensorrt/op/encoder_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/nn_ops.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "ops/encoder_layer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_base.h"
#include "src/extendrt/delegate/tensorrt/distribution/distribution_collective.h"
#include "mindspore/core/ops/op_name.h"
#include "src/fastertransformer/layers/ms_layers/encoder.h"
#include "src/fastertransformer/layers/ms_layers/BaseLayer.h"
#include "src/fastertransformer/layers/ms_layers/attention.h"
#include "src/fastertransformer/layers/ms_layers/ffn.h"
namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
}  // namespace

int EncoderTensorRT::unique_id_ = 0;
int EncoderTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  auto layer_norm = GetValue<bool>(base_operator->GetAttr(ops::kLayerNorm));
  auto position_bias = GetValue<bool>(base_operator->GetAttr(ops::kPositionBias1));
  auto use_past = GetValue<bool>(base_operator->GetAttr(ops::kUsePast));
  auto query_layer = GetValue<bool>(base_operator->GetAttr(ops::kQueryLayer));
  auto moe = GetValue<bool>(base_operator->GetAttr(ops::kMoe));
  auto embedding_layer = GetValue<bool>(base_operator->GetAttr(ops::kEmbeddingLayer));

  size_t in_num = C8NUM;  // if mask=false in_num should be 6, mask default = true
  if (use_past) in_num += C4NUM;
  if (embedding_layer && use_past) {
    in_num += C2NUM;
  }
  if (query_layer) in_num += C5NUM;
  if (moe) in_num += C1NUM;
  if (layer_norm) {
    if (position_bias)
      in_num += C1NUM;
    else
      in_num += C2NUM;
  }
  if (position_bias)
    in_num += C1NUM;
  else
    in_num += C6NUM;
  if (in_tensors.size() != in_num) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " and needs to be " << in_num;
    return RET_ERROR;
  }
  if (out_tensors.size() != C1NUM && out_tensors.size() != C3NUM) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::ITensor *EncoderTensorRT::CastTensor(TensorRTContext *ctx, const TensorInfo &ms_tensor,
                                               const std::string &op_name) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is null for ConvertConstantTensor";
    return nullptr;
  }
  nvinfer1::Dims dims = ConvertCudaDims(ms_tensor.Shape());
  if (dims.nbDims == -1) {
    MS_LOG(INFO) << ms_tensor.Name() << " ConvertCudaDims failed, convert as scalar.";
    dims.nbDims = 1;
    dims.d[0] = 1;
  }
  nvinfer1::DataType data_type = ConvertDataType(ms_tensor.DataType());
  if (!ms_tensor.IsConst()) {
    MS_LOG(ERROR) << "ConvertConstantTensor from a MSTensor with nullptr data: " << ms_tensor.Name();
    return nullptr;
  }
  nvinfer1::Weights weights{data_type, ms_tensor.Data(), ms_tensor.ElementNum()};
  if (data_type == nvinfer1::DataType::kFLOAT && runtime_->GetTransformerFfnFp16()) {
    void *data_float16 = malloc(ms_tensor.ElementNum() * sizeof(float));
    if (data_float16 == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return nullptr;
    }
    auto src = static_cast<const float *>(ms_tensor.Data());
    auto dst = static_cast<half *>(data_float16);
    for (int i = 0; i < ms_tensor.ElementNum(); i++) {
      dst[i] = static_cast<half>(src[i]);
    }
    weights.values = data_float16;
  }
  nvinfer1::IConstantLayer *constant_tensor = ctx->network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  ctx->RegisterLayer(constant_tensor, ms_tensor.Name() + "_" + op_name);
  auto tensor_ptr = constant_tensor->getOutput(0);
  return tensor_ptr;
}

int EncoderTensorRT::AddVsl(int encoder_input_idx, int input_number, TensorRTContext *ctx,
                            nvinfer1::ITensor **inputTensors, const char *name) {
  if (runtime_->GetVslEncoderPluginId() == -1) {
    auto vsl_plugin = std::make_shared<VslCompressPlugin>(name, device_id_);
    CHECK_NULL_RETURN(vsl_plugin);
    nvinfer1::ITensor *inputVsl = ctx->network()->getInput(encoder_input_idx);
    auto vsl_compress_layer = ctx->network()->addPluginV2(&inputVsl, C1NUM, *vsl_plugin);
    if (vsl_compress_layer == nullptr) {
      MS_LOG(ERROR) << " create vsl compress layer failed for: ";
      return RET_ERROR;
    }
    auto plugin_id = static_cast<int>(ctx->network()->getNbLayers() - 1);
    runtime_->SetVslEncoderPluginId(plugin_id);
    vsl_compress_layer->setName("plugin_vsl_compress");
    nvinfer1::ITensor *vsl_output_tensor = vsl_compress_layer->getOutput(0);
    ctx->RegisterTensor(ITensorHelper{vsl_output_tensor, Format::NCHW, true}, "vsl_compress_output");
    inputTensors[input_number] = vsl_output_tensor;
  } else {
    auto vsl_compress_layer = ctx->network()->getLayer(runtime_->GetVslEncoderPluginId());
    inputTensors[input_number] = vsl_compress_layer->getOutput(0);
  }
  return RET_OK;
}

int EncoderTensorRT::AddVslByBatchValidLength(int input_number, TensorRTContext *ctx, nvinfer1::ITensor **inputTensors,
                                              const char *name) {
  if (runtime_->GetVslEncoderPluginId() == -1) {
    auto vsl_plugin = std::make_shared<VslCompressPlugin>(name, device_id_, runtime_->IsTransformerOptimizeSigma());
    CHECK_NULL_RETURN(vsl_plugin);
    nvinfer1::ITensor *inputPosition = ctx->network()->getInput(C2NUM);
    nvinfer1::ITensor *inputBatchValid = ctx->network()->getInput(C3NUM);
    nvinfer1::ITensor *inputs[C2NUM] = {inputPosition, inputBatchValid};
    auto vsl_compress_layer = ctx->network()->addPluginV2(inputs, C2NUM, *vsl_plugin);

    if (vsl_compress_layer == nullptr) {
      MS_LOG(ERROR) << " create vsl compress layer failed for: ";
      return RET_ERROR;
    }
    auto plugin_id = static_cast<int>(ctx->network()->getNbLayers() - 1);
    runtime_->SetVslEncoderPluginId(plugin_id);
    vsl_compress_layer->setName("plugin_vsl_compress");
    int out_number = vsl_compress_layer->getNbOutputs();
    for (int i = 0; i < out_number; i++) {
      nvinfer1::ITensor *vsl_output_tensor = vsl_compress_layer->getOutput(i);
      ctx->RegisterTensor(ITensorHelper{vsl_output_tensor, Format::NCHW, true},
                          "vsl_compress_output_" + std::to_string(i));
      inputTensors[input_number + i] = vsl_output_tensor;
    }
  } else {
    auto vsl_compress_layer = ctx->network()->getLayer(runtime_->GetVslEncoderPluginId());
    int out_number = vsl_compress_layer->getNbOutputs();
    for (int i = 0; i < out_number; i++) {
      nvinfer1::ITensor *vsl_output_tensor = vsl_compress_layer->getOutput(i);
      inputTensors[input_number + i] = vsl_output_tensor;
    }
  }
  return RET_OK;
}

void EncoderTensorRT::CastFfnTensors(std::shared_ptr<mindspore::ops::EncoderLayer> encoder_op, TensorRTContext *ctx) {
  size_t start_fp16 = (encoder_op->get_post_layernorm()) ? C7NUM : C9NUM;
  size_t end_fp16 = (encoder_op->get_post_layernorm()) ? C11NUM : C13NUM;
  if (encoder_op->get_position_bias()) {
    start_fp16 = C6NUM;
    end_fp16 = C9NUM;
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto in_tensor = input(ctx, i);
    if (in_tensors_[i].IsConst() || in_tensor.trt_tensor_ == nullptr) {
      if (i > start_fp16 && i < end_fp16) {
        in_tensor.trt_tensor_ = CastTensor(ctx, in_tensors_[i], op_name_);
        ctx->RegisterTensor(in_tensor, in_tensors_[i].Name());
      } else {
        in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[i], op_name_);
        ctx->RegisterTensor(in_tensor, in_tensors_[i].Name());
      }
    }
  }
}

void EncoderTensorRT::BuildEncoderTensors(TensorRTContext *ctx) {
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto in_tensor = input(ctx, i);
    if (in_tensors_[i].IsConst() || in_tensor.trt_tensor_ == nullptr) {
      in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[i], op_name_);
      ctx->RegisterTensor(in_tensor, in_tensors_[i].Name());
    }
  }
}

void EncoderTensorRT::BuildUsePastTensors(TensorRTContext *ctx) {
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto in_tensor = input(ctx, i);
    if (in_tensors_[i].IsConst() || in_tensor.trt_tensor_ == nullptr) {
      if (i == C1NUM || i == C2NUM) {
        int *data = const_cast<int *>(static_cast<const int *>(in_tensors_[i].Data()));
        *data = unique_id_ | (i << C8NUM);
      }
      in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[i], op_name_);
      ctx->RegisterTensor(in_tensor, in_tensors_[i].Name());
    }
  }
}

int EncoderTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  auto encoder_op = AsOps<ops::EncoderLayer>();
  if (encoder_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  cublasHandle_t cublas_handle = GetCublasHandle();
  bool eft = false;
  int encoder_input_idx = runtime_->GetTransformerEncoderInputIdx();
  if (IsWeightInputHanledInner()) {
    if (encoder_op->get_use_past()) {
      BuildUsePastTensors(ctx);
    } else if (IsFfnMixPrecision()) {
      CastFfnTensors(encoder_op, ctx);
    } else {
      BuildEncoderTensors(ctx);
    }
  }

  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  const int input_number = inputs().size();
  const int vsl_input_number = runtime_->IsTransformerOptimizeSigma() ? C3NUM : ((encoder_input_idx == -1) ? 0 : C1NUM);

  nvinfer1::ITensor *inputTensors[input_number + vsl_input_number];
  for (int i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }

  if (encoder_op->get_embedding_layer()) {
    nvinfer1::ITensor *tmp = inputTensors[0];
    inputTensors[0] = inputTensors[C5NUM];
    inputTensors[C5NUM] = tmp;
  }

  if (encoder_input_idx != -1 && encoder_op->get_use_past() == false) {
    eft = true;
    AddVsl(encoder_input_idx, input_number, ctx, inputTensors, input_tensor->getName());
  }
  if (runtime_->IsTransformerOptimizeSigma()) {
    AddVslByBatchValidLength(input_number - 1, ctx, inputTensors, input_tensor->getName());
  }
  auto compute_type = runtime_->GetRuntimePrecisionMode();
  auto plugin = std::make_shared<EncoderPlugin>(input_tensor->getName(), compute_type, encoder_op, cublas_handle, eft,
                                                runtime_->GetTransformerFfnFp16(), device_id_);
  nvinfer1::IPluginV2Layer *encoder_layer =
    ctx->network()->addPluginV2(inputTensors, (input_number + vsl_input_number), *plugin);

  if (encoder_layer == nullptr) {
    MS_LOG(ERROR) << "add encoder op failed for TensorRT.";
    return RET_ERROR;
  }
  encoder_layer->setName((op_name_ + "plugin_encoder").c_str());
  nvinfer1::ITensor *encoder_tensor = encoder_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{encoder_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = encoder_layer;
  unique_id_++;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(EncoderPluginCreater);
template class TensorRTPluginCreater<EncoderPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int EncoderPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                           const void *const *inputs, void *const *outputs, void *workspace,
                           cudaStream_t stream) noexcept {
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    return RunCudaEncoder<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    return RunCudaEncoder<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

#ifdef LITE_CUDA_DISTRIBUTION
int allGatherFunc(const void *input_addr, void *output_addr, size_t count, nvinfer1::DataType data_type,
                  cudaStream_t stream) {
  return DistributionCollective::instance().AllGatherWrapper(input_addr, output_addr, count, data_type, stream,
                                                             NCCL_WORLD_GROUP);
}

int allReduceSumFunc(const void *input_addr, void *output_addr, size_t count, nvinfer1::DataType data_type,
                     cudaStream_t stream) {
  return DistributionCollective::instance().AllReduceWrapper(input_addr, output_addr, count, data_type, Reduce_Sum,
                                                             stream, NCCL_WORLD_GROUP);
}
#endif

template <typename T>
int EncoderPlugin::RunCudaEncoder(const nvinfer1::PluginTensorDesc *inputDesc,
                                  const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                  void *const *outputs, void *workspace, cudaStream_t stream, cublasGemmAlgo_t algoId) {
  encoder_layer_->SetParallelFunc(nullptr, nullptr);
#ifdef LITE_CUDA_DISTRIBUTION
  if (encoder_layer_->GetRankNum() > 1) {
    encoder_layer_->SetParallelFunc(allGatherFunc, allReduceSumFunc);
  }
#endif
  encoder_layer_->SetWSOffset(0);
  encoder_layer_->SetAlgo(algoId);
  std::vector<void *> inputs_forward;
  for (int i = 0; i < num_of_inputs_; i++) {
    inputs_forward.push_back(const_cast<void *>(inputs[i]));
  }
  if (encoder_op_->get_embedding_layer()) {
    void *tmp = inputs_forward[0];
    inputs_forward[0] = inputs_forward[C5NUM];
    inputs_forward[C5NUM] = tmp;
  }
  std::vector<void *> outputs_forward = {const_cast<void *>(outputs[0])};
  encoder_layer_->forward(inputs_forward, outputs_forward, workspace, cublas_handle_, stream);
  return RET_OK;
}

bool EncoderPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                              int nbOutputs) noexcept {
  auto type = (compute_type_ == RuntimePrecisionMode_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  bool res = (tensorsDesc[pos].type == nvinfer1::DataType::kINT32);
  if (eft_ && pos == (nbInputs - C1NUM)) {
    return res;
  }

  if (encoder_op_->get_use_past()) {
    // last tensors are VSL + position ids
    if ((pos >= (nbInputs - C5NUM)) && (pos <= (nbInputs - C1NUM))) {
      return (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && res;
    }
    // input ids is the 5 tensor
    if (encoder_op_->get_embedding_layer() && (pos == C5NUM)) {
      return (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && res;
    }

    if (encoder_op_->get_query_layer() && (pos == (nbInputs - C6NUM) || pos == C5NUM)) {
      return (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && res;
    }
  }
  if (encoder_op_->get_moe()) {
    int expert_id = C12NUM;
    if (encoder_op_->get_query_layer()) expert_id += C2NUM;
    if (pos == (expert_id)) {
      return res;
    }
  }
  res = (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && (tensorsDesc[pos].type == type);
  return res;
}

template <typename T>
int EncoderPlugin::InitEncoder(size_t batch_size, size_t seq_len, size_t emmbeding_size) {
  size_t h_token_num = batch_size * seq_len;
  size_t head_num = encoder_op_->get_head_num();
  size_t head_size = encoder_op_->get_head_size();
  fastertransformer::FfnBase::ActType act_type;
  if (encoder_op_->get_act_type() == ActType::ActType_FastGelu) {
    act_type = fastertransformer::FfnBase::ActType::FastGelu;
  } else if (encoder_op_->get_act_type() == ActType::ActType_Gelu) {
    act_type = fastertransformer::FfnBase::ActType::Gelu;
  } else if (encoder_op_->get_act_type() == ActType::ActType_Relu) {
    act_type = fastertransformer::FfnBase::ActType::Relu;
  } else {
    act_type = static_cast<fastertransformer::FfnBase::ActType>(encoder_op_->get_act_type());
  }
  int rank_id = 0;
  int rank_num = 1;
#ifdef LITE_CUDA_DISTRIBUTION
  rank_id = GetRankID();
  rank_num = GetGPUGroupSize();
#endif
  size_t hidden_size = head_num * head_size * rank_num;
  encoder_layer_ =
    std::make_shared<fastertransformer::Encoder<T>>(batch_size, seq_len, head_num, head_size, hidden_size);
  encoder_layer_->SetFfnParam(ffn_fp16_, encoder_op_->get_ffn_hidden_size(), act_type,
                              !encoder_op_->get_position_bias());
  encoder_layer_->SetIsLayerNorm(encoder_op_->get_layer_norm(), encoder_op_->get_eps_layernorm3());
  encoder_layer_->SetT5(encoder_op_->get_position_bias());
  encoder_layer_->SetVSL(eft_);
  encoder_layer_->SetQueryLayer(encoder_op_->get_query_layer());
  encoder_layer_->SetEps(encoder_op_->get_eps_layernorm1(), encoder_op_->get_eps_layernorm2(),
                         encoder_op_->get_eps_layernorm3());
  encoder_layer_->SetScaleAttn(encoder_op_->get_scale());
  encoder_layer_->SetUsePast(encoder_op_->get_use_past());
  encoder_layer_->SetLayerNormPost(encoder_op_->get_post_layernorm());
  encoder_layer_->SetMoeParam(encoder_op_->get_moe(), encoder_op_->get_expert_num(),
                              encoder_op_->get_expert_offset_id(), encoder_op_->get_capacity_factor(), act_type);
  encoder_layer_->SetRankParam(rank_num, rank_id);
  encoder_layer_->SetHTokenNum(h_token_num, h_token_num);
  encoder_layer_->SetEmmbedingSize(emmbeding_size);
  encoder_layer_->SetFirstLayer(encoder_op_->get_embedding_layer());
  encoder_layer_->SetAlgo(CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return RET_OK;
}
void EncoderPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  int input_idx = (encoder_op_->get_embedding_layer()) ? C5NUM : 0;
  int request_batch_size = static_cast<int>(in[input_idx].desc.dims.d[0]);
  int request_src_seq_len = static_cast<int>(in[input_idx].desc.dims.d[1]);
  int embedding_size = -1;
  if (encoder_op_->get_query_layer()) {
    request_batch_size = static_cast<int>(in[nbInputs - C5NUM].desc.dims.d[0]);
    request_src_seq_len = static_cast<int>(in[input_idx].desc.dims.d[0]) / request_batch_size;
    embedding_size = static_cast<int>(in[nbInputs - C8NUM].desc.dims.d[0]);
  }
  num_of_inputs_ = nbInputs;
  num_of_outputs_ = nbOutputs;
  int result;
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    result = InitEncoder<half>(request_batch_size, request_src_seq_len, embedding_size);
  } else {
    result = InitEncoder<float>(request_batch_size, request_src_seq_len, embedding_size);
  }
  if (result != RET_OK) {
    MS_LOG(ERROR) << "Init encoder_layer in encoder tensorrt failed.";
  }
  workspace_size_ = encoder_layer_->GetWorkspaceSize();
}

size_t EncoderPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                       const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  if (workspace_size_ != 0) {
    return workspace_size_;
  }
  size_t workspace_size = encoder_layer_->GetWorkspaceSize();
  return workspace_size;
}

nvinfer1::DimsExprs EncoderPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                       int nbInputDims, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  int input_idx = (encoder_op_->get_embedding_layer()) ? C5NUM : 0;
  if (index == 0) {
    dims.nbDims = inputs[input_idx].nbDims;
    std::vector<int> dims_value;
    dims_value.push_back(inputs[input_idx].d[0]->getConstantValue());
    dims_value.push_back(inputs[input_idx].d[C1NUM]->getConstantValue());
    if (encoder_op_->get_query_layer()) {
      dims.nbDims = C2NUM;
      dims_value[0] = inputs[nbInputDims - C5NUM].d[0]->getConstantValue();
      int embeeding_id = nbInputDims - C8NUM;
      dims_value[C1NUM] = inputs[embeeding_id].d[0]->getConstantValue();
    } else if (encoder_op_->get_use_past() && encoder_op_->get_embedding_layer()) {
      dims.nbDims = C3NUM;
      dims_value.push_back(inputs[nbInputDims - C7NUM].d[C1NUM]->getConstantValue());
    } else if (encoder_op_->get_use_past() && encoder_op_->get_layer_norm()) {
      dims.nbDims = C2NUM;
      dims_value[0] = inputs[input_idx].d[0]->getConstantValue() * inputs[input_idx].d[C1NUM]->getConstantValue();
      dims_value[C1NUM] = inputs[input_idx].d[C2NUM]->getConstantValue();
    } else {
      dims_value.push_back(inputs[input_idx].d[C2NUM]->getConstantValue());
    }
    for (int i = 0; i < dims.nbDims; i++) {
      dims.d[i] = exprBuilder.constant(dims_value[i]);
    }
  }
  return dims;
}

nvinfer1::IPluginV2DynamicExt *EncoderPlugin::clone() const noexcept {
  auto *plugin = new EncoderPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t EncoderPlugin::getSerializationSize() const noexcept {
  return sizeof(int) + sizeof(bool) + sizeof(bool) + sizeof(mindspore::ops::EncoderLayer);
}

void EncoderPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &compute_type_, sizeof(int));
  SerializeValue(&buffer, encoder_op_.get(), sizeof(mindspore::ops::EncoderLayer));
  SerializeValue(&buffer, &eft_, sizeof(bool));
  SerializeValue(&buffer, &ffn_fp16_, sizeof(bool));
}
REGISTER_TENSORRT_CREATOR(ops::kNameEncoderLayer, EncoderTensorRT)
}  // namespace mindspore::lite
