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

#include "src/extendrt/delegate/tensorrt/op/decoder_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "NvInferRuntimeCommon.h"
#include "ops/decoder_layer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
}  // namespace

int DecoderTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != C23NUM && in_tensors.size() != C16NUM && in_tensors.size() != C17NUM) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::ITensor *DecoderTensorRT::CastTensor(TensorRTContext *ctx, const TensorInfo &ms_tensor,
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

int DecoderTensorRT::AddVsl(int encoder_input_idx, int decoder_input_idx, int input_number, TensorRTContext *ctx,
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
    vsl_compress_layer->setName("plugin_encoder_vsl_compress");
    nvinfer1::ITensor *vsl_output_tensor = vsl_compress_layer->getOutput(0);
    ctx->RegisterTensor(ITensorHelper{vsl_output_tensor, Format::NCHW, true}, "vsl_compress_output");
    inputTensors[input_number] = vsl_output_tensor;
  } else {
    nvinfer1::ILayer *encoder_vsl_layer = ctx->network()->getLayer(runtime_->GetVslEncoderPluginId());
    inputTensors[input_number] = encoder_vsl_layer->getOutput(0);
  }
  if (runtime_->GetVslDecoderPluginId() == -1) {
    auto vsl_plugin = std::make_shared<VslCompressPlugin>(name, device_id_);
    CHECK_NULL_RETURN(vsl_plugin);
    nvinfer1::ITensor *inputVsl = ctx->network()->getInput(decoder_input_idx);
    auto vsl_compress_layer = ctx->network()->addPluginV2(&inputVsl, C1NUM, *vsl_plugin);
    if (vsl_compress_layer == nullptr) {
      MS_LOG(ERROR) << " create vsl compress layer failed for: ";
      return RET_ERROR;
    }
    auto plugin_id = static_cast<int>(ctx->network()->getNbLayers() - 1);
    runtime_->SetVslDecoderPluginId(plugin_id);
    vsl_compress_layer->setName("plugin_decoder_vsl_compress");
    nvinfer1::ITensor *vsl_output_tensor = vsl_compress_layer->getOutput(0);
    ctx->RegisterTensor(ITensorHelper{vsl_output_tensor, Format::NCHW, true}, "vsl_compress_output");
    inputTensors[input_number + 1] = vsl_output_tensor;
  } else {
    nvinfer1::ILayer *decoder_vsl_layer = ctx->network()->getLayer(runtime_->GetVslDecoderPluginId());
    inputTensors[input_number + 1] = decoder_vsl_layer->getOutput(0);
  }
  return RET_OK;
}
int DecoderTensorRT::InitParam(fastertransformer::decoderParamRun *params) {
  auto decoder_op = AsOps<ops::DecoderLayer>();
  if (decoder_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  cublasHandle_t cublas_handle = GetCublasHandle();
  params->common_param.eft = false;
  params->common_param.cublas_handle = cublas_handle;
  params->common_param.head_num = decoder_op->get_head_num();
  params->common_param.head_size = decoder_op->get_head_size();
  params->common_param.hidden_size = params->common_param.head_num * params->common_param.head_size;
  params->decoder.layernorm_post = decoder_op->get_post_layernorm();
  params->decoder.eps1 = decoder_op->get_eps_layernorm1();
  params->decoder.eps2 = decoder_op->get_eps_layernorm2();
  params->decoder.eps3 = decoder_op->get_eps_layernorm3();
  params->decoder.eps4 = decoder_op->get_eps_layernorm4();
  params->ffn_param.ffn_param.ffn_hidden_size = decoder_op->get_ffn_hidden_size();
  params->ffn_param.ffn_param.ffn_fp16 = runtime_->GetTransformerFfnFp16();
  params->ffn_param.ffn_param.act_type = (fastertransformer::ActType)(decoder_op->get_act_type());
  params->attn1.attn.position_bias = decoder_op->get_position_bias1();
  params->ffn_param.ffn_param.ffn_bias = !params->attn1.attn.position_bias;
  params->attn1.attn.qkv_bias = !params->attn1.attn.position_bias;
  params->attn1.attn.projection_bias = !params->attn1.attn.position_bias;
  params->attn1.attn.is_cross = false;
  params->attn1.attn.scale = decoder_op->get_scale1();
  params->attn1.attn.mask = true;
  params->attn2.attn.position_bias = decoder_op->get_position_bias2();
  params->attn2.attn.qkv_bias = !params->attn2.attn.position_bias;
  params->attn2.attn.projection_bias = !params->attn2.attn.position_bias;
  params->attn2.attn.is_cross = true;
  params->attn2.attn.scale = decoder_op->get_scale2();
  params->attn2.attn.mask = true;
  params->decoder.has_beta = !params->attn1.attn.position_bias;
  params->decoder.is_layernorm = decoder_op->get_layer_norm();
  return RET_OK;
}

void DecoderTensorRT::CastFfnTensors(fastertransformer::decoderParamRun *params, TensorRTContext *ctx) {
  size_t start_fp16 = (params->attn1.attn.position_bias) ? C13NUM : C18NUM;
  size_t end_fp16 = (params->attn1.attn.position_bias) ? C16NUM : C22NUM;
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

int DecoderTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  fastertransformer::decoderParamRun params;
  if (InitParam(&params) != RET_OK) {
    MS_LOG(ERROR) << "Init param in decoder tensorrt failed.";
    return RET_ERROR;
  }
  int encoder_input_idx = runtime_->GetTransformerEncoderInputIdx();
  int decoder_input_idx = runtime_->GetTransformerDecoderInputIdx();
  if ((encoder_input_idx != -1 && decoder_input_idx == -1) || (encoder_input_idx == -1 && decoder_input_idx != -1)) {
    MS_LOG(ERROR) << "The indexes of inputs for vsl are not correct.";
    return RET_ERROR;
  }
  if (IsWeightInputHanledInner()) {
    CastFfnTensors(&params, ctx);
  }
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  const int input_number = inputs().size();
  const int vsl_input_number = (encoder_input_idx == -1 && decoder_input_idx == -1) ? 0 : C2NUM;
  nvinfer1::ITensor *inputTensors[input_number + vsl_input_number];
  for (int i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }
  if (encoder_input_idx != -1 && decoder_input_idx != -1) {
    params.common_param.eft = true;
    AddVsl(encoder_input_idx, decoder_input_idx, input_number, ctx, inputTensors, input_tensor->getName());
  }
  auto compute_type = runtime_->GetRuntimePrecisionMode();
  auto plugin = std::make_shared<DecoderPlugin>(input_tensor->getName(), compute_type, params, device_id_);
  nvinfer1::IPluginV2Layer *decoder_layer =
    ctx->network()->addPluginV2(inputTensors, input_number + vsl_input_number, *plugin);
  if (decoder_layer == nullptr) {
    MS_LOG(ERROR) << "add decoder op failed for TensorRT.";
    return RET_ERROR;
  }
  decoder_layer->setName((op_name_ + "plugin_decoder_layer").c_str());
  nvinfer1::ITensor *decoder_tensor = decoder_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{decoder_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = decoder_layer;
  return RET_OK;
}

REGISTER_TENSORRT_PLUGIN(DecoderPluginCreater);
template class TensorRTPluginCreater<DecoderPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int DecoderPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                           const void *const *inputs, void *const *outputs, void *workspace,
                           cudaStream_t stream) noexcept {
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    return RunCudaDecoder<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    return RunCudaDecoder<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}
template <typename T>
int DecoderPlugin::RunCudaDecoder(const nvinfer1::PluginTensorDesc *inputDesc,
                                  const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                  void *const *outputs, void *workspace, cudaStream_t stream, cublasGemmAlgo_t algoId) {
  params_.common_param.algo = algoId;
  params_.common_param.stream = stream;
  void *inputs_forward[num_of_inputs_];
  for (int i = 0; i < num_of_inputs_; i++) {
    inputs_forward[i] = const_cast<void *>(inputs[i]);
  }
  void *outputs_forward[] = {outputs[0]};
  fastertransformer::forwardDecoder<T>(inputs_forward, num_of_inputs_, outputs_forward, num_of_outputs_, &params_,
                                       workspace);
  return RET_OK;
}

bool DecoderPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                              int nbOutputs) noexcept {
  auto type = (compute_type_ == RuntimePrecisionMode_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  if (params_.common_param.eft && (pos == nbInputs - 1 || pos == nbInputs - C2NUM)) {
    bool res = (tensorsDesc[pos].type == nvinfer1::DataType::kINT32) ? true : false;
    return res;
  }
  bool res = (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) && (tensorsDesc[pos].type == type);
  return res;
}

void DecoderPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                    const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  const int request_batch_size = static_cast<const int>(in[0].desc.dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(in[0].desc.dims.d[1]);
  const int request_tgt_seq_len = params_.attn1.attn.position_bias
                                    ? static_cast<const int>(in[C10NUM].desc.dims.d[C2NUM])
                                    : static_cast<const int>(in[C14NUM].desc.dims.d[C2NUM]);
  params_.common_param.batch_size = request_batch_size;
  params_.common_param.src_seq_len = request_src_seq_len;
  params_.common_param.tgt_seq_len = request_tgt_seq_len;
  num_of_inputs_ = nbInputs;
  num_of_outputs_ = nbOutputs;
}
size_t DecoderPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                       const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    return fastertransformer::GetDecoderLayerWorkspaceSize<half>(&params_);
  } else {
    return fastertransformer::GetDecoderLayerWorkspaceSize<float>(&params_);
  }
}

nvinfer1::DimsExprs DecoderPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                       int nbInputDims, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  if (index == 0) {
    int num_dims = inputs[0].nbDims;
    dims.nbDims = num_dims;
    for (int i = 0; i < num_dims; i++) {
      auto num = exprBuilder.constant(inputs[index].d[i]->getConstantValue());
      dims.d[i] = num;
    }
  }
  return dims;
}

nvinfer1::IPluginV2DynamicExt *DecoderPlugin::clone() const noexcept {
  auto *plugin = new DecoderPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  plugin->params_.attn1.common_param = &plugin->params_.common_param;
  plugin->params_.attn2.common_param = &plugin->params_.common_param;
  plugin->params_.ffn_param.common_param = &plugin->params_.common_param;
  return plugin;
}

size_t DecoderPlugin::getSerializationSize() const noexcept {
  return sizeof(int) + sizeof(fastertransformer::decoderParamRun);
}

void DecoderPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &compute_type_, sizeof(int));
  SerializeValue(&buffer, &params_, sizeof(fastertransformer::decoderParamRun));
}
REGISTER_TENSORRT_CREATOR(ops::kNameDecoderLayer, DecoderTensorRT)
}  // namespace mindspore::lite
