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
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/mha_tensorrt.h"
#include "ops/attention.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/allocator.h"

namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
constexpr std::size_t kThree = 3;
}  // namespace

// Multi Head Attention TensorRT op
int MhaTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 8 && in_tensors.size() != 6) {
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
  // get attribute for Attn op - TODO - add attribute in op
  int head_number = mha_op->get_head_num();
  int head_size = mha_op->get_head_size();
  int compute_type = 1;  //  mha_op->get_compute_type();
  int is_cross = mha_op->get_cross();
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;

  auto plugin = std::make_shared<MhaPlugin>(input_tensor->getName(), compute_type, head_number, head_size, is_cross,
                                            GetCublasHandle(), GetCublasLtHandle(), device_id_);
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
  mha_layer->setName(op_name_.c_str());
  // TODO(haim) one output
  nvinfer1::ITensor *attn_tensor = mha_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{attn_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  // nvinfer1::ITensor *key_tensor = mha_layer->getOutput(1);
  // ctx->RegisterTensor(ITensorHelper{key_tensor, Format::NCHW, true}, out_tensors_[1].Name());
  // nvinfer1::ITensor *value_tensor = mha_layer->getOutput(kTwo);
  // ctx->RegisterTensor(ITensorHelper{value_tensor, Format::NCHW, true}, out_tensors_[kTwo].Name());
  this->layer_ = mha_layer;

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
  return RunCudaMha(inputDesc, outputDesc, inputs, outputs, workspace, stream);
}

int MhaPlugin::RunCudaMha(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
  // inputs order:
  // 0] Q
  // 1] K
  // 2] V
  // 3] W
  // 4] PW
  // 5] B
  // 6] PB
  // 7] AttnMask

  cublasSetStream(cublas_handle_, stream);

  // TODO(Haim) - Fix tensor ids according to cross flag
  const int from_tensor_idx = 0;
  // const int encoder_tensor_idx = 1;
  const int weight_qkv_tensor_idx = 3;
  const int weight_projection_tensor_idx = 4;
  const int bias_qkv_tensor_idx = 5;
  const int bias_projection_tensor_idx = 6;
  const int attn_mask_tensor_idx = 7;

  auto from_tensor = static_cast<const float *>(inputs[from_tensor_idx]);
  auto attention_mask = static_cast<const float *>(inputs[attn_mask_tensor_idx]);
  auto weight_qkv = static_cast<const float *>(inputs[weight_qkv_tensor_idx]);
  auto bias_qkv = static_cast<const float *>(inputs[bias_qkv_tensor_idx]);
  auto weight_projection = static_cast<const float *>(inputs[weight_projection_tensor_idx]);
  auto bias_projection = static_cast<const float *>(inputs[bias_projection_tensor_idx]);

  auto output0 = static_cast<float *>(outputs[0]);
  // auto output1 = static_cast<float *>(outputs[1]);
  // auto output2 = static_cast<float *>(outputs[2]);

  auto attn_dim_size = inputDesc[attn_mask_tensor_idx].dims.nbDims;
  const int request_batch_size = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 2]);
  const int request_tgt_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 1]);
  auto hidden_size = static_cast<const int>(head_number_ * head_size_);

  // TODO(NIZZAN): fix allocator
  size_t extra_tmp_size = request_batch_size * head_number_ * head_size_ * request_tgt_seq_len;

  size_t size_q = request_batch_size * request_src_seq_len * hidden_size;
  size_t size_k = request_batch_size * request_tgt_seq_len * hidden_size;
  size_t size_v = size_k;

  size_t qkv_len = size_q + size_k + size_v;
  size_t q_buf_2_len = size_q;
  size_t qk_buf_len = request_batch_size * head_number_ * request_src_seq_len * request_tgt_seq_len;
  size_t qkv_buf_2_len = request_batch_size * request_src_seq_len * hidden_size;
  size_t qkv_buf_3_len = qkv_buf_2_len;
  size_t buff_size = qkv_len + q_buf_2_len + qk_buf_len + qkv_buf_2_len + qkv_buf_3_len;
  qkv_buf_ = workspace;
  q_buf_2_ = static_cast<float *>(qkv_buf_) + qkv_len;
  qk_buf_ = static_cast<float *>(q_buf_2_) + q_buf_2_len;
  qkv_buf_2_ = static_cast<float *>(qk_buf_) + qk_buf_len;
  qkv_buf_3_ = static_cast<float *>(qkv_buf_2_) + qkv_buf_2_len;
  output1_ = static_cast<float *>(workspace) + buff_size;
  output2_ = static_cast<float *>(output1_) + extra_tmp_size;

  int gemm_dims[3] = {3 * hidden_size, request_batch_size * request_src_seq_len, hidden_size};
  int gemm_lds[3] = {3 * hidden_size, hidden_size, 3 * hidden_size};

  cublasOperation_t gemm_ops[2] = {CUBLAS_OP_N, CUBLAS_OP_N};
  const cudaDataType gemm_data_types[] = {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F};
  float alpha = 1.0f;
  float beta = 0.0f;

  CublasGemmWrapper(weight_qkv, from_tensor, qkv_buf_, gemm_dims, gemm_lds, gemm_ops, gemm_data_types, &alpha, &beta,
                    cublas_handle_);

  fastertransformer::invokeAddFusedQKVBiasTranspose(static_cast<float *>(q_buf_2_), static_cast<float *>(output1_),
                                                    static_cast<float *>(output2_), static_cast<float *>(qkv_buf_),
                                                    bias_qkv, request_batch_size, request_src_seq_len, head_number_,
                                                    head_size_, 0, stream);
  gemm_ops[0] = CUBLAS_OP_T;
  gemm_ops[1] = CUBLAS_OP_N;
  gemm_dims[0] = request_tgt_seq_len;
  gemm_dims[1] = request_src_seq_len;
  gemm_dims[THIRD_INPUT] = head_size_;

  gemm_lds[0] = head_size_;
  gemm_lds[1] = head_size_;
  gemm_lds[THIRD_INPUT] = request_tgt_seq_len;

  int gemm_strides[] = {request_tgt_seq_len * head_size_, request_src_seq_len * head_size_,
                        request_src_seq_len * request_tgt_seq_len};
  CublasGemmStridedBatchedWrapper(output1_, q_buf_2_, qk_buf_, gemm_dims, gemm_lds, gemm_ops, gemm_strides,
                                  gemm_data_types, &alpha, &beta, request_batch_size * head_number_, cublas_handle_);

  float scalar = (1.0f / sqrtf(static_cast<float>(head_size_) * 1.0f));
  fastertransformer::invokeMixMaskedSoftMax(static_cast<float *>(qk_buf_), attention_mask, request_batch_size,
                                            request_src_seq_len, request_tgt_seq_len, head_number_, scalar, stream);
  gemm_ops[0] = CUBLAS_OP_N;
  gemm_ops[1] = CUBLAS_OP_N;
  gemm_dims[0] = head_size_;
  gemm_dims[1] = request_src_seq_len;
  gemm_dims[THIRD_INPUT] = request_tgt_seq_len;

  gemm_lds[0] = head_size_;
  gemm_lds[1] = request_tgt_seq_len;
  gemm_lds[THIRD_INPUT] = head_size_;

  gemm_strides[0] = request_tgt_seq_len * head_size_;
  gemm_strides[1] = request_src_seq_len * request_tgt_seq_len;
  gemm_strides[THIRD_INPUT] = request_src_seq_len * head_size_;

  CublasGemmStridedBatchedWrapper(output2_, qk_buf_, qkv_buf_2_, gemm_dims, gemm_lds, gemm_ops, gemm_strides,
                                  gemm_data_types, &alpha, &beta, request_batch_size * head_number_, cublas_handle_);

  fastertransformer::invokeTransposeQKV(static_cast<float *>(qkv_buf_3_), static_cast<float *>(qkv_buf_2_),
                                        request_batch_size, request_src_seq_len, head_number_, head_size_, stream);

  gemm_ops[0] = CUBLAS_OP_N;
  gemm_ops[1] = CUBLAS_OP_N;
  gemm_dims[0] = hidden_size;
  gemm_dims[1] = request_batch_size * request_src_seq_len;
  gemm_dims[THIRD_INPUT] = hidden_size;

  gemm_lds[0] = hidden_size;
  gemm_lds[1] = hidden_size;
  gemm_lds[THIRD_INPUT] = hidden_size;

  CublasGemmWrapper(weight_projection, qkv_buf_3_, output0, gemm_dims, gemm_lds, gemm_ops, gemm_data_types, &alpha,
                    &beta, cublas_handle_);
  int len = request_batch_size * request_src_seq_len;
  fastertransformer::invokeAddBias(reinterpret_cast<float *>(output0), reinterpret_cast<const float *>(bias_projection),
                                   len, hidden_size, stream);

  return RET_OK;
}

int MhaPlugin::RunCudaCrossMha(const nvinfer1::PluginTensorDesc *inputDesc,
                               const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                               void *const *outputs, void *workspace, cudaStream_t stream) {
  //  Add Cross Mha Layer here
  return RET_OK;
}

size_t MhaPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  auto attn_dim_size = inputs[nbInputs - 1].dims.nbDims;
  const int request_batch_size = static_cast<const int>(inputs[nbInputs - 1].dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(inputs[nbInputs - 1].dims.d[attn_dim_size - 2]);
  const int request_tgt_seq_len = static_cast<const int>(inputs[nbInputs - 1].dims.d[attn_dim_size - 1]);
  auto hidden_size = static_cast<const int>(head_number_ * head_size_);

  // TODO(NIZZAN) Fix efficient allocator
  // size_t buff_size = request_batch_size * head_number_ * request_src_seq_len * request_tgt_seq_len +
  //             request_batch_size * request_src_seq_len * hidden_size;

  size_t size_q = request_batch_size * request_src_seq_len * hidden_size;
  size_t size_k = request_batch_size * request_tgt_seq_len * hidden_size;
  size_t size_v = size_k;

  size_t qkv_len = size_q + size_k + size_v;
  size_t q_buf_2_len = size_q;
  size_t qk_buf_len = request_batch_size * head_number_ * request_src_seq_len * request_tgt_seq_len;
  size_t qkv_buf_2_len = request_batch_size * request_src_seq_len * hidden_size;
  size_t qkv_buf_3_len = qkv_buf_2_len;
  size_t buff_size = qkv_len + q_buf_2_len + qk_buf_len + qkv_buf_2_len + qkv_buf_3_len;

  size_t extra_tmp_size = request_batch_size * head_number_ * head_size_ * request_tgt_seq_len;

  return (buff_size + extra_tmp_size + extra_tmp_size) * sizeof(float);
}

nvinfer1::DimsExprs MhaPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                                   nvinfer1::IExprBuilder &exprBuilder) noexcept {
  // MHA inputs:
  //      from_tensor [batch_size, src_seq_len, hidden_size_] or [batch_size * src_seq_len, hidden_size_]
  //      encoder_output [batch_size, tgt_seq_len, hidden_size_] or [batch_size * tgt_seq_len, hidden_size_]--> only in
  //      cross MHA attention_mask [batch_size, 1, src_seq_len, tgt_seq_len] or [batch_size, src_seq_len, tgt_seq_len]

  // MHA output_tensors:
  //      attention_out [batch_size, src_seq_len, hidden_size_]
  //      key_cache [batch, head_num, size_per_head]
  //      value_cache [batch, head_num, tgt_seq_len, size_per_head]
  nvinfer1::DimsExprs dims;
  if (index == 0) {
    // if (inputs[0].nbDims == 2) {
    //   dims.nbDims = INPUT_SIZE2;
    //   dims.d[0] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
    //   auto hidden_size = exprBuilder.constant(head_size_ * head_number_);
    //   dims.d[1] = hidden_size;
    // } else
    {
      dims.nbDims = INPUT_SIZE3;
      dims.d[0] = inputs[nbInputDims - 1].d[0];  // batch
      dims.d[1] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
      auto hidden_size = exprBuilder.constant(head_size_ * head_number_);
      dims.d[kTwo] = hidden_size;
    }
  } else {
    // TODO(Haim) - Fix size in case of 2d input
    dims.nbDims = INPUT_SIZE4;
    dims.d[0] = inputs[nbInputDims - 1].d[0];  // batch
    dims.d[1] = exprBuilder.constant(head_number_);
    dims.d[kTwo] = inputs[nbInputDims - 1].d[(inputs[nbInputDims - 1].nbDims) - 1];
    dims.d[kThree] = exprBuilder.constant(head_size_);
  }
  return dims;
}

nvinfer1::IPluginV2DynamicExt *MhaPlugin::clone() const noexcept {
  auto *plugin = new MhaPlugin(*this);  // TODO(haim) CopyConstructor
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

void MhaPlugin::terminate() noexcept {}

size_t MhaPlugin::getSerializationSize() const noexcept { return INPUT_SIZE4 * sizeof(int); }

void MhaPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &compute_type_, sizeof(int));
  SerializeValue(&buffer, &head_number_, sizeof(int));
  SerializeValue(&buffer, &head_size_, sizeof(int));
  SerializeValue(&buffer, &is_cross_, sizeof(int));
}
REGISTER_TENSORRT_CREATOR(ops::kNameAttention, MhaTensorRT)
}  // namespace mindspore::lite
