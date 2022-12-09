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

#define SET_GEMM_PARAMS(gemm_ops_, gemm_lds_, gemm_op1_, gemm_op2_, gemm_ld1_, gemm_ld2_, gemm_ld3_) \
  do {                                                                                               \
    gemm_ops_[0] = gemm_op1_;                                                                        \
    gemm_ops_[1] = gemm_op2_;                                                                        \
    gemm_lds_[0] = gemm_ld1_;                                                                        \
    gemm_lds_[1] = gemm_ld2_;                                                                        \
    gemm_lds_[2] = gemm_ld3_;                                                                        \
  } while (0)

#define SET_GEMM_DIMS(gemm_dims_, gemm_dim1_, gemm_dim2_, gemm_dim3_) \
  do {                                                                \
    gemm_dims_[0] = gemm_dim1_;                                       \
    gemm_dims_[1] = gemm_dim2_;                                       \
    gemm_dims_[2] = gemm_dim3_;                                       \
  } while (0)

// Multi Head Attention TensorRT op
int MhaTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() < 7 || in_tensors.size() > 9) {  // T5 has 6 or 7 inputs, other models have 8 or 9 inputs
    MS_LOG(ERROR) << "Unsupported number of inputs, size is " << in_tensors.size();
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
  auto compute_type = runtime_->GetRuntimePrecisionMode();  //  mha_op->get_compute_type();
  bool is_cross = mha_op->get_cross();
  const int input_number = inputs().size();
  bool is_position_bias = (((input_number == 8) && is_cross) || ((input_number == 7) && !is_cross)) ? true : false;
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto plugin = std::make_shared<MhaPlugin>(input_tensor->getName(), compute_type, head_number, head_size, is_cross,
                                            is_position_bias, GetCublasHandle(), GetCublasLtHandle(), device_id_);
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
  // TODO(haim) one output
  nvinfer1::ITensor *attn_tensor = mha_layer->getOutput(0);
#ifndef TEST_
  ctx->RegisterTensor(ITensorHelper{attn_tensor, Format::NCHW, true}, out_tensors_[0].Name());
#else  /* TEST_ */
  ctx->RegisterTensor(ITensorHelper{attn_tensor, Format::NCHW, true}, out_tensors_[0].Name() + "attn");
#endif /* TEST_ */
  // nvinfer1::ITensor *key_tensor = mha_layer->getOutput(1);
  // ctx->RegisterTensor(ITensorHelper{key_tensor, Format::NCHW, true}, out_tensors_[1].Name());
  // nvinfer1::ITensor *value_tensor = mha_layer->getOutput(kTwo);
  // ctx->RegisterTensor(ITensorHelper{value_tensor, Format::NCHW, true}, out_tensors_[kTwo].Name());
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
    return RunCudaMha<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream, fast_algo_gemm);
  } else {
    return RunCudaMha<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream, fast_algo_gemm);
  }
}

template <typename T>
void MhaPlugin::SetInnerAddr(void *workspace, size_t size_q, size_t size_k, size_t qk_buf_len, size_t qkv_buf_2_len,
                             size_t extra_size) {
  size_t qkv_len = size_q + (size_k * 2);  // size_v is equal to size_k
  size_t q_buf_2_len = size_q;
  auto buff_size =
    qkv_len + q_buf_2_len + qk_buf_len + (qkv_buf_2_len * 2);  // qkv_buf_3_ len is equal to qkv_buf_2_len
  qkv_buf_ = workspace;
  q_buf_2_ = static_cast<T *>(qkv_buf_) + qkv_len;
  qk_buf_ = static_cast<T *>(q_buf_2_) + q_buf_2_len;
  qkv_buf_2_ = static_cast<T *>(qk_buf_) + qk_buf_len;
  qkv_buf_3_ = static_cast<T *>(qkv_buf_2_) + qkv_buf_2_len;
  output1_ = static_cast<T *>(workspace) + buff_size;
  output2_ = static_cast<T *>(output1_) + extra_size;
}

template <typename T>
void MhaPlugin::RunPhase1GEMM(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, int *gemm_dims,
                              int *gemm_lds, cublasOperation_t *gemm_ops, cudaDataType *gemm_data_types, void *alpha,
                              void *beta, cublasGemmAlgo_t algoId, cudaStream_t stream) {
  int cross_tensor_offset = 0;
  if (is_cross_) cross_tensor_offset = 1;
  const int from_tensor_idx = 0, encoder_tensor_idx = 1, weight_qkv_tensor_idx = 3;
  const int weight_qkv_tensor_idx_cross = 3 + cross_tensor_offset;
  const int bias_qkv_tensor_idx = 5 + cross_tensor_offset;
  const int attn_mask_tensor_idx = (is_position_bias_) ? 6 + cross_tensor_offset : 7 + cross_tensor_offset;

  auto from_tensor = static_cast<const T *>(inputs[from_tensor_idx]);
  auto encoder_output_tensor = static_cast<const T *>(inputs[encoder_tensor_idx]);
  auto weight_q = static_cast<const T *>(inputs[weight_qkv_tensor_idx]);
  auto weight_kv = static_cast<const T *>(inputs[weight_qkv_tensor_idx_cross]);
  auto weight_qkv = static_cast<const T *>(inputs[weight_qkv_tensor_idx_cross]);
  auto bias_qkv = (is_position_bias_) ? nullptr : static_cast<const T *>(inputs[bias_qkv_tensor_idx]);

  auto attn_dim_size = inputDesc[attn_mask_tensor_idx].dims.nbDims;
  const int request_batch_size = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 2]);
  const int request_tgt_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 1]);
  auto hidden_size = static_cast<const int>(head_number_ * head_size_);
  if (is_cross_) {
    SET_GEMM_PARAMS(gemm_ops, gemm_lds, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, hidden_size, hidden_size);
    SET_GEMM_DIMS(gemm_dims, hidden_size, request_batch_size * request_src_seq_len, hidden_size);
    CublasGemmWrapper(weight_q, from_tensor, qkv_buf_, gemm_dims, gemm_lds, gemm_ops,
                      const_cast<const cudaDataType *>(gemm_data_types), alpha, beta, cublas_handle_);
    SET_GEMM_DIMS(gemm_dims, C2NUM * hidden_size, request_batch_size * request_tgt_seq_len, hidden_size);
    gemm_lds[0] = gemm_lds[THIRD_INPUT] = C2NUM * hidden_size;

    CublasGemmWrapper(weight_kv, encoder_output_tensor,
                      static_cast<T *>(qkv_buf_) + (request_batch_size * request_src_seq_len) * hidden_size, gemm_dims,
                      gemm_lds, gemm_ops, const_cast<const cudaDataType *>(gemm_data_types), alpha, beta,
                      cublas_handle_);
    fastertransformer::invokeCrossAddFusedQKVBiasTranspose(
      static_cast<T *>(q_buf_2_), static_cast<T *>(output1_), static_cast<T *>(output2_), static_cast<T *>(qkv_buf_),
      bias_qkv, request_batch_size, request_src_seq_len, request_tgt_seq_len, head_number_, head_size_, stream);
  } else {
    CublasGemmWrapper(weight_qkv, from_tensor, qkv_buf_, gemm_dims, gemm_lds, gemm_ops,
                      const_cast<const cudaDataType *>(gemm_data_types), alpha, beta, cublas_handle_, algoId);
    fastertransformer::invokeAddFusedQKVBiasTranspose(
      static_cast<T *>(q_buf_2_), static_cast<T *>(output1_), static_cast<T *>(output2_), static_cast<T *>(qkv_buf_),
      bias_qkv, request_batch_size, request_src_seq_len, head_number_, head_size_, 0, stream);
  }
}

template <typename T>
int MhaPlugin::RunCudaMha(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream,
                          cublasGemmAlgo_t *algoId) {
  // inputs order:
  // 0] Q
  // 1] K
  // 2] V
  // 3] W
  // 4] PW
  // 5] B
  // 6] PB
  // 7] AttnMask
  // inputs order cross:
  // 0] Q
  // 1] K enco output
  // 2] V
  // 3] Wq
  // 4] Wkv
  // 5] PW
  // 6] Bqkv
  // 7] PB
  // 8] AttnMask
  int cross_tensor_offset = 0;
  cublasSetStream(cublas_handle_, stream);
  if (is_cross_) cross_tensor_offset = 1;
  const int weight_projection_tensor_idx = 4 + cross_tensor_offset;
  const int bias_projection_tensor_idx = 6 + cross_tensor_offset;
  const int attn_mask_tensor_idx = (is_position_bias_) ? 6 + cross_tensor_offset : 7 + cross_tensor_offset;
  const int bias_position_tensor_idx = 5 + cross_tensor_offset;

  auto attention_mask = static_cast<const T *>(inputs[attn_mask_tensor_idx]);
  auto weight_projection = static_cast<const T *>(inputs[weight_projection_tensor_idx]);
  auto bias_projection = (is_position_bias_) ? nullptr : static_cast<const T *>(inputs[bias_projection_tensor_idx]);
  auto bias_position = (is_position_bias_) ? static_cast<const T *>(inputs[bias_position_tensor_idx]) : nullptr;
  auto output0 = static_cast<T *>(outputs[0]);
  auto attn_dim_size = inputDesc[attn_mask_tensor_idx].dims.nbDims;
  const int request_batch_size = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 2]);
  const int request_tgt_seq_len = static_cast<const int>(inputDesc[attn_mask_tensor_idx].dims.d[attn_dim_size - 1]);
  auto hidden_size = static_cast<const int>(head_number_ * head_size_);
  auto extra_tmp_size = request_batch_size * head_number_ * head_size_ * request_tgt_seq_len;

  size_t size_q = request_batch_size * request_src_seq_len * hidden_size;
  size_t size_k = request_batch_size * request_tgt_seq_len * hidden_size;
  size_t qk_buf_len = request_batch_size * head_number_ * request_src_seq_len * request_tgt_seq_len;
  size_t qkv_buf_2_len = request_batch_size * request_src_seq_len * hidden_size;
  SetInnerAddr<T>(workspace, size_q, size_k, qk_buf_len, qkv_buf_2_len, extra_tmp_size);

  cublasOperation_t gemm_ops[] = {CUBLAS_OP_N, CUBLAS_OP_N};
  cudaDataType gemm_data_types[] = {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F};
  if constexpr (std::is_same<T, half>::value)
    std::fill(std::begin(gemm_data_types), std::end(gemm_data_types), CUDA_R_16F);
  float alpha = 1.0f, beta = 0.0f;
  int gemm_dims[] = {3 * hidden_size, request_batch_size * request_src_seq_len, hidden_size};
  int gemm_lds[] = {3 * hidden_size, hidden_size, 3 * hidden_size};

  RunPhase1GEMM<T>(inputDesc, inputs, gemm_dims, gemm_lds, gemm_ops, gemm_data_types, &alpha, &beta, algoId[0], stream);

  SET_GEMM_PARAMS(gemm_ops, gemm_lds, CUBLAS_OP_T, CUBLAS_OP_N, head_size_, head_size_, request_tgt_seq_len);
  SET_GEMM_DIMS(gemm_dims, request_tgt_seq_len, request_src_seq_len, head_size_);
  int gemm_strides[] = {request_tgt_seq_len * head_size_, request_src_seq_len * head_size_,
                        request_src_seq_len * request_tgt_seq_len};

  CublasGemmStridedBatchedWrapper(output1_, q_buf_2_, qk_buf_, gemm_dims, gemm_lds, gemm_ops, gemm_strides,
                                  const_cast<const cudaDataType *>(gemm_data_types), &alpha, &beta,
                                  request_batch_size * head_number_, cublas_handle_, algoId[1]);

  T scalar = static_cast<T>(1.0f / sqrtf(head_size_ * 1.0f));
  fastertransformer::invokeMixMaskedSoftMax(static_cast<T *>(qk_buf_), attention_mask, bias_position,
                                            request_batch_size, request_src_seq_len, request_tgt_seq_len, head_number_,
                                            scalar, stream);
  SET_GEMM_PARAMS(gemm_ops, gemm_lds, CUBLAS_OP_N, CUBLAS_OP_N, head_size_, request_tgt_seq_len, head_size_);
  SET_GEMM_DIMS(gemm_dims, head_size_, request_src_seq_len, request_tgt_seq_len);
  gemm_strides[1] = request_src_seq_len * request_tgt_seq_len;
  gemm_strides[THIRD_INPUT] = request_src_seq_len * head_size_;

  CublasGemmStridedBatchedWrapper(output2_, qk_buf_, qkv_buf_2_, gemm_dims, gemm_lds, gemm_ops, gemm_strides,
                                  const_cast<const cudaDataType *>(gemm_data_types), &alpha, &beta,
                                  request_batch_size * head_number_, cublas_handle_, algoId[2]);
  fastertransformer::invokeTransposeQKV(static_cast<T *>(qkv_buf_3_), static_cast<T *>(qkv_buf_2_), request_batch_size,
                                        request_src_seq_len, head_number_, head_size_, stream);
  SET_GEMM_PARAMS(gemm_ops, gemm_lds, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size, hidden_size, hidden_size);
  SET_GEMM_DIMS(gemm_dims, hidden_size, request_batch_size * request_src_seq_len, hidden_size);
  CublasGemmWrapper(weight_projection, qkv_buf_3_, output0, gemm_dims, gemm_lds, gemm_ops,
                    const_cast<const cudaDataType *>(gemm_data_types), &alpha, &beta, cublas_handle_, algoId[3]);
  if (!is_position_bias_) {
    int len = request_batch_size * request_src_seq_len;
    fastertransformer::invokeAddBias(reinterpret_cast<T *>(output0), reinterpret_cast<const T *>(bias_projection), len,
                                     hidden_size, stream);
  }
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
                                const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {}

size_t MhaPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  auto attn_dim_size = inputs[nbInputs - 1].dims.nbDims;
  const int request_batch_size = static_cast<const int>(inputs[nbInputs - 1].dims.d[0]);
  const int request_src_seq_len = static_cast<const int>(inputs[nbInputs - 1].dims.d[attn_dim_size - 2]);
  const int request_tgt_seq_len = static_cast<const int>(inputs[nbInputs - 1].dims.d[attn_dim_size - 1]);
  auto hidden_size = static_cast<const int>(head_number_ * head_size_);

  // TODO(NIZZAN) Fix efficient allocator
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
  int elem_size = sizeof(float);
  if (compute_type_ == RuntimePrecisionMode_FP16) {
    elem_size = sizeof(half);
  }

  return (buff_size + extra_tmp_size + extra_tmp_size) * elem_size;
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
#ifndef TEST_
    int num_dims = inputs[0].nbDims;
    dims.nbDims = num_dims;
    if (num_dims == INPUT_SIZE2) {
      dims.d[0] = exprBuilder.constant(inputs[nbInputDims - 1].d[0]->getConstantValue() *
                                       inputs[nbInputDims - 1].d[1]->getConstantValue());
      auto hidden_size = exprBuilder.constant(head_size_ * head_number_);
      dims.d[1] = hidden_size;
    } else if (num_dims == INPUT_SIZE3) {
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
  auto *plugin = new MhaPlugin(*this);  // TODO(haim) CopyConstructor
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

int MhaPlugin::initialize() noexcept { return 0; }

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
