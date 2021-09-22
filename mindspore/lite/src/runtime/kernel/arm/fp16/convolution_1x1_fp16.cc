/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int Convolution1x1FP16CPUKernel::InitMatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  matmul_param_->col_align_ = UP_ROUND(matmul_param_->col_, col_tile_);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return RET_OK;
}

Convolution1x1FP16CPUKernel::~Convolution1x1FP16CPUKernel() {
  FreeTmpBuffer();
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  return;
}

int Convolution1x1FP16CPUKernel::InitConv1x1Param() {
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  if ((matmul_param_->row_ > (row_tile_ * op_parameter_->thread_num_)) && (matmul_param_->row_ > matmul_param_->col_)) {
    multi_thread_by_hw_ = true;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->row_, row_tile_));
    if (thread_count_ <= 0) {
      MS_LOG(ERROR) << "thread_count_ must be greater than 0!";
      return RET_ERROR;
    }
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->row_, row_tile_), thread_count_) * row_tile_;
  } else {
    multi_thread_by_hw_ = false;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, col_tile_));
    if (thread_count_ <= 0) {
      MS_LOG(ERROR) << "thread_count_ must be greater than 0!";
      return RET_ERROR;
    }
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, col_tile_), thread_count_) * col_tile_;
  }
  matmul_param_->op_parameter_.thread_num_ = thread_count_;

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float16_t *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float16_t));
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Channel();
  auto output_channel = weight_tensor->Batch();

  size_t size = input_channel * UP_ROUND(output_channel, col_tile_) * sizeof(float16_t);
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      packed_weight_ = malloc(size);
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "Conv1x1 Malloc packed_weight_ error!";
        return RET_ERROR;
      }
    }
    memset(packed_weight_, 0, size);
  }

  if (in_tensors_.size() == kInputSize2) {
    size = UP_ROUND(output_channel, col_tile_) * sizeof(float16_t);
    if (bias_data_ == nullptr) {
      bias_data_ = malloc(size);
      if (bias_data_ == nullptr) {
        MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
        return RET_ERROR;
      }
    }
    memset(bias_data_, 0, size);
  }
  return RET_OK;
}

void Convolution1x1FP16CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Channel();
  auto output_channel = weight_tensor->Batch();
  void *weight_origin = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(weight_origin != nullptr);
#ifdef ENABLE_ARM64
  if (out_tensors_.front()->format() == NC4HW4) {
    ColMajor2Row8MajorFp16(weight_origin, reinterpret_cast<float16_t *>(packed_weight_), input_channel, output_channel,
                           true);
  } else {
    RowMajor2Col16MajorFp16Opt(static_cast<const float16_t *>(weight_origin),
                               reinterpret_cast<float16_t *>(packed_weight_), output_channel, input_channel);
  }
#else
  ColMajor2Row8MajorFp16(weight_origin, reinterpret_cast<float16_t *>(packed_weight_), input_channel, output_channel,
                         true);
#endif
}

int Convolution1x1FP16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  UpdateOriginWeightAndBias();
#ifdef ENABLE_ARM64
  if (out_tensors_.front()->format() == NC4HW4) {
    row_tile_ = C16NUM;
    col_tile_ = C8NUM;
  } else {
    row_tile_ = C12NUM;
    col_tile_ = C16NUM;
  }
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(weight_tensor);
    auto input_channel = weight_tensor->Channel();
    auto output_channel = weight_tensor->Batch();
    size_t size = input_channel * UP_ROUND(output_channel, col_tile_) * sizeof(float16_t);
    set_workspace_size(size);
  }
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Init matmul_param_ failed.";
    return RET_ERROR;
  }
  int ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return ret;
  }
  return RET_OK;
}

void Convolution1x1FP16CPUKernel::FreeTmpBuffer() {
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

int Convolution1x1FP16CPUKernel::ReSize() {
  FreeTmpBuffer();
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = InitMatmulParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init matmul param failed.";
    return ret;
  }
  ret = InitConv1x1Param();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init conv1x1 param failed.";
    return ret;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::RunOc(int task_id) {
  int cur_stride = matmul_param_->col_ - task_id * thread_stride_;
  int cur_oc = MSMIN(thread_stride_, cur_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float16_t *>(bias_data_) + thread_stride_ * task_id;

  if (out_tensors_.front()->format() == NC4HW4) {
    Conv1x1OutNc8hw8MultiThreadByWeightFp16(input_ptr_, pack_input_, reinterpret_cast<float16_t *>(packed_weight_),
                                            reinterpret_cast<float16_t *>(bias_data_), output_ptr_, task_id,
                                            matmul_param_);
  } else {
#ifdef ENABLE_ARM64
    MatMul12x16Fp16Opt(pack_input_,
                       reinterpret_cast<float16_t *>(packed_weight_) + task_id * thread_stride_ * matmul_param_->deep_,
                       output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
                       matmul_param_->row_, cur_oc, matmul_param_->col_, OutType_Nhwc);
#else
    MatMul12x8A32Fp16(pack_input_,
                      reinterpret_cast<float16_t *>(packed_weight_) + task_id * thread_stride_ * matmul_param_->deep_,
                      output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
                      matmul_param_->row_, cur_oc, matmul_param_->col_, OutType_Nhwc);
#endif
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::RunHw(int task_id) {
  if (out_tensors_.front()->format() == NC4HW4) {
    Conv1x1OutNc8hw8MultiThreadByInputFp16(input_ptr_, pack_input_, reinterpret_cast<float16_t *>(packed_weight_),
                                           reinterpret_cast<float16_t *>(bias_data_), output_ptr_, task_id,
                                           matmul_param_);
  } else {
    int res_stride = matmul_param_->row_ - task_id * thread_stride_;
    int cur_hw_ = MSMIN(thread_stride_, res_stride);
    if (cur_hw_ <= 0) {
      return RET_OK;
    }
    float16_t *thread_input_ptr = input_ptr_ + task_id * thread_stride_ * matmul_param_->deep_;
    float16_t *thread_pack_input = pack_input_ + task_id * thread_stride_ * matmul_param_->deep_;
    float16_t *thread_output_ptr = output_ptr_ + task_id * thread_stride_ * matmul_param_->col_;

    RowMajor2Col12MajorFp16Opt(thread_input_ptr, thread_pack_input, cur_hw_, matmul_param_->deep_);
#ifdef ENABLE_ARM64
    MatMul12x16Fp16Opt(thread_pack_input, reinterpret_cast<float16_t *>(packed_weight_), thread_output_ptr,
                       reinterpret_cast<float16_t *>(bias_data_), matmul_param_->act_type_, matmul_param_->deep_,
                       cur_hw_, matmul_param_->col_, matmul_param_->col_, OutType_Nhwc);
#else
    MatMul12x8A32Fp16(thread_pack_input, reinterpret_cast<float16_t *>(packed_weight_), thread_output_ptr,
                      reinterpret_cast<float16_t *>(bias_data_), matmul_param_->act_type_, matmul_param_->deep_,
                      cur_hw_, matmul_param_->col_, matmul_param_->col_, OutType_Nhwc);
#endif
  }
  return RET_OK;
}

static int Convolution1x1Fp16RunOc(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<Convolution1x1FP16CPUKernel *>(cdata);
  auto error_code = conv->RunOc(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

static int Convolution1x1Fp16RunHw(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<Convolution1x1FP16CPUKernel *>(cdata);
  auto error_code = conv->RunHw(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 Run hw error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1FP16CPUKernel::Run() {
  auto input_data = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  auto output_data = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  MS_ASSERT(input_data != nullptr);
  MS_ASSERT(output_data != nullptr);
  if (input_data == nullptr || output_data == nullptr) {
    MS_LOG(ERROR) << "Convolution1x1 Fp16 get null tensor data!";
    return RET_ERROR;
  }
  pack_input_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(matmul_param_->row_align_ * matmul_param_->deep_ * sizeof(float16_t)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    output_ptr_ = output_data + batch_index * matmul_param_->row_ * matmul_param_->col_;
    float16_t *batch_in =
      input_data + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_;
    if (pre_trans_input_) {
      Conv1x1InputPack(batch_in, input_ptr_, conv_param_, sizeof(float16_t));
    } else {
      input_ptr_ = batch_in;
    }

    int ret = RET_ERROR;
    if (multi_thread_by_hw_) {
      ret = ParallelLaunch(this->ms_context_, Convolution1x1Fp16RunHw, this, thread_count_);
    } else {
      if (out_tensors_.front()->format() == NC4HW4) {
#ifdef ENABLE_ARM64
        RowMajor2Col16MajorFp16Opt(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#else
        RowMajor2Col12MajorFp16Opt(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#endif
      } else {
        RowMajor2Col12MajorFp16Opt(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
      }
      ret = ParallelLaunch(this->ms_context_, Convolution1x1Fp16RunOc, this, thread_count_);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ParallelLaunch failed.";
      ctx_->allocator->Free(pack_input_);
      pack_input_ = nullptr;
      return ret;
    }
  }
  ctx_->allocator->Free(pack_input_);
  pack_input_ = nullptr;
  return RET_OK;
}
}  // namespace mindspore::kernel
