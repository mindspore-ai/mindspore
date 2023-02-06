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

#include "src/litert/kernel/cpu/fp32/convolution_1x1_fp32.h"
#include "src/litert/pack_weight_manager.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
Convolution1x1CPUKernel::~Convolution1x1CPUKernel() {
  FreeTmpBuffer();

  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
}

void Convolution1x1CPUKernel::FreeTmpBuffer() {
  if (pre_trans_input_ && input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

int Convolution1x1CPUKernel::ReSize() {
  FreeTmpBuffer();
  auto error_code = ConvolutionBaseCPUKernel::Prepare();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return error_code;
  }
  error_code = InitConv1x1MatmulParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "init convolution 1x1(matmul) parameters failed.";
    return error_code;
  }
  error_code = InitConv1x1Param();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1MatmulParam() {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  matmul_param_->col_align_ = UP_ROUND(matmul_param_->col_, col_tile_);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1Param() {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(row_tile_, op_parameter_->thread_num_, RET_ERROR);
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

  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_param_->row_, matmul_param_->deep_, RET_ERROR);
    input_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float));
  }

  return RET_OK;
}

int Convolution1x1CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
#ifdef ENABLE_AVX
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_SSE)
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#elif defined(ENABLE_ARM32)
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  if (matmul_param_ == nullptr) {
    matmul_param_ = new (std::nothrow) MatMulParameter;
    if (matmul_param_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  if (op_parameter_->is_train_session_) {
    auto filter_tensor = in_tensors_.at(kWeightIndex);
    CHECK_NULL_RETURN(filter_tensor);
    auto input_channel = filter_tensor->Channel();
    auto output_channel = filter_tensor->Batch();
    int output_tile_size = UP_ROUND(output_channel, col_tile_);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, output_tile_size, RET_ERROR);
    size_t size = static_cast<size_t>(input_channel * output_tile_size) * sizeof(float);
    set_workspace_size(size);
  }
  int error_code = InitConvWeightBias();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 init weight and bias failed.";
    return error_code;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::PackMatmulInput(const float *src_ptr, float *dst_ptr, int row, int col) const {
#ifdef ENABLE_AVX
  RowMajor2Col6Major(src_ptr, dst_ptr, row, col);
#elif defined(ENABLE_SSE)
  RowMajor2Col4Major(src_ptr, dst_ptr, row, col);
#else
  RowMajor2Col12Major(src_ptr, dst_ptr, row, col);
#endif
}

int Convolution1x1CPUKernel::DoConv1x1(int task_id) {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_stride_, RET_ERROR);
  int total_thead_stride_ = task_id * thread_stride_;
  int res_stride = matmul_param_->col_ - total_thead_stride_;
  int cur_oc = MSMIN(thread_stride_, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  CHECK_NULL_RETURN(out_tensors()[0]);
  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul_param_->deep_, RET_ERROR);
  if (out_tensors()[0]->format() == NC4HW4) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul_param_->row_, RET_ERROR);
    MatMulOpt(pack_input_, reinterpret_cast<float *>(packed_weight_) + total_thead_stride_ * matmul_param_->deep_,
              output_ptr_ + total_thead_stride_ * matmul_param_->row_, bias, matmul_param_->act_type_,
              matmul_param_->deep_, matmul_param_->row_, cur_oc, matmul_param_->row_, OutType_NC4HW4);
  } else {
    MatMulOpt(pack_input_, reinterpret_cast<float *>(packed_weight_) + total_thead_stride_ * matmul_param_->deep_,
              output_ptr_ + total_thead_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
              matmul_param_->row_, cur_oc, matmul_param_->col_, OutType_Nhwc);
  }
  return RET_OK;
}

int Convolution1x1Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoConv1x1(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::DoConv1x1Hw(int task_id) {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_stride_, RET_ERROR);
  int total_thead_stride_ = task_id * thread_stride_;
  int res_stride = matmul_param_->row_ - total_thead_stride_;
  int cur_hw_ = MSMIN(thread_stride_, res_stride);
  if (cur_hw_ <= 0) {
    return RET_OK;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul_param_->deep_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, row_tile_, RET_ERROR);
  int total_row_tile_ = task_id * row_tile_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_row_tile_, matmul_param_->deep_, RET_ERROR);
  float *thread_input_ptr = input_ptr_ + total_thead_stride_ * matmul_param_->deep_;
  float *thread_pack_input = pack_input_ + total_row_tile_ * matmul_param_->deep_;
  float *thread_output_ptr = nullptr;
  if (out_tensors()[0]->format() != NC4HW4) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul_param_->col_, RET_ERROR);
    thread_output_ptr = output_ptr_ + total_thead_stride_ * matmul_param_->col_;
  } else {
    auto col_min = MSMIN(matmul_param_->col_, C4NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, col_min, RET_ERROR);
    thread_output_ptr = output_ptr_ + total_thead_stride_ * col_min;
  }
  float *cur_intput = thread_input_ptr;
  float *cur_output = thread_output_ptr;
  for (int i = 0; i < cur_hw_; i += row_tile_) {
    int cur_rows = (cur_hw_ - i >= row_tile_) ? row_tile_ : (cur_hw_ - i);
    PackMatmulInput(cur_intput, thread_pack_input, cur_rows, matmul_param_->deep_);
    if (out_tensors()[0]->format() == NC4HW4) {
      MatMulOpt(thread_pack_input, reinterpret_cast<float *>(packed_weight_), cur_output,
                reinterpret_cast<float *>(bias_data_), matmul_param_->act_type_, matmul_param_->deep_, cur_rows,
                matmul_param_->col_, matmul_param_->row_, OutType_NC4HW4);
      cur_output += row_tile_ * MSMIN(matmul_param_->col_, C4NUM);
    } else {
      MatMulOpt(thread_pack_input, reinterpret_cast<float *>(packed_weight_), cur_output,
                reinterpret_cast<float *>(bias_data_), matmul_param_->act_type_, matmul_param_->deep_, cur_rows,
                matmul_param_->col_, matmul_param_->col_, OutType_Nhwc);
      cur_output += row_tile_ * matmul_param_->col_;
    }
    cur_intput += row_tile_ * matmul_param_->deep_;
  }

  return RET_OK;
}

int Convolution1x1RunHw(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoConv1x1Hw(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  auto src_in = reinterpret_cast<float *>(in_tensors_[0]->data());
  auto src_out = reinterpret_cast<float *>(out_tensors_[0]->data());
  CHECK_NULL_RETURN(src_in);
  CHECK_NULL_RETURN(src_out);
  int pack_input_size = 0;
  if (multi_thread_by_hw_) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(thread_count_, row_tile_, RET_ERROR);
    int total_row_tile_ = thread_count_ * row_tile_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(total_row_tile_, matmul_param_->deep_, RET_ERROR);
    pack_input_size = total_row_tile_ * matmul_param_->deep_;
  } else {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_param_->row_align_, matmul_param_->deep_, RET_ERROR);
    pack_input_size = matmul_param_->row_align_ * matmul_param_->deep_;
  }
  pack_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(pack_input_size * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_param_->row_, matmul_param_->col_, RET_ERROR);
  int matmul_size = matmul_param_->row_ * matmul_param_->col_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW((conv_param_->input_batch_ - 1), matmul_size, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
  int conv_input_hw = conv_param_->input_h_ * conv_param_->input_w_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_hw, conv_param_->input_channel_, RET_ERROR);
  int conv_input_bhw = conv_input_hw * conv_param_->input_channel_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW((conv_param_->input_batch_ - 1), conv_input_bhw, RET_ERROR);
  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    output_ptr_ = src_out + batch_index * matmul_size;
    auto tmp_in = src_in + batch_index * conv_input_bhw;
    if (pre_trans_input_) {
      Conv1x1InputPack(tmp_in, input_ptr_, conv_param_, sizeof(float));
    } else {
      input_ptr_ = tmp_in;
    }
    int ret = 0;
    if (multi_thread_by_hw_) {
      ret = ParallelLaunch(this->ms_context_, Convolution1x1RunHw, this, thread_count_);
    } else {
      PackMatmulInput(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
      ret = ParallelLaunch(this->ms_context_, Convolution1x1Run, this, thread_count_);
    }
    if (ret != RET_OK) {
      if (pack_input_ != nullptr) {
        ctx_->allocator->Free(pack_input_);
        pack_input_ = nullptr;
      }
      return ret;
    }
  }
  if (pack_input_ != nullptr) {
    ctx_->allocator->Free(pack_input_);
    pack_input_ = nullptr;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::PackWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  if (input_channel < 0) {
    MS_LOG(ERROR) << "get channel failed from filter_tensor.";
    return;
  }
  auto output_channel = filter_tensor->Batch();
  if (output_channel < 0) {
    MS_LOG(ERROR) << "get channel failed from filter_tensor.";
    return;
  }

  void *origin_weight = (op_parameter_->is_train_session_) ? filter_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
#ifdef ENABLE_AVX
  RowMajor2Col16Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                      output_channel, input_channel);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                     output_channel, input_channel);
#else
  RowMajor2Col8Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                     output_channel, input_channel);
#endif
}

int Convolution1x1CPUKernel::MallocWeightBiasData() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  MS_CHECK_TRUE_RET(input_channel > 0 && output_channel > 0, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, UP_ROUND(output_channel, col_tile_), RET_ERROR);
  size_t size = static_cast<size_t>(input_channel * UP_ROUND(output_channel, col_tile_)) * sizeof(float);
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, size);
    packed_weight_ =
      lite::PackWeightManager::GetInstance()->GetPackData(in_tensors_[1]->data(), size, &weight_is_packed_);
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc packed_weight_ error!";
      return RET_ERROR;
    }
  }

  if (in_tensors_.size() == kInputSize2) {
    size = UP_ROUND(output_channel, col_tile_) * sizeof(float);
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, size);
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
}  // namespace mindspore::kernel
