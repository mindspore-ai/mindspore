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

#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
Convolution1x1CPUKernel::~Convolution1x1CPUKernel() {
  FreeTmpBuffer();
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
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
  auto error_code = ConvolutionBaseCPUKernel::Init();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return error_code;
  }
  InitConv1x1MatmulParam();
  error_code = InitConv1x1Param();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution base init failed.";
    return error_code;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::InitConv1x1MatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  matmul_param_->col_align_ = UP_ROUND(matmul_param_->col_, col_tile_);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return;
}

int Convolution1x1CPUKernel::InitConv1x1BiasWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();

  if (in_tensors_.size() == 3) {
    int size = UP_ROUND(output_channel, col_tile_) * sizeof(float);
    int weight_size = output_channel * sizeof(float);
    bias_data_ = malloc(size);
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc bias_ptr_ error!";
      return RET_ERROR;
    }
    memcpy(bias_data_, origin_bias_, weight_size);
    memset(reinterpret_cast<char *>(bias_data_) + weight_size, 0, size - weight_size);
  }

  int size = input_channel * UP_ROUND(output_channel, col_tile_) * sizeof(float);
  int down_size = input_channel * DOWN_DIV(output_channel, col_tile_) * col_tile_ * sizeof(float);
  weight_ptr_ = reinterpret_cast<float *>(malloc(size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(reinterpret_cast<char *>(weight_ptr_) + down_size, 0, size - down_size);
#ifdef ENABLE_AVX
  RowMajor2Col16Major(origin_weight_, weight_ptr_, output_channel, input_channel);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major(origin_weight_, weight_ptr_, output_channel, input_channel);
#else
  RowMajor2Col8Major(origin_weight_, weight_ptr_, output_channel, input_channel);
#endif
  return RET_OK;
}

int Convolution1x1CPUKernel::InitConv1x1Param() {
  if ((matmul_param_->row_ > (row_tile_ * op_parameter_->thread_num_)) && (matmul_param_->row_ > matmul_param_->col_)) {
    multi_thread_by_hw_ = true;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->row_, row_tile_));
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->row_, row_tile_), thread_count_) * row_tile_;
  } else {
    multi_thread_by_hw_ = false;
    thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(matmul_param_->col_, col_tile_));
    thread_stride_ = UP_DIV(UP_DIV(matmul_param_->col_, col_tile_), thread_count_) * col_tile_;
  }

  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<float *>(malloc(matmul_param_->row_ * matmul_param_->deep_ * sizeof(float)));
    if (input_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Conv1x1 Malloc input_ptr_ error!";
      return RET_MEMORY_FAILED;
    }
    memset(input_ptr_, 0, matmul_param_->row_ * matmul_param_->deep_ * sizeof(float));
  }

  return RET_OK;
}

int Convolution1x1CPUKernel::Init() {
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
  matmul_param_ = new (std::nothrow) MatMulParameter;
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  int error_code = InitConv1x1BiasWeight();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1 init weight and bias failed.";
    return error_code;
  }
  return RET_OK;
}

void Convolution1x1CPUKernel::PackMatmulInput(const float *src_ptr, float *dst_ptr, int row, int col) {
#if ENABLE_AVX
  RowMajor2Col6Major(src_ptr, dst_ptr, row, col);
#elif defined(ENABLE_SSE)
  RowMajor2Col4Major(src_ptr, dst_ptr, row, col);
#else
  RowMajor2Col12Major(src_ptr, dst_ptr, row, col);
#endif
}

int Convolution1x1CPUKernel::DoConv1x1(int task_id) {
  int res_stride = matmul_param_->col_ - task_id * thread_stride_;
  int cur_oc = MSMIN(thread_stride_, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto bias = (bias_data_ == nullptr) ? nullptr : reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id;
  MatMulOpt(pack_input_, weight_ptr_ + task_id * thread_stride_ * matmul_param_->deep_,
            output_ptr_ + task_id * thread_stride_, bias, matmul_param_->act_type_, matmul_param_->deep_,
            matmul_param_->row_, cur_oc, matmul_param_->col_, OutType_Nhwc);
  return RET_OK;
}

int Convolution1x1Run(void *cdata, int task_id) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoConv1x1(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::DoConv1x1Hw(int task_id) {
  int res_stride = matmul_param_->row_ - task_id * thread_stride_;
  int cur_hw_ = MSMIN(thread_stride_, res_stride);
  if (cur_hw_ <= 0) {
    return RET_OK;
  }

  float *thread_input_ptr = input_ptr_ + task_id * thread_stride_ * matmul_param_->deep_;
  float *thread_pack_input = pack_input_ + task_id * row_tile_ * matmul_param_->deep_;
  float *thread_output_ptr = output_ptr_ + task_id * thread_stride_ * matmul_param_->col_;
  float *cur_intput = thread_input_ptr;
  float *cur_output = thread_output_ptr;
  for (int i = 0; i < cur_hw_; i += row_tile_) {
    int cur_rows = (cur_hw_ - i >= row_tile_) ? row_tile_ : (cur_hw_ - i);
    PackMatmulInput(cur_intput, thread_pack_input, cur_rows, matmul_param_->deep_);
    MatMulOpt(thread_pack_input, weight_ptr_, cur_output, reinterpret_cast<float *>(bias_data_),
              matmul_param_->act_type_, matmul_param_->deep_, cur_rows, matmul_param_->col_, matmul_param_->col_,
              OutType_Nhwc);
    cur_intput += row_tile_ * matmul_param_->deep_;
    cur_output += row_tile_ * matmul_param_->col_;
  }

  return RET_OK;
}

int Convolution1x1RunHw(void *cdata, int task_id) {
  auto conv1x1 = reinterpret_cast<Convolution1x1CPUKernel *>(cdata);
  auto error_code = conv1x1->DoConv1x1Hw(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution1x1Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution1x1CPUKernel::Run() {
  auto src_in = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto src_out = reinterpret_cast<float *>(out_tensors_[0]->MutableData());
  int pack_input_size = multi_thread_by_hw_ ? (thread_count_ * row_tile_ * matmul_param_->deep_)
                                            : (matmul_param_->row_align_ * matmul_param_->deep_);
  pack_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(pack_input_size * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc pack_input_ error!";
    return RET_MEMORY_FAILED;
  }
  if (IsTrain() && is_trainable()) {
    PackWeight();
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    output_ptr_ = src_out + batch_index * matmul_param_->row_ * matmul_param_->col_;
    auto tmp_in = src_in + batch_index * conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_;
    if (pre_trans_input_) {
      Conv1x1InputPack(tmp_in, input_ptr_, conv_param_, sizeof(float));
    } else {
      input_ptr_ = tmp_in;
    }

    if (multi_thread_by_hw_) {
      ParallelLaunch(this->context_->thread_pool_, Convolution1x1RunHw, this, thread_count_);
    } else {
      PackMatmulInput(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
      ParallelLaunch(this->context_->thread_pool_, Convolution1x1Run, this, thread_count_);
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
  auto output_channel = filter_tensor->Batch();

  int size = input_channel * UP_ROUND(output_channel, col_tile_) * sizeof(float);
  int down_size = input_channel * DOWN_DIV(output_channel, col_tile_) * col_tile_ * sizeof(float);
  memset(reinterpret_cast<char *>(weight_ptr_) + down_size, 0, size - down_size);
#ifdef ENABLE_AVX
  RowMajor2Col16Major(reinterpret_cast<float *>(filter_tensor->MutableData()), weight_ptr_, output_channel,
                      input_channel);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major(reinterpret_cast<float *>(filter_tensor->MutableData()), weight_ptr_, output_channel,
                     input_channel);
#else
  RowMajor2Col8Major(reinterpret_cast<float *>(filter_tensor->MutableData()), weight_ptr_, output_channel,
                     input_channel);
#endif
}

int Convolution1x1CPUKernel::Eval() {
  LiteKernel::Eval();
  if (is_trainable()) {
    PackWeight();
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
