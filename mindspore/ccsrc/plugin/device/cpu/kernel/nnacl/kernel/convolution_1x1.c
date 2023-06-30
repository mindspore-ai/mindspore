/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/convolution_1x1.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/fp32/matmul_fp32.h"

int Conv1x1Run(void *cdata, int task_id, float l, float r) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);
  MatMulParameter *matmul = &conv_1x1->matmul_param_;

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, conv_1x1->thread_stride_, NNACL_ERR);
  int total_thead_stride_ = task_id * conv_1x1->thread_stride_;
  int res_stride = matmul->col_ - total_thead_stride_;
  int cur_oc = MSMIN(conv_1x1->thread_stride_, res_stride);
  if (cur_oc <= 0) {
    return NNACL_OK;
  }

  TensorC *out_tensor = conv_1x1->conv_.base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  float *bias = conv_1x1->conv_.bias_data_ == NULL
                  ? NULL
                  : (float *)conv_1x1->conv_.bias_data_ + conv_1x1->thread_stride_ * task_id;
  float *weight = (float *)conv_1x1->conv_.packed_weight_ + total_thead_stride_ * matmul->deep_;

  if (out_tensor->format_ == Format_NC4HW4) {
    MatMulOpt(conv_1x1->pack_input_, weight, conv_1x1->output_ptr_ + total_thead_stride_ * matmul->row_, bias,
              matmul->act_type_, matmul->deep_, matmul->row_, cur_oc, matmul->row_, OutType_NC4HW4);
  } else {
    MatMulOpt(conv_1x1->pack_input_, weight, conv_1x1->output_ptr_ + total_thead_stride_, bias, matmul->act_type_,
              matmul->deep_, matmul->row_, cur_oc, matmul->col_, OutType_Nhwc);
  }
  return NNACL_OK;
}

void Conv1x1PackMatmulInput(const float *src_ptr, float *dst_ptr, int row, int col) {
#ifdef ENABLE_AVX
  RowMajor2Col6Major(src_ptr, dst_ptr, row, col);
#elif defined(ENABLE_SSE)
  RowMajor2Col4Major(src_ptr, dst_ptr, row, col);
#else
  RowMajor2Col12Major(src_ptr, dst_ptr, row, col);
#endif
}

int Conv1x1RunHw(void *cdata, int task_id, float l, float r) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);
  MatMulParameter *matmul = &conv_1x1->matmul_param_;
  TensorC *output_tensor = conv_1x1->conv_.base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, conv_1x1->thread_stride_, NNACL_ERR);
  int total_thead_stride_ = task_id * conv_1x1->thread_stride_;
  int res_stride = matmul->row_ - total_thead_stride_;
  int cur_hw_ = MSMIN(conv_1x1->thread_stride_, res_stride);
  if (cur_hw_ <= 0) {
    return NNACL_OK;
  }

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul->deep_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, conv_1x1->row_tile_, NNACL_ERR);
  int total_row_tile_ = task_id * conv_1x1->row_tile_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_row_tile_, matmul->deep_, NNACL_ERR);
  float *thread_input_ptr = conv_1x1->input_ptr_ + total_thead_stride_ * matmul->deep_;
  float *thread_pack_input = conv_1x1->pack_input_ + total_row_tile_ * matmul->deep_;
  float *thread_output_ptr = NULL;
  if (output_tensor->format_ != Format_NC4HW4) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, matmul->col_, NNACL_ERR);
    thread_output_ptr = conv_1x1->output_ptr_ + total_thead_stride_ * matmul->col_;
  } else {
    int col_min = MSMIN(matmul->col_, C4NUM);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, col_min, NNACL_ERR);
    thread_output_ptr = conv_1x1->output_ptr_ + total_thead_stride_ * col_min;
  }
  float *cur_intput = thread_input_ptr;
  float *cur_output = thread_output_ptr;
  float *bias = (float *)conv_1x1->conv_.bias_data_;
  for (int i = 0; i < cur_hw_; i += conv_1x1->row_tile_) {
    int cur_rows = (cur_hw_ - i >= conv_1x1->row_tile_) ? conv_1x1->row_tile_ : (cur_hw_ - i);
    Conv1x1PackMatmulInput(cur_intput, thread_pack_input, cur_rows, matmul->deep_);
    if (output_tensor->format_ == Format_NC4HW4) {
      MatMulOpt(thread_pack_input, (float *)conv_1x1->conv_.packed_weight_, cur_output, bias, matmul->act_type_,
                matmul->deep_, cur_rows, matmul->col_, matmul->row_, OutType_NC4HW4);
      cur_output += conv_1x1->row_tile_ * MSMIN(matmul->col_, C4NUM);
    } else {
      MatMulOpt(thread_pack_input, (float *)conv_1x1->conv_.packed_weight_, cur_output, bias, matmul->act_type_,
                matmul->deep_, cur_rows, matmul->col_, matmul->col_, OutType_Nhwc);
      cur_output += conv_1x1->row_tile_ * matmul->col_;
    }
    cur_intput += conv_1x1->row_tile_ * matmul->deep_;
  }

  return NNACL_OK;
}

void Conv1x1PackWeight(ConvolutionBaseStruct *conv) {
  TensorC *filter_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(filter_tensor);
  ConvComputeParam *compute = &conv->compute_;
  NNACL_CHECK_NULL_RETURN_VOID(compute);

  if (compute->in_c_ <= 0 || compute->out_c_ <= 0) {
    return;
  }

  void *origin_weight = conv->base_.train_session_ ? filter_tensor->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);

#ifdef ENABLE_AVX
  RowMajor2Col16Major((float *)origin_weight, (float *)conv->packed_weight_, compute->out_c_, compute->in_c_);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major((float *)origin_weight, (float *)conv->packed_weight_, compute->out_c_, compute->in_c_);
#else
  RowMajor2Col8Major((float *)origin_weight, (float *)conv->packed_weight_, compute->out_c_, compute->in_c_);
#endif
}

int Conv1x1MallocWeightBiasData(ConvolutionBaseStruct *conv) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);

  int size = conv->compute_.in_c_ * UP_ROUND(conv->compute_.out_c_, conv_1x1->col_tile_) * sizeof(float);
  if (!conv->base_.train_session_) {
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  if (conv->base_.in_size_ == THREE_TENSOR) {
    size = UP_ROUND(conv->compute_.out_c_, conv_1x1->col_tile_) * sizeof(float);
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
    memset(conv->bias_data_, 0, size);
  }
  return NNACL_OK;
}

void Conv1x1FreeTmpBuffer(Convolution1x1Struct *conv_1x1) {
  if (conv_1x1->pre_trans_input_ && conv_1x1->input_ptr_ != NULL) {
    conv_1x1->conv_.base_.env_->Free(conv_1x1->conv_.base_.env_->allocator_, conv_1x1->input_ptr_);
    conv_1x1->input_ptr_ = NULL;
  }
  return;
}

int InitConv1x1MatmulParam(Convolution1x1Struct *conv_1x1) {
  ConvParameter *conv_param = (ConvParameter *)conv_1x1->conv_.base_.param_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->output_h_, conv_param->output_w_, NNACL_ERR);
  conv_1x1->matmul_param_.row_ = conv_param->output_h_ * conv_param->output_w_;
  conv_1x1->matmul_param_.col_ = conv_param->output_channel_;
  conv_1x1->matmul_param_.deep_ = conv_param->input_channel_;
  conv_1x1->matmul_param_.row_align_ = UP_ROUND(conv_1x1->matmul_param_.row_, conv_1x1->row_tile_);
  conv_1x1->matmul_param_.col_align_ = UP_ROUND(conv_1x1->matmul_param_.col_, conv_1x1->col_tile_);
  conv_1x1->matmul_param_.act_type_ = conv_param->act_type_;
  return NNACL_OK;
}

int InitConv1x1Param(Convolution1x1Struct *conv_1x1) {
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->row_tile_, conv_1x1->conv_.base_.thread_nr_, NNACL_ERR);
  if ((conv_1x1->matmul_param_.row_ > (conv_1x1->row_tile_ * conv_1x1->conv_.base_.thread_nr_)) &&
      (conv_1x1->matmul_param_.row_ > conv_1x1->matmul_param_.col_)) {
    conv_1x1->multi_thread_by_hw_ = true;
    conv_1x1->conv_.base_.thread_nr_ =
      MSMIN(conv_1x1->conv_.base_.thread_nr_, UP_DIV(conv_1x1->matmul_param_.row_, conv_1x1->row_tile_));
    if (conv_1x1->conv_.base_.thread_nr_ <= 0) {
      return NNACL_ERR;
    }
    conv_1x1->thread_stride_ =
      UP_DIV(UP_DIV(conv_1x1->matmul_param_.row_, conv_1x1->row_tile_), conv_1x1->conv_.base_.thread_nr_) *
      conv_1x1->row_tile_;
  } else {
    conv_1x1->multi_thread_by_hw_ = false;
    conv_1x1->conv_.base_.thread_nr_ =
      MSMIN(conv_1x1->conv_.base_.thread_nr_, UP_DIV(conv_1x1->matmul_param_.col_, conv_1x1->col_tile_));
    if (conv_1x1->conv_.base_.thread_nr_ <= 0) {
      return NNACL_ERR;
    }
    conv_1x1->thread_stride_ =
      UP_DIV(UP_DIV(conv_1x1->matmul_param_.col_, conv_1x1->col_tile_), conv_1x1->conv_.base_.thread_nr_) *
      conv_1x1->col_tile_;
  }

  ConvParameter *conv_param = (ConvParameter *)conv_1x1->conv_.base_.param_;
  conv_1x1->pre_trans_input_ =
    (conv_param->pad_u_ != 0 || conv_param->pad_l_ != 0 || conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1);
  if (conv_1x1->pre_trans_input_) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->matmul_param_.row_, conv_1x1->matmul_param_.deep_, NNACL_ERR);
    conv_1x1->input_ptr_ = (float *)(conv_1x1->conv_.base_.env_->Alloc(
      conv_1x1->conv_.base_.env_->allocator_,
      conv_1x1->matmul_param_.row_ * conv_1x1->matmul_param_.deep_ * sizeof(float)));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_1x1->input_ptr_);
    memset(conv_1x1->input_ptr_, 0, conv_1x1->matmul_param_.row_ * conv_1x1->matmul_param_.deep_ * sizeof(float));
  }

  return NNACL_OK;
}

int Convolution1x1Resize(KernelBase *self) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);

  Conv1x1FreeTmpBuffer(conv_1x1);
  int error_code = ConvBasePrepare(&conv_1x1->conv_);
  if (error_code != NNACL_OK) {
    return error_code;
  }

  error_code = InitConv1x1MatmulParam(conv_1x1);
  if (error_code != NNACL_OK) {
    return error_code;
  }

  error_code = InitConv1x1Param(conv_1x1);
  if (error_code != NNACL_OK) {
    return error_code;
  }

  return NNACL_OK;
}

int Convolution1x1Prepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);

#ifdef ENABLE_AVX
  conv_1x1->row_tile_ = C6NUM;
  conv_1x1->col_tile_ = C16NUM;
#elif defined(ENABLE_SSE)
  conv_1x1->row_tile_ = C4NUM;
  conv_1x1->col_tile_ = C8NUM;
#elif defined(ENABLE_ARM32)
  conv_1x1->row_tile_ = C12NUM;
  conv_1x1->col_tile_ = C4NUM;
#else
  conv_1x1->row_tile_ = C12NUM;
  conv_1x1->col_tile_ = C8NUM;
#endif

  if (self->train_session_) {
    int output_tile_size = UP_ROUND(conv_1x1->conv_.compute_.out_c_, conv_1x1->col_tile_);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->conv_.compute_.in_c_, output_tile_size, NNACL_ERR);
    size_t size = conv_1x1->conv_.compute_.in_c_ * output_tile_size * sizeof(float);
    conv_1x1->conv_.base_.work_size_ = size;
  }

  int error_code = ConvBaseInitConvWeightBias(&conv_1x1->conv_);
  if (error_code != NNACL_OK) {
    return error_code;
  }
  return NNACL_OK;
}

int Convolution1x1Release(KernelBase *self) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);
  Conv1x1FreeTmpBuffer(conv_1x1);
  ConvBaseRelease(&conv_1x1->conv_);
  return NNACL_OK;
}

int Convolution1x1Compute(KernelBase *self) {
  Convolution1x1Struct *conv_1x1 = (Convolution1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_1x1);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);

  float *src_in = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_in);
  float *src_out = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_out);

  int pack_input_size = 0;
  if (conv_1x1->multi_thread_by_hw_) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->conv_.base_.thread_nr_, conv_1x1->row_tile_, NNACL_ERR);
    int total_row_tile_ = conv_1x1->conv_.base_.thread_nr_ * conv_1x1->row_tile_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_row_tile_, conv_1x1->matmul_param_.deep_, NNACL_ERR);
    pack_input_size = total_row_tile_ * conv_1x1->matmul_param_.deep_;
  } else {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->matmul_param_.row_align_, conv_1x1->matmul_param_.deep_, NNACL_ERR);
    pack_input_size = conv_1x1->matmul_param_.row_align_ * conv_1x1->matmul_param_.deep_;
  }
  conv_1x1->pack_input_ =
    (float *)conv_1x1->conv_.base_.env_->Alloc(conv_1x1->conv_.base_.env_->allocator_, pack_input_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_1x1->pack_input_);

  int ret = ConvBaseRepackWeight(&conv_1x1->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_1x1->matmul_param_.row_, conv_1x1->matmul_param_.col_, NNACL_ERR);
  int matmul_size = conv_1x1->matmul_param_.row_ * conv_1x1->matmul_param_.col_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->input_batch_ - 1, matmul_size, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->input_h_, conv_param->input_w_, NNACL_ERR);
  int conv_input_hw = conv_param->input_h_ * conv_param->input_w_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_input_hw, conv_param->input_channel_, NNACL_ERR);
  int conv_input_bhw = conv_input_hw * conv_param->input_channel_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->input_batch_ - 1, conv_input_bhw, NNACL_ERR);
  for (int batch_index = 0; batch_index < conv_param->input_batch_; batch_index++) {
    conv_1x1->output_ptr_ = src_out + batch_index * matmul_size;
    float *tmp_in = src_in + batch_index * conv_input_bhw;
    if (conv_1x1->pre_trans_input_) {
      Conv1x1InputPack(tmp_in, conv_1x1->input_ptr_, conv_param, sizeof(float));
    } else {
      conv_1x1->input_ptr_ = tmp_in;
    }
    if (conv_1x1->multi_thread_by_hw_) {
      ret = self->env_->ParallelLaunch(self->env_->thread_pool_, Conv1x1RunHw, self, self->thread_nr_);
    } else {
      Conv1x1PackMatmulInput(conv_1x1->input_ptr_, conv_1x1->pack_input_, conv_1x1->matmul_param_.row_,
                             conv_1x1->matmul_param_.deep_);
      ret = self->env_->ParallelLaunch(self->env_->thread_pool_, Conv1x1Run, self, self->thread_nr_);
    }
    if (ret != NNACL_OK) {
      break;
    }
  }

  if (conv_1x1->pack_input_ != NULL) {
    self->env_->Free(self->env_->allocator_, conv_1x1->pack_input_);
    conv_1x1->pack_input_ = NULL;
  }
  return ret;
}

ConvolutionBaseStruct *CreateConvolution1x1(ConvParameter *conv_param) {
  Convolution1x1Struct *conv1x1 = (Convolution1x1Struct *)malloc(sizeof(Convolution1x1Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv1x1);
  memset(conv1x1, 0, sizeof(Convolution1x1Struct));

  conv1x1->conv_.is_sharing_pack_ = false;
  conv1x1->conv_.malloc_weight_bias_ = Conv1x1MallocWeightBiasData;
  conv1x1->conv_.pack_weight_ = Conv1x1PackWeight;

  conv1x1->conv_.base_.Resize = Convolution1x1Resize;
  conv1x1->conv_.base_.Prepare = Convolution1x1Prepare;
  conv1x1->conv_.base_.Release = Convolution1x1Release;
  conv1x1->conv_.base_.Compute = Convolution1x1Compute;

  return (ConvolutionBaseStruct *)conv1x1;
}
