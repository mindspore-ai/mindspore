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

#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_sw_1x1.h"
#include "nnacl/kernel/matmul_f32_base.h"

int ConvSW1x1Prepare(ConvolutionSW1x1Struct *sw_1x1) {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)sw_1x1->matmul_;
  NNACL_CHECK_NULL_RETURN_ERR(matmul);

  if (sw_1x1->conv_.infershape_done_ || matmul->a_const_) {
    TensorC *input_tensor = sw_1x1->conv_.base_.in_[FIRST_INPUT];
    matmul->compute_.row_ = GetBatch(input_tensor) * GetHeight(input_tensor) * GetWidth(input_tensor);
    matmul->compute_.deep_ = GetChannel(input_tensor);
  }

  if (sw_1x1->conv_.infershape_done_ || matmul->b_const_) {
    TensorC *weight_tensor = sw_1x1->conv_.base_.in_[SECOND_INPUT];
    matmul->compute_.col_ = GetBatch(weight_tensor);
    matmul->compute_.deep_ = GetChannel(weight_tensor);
  }

  matmul->a_batch_ = 1;
  matmul->b_batch_ = 1;
  matmul->batch_ = 1;

  int ret = MatmulFP32Base_MallocBatchOffset(sw_1x1->matmul_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ((MatMulParameter *)matmul->base_.param_)->a_transpose_ = false;
  ((MatMulParameter *)matmul->base_.param_)->b_transpose_ = true;

  return sw_1x1->matmul_->base_.prepare(&sw_1x1->matmul_->base_);
}

int convolution_sw1x1_compute(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  sw_1x1->matmul_->base_.in_ = self->in_;
  sw_1x1->matmul_->base_.in_size_ = self->in_size_;
  sw_1x1->matmul_->base_.out_ = self->out_;
  sw_1x1->matmul_->base_.out_size_ = self->out_size_;
  sw_1x1->matmul_->base_.workspace_ = self->workspace_;
  return sw_1x1->matmul_->base_.compute(&sw_1x1->matmul_->base_);
}

int convolution_sw1x1_resize(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  MatMulParameter *matmul_param = (MatMulParameter *)sw_1x1->matmul_->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(matmul_param);

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  matmul_param->row_ = GetBatch(input_tensor) * GetHeight(input_tensor) * GetWidth(input_tensor);
  matmul_param->deep_ = GetChannel(input_tensor);

  TensorC *weight_tensor = self->in_[SECOND_INPUT];
  matmul_param->col_ = GetBatch(weight_tensor);

  sw_1x1->matmul_->base_.in_ = self->in_;
  sw_1x1->matmul_->base_.in_size_ = self->in_size_;
  sw_1x1->matmul_->base_.out_ = self->out_;
  sw_1x1->matmul_->base_.out_size_ = self->out_size_;
  sw_1x1->matmul_->base_.workspace_ = self->workspace_;
  return sw_1x1->matmul_->base_.resize(&sw_1x1->matmul_->base_);
}

int convolution_sw1x1_prepare(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  sw_1x1->matmul_->matrix_b_.origin_ptr_ = sw_1x1->conv_.origin_weight_;
  sw_1x1->matmul_->matrix_b_.has_origin_ = true;
  sw_1x1->matmul_->matrix_c_.origin_ptr_ = sw_1x1->conv_.origin_bias_;
  sw_1x1->matmul_->matrix_c_.has_origin_ = true;

  sw_1x1->matmul_->a_const_ = false;
  sw_1x1->matmul_->b_const_ = true;

  sw_1x1->matmul_->base_.in_ = self->in_;
  sw_1x1->matmul_->base_.in_size_ = self->in_size_;
  sw_1x1->matmul_->base_.out_ = self->out_;
  sw_1x1->matmul_->base_.out_size_ = self->out_size_;
  sw_1x1->matmul_->base_.workspace_ = self->workspace_;
  sw_1x1->matmul_->base_.train_session_ = self->train_session_;
  sw_1x1->matmul_->base_.thread_nr_ = self->thread_nr_;
  sw_1x1->matmul_->base_.env_ = self->env_;

  return ConvSW1x1Prepare(sw_1x1);
}

int convolution_sw1x1_release(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);

  MatmulFp32Base_FreeBatchOffset(sw_1x1->matmul_);

  if (sw_1x1->matmul_ != NULL) {
    if (sw_1x1->matmul_->base_.param_ != NULL) {
      free(sw_1x1->matmul_->base_.param_);
      sw_1x1->matmul_->base_.param_ = NULL;
    }
    free(sw_1x1->matmul_);
    sw_1x1->matmul_ = NULL;
  }

  ConvBaseRelease(&sw_1x1->conv_);
  return NNACL_OK;
}

ConvolutionBaseStruct *CreateConvolutionSW1x1(ConvParameter *conv_param) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)malloc(sizeof(ConvolutionSW1x1Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(sw_1x1);
  memset(sw_1x1, 0, sizeof(ConvolutionSW1x1Struct));

  sw_1x1->conv_.is_sharing_pack_ = false;
  sw_1x1->conv_.base_.compute = convolution_sw1x1_compute;
  sw_1x1->conv_.base_.resize = convolution_sw1x1_resize;
  sw_1x1->conv_.base_.prepare = convolution_sw1x1_prepare;
  sw_1x1->conv_.base_.release = convolution_sw1x1_release;

  MatMulParameter *matmul_param = (MatMulParameter *)malloc(sizeof(MatMulParameter));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul_param);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->output_h_, conv_param->output_w_, NULL);
  matmul_param->row_ = conv_param->output_h_ * conv_param->output_w_;
  matmul_param->col_ = conv_param->output_channel_;
  matmul_param->deep_ = conv_param->input_channel_;
  matmul_param->batch = conv_param->input_batch_;
  matmul_param->op_parameter_ = conv_param->op_parameter_;
  matmul_param->act_type_ = conv_param->act_type_;
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;

  KernelBase *matmul = CreateMatmulFp32();
  matmul->param_ = (OpParameter *)matmul_param;
  ((MatmulFp32Struct *)matmul)->is_sharing_pack_ = false;
  sw_1x1->matmul_ = (MatmulFp32Struct *)matmul;
  return (ConvolutionBaseStruct *)sw_1x1;
}
#endif
