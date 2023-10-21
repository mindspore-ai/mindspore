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
#include "nnacl/kernel/matmul_base.h"
#include "nnacl/kernel/matmul_create.h"

int MatmulConv1x1Prelare(ConvolutionSW1x1Struct *sw_1x1) {
  sw_1x1->matmul_->batch_ = 1;
  sw_1x1->matmul_->a_batch_ = 1;
  sw_1x1->matmul_->b_batch_ = 1;

  sw_1x1->matmul_->compute_.deep_ = sw_1x1->conv_.compute_.in_c_;
  sw_1x1->matmul_->compute_.col_ = sw_1x1->conv_.compute_.out_c_;
  sw_1x1->matmul_->compute_.row_ = sw_1x1->conv_.compute_.in_hw_ * sw_1x1->conv_.compute_.in_n_;

  return sw_1x1->matmul_->base_.Prepare(&sw_1x1->matmul_->base_);
}

int MatmulConv1x1Resize(ConvolutionSW1x1Struct *sw_1x1) {
  sw_1x1->matmul_->compute_.deep_ = sw_1x1->conv_.compute_.in_c_;
  sw_1x1->matmul_->compute_.col_ = sw_1x1->conv_.compute_.out_c_;
  sw_1x1->matmul_->compute_.row_ = sw_1x1->conv_.compute_.in_hw_ * sw_1x1->conv_.compute_.in_n_;

  MatmulBaseFreeBatchOffset(sw_1x1->matmul_);
  int ret = MatmulBaseMallocBatchOffset(sw_1x1->matmul_);
  if (ret != NNACL_OK) {
    return ret;
  }

  return sw_1x1->matmul_->base_.Resize(&sw_1x1->matmul_->base_);
}

void UpdateTensorInfo(KernelBase *self, ConvolutionSW1x1Struct *sw_1x1) {
  sw_1x1->matmul_->base_.in_ = self->in_;
  sw_1x1->matmul_->base_.in_size_ = self->in_size_;
  sw_1x1->matmul_->base_.out_ = self->out_;
  sw_1x1->matmul_->base_.out_size_ = self->out_size_;
  sw_1x1->matmul_->base_.workspace_ = self->workspace_;
}

int ConvolutionSW1x1Compute(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  UpdateTensorInfo(self, sw_1x1);
  return sw_1x1->matmul_->base_.Compute(&sw_1x1->matmul_->base_);
}

int ConvolutionSW1x1Resize(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  UpdateTensorInfo(self, sw_1x1);
  return MatmulConv1x1Resize(sw_1x1);
}

int ConvolutionSW1x1Prepare(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1->matmul_);

  sw_1x1->matmul_->matrix_b_.origin_ptr_ = sw_1x1->conv_.origin_weight_;
  sw_1x1->matmul_->matrix_b_.origin_need_free_ = false;
  sw_1x1->matmul_->matrix_c_.origin_ptr_ = sw_1x1->conv_.origin_bias_;
  sw_1x1->matmul_->matrix_c_.origin_need_free_ = false;

  sw_1x1->matmul_->infer_shape_ = sw_1x1->conv_.infershape_done_;
  sw_1x1->matmul_->base_.train_session_ = self->train_session_;
  sw_1x1->matmul_->base_.thread_nr_ = self->thread_nr_;
  sw_1x1->matmul_->base_.env_ = self->env_;

  UpdateTensorInfo(self, sw_1x1);
  return MatmulConv1x1Prelare(sw_1x1);
}

int ConvolutionSW1x1Release(KernelBase *self) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(sw_1x1);

  if (sw_1x1->matmul_ != NULL) {
    sw_1x1->matmul_->matrix_b_.origin_ptr_ = NULL;
    sw_1x1->matmul_->matrix_c_.origin_ptr_ = NULL;

    (void)sw_1x1->matmul_->base_.Release(&sw_1x1->matmul_->base_);

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

ConvolutionBaseStruct *CreateConvolutionSW1x1(ConvParameter *conv_param, bool input_const, bool weight_const) {
  ConvolutionSW1x1Struct *sw_1x1 = (ConvolutionSW1x1Struct *)malloc(sizeof(ConvolutionSW1x1Struct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(sw_1x1);
  memset(sw_1x1, 0, sizeof(ConvolutionSW1x1Struct));

  sw_1x1->conv_.is_sharing_pack_ = false;
  sw_1x1->conv_.base_.Compute = ConvolutionSW1x1Compute;
  sw_1x1->conv_.base_.Resize = ConvolutionSW1x1Resize;
  sw_1x1->conv_.base_.Prepare = ConvolutionSW1x1Prepare;
  sw_1x1->conv_.base_.Release = ConvolutionSW1x1Release;

  MatMulParameter *matmul_param = (MatMulParameter *)malloc(sizeof(MatMulParameter));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul_param);
  matmul_param->op_parameter_ = conv_param->op_parameter_;
  matmul_param->act_type_ = conv_param->act_type_;
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;

  KernelBase *matmul = CreateMatmulKernel();
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(matmul);
  matmul->param_ = (OpParameter *)matmul_param;
  ((MatmulStruct *)matmul)->is_sharing_pack_ = false;
  ((MatmulStruct *)matmul)->a_const_ = input_const;
  ((MatmulStruct *)matmul)->b_const_ = weight_const;
  sw_1x1->matmul_ = (MatmulStruct *)matmul;
  return (ConvolutionBaseStruct *)sw_1x1;
}
#endif
