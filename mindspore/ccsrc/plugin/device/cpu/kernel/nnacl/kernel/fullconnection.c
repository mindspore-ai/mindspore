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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/fullconnection.h"
#include "nnacl/kernel/matmul_base.h"
#include "nnacl/kernel/matmul_create.h"

int FullConnectionPrepare(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;

  NNACL_CHECK_FALSE(self->in_size_ < C2NUM, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < C1NUM, NNACL_ERR);

  if (matmul->a_const_ || matmul->infer_shape_) {
    int *a_shape = self->in_[FIRST_INPUT]->shape_;
    matmul->compute_.row_ = a_shape[0];
    matmul->compute_.deep_ = a_shape[1];
  }

  if (matmul->b_const_ || matmul->infer_shape_) {
    int *b_shape = self->in_[SECOND_INPUT]->shape_;
    matmul->compute_.col_ = b_shape[0];
    matmul->compute_.deep_ = b_shape[1];
  }

  matmul->batch_ = 1;
  matmul->a_batch_ = 1;
  matmul->b_batch_ = 1;

  MatMulParameter *param = (MatMulParameter *)matmul->base_.param_;
  param->a_transpose_ = false;
  param->b_transpose_ = true;

  int ret = MatmulBaseMallocBatchOffset(matmul);
  if (ret != NNACL_OK) {
    return ret;
  }

  return MatmulBasePrepare(self);
}

int FullConnectionResize(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;
  NNACL_CHECK_TRUE_RET(self->out_[0]->shape_size_ > 0, NNACL_ERR);

  int row = 1;
  for (size_t i = 0; i < self->out_[0]->shape_size_ - 1; ++i) {
    row *= (self->out_[OUTPUT_INDEX]->shape_)[i];
  }
  matmul->compute_.row_ = row;
  matmul->compute_.col_ = (self->out_[OUTPUT_INDEX]->shape_)[self->out_[0]->shape_size_ - 1];
  matmul->compute_.deep_ = self->in_[SECOND_INPUT]->shape_[SECOND_INPUT];

  return MatmulBaseResize(self);
}

KernelBase *CreateFullconnection(OpParameter *param, int data_type) {
  KernelBase *kernel = NULL;
  if (data_type == kNumberTypeFloat32) {
    kernel = CreateMatmulKernel();
    NNACL_MALLOC_CHECK_NULL_RETURN_NULL(kernel);
    kernel->Prepare = FullConnectionPrepare;
    kernel->Resize = FullConnectionResize;
  }
  return kernel;
}

REG_KERNEL_CREATOR(PrimType_FullConnection, kNumberTypeFloat32, CreateFullconnection);
