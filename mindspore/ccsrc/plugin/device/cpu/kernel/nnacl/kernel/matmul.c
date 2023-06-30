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

#include "nnacl/kernel/matmul.h"
#include "nnacl/kernel/matmul_base.h"
#include "nnacl/kernel/matmul_create.h"

void MatmulInitShapeA(MatmulStruct *matmul) {
  int *a_shape = matmul->base_.in_[kInputIndex]->shape_;
  size_t a_shape_size = matmul->base_.in_[kInputIndex]->shape_size_;
  int batch = 1;
  NNACL_CHECK_TRUE_RET_VOID(a_shape_size >= C2NUM);
  for (size_t i = 0; i < a_shape_size - C2NUM; ++i) {
    batch *= a_shape[i];
  }
  matmul->a_batch_ = batch;
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->compute_.row_ = param->a_transpose_ ? a_shape[a_shape_size - 1] : a_shape[a_shape_size - C2NUM];
  matmul->compute_.deep_ = param->a_transpose_ ? a_shape[a_shape_size - C2NUM] : a_shape[a_shape_size - 1];
}

void MatmulInitShapeB(MatmulStruct *matmul) {
  int *b_shape = matmul->base_.in_[kWeightIndex]->shape_;
  size_t b_shape_size = matmul->base_.in_[kWeightIndex]->shape_size_;
  int batch = 1;
  NNACL_CHECK_TRUE_RET_VOID(b_shape_size >= C2NUM);
  for (size_t i = 0; i < b_shape_size - C2NUM; ++i) {
    batch *= b_shape[i];
  }
  matmul->b_batch_ = batch;
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->compute_.col_ = param->b_transpose_ ? b_shape[b_shape_size - C2NUM] : b_shape[b_shape_size - 1];
  matmul->compute_.deep_ = param->b_transpose_ ? b_shape[b_shape_size - 1] : b_shape[b_shape_size - C2NUM];
}

int MatmulInitBroadcastParams(MatmulStruct *matmul) {
  TensorC *a = matmul->base_.in_[FIRST_INPUT];
  TensorC *b = matmul->base_.in_[SECOND_INPUT];

  int max_dim_size = (int)NNACL_MAX(a->shape_size_, b->shape_size_);
  max_dim_size = NNACL_MAX(max_dim_size, COMM_SHAPE_SIZE);

  int a_shape[MAX_SHAPE_SIZE] = {0};
  int index = max_dim_size - 1;
  for (int i = (int)a->shape_size_ - 1; i >= 0; i--) {
    a_shape[index--] = a->shape_[i];
  }
  for (; index >= 0;) {
    a_shape[index--] = 1;
  }

  int b_shape[MAX_SHAPE_SIZE] = {0};
  index = max_dim_size - 1;
  for (int i = (int)b->shape_size_ - 1; i >= 0; i--) {
    b_shape[index--] = b->shape_[i];
  }
  for (; index >= 0;) {
    b_shape[index--] = 1;
  }

  int batch_sizes[MAX_SHAPE_SIZE] = {0};
  int a_batch_sizes[MAX_SHAPE_SIZE] = {0};
  int b_batch_sizes[MAX_SHAPE_SIZE] = {0};
  for (int i = max_dim_size - Num3; i >= 0; --i) {
    if (max_dim_size - Num3 == i) {
      batch_sizes[i] = NNACL_MAX(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_shape[i];
      b_batch_sizes[i] = b_shape[i];
    } else {
      batch_sizes[i] = batch_sizes[i + 1] * NNACL_MAX(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_batch_sizes[i + 1] * a_shape[i];
      b_batch_sizes[i] = b_batch_sizes[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (int i = 0; i < max_dim_size - Num2; ++i) {
    int max_v = NNACL_MAX(a_shape[i], b_shape[i]);
    int min_v = NNACL_MIN(a_shape[i], b_shape[i]) > 0 ? NNACL_MIN(a_shape[i], b_shape[i]) : 1;
    out_batch *= max_v;
    if ((max_v != min_v) && ((max_v % min_v) != 0)) {
      return NNACL_ERR;
    }
  }
  matmul->batch_ = out_batch;

  MatmulBaseFreeBatchOffset(matmul);
  int ret = MatmulBaseMallocBatchOffset(matmul);
  if (ret != NNACL_OK) {
    return ret;
  }

  for (int i = 0; i < matmul->batch_; ++i) {
    int delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (int j = 0; j < max_dim_size - Num2; ++j) {
      if (j > 0) {
        delta = delta % batch_sizes[j];
      }
      if (j >= (MAX_SHAPE_SIZE - 1)) {
        return NNACL_ERR;
      }
      if (j < (max_dim_size - Num3)) {
        a_offset +=
          (delta / batch_sizes[j + 1] * a_shape[j] / NNACL_MAX(a_shape[j], b_shape[j])) * a_batch_sizes[j + 1];
        b_offset +=
          (delta / batch_sizes[j + 1] * b_shape[j] / NNACL_MAX(a_shape[j], b_shape[j])) * b_batch_sizes[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / NNACL_MAX(a_shape[j], b_shape[j]));
        b_offset += (delta * b_shape[j] / NNACL_MAX(a_shape[j], b_shape[j]));
      }
    }
    matmul->a_offset_[i] = a_offset;
    matmul->b_offset_[i] = b_offset;
  }
  return NNACL_OK;
}

int MatmulPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < C2NUM, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < C1NUM, NNACL_ERR);

  MatmulStruct *matmul = (MatmulStruct *)self;
  if (matmul->a_const_ || matmul->infer_shape_) {
    MatmulInitShapeA(matmul);
  }

  if (matmul->b_const_ || matmul->infer_shape_) {
    MatmulInitShapeB(matmul);
  }

  return MatmulBasePrepare(self);
}

int MatmulResize(KernelBase *self) {
  MatmulStruct *matmul = (MatmulStruct *)self;
  MatmulInitShapeA(matmul);
  MatmulInitShapeB(matmul);

  int ret = MatmulInitBroadcastParams(matmul);
  NNACL_CHECK_FALSE(ret != NNACL_OK, ret);
  return MatmulBaseResize(self);
}

int MatmulRelease(KernelBase *self) {
  MatmulBaseFreeBatchOffset((MatmulStruct *)self);
  return MatmulBaseRelease(self);
}

KernelBase *CreateMatmul(OpParameter *param, int data_type) {
  KernelBase *kernel = NULL;
  if (data_type == kNumberTypeFloat32) {
    kernel = CreateMatmulKernel();
    NNACL_MALLOC_CHECK_NULL_RETURN_NULL(kernel);
    kernel->Prepare = MatmulPrepare;
    kernel->Resize = MatmulResize;
    kernel->Release = MatmulRelease;
  }
  return kernel;
}

REG_KERNEL_CREATOR(PrimType_MatMulFusion, kNumberTypeFloat32, CreateMatmul);
