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
#include "nnacl/kernel/matmul_f32_base.h"

void Matmul_InitShapeA(MatmulFp32Struct *matmul) {
  int *a_shape = matmul->base_.in_[kInputIndex]->shape_;
  int a_shape_size = matmul->base_.in_[kInputIndex]->shape_size_;
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(a_shape_size >= C2NUM);
  for (size_t i = 0; i < a_shape_size - C2NUM; ++i) {
    batch *= a_shape[i];
  }
  matmul->a_batch_ = batch;
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->row_ = param->a_transpose_ ? a_shape[a_shape_size - 1] : a_shape[a_shape_size - C2NUM];
  matmul->deep_ = param->a_transpose_ ? a_shape[a_shape_size - C2NUM] : a_shape[a_shape_size - 1];
}

void Matmul_InitShapeB(MatmulFp32Struct *matmul) {
  int *b_shape = matmul->base_.in_[kWeightIndex]->shape_;
  int b_shape_size = matmul->base_.in_[kWeightIndex]->shape_size_;
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(b_shape_size >= C2NUM);
  for (size_t i = 0; i < b_shape_size - C2NUM; ++i) {
    batch *= b_shape[i];
  }
  matmul->b_batch_ = batch;
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->col_ = param->b_transpose_ ? b_shape[b_shape_size - C2NUM] : b_shape[b_shape_size - 1];
  matmul->deep_ = param->b_transpose_ ? b_shape[b_shape_size - 1] : b_shape[b_shape_size - C2NUM];
}

int Matmul_InitBroadcastParams(MatmulFp32Struct *matmul) {
  TensorC *a = matmul->base_.in_[FIRST_INPUT];
  TensorC *b = matmul->base_.in_[SECOND_INPUT];

  size_t max_dim_size = MSMAX(a->shape_size_, b->shape_size_);
  max_dim_size = MSMAX(max_dim_size, COMM_SHAPE_SIZE);

  int a_shape[MAX_SHAPE_SIZE] = {0};
  int index = max_dim_size - 1;
  for (int i = a->shape_size_ - 1; i >= 0; i--) {
    a_shape[index--] = a->shape_[i];
  }
  for (; index >= 0;) {
    a_shape[index--] = 1;
  }

  int b_shape[MAX_SHAPE_SIZE] = {0};
  index = max_dim_size - 1;
  for (int i = b->shape_size_ - 1; i >= 0; i--) {
    b_shape[index--] = b->shape_[i];
  }
  for (; index >= 0;) {
    b_shape[index--] = 1;
  }

  int batch_sizes[MAX_SHAPE_SIZE] = {0};
  int a_batch_sizes[MAX_SHAPE_SIZE] = {0};
  int b_batch_sizes[MAX_SHAPE_SIZE] = {0};
  for (int i = max_dim_size - DIMENSION_3D; i >= 0; --i) {
    if (max_dim_size - DIMENSION_3D == i) {
      batch_sizes[i] = MSMAX(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_shape[i];
      b_batch_sizes[i] = b_shape[i];
    } else {
      batch_sizes[i] = batch_sizes[i + 1] * MSMAX(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_batch_sizes[i + 1] * a_shape[i];
      b_batch_sizes[i] = b_batch_sizes[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (size_t i = 0; i < max_dim_size - DIMENSION_2D; ++i) {
    int max_v = MSMAX(a_shape[i], b_shape[i]);
    int min_v = MSMIN(a_shape[i], b_shape[i]) > 0 ? MSMIN(a_shape[i], b_shape[i]) : 1;
    out_batch *= max_v;
    if ((max_v != min_v) && ((max_v % min_v) != 0)) {
      return NNACL_ERR;
    }
  }
  matmul->batch_ = out_batch;

  memset(matmul->a_offset_, 0, MAX_BATCH_SIZE * sizeof(int));
  memset(matmul->b_offset_, 0, MAX_BATCH_SIZE * sizeof(int));
  for (int i = 0; i < matmul->batch_; ++i) {
    int delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (size_t j = 0; j < max_dim_size - DIMENSION_2D; ++j) {
      if (j > 0) {
        delta = delta % batch_sizes[j];
      }
      if (j >= (MAX_SHAPE_SIZE - 1)) {
        return NNACL_ERR;
      }
      if (j < (max_dim_size - DIMENSION_3D)) {
        a_offset += (delta / batch_sizes[j + 1] * a_shape[j] / MSMAX(a_shape[j], b_shape[j])) * a_batch_sizes[j + 1];
        b_offset += (delta / batch_sizes[j + 1] * b_shape[j] / MSMAX(a_shape[j], b_shape[j])) * b_batch_sizes[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / MSMAX(a_shape[j], b_shape[j]));
        b_offset += (delta * b_shape[j] / MSMAX(a_shape[j], b_shape[j]));
      }
    }
    matmul->a_offset_[i] = a_offset;
    matmul->b_offset_[i] = b_offset;
  }
  return NNACL_OK;
}

int matmul_prepare(KernelBase *self) {
  MS_CHECK_FALSE(self->in_size_ < C2NUM, NNACL_ERR);
  MS_CHECK_FALSE(self->out_size_ < C1NUM, NNACL_ERR);

  MatmulFp32Struct *matmul = (MatmulFp32Struct *)self;
  if (matmul->a_const_ || matmul->infer_shape_) {
    Matmul_InitShapeA(matmul);
  }

  if (matmul->b_const_ || matmul->infer_shape_) {
    Matmul_InitShapeB(matmul);
  }

  return matmul_f32_prepare(self);
}

int matmul_resize(KernelBase *self) {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)self;
  Matmul_InitShapeA(matmul);
  Matmul_InitShapeB(matmul);

  int ret = Matmul_InitBroadcastParams(matmul);
  MS_CHECK_FALSE(ret != NNACL_OK, ret);
  return matmul_f32_resize(self);
}

KernelBase *CreateMatmul(OpParameter *param, int data_type) {
  KernelBase *kernel = NULL;
  if (data_type == kNumberTypeFloat32) {
    kernel = CreateMatmulFp32();
    kernel->prepare = matmul_prepare;
    kernel->resize = matmul_resize;
  }
  return kernel;
}

REG_KERNEL_CREATOR(PrimType_MatMulFusion, kNumberTypeFloat32, CreateMatmul);
