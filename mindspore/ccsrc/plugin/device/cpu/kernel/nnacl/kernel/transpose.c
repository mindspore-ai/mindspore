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

#include "nnacl/kernel/transpose.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/transpose_fp16.h"
#endif

/* opt perm: { 0, 2, 1 } */
#define OPT_PERM_0 0
#define OPT_PERM_1 2
#define OPT_PERM_2 1

int TransposeComputeinMultiThread(TransposeStruct *transpose, int task_id) {
  void *in = transpose->base_.in_[FIRST_INPUT]->data_;
  void *out = transpose->base_.out_[OUTPUT_INDEX]->data_;

  if (transpose->opt_run_) {
    transpose->nhwc2nchw_(in, out, transpose->opt_perm_[FIRST_INPUT], transpose->opt_perm_[SECOND_INPUT],
                          transpose->opt_perm_[THIRD_INPUT], task_id, transpose->base_.thread_nr_);
  } else {
    transpose->optimize_(in, out, transpose->out_shape_, transpose->perm_, transpose->strides_, transpose->out_strides_,
                         transpose->num_axes_, task_id, transpose->base_.thread_nr_);
  }
  return NNACL_OK;
}

int TransposeComputeinSingleThread(TransposeStruct *transpose) {
  if (transpose->opt_run_ || transpose->num_axes_ > DIMENSION_6D) {
    return TransposeComputeinMultiThread(transpose, 0);
  }

  void *in = transpose->base_.in_[FIRST_INPUT]->data_;
  void *out = transpose->base_.out_[OUTPUT_INDEX]->data_;
  return transpose->compute_(in, out, transpose->out_shape_, transpose->perm_, transpose->strides_,
                             transpose->out_strides_, transpose->data_num_, transpose->num_axes_);
}

int ResetTransposeStatus(TransposeStruct *transpose) {
  transpose->num_axes_ = 0;
  if (transpose->base_.in_size_ == C2NUM) {
    transpose->num_axes_ = GetElementNum(transpose->base_.in_[SECOND_INPUT]);
    transpose->perm_size_ = transpose->base_.in_[SECOND_INPUT]->shape_[0];
  }

  TensorC *in_tensor = transpose->base_.in_[FIRST_INPUT];
  if (in_tensor->shape_size_ > MAX_TRANSPOSE_DIM_SIZE) {
    return NNACL_TRANSPOSE_INSHAPE_OUT_OF_RANGE;
  }

  int trans_nd[MAX_TRANSPOSE_DIM_SIZE] = {0, 2, 1};
  int *perm_data;
  if ((int)in_tensor->shape_size_ != transpose->num_axes_) {
    perm_data = trans_nd;
    if (in_tensor->shape_size_ == Num3 && transpose->num_axes_ == Num4) {
      transpose->num_axes_ = Num3;
    }
    if (transpose->num_axes_ == 0) {
      for (size_t i = 0; i < in_tensor->shape_size_; ++i) {
        trans_nd[i] = (int)in_tensor->shape_size_ - 1 - (int)i;
      }
      transpose->num_axes_ = (int)in_tensor->shape_size_;
    }
  } else {
    NNACL_CHECK_TRUE_RET(transpose->base_.in_size_ == TWO_TENSOR, NNACL_TRANSPOSE_INPUT_TENSOR_NUM_INVALID);
    TensorC *perm_tensor = transpose->base_.in_[SECOND_INPUT];
    if (perm_tensor->data_type_ != kNumberTypeInt32) {
      return NNACL_TRANSPOSE_PERM_TENSOR_INVALID;
    }
    perm_data = (int *)(perm_tensor->data_);
    NNACL_CHECK_NULL_RETURN_ERR(perm_data);
    int ele_num = GetElementNum(perm_tensor);
    for (int i = 0; i < ele_num; i++) {
      for (int j = 0; j < ele_num; j++) {
        if (i == perm_data[j]) {
          break;
        }
        if (j == ele_num - 1) {
          return NNACL_TRANSPOSE_PERM_TENSOR_VALUE_INVALID;
        }
      }
    }
  }

  NNACL_CHECK_TRUE_RET(transpose->num_axes_ <= MAX_TRANSPOSE_DIM_SIZE, NNACL_TRANSPOSE_PERM_DIMS_INVALID);
  for (int i = 0; i < transpose->num_axes_; ++i) {
    transpose->perm_[i] = perm_data[i];
  }
  return NNACL_OK;
}

void TransposeFreeSegments(int **segments, int segments_size) {
  for (int i = 0; i < segments_size; i++) {
    if (segments[i] != NULL) {
      free(segments[i]);
      segments[i] = NULL;
    }
  }
}

int TransposeOptimizeShape(TransposeStruct *transpose) {
  TensorC *in_tensor = transpose->base_.in_[FIRST_INPUT];
  int *in_shape = in_tensor->shape_;

  // first step, delete dimension where value is 1.
  int in_shape_temp[MAX_TRANSPOSE_DIM_SIZE] = {0};
  int in_shape_temp_size = 0;
  int perm_diff[MAX_TRANSPOSE_DIM_SIZE] = {0};
  for (size_t i = 0; i < in_tensor->shape_size_; ++i) {
    if (in_shape[i] != 1) {
      in_shape_temp[in_shape_temp_size++] = in_shape[i];
      continue;
    }
    for (size_t j = 0; j < in_tensor->shape_size_; ++j) {
      if (transpose->perm_[j] < (int)(i)) {
        continue;
      }
      if (transpose->perm_[j] == (int)(i)) {
        perm_diff[j] = (int)(i) + 1;
      } else {
        perm_diff[j] += 1;
      }
    }
  }

  int perm_temp[MAX_TRANSPOSE_DIM_SIZE] = {0};
  int perm_temp_size = 0;
  for (size_t i = 0; i < in_tensor->shape_size_; ++i) {
    int diff = transpose->perm_[i] - perm_diff[i];
    if (diff < 0) {
      continue;
    }
    perm_temp[perm_temp_size++] = diff;
  }

  NNACL_CHECK_TRUE_RET(in_shape_temp_size == perm_temp_size, NNACL_TRANSPOSE_PERM_DELETE_DIMENSION_FAILED);

  // second step, fuse continuous dimension.;
  int axis_num = in_shape_temp_size;
  int *segments[MAX_TRANSPOSE_DIM_SIZE];
  int segment_sizes[MAX_TRANSPOSE_DIM_SIZE];
  int segments_size = 0;
  for (int i = 0; i < axis_num;) {
    int segment[MAX_TRANSPOSE_DIM_SIZE];
    int segment_size = 0;
    segment[segment_size++] = perm_temp[i];
    ++i;
    for (; i < axis_num; ++i) {
      if (perm_temp[i] - 1 != perm_temp[i - 1]) {
        break;
      }
      segment[segment_size++] = perm_temp[i];
    }

    segments[segments_size] = malloc(segment_size * sizeof(int));
    if (segments[segments_size] == NULL) {
      TransposeFreeSegments(segments, segments_size);
      return NNACL_NULL_PTR;
    }
    memcpy(segments[segments_size], segment, segment_size * sizeof(int));
    segment_sizes[segments_size] = segment_size;
    segments_size++;
  }

  transpose->in_shape_size_ = segments_size;
  transpose->perm_size_ = segments_size;
  for (int i = 0; i < segments_size; i++) {
    transpose->in_shape_[i] = 1;
    transpose->perm_[i] = 0;
  }
  for (int i = 0; i < segments_size; ++i) {
    for (int j = 0; j < segments_size; ++j) {
      transpose->perm_[i] += (segments[j][FIRST_INPUT] < segments[i][FIRST_INPUT] ? 1 : 0);
    }
    for (int k = 0; k < segment_sizes[i]; ++k) {
      transpose->in_shape_[transpose->perm_[i]] *= in_shape_temp[segments[i][k]];
    }
  }
  TransposeFreeSegments(segments, segments_size);
  return NNACL_OK;
}

void SetTransposeOptInfo(TransposeStruct *transpose) {
  // now perm is [1, 0] or [0, 2, 1]
  if (transpose->perm_size_ == C2NUM) {
    transpose->opt_perm_[FIRST_INPUT] = 1;
    transpose->opt_perm_[SECOND_INPUT] = transpose->in_shape_[FIRST_INPUT];
    transpose->opt_perm_[THIRD_INPUT] = transpose->in_shape_[transpose->in_shape_size_ - 1];
  } else {
    transpose->opt_perm_[FIRST_INPUT] = transpose->in_shape_[FIRST_INPUT];
    transpose->opt_perm_[SECOND_INPUT] = transpose->in_shape_[SECOND_INPUT];
    transpose->opt_perm_[THIRD_INPUT] = transpose->in_shape_[transpose->in_shape_size_ - 1];
  }
}

bool TransposeOpt(TransposeStruct *transpose) {
  if (transpose->perm_size_ == DIMENSION_2D) {
    return true;
  }
  if (transpose->perm_size_ == DIMENSION_3D && transpose->perm_[FIRST_INPUT] == OPT_PERM_0 &&
      transpose->perm_[SECOND_INPUT] == OPT_PERM_1 && transpose->perm_[THIRD_INPUT] == OPT_PERM_2) {
    return true;
  }
  return false;
}

int TransposeComputeOfflineInfo(TransposeStruct *transpose) {
  transpose->num_axes_ = transpose->in_shape_size_;
  NNACL_CHECK_TRUE_RET(transpose->num_axes_ >= DIMENSION_3D, NNACL_TRANSPOSE_INSHAPE_OUT_OF_RANGE);

  for (int i = 0; i < transpose->num_axes_; ++i) {
    transpose->out_shape_[i] = transpose->in_shape_[transpose->perm_[i]];
  }
  transpose->strides_[transpose->num_axes_ - 1] = 1;
  transpose->out_strides_[transpose->num_axes_ - 1] = 1;
  transpose->data_num_ = GetElementNum(transpose->base_.in_[FIRST_INPUT]);
  for (int i = transpose->num_axes_ - 2; i >= 0; i--) {
    transpose->strides_[i] = transpose->in_shape_[i + 1] * transpose->strides_[i + 1];
    transpose->out_strides_[i] = transpose->out_shape_[i + 1] * transpose->out_strides_[i + 1];
  }
  return NNACL_OK;
}

int TransposeCopyInputToOutput(TransposeStruct *transpose) {
  TensorC *in_tensor = transpose->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor->data_);
  TensorC *out_tensor = transpose->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor->data_);

  NNACL_CHECK_FALSE(GetSize(in_tensor) == 0, NNACL_TRANSPOSE_INPUT_TENSOR_VALUD_INVALID);
  if (in_tensor->data_ != out_tensor->data_) {
    (void)memcpy(out_tensor->data_, in_tensor->data_, GetSize(in_tensor));
  }
  return NNACL_OK;
}

int TransposeImpl(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  TransposeStruct *transpose = (TransposeStruct *)cdata;
  return TransposeComputeinMultiThread(transpose, task_id);
}

int TransposeCompute(struct KernelBase *self) {
  TransposeStruct *transpose = (TransposeStruct *)self;
  if (!transpose->is_valid_) {
    return TransposeCopyInputToOutput(transpose);
  }
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]->data_);
  if (self->thread_nr_ == 1) {
    return TransposeComputeinSingleThread(transpose);
  }
  return self->env_->ParallelLaunch(self->env_->thread_pool_, TransposeImpl, self, self->thread_nr_);
}

int TransposeResize(struct KernelBase *self) {
  TransposeStruct *transpose = (TransposeStruct *)self;
  int ret = ResetTransposeStatus(transpose);
  if (ret != NNACL_OK) {
    return ret;
  }
  transpose->is_valid_ = (int)transpose->base_.in_[FIRST_INPUT]->shape_size_ == transpose->num_axes_ &&
                         (int)transpose->base_.in_[FIRST_INPUT]->shape_size_ == transpose->perm_size_;
  if (!transpose->is_valid_) {
    return NNACL_OK;
  }

  ret = TransposeOptimizeShape(transpose);
  if (ret != NNACL_OK) {
    return ret;
  }

  transpose->is_valid_ = transpose->perm_size_ > DIMENSION_1D;
  if (!transpose->is_valid_) {
    return NNACL_OK;
  }

  transpose->opt_run_ = TransposeOpt(transpose);
  if (transpose->opt_run_) {
    SetTransposeOptInfo(transpose);
    return NNACL_OK;
  }

  ret = TransposeComputeOfflineInfo(transpose);
  if (ret != NNACL_OK) {
    return ret;
  }

  self->thread_nr_ = (!transpose->opt_run_ && transpose->num_axes_ <= DIMENSION_6D) ? 1 : self->thread_nr_;
  return NNACL_OK;
}

int TransposePrepare(struct KernelBase *self) {
  int ret = DefaultPrepare1In1Out(self);
  if (ret != NNACL_OK) {
    return ret;
  }
  TransposeStruct *transpose = (TransposeStruct *)self;
  TransposeParameter *param = (TransposeParameter *)transpose->base_.param_;
  if (param->perm_size_ > INT32_MAX) {
    return NNACL_TRANSPOSE_PERM_DIMS_INVALID;
  }
  transpose->perm_size_ = (int)param->perm_size_;
  for (int i = 0; i < transpose->perm_size_; i++) {
    transpose->perm_[i] = param->perm_[i];
  }
  return NNACL_OK;
}

KernelBase *CreateTranspose(OpParameter *param, int data_type) {
  TransposeStruct *transpose = (TransposeStruct *)malloc(sizeof(TransposeStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(transpose);
  transpose->nhwc2nchw_ = PackNHWCToNCHWFp32;
  transpose->optimize_ = TransposeDimsFp32;
  transpose->compute_ = DoTransposeFp32;
  transpose->base_.Release = DefaultRelease;
  transpose->base_.Prepare = TransposePrepare;
  transpose->base_.Resize = TransposeResize;
  transpose->base_.Compute = TransposeCompute;
  if (data_type == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    transpose->nhwc2nchw_ = PackNHWCToNCHWFp16;
    transpose->optimize_ = TransposeDimsFp16;
    transpose->compute_ = DoTransposeFp16;
#else
    free(transpose);
    return NULL;
#endif
  }
  return (KernelBase *)transpose;
}

REG_KERNEL_CREATOR(PrimType_Transpose, kNumberTypeFloat32, CreateTranspose)
REG_KERNEL_CREATOR(PrimType_Transpose, kNumberTypeFloat16, CreateTranspose)
REG_KERNEL_CREATOR(PrimType_Transpose, kNumberTypeInt32, CreateTranspose)
