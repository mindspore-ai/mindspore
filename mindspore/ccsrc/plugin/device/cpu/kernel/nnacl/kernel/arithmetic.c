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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either arithmeticress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/arithmetic.h"
#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/mul_fp32.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/arithmetic_fp16.h"
#endif

void InitArithmeticRunFunction(KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;

  ArithmeticFuncions fun_table[] = {
    {PrimType_MulFusion, ActType_Relu, ElementMulRelu, ElementMulReluInt, NULL, ElementOptMulRelu, ElementOptMulReluInt,
     NULL},
    {PrimType_MulFusion, ActType_Relu6, ElementMulRelu6, ElementMulRelu6Int, NULL, ElementOptMulRelu6,
     ElementOptMulRelu6Int, NULL},
    {PrimType_MulFusion, ActType_No, ElementMul, ElementMulInt, NULL, ElementOptMul, ElementOptMulInt, NULL},
    {PrimType_AddFusion, ActType_Relu, ElementAddRelu, NULL, NULL, ElementOptAddRelu, NULL, NULL},
    {PrimType_AddFusion, ActType_Relu6, ElementAddRelu6, NULL, NULL, ElementOptAddRelu6, NULL, NULL},
    {PrimType_AddFusion, ActType_No, ElementAdd, ElementAddInt, NULL, ElementOptAdd, ElementOptAddInt, NULL},
    {PrimType_SubFusion, ActType_Relu, ElementSubRelu, NULL, NULL, ElementOptSubRelu, NULL, NULL},
    {PrimType_SubFusion, ActType_Relu6, ElementSubRelu6, NULL, NULL, ElementOptSubRelu6, NULL, NULL},
    {PrimType_SubFusion, ActType_No, ElementSub, ElementSubInt, NULL, ElementOptSub, ElementOptSubInt, NULL},
    {PrimType_DivFusion, ActType_Relu, ElementDivRelu, NULL, NULL, ElementOptDivRelu, NULL, NULL},
    {PrimType_DivFusion, ActType_Relu6, ElementDivRelu6, NULL, NULL, ElementOptDivRelu6, NULL, NULL},
    {PrimType_DivFusion, ActType_No, ElementDiv, NULL, NULL, ElementOptDiv, ElementOptDivInt, NULL},
    {PrimType_RealDiv, ActType_Relu, ElementDivRelu, NULL, NULL, ElementOptDivRelu, NULL, NULL},
    {PrimType_RealDiv, ActType_Relu6, ElementDivRelu6, NULL, NULL, ElementOptDivRelu6, NULL, NULL},
    {PrimType_RealDiv, ActType_No, ElementDiv, NULL, NULL, ElementOptDiv, ElementOptDivInt, NULL},
    {PrimType_LogicalAnd, ActType_No, ElementLogicalAnd, ElementLogicalAndInt, ElementLogicalAndBool,
     ElementOptLogicalAnd, ElementOptLogicalAndInt, ElementOptLogicalAndBool},
    {PrimType_LogicalOr, ActType_No, ElementLogicalOr, NULL, ElementLogicalOrBool, NULL, NULL, ElementOptLogicalOrBool},
    {PrimType_Maximum, ActType_No, ElementMaximum, ElementMaximumInt, NULL, ElementOptMaximum, ElementOptMaximumInt,
     NULL},
    {PrimType_Minimum, ActType_No, ElementMinimum, ElementMinimumInt, NULL, ElementOptMinimum, ElementOptMinimumInt,
     NULL},
    {PrimType_FloorMod, ActType_No, ElementFloorMod, ElementFloorModInt, NULL, ElementOptFloorMod,
     ElementOptFloorModInt, NULL},
    {PrimType_FloorDiv, ActType_No, ElementFloorDiv, ElementFloorDivInt, NULL, ElementOptFloorDiv,
     ElementOptFloorDivInt, NULL},
    {PrimType_Mod, ActType_No, ElementMod, ElementModInt, NULL, ElementOptMod, ElementOptModInt, NULL},
    {PrimType_SquaredDifference, ActType_No, ElementSquaredDifference, NULL, NULL, ElementOptSquaredDifference, NULL,
     NULL}};

  size_t length = sizeof(fun_table) / sizeof(ArithmeticFuncions);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == arithmetic->primitive_type_ &&
        fun_table[i].activation_type_ == ((ArithmeticParameter *)(arithmetic->base_.param_))->activation_type_) {
      arithmetic->functions_ = fun_table[i];
      return;
    }
  }
}

int ArithmeticRelease(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  for (int i = 0; i < TWO_TENSOR; i++) {
    if (arithmetic->broadcast_buffer_[i] != NULL) {
      self->env_->Free(self->env_->allocator_, arithmetic->broadcast_buffer_[i]);
      arithmetic->broadcast_buffer_[i] = NULL;
    }
  }

  for (int i = 0; i < arithmetic->block_boundary_infos_size_; i++) {
    if (arithmetic->block_boundary_infos_[i].a_offset_ != NULL) {
      self->env_->Free(self->env_->allocator_, arithmetic->block_boundary_infos_[i].a_offset_);
      arithmetic->block_boundary_infos_[i].a_offset_ = NULL;
    }
    if (arithmetic->block_boundary_infos_[i].b_offset_ != NULL) {
      self->env_->Free(self->env_->allocator_, arithmetic->block_boundary_infos_[i].b_offset_);
      arithmetic->block_boundary_infos_[i].b_offset_ = NULL;
    }
  }
  arithmetic->block_boundary_infos_size_ = 0;

  if (arithmetic->a_matrix_.batch_post_sum_ != NULL) {
    self->env_->Free(self->env_->allocator_, arithmetic->a_matrix_.batch_post_sum_);
    arithmetic->a_matrix_.batch_post_sum_ = NULL;
  }

  if (arithmetic->b_matrix_.batch_post_sum_ != NULL) {
    self->env_->Free(self->env_->allocator_, arithmetic->b_matrix_.batch_post_sum_);
    arithmetic->b_matrix_.batch_post_sum_ = NULL;
  }

  if (arithmetic->c_matrix_.batch_post_sum_ != NULL) {
    self->env_->Free(self->env_->allocator_, arithmetic->c_matrix_.batch_post_sum_);
    arithmetic->c_matrix_.batch_post_sum_ = NULL;
  }
  return NNACL_OK;
}

void ArithmeticComputeOffset(ArithmeticStruct *arithmetic, int task_id) {
  ArithmeticBlockBoundaryInfo *block_info = &arithmetic->block_boundary_infos_[task_id];
  block_info->init_offset_ = true;

  int64_t b_start = block_info->batch_begin_;
  int64_t b_end = block_info->batch_end_;
  int64_t s_end = block_info->size_end_;
  if (s_end != 0) {
    ++b_end;
  }
  int offset_index = 0;
  for (; b_start < b_end; ++b_start) {
    int64_t delta = b_start;
    int64_t a_offset = 0;
    int64_t b_offset = 0;
    for (int j = 0; j <= arithmetic->batch_tail_dim_; ++j) {
      if (j > 0) {
        delta = delta % arithmetic->c_matrix_.batch_post_sum_[j];
      }
      if (j < arithmetic->batch_tail_dim_) {
        a_offset += (delta / arithmetic->c_matrix_.batch_post_sum_[j + 1] * arithmetic->a_matrix_.shape_[j] /
                     arithmetic->c_matrix_.shape_[j]) *
                    arithmetic->a_matrix_.batch_post_sum_[j + 1];
        b_offset += (delta / arithmetic->c_matrix_.batch_post_sum_[j + 1] * arithmetic->b_matrix_.shape_[j] /
                     arithmetic->c_matrix_.shape_[j]) *
                    arithmetic->b_matrix_.batch_post_sum_[j + 1];
      } else {
        a_offset += (delta * arithmetic->a_matrix_.shape_[j] / arithmetic->c_matrix_.shape_[j]);
        b_offset += (delta * arithmetic->b_matrix_.shape_[j] / arithmetic->c_matrix_.shape_[j]);
      }
    }
    block_info->a_offset_[offset_index] = a_offset * arithmetic->a_matrix_.inner_size_ * arithmetic->in_data_size_;
    block_info->b_offset_[offset_index] = b_offset * arithmetic->b_matrix_.inner_size_ * arithmetic->in_data_size_;
    offset_index++;
  }
}

int ArithmeticDoExecute(KernelBase *base, const void *input0, const void *input1, void *output, int64_t size) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)base;
  int data_type = arithmetic->base_.in_[FIRST_INPUT]->data_type_;
  NNACL_CHECK_NULL_RETURN_ERR(input0);
  NNACL_CHECK_NULL_RETURN_ERR(input1);

  if (data_type == kNumberTypeFloat32) {
    if (arithmetic->scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.optimzie_f32_);
      return arithmetic->functions_.optimzie_f32_((const float *)input0, (const float *)input1, (float *)output, size,
                                                  arithmetic->in_elements_num0_ == 1);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.compute_f32_);
      return arithmetic->functions_.compute_f32_((const float *)input0, (const float *)input1, (float *)output, size);
    }
  }

  if (data_type == kNumberTypeBool) {
    if (arithmetic->scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.optimzie_bool_);
      return arithmetic->functions_.optimzie_bool_((const bool *)input0, (const bool *)input1, (bool *)output, size,
                                                   arithmetic->in_elements_num0_ == 1);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.compute_bool_);
      return arithmetic->functions_.compute_bool_((const bool *)input0, (const bool *)input1, (bool *)output, size);
    }
  }

  if (data_type == kNumberTypeInt32) {
    if (arithmetic->scalar_opt_) {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.optimzie_int_);
      return arithmetic->functions_.optimzie_int_((const int *)input0, (const int *)input1, (int *)output, size,
                                                  arithmetic->in_elements_num0_ == 1);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(arithmetic->functions_.compute_int_);
      return arithmetic->functions_.compute_int_((const int *)input0, (const int *)input1, (int *)output, size);
    }
  }

  return NNACL_UNSUPPORTED_DATA_TYPE;
}

int ArithmeticRun(void *cdata, int task_id, float l, float r) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)cdata;
  NNACL_CHECK_FALSE(task_id < 0, NNACL_ERR);
  NNACL_CHECK_FALSE(task_id >= arithmetic->block_boundary_infos_size_, NNACL_ERR);

  if (arithmetic->block_boundary_infos_[task_id].init_offset_ == false) {
    ArithmeticComputeOffset(arithmetic, task_id);
  }

  ArithmeticBlockBoundaryInfo *block_info = &arithmetic->block_boundary_infos_[task_id];
  int64_t b_start = block_info->batch_begin_;
  int64_t s_start = block_info->size_begin_;
  int64_t s_end = block_info->size_end_;
  int64_t index_start = 0;
  int64_t index_end = block_info->batch_end_ - b_start;
  uint8_t *a_ptr = (uint8_t *)(arithmetic->a_matrix_.data_) + block_info->a_offset_[index_start];
  uint8_t *b_ptr = (uint8_t *)(arithmetic->b_matrix_.data_) + block_info->b_offset_[index_start];
  uint8_t *c_ptr = (uint8_t *)(arithmetic->c_matrix_.data_) +
                   (b_start * arithmetic->c_matrix_.inner_size_ + s_start) * arithmetic->out_data_size_;
  if (arithmetic->a_matrix_.inner_size_ > 1) {
    a_ptr += s_start * arithmetic->in_data_size_;
  }
  if (arithmetic->b_matrix_.inner_size_ > 1) {
    b_ptr += s_start * arithmetic->in_data_size_;
  }

  if (index_start == index_end) {
    return arithmetic->execute_((KernelBase *)arithmetic, a_ptr, b_ptr, c_ptr, s_end - s_start);
  }

  int64_t size = arithmetic->c_matrix_.inner_size_ - s_start;
  int ret = arithmetic->execute_((KernelBase *)arithmetic, a_ptr, b_ptr, c_ptr, size);
  if (ret != NNACL_OK) {
    return ret;
  }

  ++index_start;
  c_ptr += size * arithmetic->out_data_size_;
  int64_t c_stride = arithmetic->c_matrix_.inner_size_ * arithmetic->out_data_size_;
  for (; index_start < index_end; ++index_start) {
    a_ptr = (uint8_t *)(arithmetic->a_matrix_.data_) + block_info->a_offset_[index_start];
    b_ptr = (uint8_t *)(arithmetic->b_matrix_.data_) + block_info->b_offset_[index_start];
    ret = arithmetic->execute_((KernelBase *)arithmetic, a_ptr, b_ptr, c_ptr, arithmetic->c_matrix_.inner_size_);
    if (ret != NNACL_OK) {
      return ret;
    }
    c_ptr += c_stride;
  }
  if (s_end == 0) {
    return NNACL_OK;
  }
  a_ptr = (uint8_t *)(arithmetic->a_matrix_.data_) + block_info->a_offset_[index_start];
  b_ptr = (uint8_t *)(arithmetic->b_matrix_.data_) + block_info->b_offset_[index_start];
  return arithmetic->execute_((KernelBase *)arithmetic, a_ptr, b_ptr, c_ptr, s_end);
}

void ResetArithmeticMatric(KernelBase *base, ArithmeticMatrixInfo *matrix) {
  matrix->is_valid_ = false;
  matrix->data_ = NULL;
  matrix->inner_size_ = 1;
  matrix->shape_size_ = 0;

  if (matrix->batch_post_sum_ != NULL) {
    base->env_->Free(base->env_->allocator_, matrix->batch_post_sum_);
    matrix->batch_post_sum_ = NULL;
  }
}

int UpdateArithmeticParameter(ArithmeticStruct *arithmetic) {
  NNACL_CHECK_TRUE_RET(arithmetic->a_matrix_.shape_size_ == arithmetic->b_matrix_.shape_size_,
                       NNACL_ARITHMETIC_SHAPE_INVALID);

  arithmetic->ndim_ = arithmetic->a_matrix_.shape_size_;
  ResetArithmeticMatric(&arithmetic->base_, &arithmetic->c_matrix_);

  for (size_t i = 0; i < arithmetic->ndim_; ++i) {
    NNACL_CHECK_TRUE_RET(arithmetic->a_matrix_.shape_[i] <= INT_MAX, NNACL_ARITHMETIC_SHAPE_INVALID);
    NNACL_CHECK_TRUE_RET(arithmetic->b_matrix_.shape_[i] <= INT_MAX, NNACL_ARITHMETIC_SHAPE_INVALID);
    arithmetic->in_shape0_[i] = arithmetic->a_matrix_.shape_[i];
    arithmetic->in_shape1_[i] = arithmetic->b_matrix_.shape_[i];
    arithmetic->out_shape_[i] = MSMAX(arithmetic->in_shape0_[i], arithmetic->in_shape1_[i]);
    arithmetic->c_matrix_.shape_[arithmetic->c_matrix_.shape_size_++] =
      MSMAX(arithmetic->a_matrix_.shape_[i], arithmetic->b_matrix_.shape_[i]);
  }
  return NNACL_OK;
}

int OptimizeArithmeticShape(ArithmeticStruct *arithmetic) {
  ArithmeticMatrixInfo *a = &arithmetic->a_matrix_;
  ArithmeticMatrixInfo *b = &arithmetic->b_matrix_;
  arithmetic->ndim_ = a->shape_size_ >= b->shape_size_ ? a->shape_size_ : b->shape_size_;

  int shape0[MAX_LEN] = {0};
  int shape1[MAX_LEN] = {0};
  /* init a & b shape */
  int i = 0;
  for (; i < arithmetic->ndim_; ++i) {
    shape0[i] = 1;
    shape1[i] = 1;
  }

  /* init matrix shape dim */
  int a_matrix_size = arithmetic->ndim_ - a->shape_size_;
  for (i = a_matrix_size; i < arithmetic->ndim_; i++) {
    shape0[i] = a->shape_[i - a_matrix_size];
  }

  int b_matrix_size = arithmetic->ndim_ - b->shape_size_;
  for (i = b_matrix_size; i < arithmetic->ndim_; i++) {
    shape1[i] = b->shape_[i - b_matrix_size];
  }

  /* horizontal shape dims */
  int shape0_temp[MAX_LEN] = {0};
  int shape1_temp[MAX_LEN] = {0};
  int shape_temp_size = 0;
  for (i = 0; i < arithmetic->ndim_;) {  // horizontal comparison, merge the part of continuous 1.
    shape0_temp[shape_temp_size] = shape0[i];
    shape1_temp[shape_temp_size] = shape1[i];
    shape_temp_size++;
    if (shape0[i] != 1 && shape1[i] != 1) {
      ++i;
      continue;
    }

    size_t j0 = i;
    while (j0 < arithmetic->ndim_ && shape0[j0] == 1) {
      ++j0;
    }
    size_t j1 = i;
    while (j1 < arithmetic->ndim_ && shape1[j1] == 1) {
      ++j1;
    }
    size_t j = MSMAX(j0, j1);
    while ((++i) < j) {
      shape0_temp[shape_temp_size - 1] *= shape0[i];
      shape1_temp[shape_temp_size - 1] *= shape1[i];
    }
  }

  arithmetic->a_matrix_.shape_size_ = 0;
  arithmetic->b_matrix_.shape_size_ = 0;

  for (i = 0; i < shape_temp_size;) {  // vertical comparison, merge the part of continuous equation.
    if (shape0_temp[i] == 1 && shape1_temp[i] == 1) {
      ++i;
      continue;
    }
    shape0[arithmetic->a_matrix_.shape_size_++] = shape0_temp[i];
    shape1[arithmetic->b_matrix_.shape_size_++] = shape1_temp[i];
    if (shape0_temp[i] != shape1_temp[i]) {
      ++i;
      continue;
    }
    while ((++i) < shape_temp_size) {
      if (shape0_temp[i] != shape1_temp[i]) {
        break;
      }
      shape0[arithmetic->a_matrix_.shape_size_ - 1] *= shape0_temp[i];
      shape1[arithmetic->b_matrix_.shape_size_ - 1] *= shape1_temp[i];
    }
  }

  memcpy(arithmetic->a_matrix_.shape_, shape0, arithmetic->a_matrix_.shape_size_ * sizeof(int));
  memcpy(arithmetic->b_matrix_.shape_, shape1, arithmetic->b_matrix_.shape_size_ * sizeof(int));

  return UpdateArithmeticParameter(arithmetic);
}

int ResetArithmeticStatus(ArithmeticStruct *arithmetic) {
  ResetArithmeticMatric(&arithmetic->base_, &arithmetic->a_matrix_);
  ResetArithmeticMatric(&arithmetic->base_, &arithmetic->b_matrix_);
  ResetArithmeticMatric(&arithmetic->base_, &arithmetic->c_matrix_);

  arithmetic->a_matrix_.shape_size_ = arithmetic->base_.in_[FIRST_INPUT]->shape_size_;
  memcpy(arithmetic->a_matrix_.shape_, arithmetic->base_.in_[FIRST_INPUT]->shape_,
         arithmetic->a_matrix_.shape_size_ * sizeof(int));
  arithmetic->b_matrix_.shape_size_ = arithmetic->base_.in_[SECOND_INPUT]->shape_size_;
  memcpy(arithmetic->b_matrix_.shape_, arithmetic->base_.in_[SECOND_INPUT]->shape_,
         arithmetic->b_matrix_.shape_size_ * sizeof(int));

  return OptimizeArithmeticShape(arithmetic);
}

void ArithmeticDoBroadcast(ArithmeticStruct *arithmetic, void *in_data, void *out_data, int input_index) {
  int *in_shape = input_index == FIRST_INPUT ? arithmetic->in_shape0_ : arithmetic->in_shape1_;
  int *in_stride = input_index == FIRST_INPUT ? arithmetic->in_strides0_ : arithmetic->in_strides1_;
  int *multiples = input_index == FIRST_INPUT ? arithmetic->multiples0_ : arithmetic->multiples1_;
  return arithmetic->tile_function_(in_data, out_data, 0, arithmetic->ndim_, in_shape, in_stride,
                                    arithmetic->out_strides_, multiples);
}

int ArithmeticBroadCastConstTensor(ArithmeticStruct *arithmetic) {
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);

  CalcStructMultiplesAndStrides(arithmetic);

#ifdef PARALLEL_INFERENCE
  bool prefer_explicit_broadcast = false;
#else
  bool prefer_explicit_broadcast = arithmetic->ndim_ != 1;
#endif
  prefer_explicit_broadcast =
    prefer_explicit_broadcast && (arithmetic->base_.in_[FIRST_INPUT]->data_type_ != kNumberTypeBool);

  bool exist_broadcast_ = false;
  int buffer_size = GetElementNum(arithmetic->base_.out_[OUTPUT_INDEX]) * arithmetic->in_data_size_;
  if (arithmetic->a_matrix_.is_const_) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic->base_.in_[FIRST_INPUT]->data_);
    if (arithmetic->in_elements_num0_ != arithmetic->out_elements_num_ && prefer_explicit_broadcast) {
      exist_broadcast_ = true;

      arithmetic->a_matrix_.data_ = arithmetic->base_.env_->Alloc(arithmetic->base_.env_->allocator_, buffer_size);
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arithmetic->a_matrix_.data_);
      arithmetic->broadcast_buffer_[Index0] = arithmetic->a_matrix_.data_;

      ArithmeticDoBroadcast(arithmetic, arithmetic->base_.in_[FIRST_INPUT]->data_, arithmetic->a_matrix_.data_, Index0);
      arithmetic->in_elements_num0_ = arithmetic->out_elements_num_;

      // shape must be equal to out
      for (size_t i = 0; i < arithmetic->ndim_; ++i) {
        arithmetic->in_shape0_[i] = arithmetic->out_shape_[i];
        arithmetic->in_strides0_[i] = arithmetic->out_strides_[i];
      }
      memcpy(arithmetic->a_matrix_.shape_, arithmetic->c_matrix_.shape_, arithmetic->ndim_ * sizeof(int));
      arithmetic->a_matrix_.is_valid_ = true;
    }
  }

  if (arithmetic->b_matrix_.is_const_) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic->base_.in_[SECOND_INPUT]->data_);
    if (arithmetic->in_elements_num1_ != arithmetic->out_elements_num_ && prefer_explicit_broadcast) {
      exist_broadcast_ = true;

      arithmetic->b_matrix_.data_ = arithmetic->base_.env_->Alloc(arithmetic->base_.env_->allocator_, buffer_size);
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arithmetic->b_matrix_.data_);
      arithmetic->broadcast_buffer_[Index1] = arithmetic->b_matrix_.data_;

      ArithmeticDoBroadcast(arithmetic, arithmetic->base_.in_[Index1]->data_, arithmetic->b_matrix_.data_, Index1);
      arithmetic->in_elements_num1_ = arithmetic->out_elements_num_;
      // shape must be equal to out
      for (size_t i = 0; i < arithmetic->ndim_; ++i) {
        arithmetic->in_shape1_[i] = arithmetic->out_shape_[i];
        arithmetic->in_strides1_[i] = arithmetic->out_strides_[i];
      }

      memcpy(arithmetic->b_matrix_.shape_, arithmetic->c_matrix_.shape_, arithmetic->ndim_ * sizeof(int));
      arithmetic->b_matrix_.is_valid_ = true;
    }
  }
  if (!exist_broadcast_) {
    return NNACL_OK;
  }
  return OptimizeArithmeticShape(arithmetic);
}

int ArithmeticComputeOfflineInfo(ArithmeticStruct *arithmetic) {
  int bread_pos = -1;
  int last_dim = arithmetic->a_matrix_.shape_size_ - 1;
  for (int i = last_dim; i >= 0; --i) {
    if (arithmetic->a_matrix_.shape_[i] != arithmetic->b_matrix_.shape_[i]) {
      bread_pos = i;
      break;
    }
  }
  arithmetic->batch_tail_dim_ = bread_pos;
  if (bread_pos == last_dim && arithmetic->batch_tail_dim_ >= 0) {
    --arithmetic->batch_tail_dim_;
  }

  for (int i = last_dim; i > arithmetic->batch_tail_dim_; --i) {
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(arithmetic->a_matrix_.inner_size_, arithmetic->a_matrix_.shape_[i], NNACL_ERR);
    arithmetic->a_matrix_.inner_size_ *= arithmetic->a_matrix_.shape_[i];
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(arithmetic->b_matrix_.inner_size_, arithmetic->b_matrix_.shape_[i], NNACL_ERR);
    arithmetic->b_matrix_.inner_size_ *= arithmetic->b_matrix_.shape_[i];
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(arithmetic->c_matrix_.inner_size_, arithmetic->c_matrix_.shape_[i], NNACL_ERR);
    arithmetic->c_matrix_.inner_size_ *= arithmetic->c_matrix_.shape_[i];
  }

  arithmetic->a_matrix_.batch_post_sum_ = arithmetic->base_.env_->Alloc(
    arithmetic->base_.env_->allocator_, (arithmetic->a_matrix_.shape_size_ + 1) * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arithmetic->a_matrix_.batch_post_sum_);
  for (int i = 0; i < arithmetic->a_matrix_.shape_size_ + 1; i++) {
    arithmetic->a_matrix_.batch_post_sum_[i] = 1;
  }

  arithmetic->b_matrix_.batch_post_sum_ = arithmetic->base_.env_->Alloc(
    arithmetic->base_.env_->allocator_, (arithmetic->b_matrix_.shape_size_ + 1) * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arithmetic->b_matrix_.batch_post_sum_);
  for (int i = 0; i < arithmetic->b_matrix_.shape_size_ + 1; i++) {
    arithmetic->b_matrix_.batch_post_sum_[i] = 1;
  }

  arithmetic->c_matrix_.batch_post_sum_ = arithmetic->base_.env_->Alloc(
    arithmetic->base_.env_->allocator_, (arithmetic->c_matrix_.shape_size_ + 1) * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(arithmetic->c_matrix_.batch_post_sum_);
  for (int i = 0; i < arithmetic->c_matrix_.shape_size_ + 1; i++) {
    arithmetic->c_matrix_.batch_post_sum_[i] = 1;
  }

  for (int i = arithmetic->batch_tail_dim_; i >= 0; --i) {
    if (i == arithmetic->batch_tail_dim_) {
      arithmetic->a_matrix_.batch_post_sum_[i] = arithmetic->a_matrix_.shape_[i];
      arithmetic->b_matrix_.batch_post_sum_[i] = arithmetic->b_matrix_.shape_[i];
      arithmetic->c_matrix_.batch_post_sum_[i] = arithmetic->c_matrix_.shape_[i];
    } else {
      arithmetic->a_matrix_.batch_post_sum_[i] =
        arithmetic->a_matrix_.shape_[i] * arithmetic->a_matrix_.batch_post_sum_[i + 1];
      arithmetic->b_matrix_.batch_post_sum_[i] =
        arithmetic->b_matrix_.shape_[i] * arithmetic->b_matrix_.batch_post_sum_[i + 1];
      arithmetic->c_matrix_.batch_post_sum_[i] =
        arithmetic->c_matrix_.shape_[i] * arithmetic->c_matrix_.batch_post_sum_[i + 1];
    }
  }

  arithmetic->scalar_opt_ = false;
  if (arithmetic->a_matrix_.inner_size_ == 1) {
    arithmetic->in_elements_num0_ = 1;
    arithmetic->scalar_opt_ = true;
  }
  if (arithmetic->b_matrix_.inner_size_ == 1) {
    arithmetic->in_elements_num1_ = 1;
    arithmetic->scalar_opt_ = true;
  }
  return NNACL_OK;
}

int ArithmeticChooseThreadCuttingStrategy(ArithmeticStruct *arithmetic) {
  int total_num = GetElementNum(arithmetic->base_.out_[OUTPUT_INDEX]);
  arithmetic->base_.thread_nr_ =
    arithmetic->base_.UpdateThread(TC_TYPE(arithmetic->primitive_type_, arithmetic->functions_.activation_type_), 1, 1,
                                   total_num, arithmetic->base_.thread_nr_);

  int64_t block_size = UP_DIV(total_num, arithmetic->base_.thread_nr_);
  int64_t split_point = 0;
  while (split_point < total_num) {
    int64_t start = split_point;
    int64_t end = start + block_size;
    if (end > total_num) {
      end = total_num;
    }
    ArithmeticBlockBoundaryInfo block_boundary_info;
    block_boundary_info.size_begin_ = start % arithmetic->c_matrix_.inner_size_;
    block_boundary_info.size_end_ = end % arithmetic->c_matrix_.inner_size_;
    block_boundary_info.batch_begin_ = start / arithmetic->c_matrix_.inner_size_;
    block_boundary_info.batch_end_ = end / arithmetic->c_matrix_.inner_size_;
    block_boundary_info.init_offset_ = false;

    int max_offset_size = block_boundary_info.batch_end_ - block_boundary_info.batch_begin_ + TWO_TENSOR;
    block_boundary_info.a_offset_ =
      (int *)arithmetic->base_.env_->Alloc(arithmetic->base_.env_->allocator_, max_offset_size * sizeof(int));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(block_boundary_info.a_offset_);
    block_boundary_info.b_offset_ =
      (int *)arithmetic->base_.env_->Alloc(arithmetic->base_.env_->allocator_, max_offset_size * sizeof(int));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(block_boundary_info.b_offset_);

    arithmetic->block_boundary_infos_[arithmetic->block_boundary_infos_size_++] = block_boundary_info;
    split_point = end;
  }

  arithmetic->base_.thread_nr_ = arithmetic->block_boundary_infos_size_;
  return NNACL_OK;
}

int ArithmeticResize(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);

  ArithmeticRelease(&arithmetic->base_);

  NNACL_CHECK_TRUE_RET(arithmetic->in_data_size_ != 0, NNACL_UNSUPPORTED_DATA_TYPE);
  NNACL_CHECK_TRUE_RET(arithmetic->out_data_size_ != 0, NNACL_UNSUPPORTED_DATA_TYPE);
  arithmetic->in_elements_num0_ = GetElementNum(self->in_[FIRST_INPUT]);
  arithmetic->in_elements_num1_ = GetElementNum(self->in_[SECOND_INPUT]);
  arithmetic->out_elements_num_ = GetElementNum(self->in_[OUTPUT_INDEX]);

  int ret = ResetArithmeticStatus(arithmetic);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ArithmeticBroadCastConstTensor(arithmetic);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ArithmeticComputeOfflineInfo(arithmetic);
  if (ret != NNACL_OK) {
    return ret;
  }

  return ArithmeticChooseThreadCuttingStrategy(arithmetic);
}

int ArithmeticPrepare(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);

  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ < kNumberTypeBegin, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[SECOND_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[SECOND_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);

  if (self->param_->quant_type_ != Quant_None) {
    return NNACL_ERR;
  }

  arithmetic->primitive_type_ = self->param_->type_;
  if (self->param_->type_ == PrimType_Eltwise) {
    switch (((ArithmeticParameter *)(self->param_))->eltwise_mode_) {
      case Eltwise_PROD:
        arithmetic->primitive_type_ = PrimType_MulFusion;
        break;
      case Eltwise_SUM:
        arithmetic->primitive_type_ = PrimType_AddFusion;
        break;
      case Eltwise_MAXIMUM:
        arithmetic->primitive_type_ = PrimType_Maximum;
        break;
      default:
        return NNACL_ELTWISE_INVALID_MOD;
    }
  }
  arithmetic->init_function_(self);

  arithmetic->a_matrix_.is_const_ = IsConst(self->in_[FIRST_INPUT]);
  arithmetic->b_matrix_.is_const_ = IsConst(self->in_[SECOND_INPUT]);
  return NNACL_OK;
}

int ArithmeticCompute(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ != self->in_[SECOND_INPUT]->data_type_,
                    NNACL_ARITHMETIC_DATA_TYPE_UNMATCH);

  if (self->train_session_) {
    arithmetic->in_data_size_ = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  }

  if (false == arithmetic->a_matrix_.is_valid_) {
    arithmetic->a_matrix_.data_ = self->in_[FIRST_INPUT]->data_;
  }
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic->a_matrix_.data_);

  if (false == arithmetic->b_matrix_.is_valid_) {
    arithmetic->b_matrix_.data_ = self->in_[SECOND_INPUT]->data_;
  }
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic->b_matrix_.data_);

  arithmetic->c_matrix_.data_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic->c_matrix_.data_);

  return self->env_->ParallelLaunch(self->env_->thread_pool_, ArithmeticRun, self, self->thread_nr_);
}

KernelBase *CreateArithmetic(OpParameter *param, int data_type) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)malloc(sizeof(ArithmeticStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(arithmetic);
  memset(arithmetic, 0, sizeof(ArithmeticStruct));
  arithmetic->in_data_size_ = DataTypeCSize(data_type);
  arithmetic->out_data_size_ = DataTypeCSize(data_type);
  arithmetic->block_boundary_infos_size_ = 0;
  arithmetic->a_matrix_.batch_post_sum_ = NULL;
  arithmetic->b_matrix_.batch_post_sum_ = NULL;
  arithmetic->c_matrix_.batch_post_sum_ = NULL;
  arithmetic->broadcast_buffer_[FIRST_INPUT] = NULL;
  arithmetic->broadcast_buffer_[SECOND_INPUT] = NULL;
  arithmetic->tile_function_ = TileOneDimensionFp32;
  arithmetic->init_function_ = InitArithmeticRunFunction;
  arithmetic->execute_ = ArithmeticDoExecute;
  arithmetic->base_.Prepare = ArithmeticPrepare;
  arithmetic->base_.Resize = ArithmeticResize;
  arithmetic->base_.Release = ArithmeticRelease;
  arithmetic->base_.Compute = ArithmeticCompute;
  return (KernelBase *)arithmetic;
}

REG_KERNEL_CREATOR(PrimType_MulFusion, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_MulFusion, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_AddFusion, kNumberTypeBool, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_AddFusion, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_AddFusion, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_SubFusion, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_SubFusion, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_DivFusion, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_RealDiv, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Mod, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Mod, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_LogicalAnd, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_LogicalAnd, kNumberTypeBool, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_LogicalAnd, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_LogicalOr, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_LogicalOr, kNumberTypeBool, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Maximum, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Minimum, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Maximum, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Minimum, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_FloorDiv, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_FloorMod, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_FloorDiv, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_FloorMod, kNumberTypeInt32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_SquaredDifference, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_Eltwise, kNumberTypeFloat32, CreateArithmetic)
REG_KERNEL_CREATOR(PrimType_DivFusion, kNumberTypeInt32, CreateArithmetic)
