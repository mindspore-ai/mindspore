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
#include "nnacl/arithmetic.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/mul_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/arithmetic_fp16.h"
#endif

void InitArithmeticRunFunction(KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;

  ArithmeticFuncions fun_table[] = {{PrimType_Stack, ActType_No, NULL, NULL, NULL, NULL, NULL, NULL}};

  size_t length = sizeof(fun_table) / sizeof(ArithmeticFuncions);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == arithmetic->primitive_type_ &&
        fun_table[i].activation_type_ == ((ArithmeticParameter *)(arithmetic->base_.param_))->activation_type_) {
      arithmetic->functions_ = fun_table[i];
      return;
    }
  }
}

int arithmetic_release(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  for (int i = 0; i < TWO_TENSOR; i++) {
    if (arithmetic->broadcast_buffer_[i] != NULL) {
      self->env_->free(self->env_->allocator_, arithmetic->broadcast_buffer_[i]);
      arithmetic->broadcast_buffer_[i] = NULL;
    }
  }

  for (int i = 0; i < arithmetic->block_boundary_infos_size_; i++) {
    if (arithmetic->block_boundary_infos_[i].a_offset_ != NULL) {
      self->env_->free(self->env_->allocator_, arithmetic->block_boundary_infos_[i].a_offset_);
      arithmetic->block_boundary_infos_[i].a_offset_ = NULL;
    }
    if (arithmetic->block_boundary_infos_[i].b_offset_ != NULL) {
      self->env_->free(self->env_->allocator_, arithmetic->block_boundary_infos_[i].b_offset_);
      arithmetic->block_boundary_infos_[i].b_offset_ = NULL;
    }
  }
  arithmetic->block_boundary_infos_size_ = 0;

  if (arithmetic->a_matrix_.batch_post_sum_ != NULL) {
    self->env_->free(self->env_->allocator_, arithmetic->a_matrix_.batch_post_sum_);
    arithmetic->a_matrix_.batch_post_sum_ = NULL;
  }

  if (arithmetic->b_matrix_.batch_post_sum_ != NULL) {
    self->env_->free(self->env_->allocator_, arithmetic->b_matrix_.batch_post_sum_);
    arithmetic->b_matrix_.batch_post_sum_ = NULL;
  }

  if (arithmetic->c_matrix_.batch_post_sum_ != NULL) {
    self->env_->free(self->env_->allocator_, arithmetic->c_matrix_.batch_post_sum_);
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
    int delta = b_start;
    int a_offset = 0;
    int b_offset = 0;
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

  return NNACL_ARITHMETIC_DATA_TYPE_INVALID;
}

int ArithmeticRun(void *cdata, int task_id, float l, float r) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)cdata;
  NNACL_CHECK_FALSE(task_id < 0, NNACL_ERR);
  NNACL_CHECK_FALSE(task_id >= arithmetic->block_boundary_infos_size_, NNACL_ERR);
  return NNACL_OK;
}

void ResetArithmeticMatric(KernelBase *base, ArithmeticMatrixInfo *matrix) {
  matrix->is_valid_ = false;
  matrix->data_ = NULL;
  matrix->inner_size_ = 1;
  matrix->shape_size_ = 0;

  if (matrix->batch_post_sum_ != NULL) {
    base->env_->free(base->env_->allocator_, matrix->batch_post_sum_);
    matrix->batch_post_sum_ = NULL;
  }
}

int UpdateArithmeticParameter(ArithmeticStruct *arithmetic) {
  NNACL_CHECK_TRUE_RET(arithmetic->a_matrix_.shape_size_ == arithmetic->b_matrix_.shape_size_,
                       NNACL_ARITHMETIC_SHAPE_INVALID);

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
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  return NNACL_OK;
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
  return NNACL_OK;
}

int ArithmeticComputeOfflineInfo(ArithmeticStruct *arithmetic) {
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);
  return NNACL_OK;
}

int ArithmeticChooseThreadCuttingStrategy(ArithmeticStruct *arithmetic) {
  int total_num = GetElementNum(arithmetic->base_.out_[OUTPUT_INDEX]);
  arithmetic->base_.thread_nr_ =
    arithmetic->base_.update_thread_(TC_TYPE(arithmetic->primitive_type_, arithmetic->functions_.activation_type_), 1,
                                     1, total_num, arithmetic->base_.thread_nr_);

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

    int max_offset_size = block_boundary_info.batch_end_ / block_boundary_info.batch_begin_ + TWO_TENSOR;
    block_boundary_info.a_offset_ =
      (int *)arithmetic->base_.env_->alloc(arithmetic->base_.env_->allocator_, max_offset_size * sizeof(int));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(block_boundary_info.a_offset_);
    block_boundary_info.b_offset_ =
      (int *)arithmetic->base_.env_->alloc(arithmetic->base_.env_->allocator_, max_offset_size * sizeof(int));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(block_boundary_info.b_offset_);

    arithmetic->block_boundary_infos_[arithmetic->block_boundary_infos_size_++] = block_boundary_info;
    split_point = end;
  }

  arithmetic->base_.thread_nr_ = arithmetic->block_boundary_infos_size_;
  return NNACL_OK;
}

int arithmetic_resize(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);

  arithmetic_release(&arithmetic->base_);

  arithmetic->in_data_size_ = DataTypeCSize(self->in_[FIRST_INPUT]->data_type_);
  NNACL_CHECK_TRUE_RET(arithmetic->in_data_size_ != 0, NNACL_ARITHMETIC_DATA_TYPE_INVALID);
  arithmetic->out_data_size_ = DataTypeCSize(self->out_[OUTPUT_INDEX]->data_type_);
  NNACL_CHECK_TRUE_RET(arithmetic->out_data_size_ != 0, NNACL_ARITHMETIC_DATA_TYPE_INVALID);

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

int arithmetic_prepare(struct KernelBase *self) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic);

  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ < kNumberTypeBegin, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[SECOND_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);
  NNACL_CHECK_FALSE(self->in_[SECOND_INPUT]->data_type_ > kNumberTypeEnd, NNACL_ERR);

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
  arithmetic->a_matrix_.is_const_ = self->in_[FIRST_INPUT]->data_ != NULL;
  arithmetic->b_matrix_.is_const_ = self->in_[SECOND_INPUT]->data_ != NULL;
  return NNACL_OK;
}

int arithmetic_compute(struct KernelBase *self) {
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

  return self->env_->parallel_launch(self->env_->thread_pool_, ArithmeticRun, self, self->thread_nr_);
}

KernelBase *CreateArithmetic(OpParameter *param, int data_type) {
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)malloc(sizeof(ArithmeticStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(arithmetic);
  memset(arithmetic, 0, sizeof(ArithmeticStruct));
  arithmetic->block_boundary_infos_size_ = 0;
  arithmetic->a_matrix_.batch_post_sum_ = NULL;
  arithmetic->b_matrix_.batch_post_sum_ = NULL;
  arithmetic->c_matrix_.batch_post_sum_ = NULL;
  arithmetic->broadcast_buffer_[FIRST_INPUT] = NULL;
  arithmetic->broadcast_buffer_[SECOND_INPUT] = NULL;
  arithmetic->init_function_ = InitArithmeticRunFunction;
  arithmetic->execute_ = ArithmeticDoExecute;
  arithmetic->base_.prepare = arithmetic_prepare;
  arithmetic->base_.resize = arithmetic_resize;
  arithmetic->base_.release = arithmetic_release;
  arithmetic->base_.compute = arithmetic_compute;
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
