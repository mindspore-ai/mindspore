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

#include "nnacl/kernel/f16/arithmetic_f16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/utils_fp16.h"
#include "nnacl/tensor_c_utils.h"

void InitArithmeticF16RunFunction(KernelBase *base) {
  ArithmeticF16Struct *arithmetic_f16 = (ArithmeticF16Struct *)base;

  ArithmeticF16Funcions f16_fun_table[] = {
    {PrimType_MulFusion, ActType_Relu, ElementMulReluFp16, ElementOptMulReluFp16},
    {PrimType_MulFusion, ActType_Relu6, ElementMulRelu6Fp16, ElementOptMulRelu6Fp16},
    {PrimType_MulFusion, ActType_No, ElementMulFp16, ElementOptMulFp16},
    {PrimType_AddFusion, ActType_Relu, ElementAddReluFp16, ElementOptAddReluFp16},
    {PrimType_AddFusion, ActType_Relu6, ElementAddRelu6Fp16, ElementOptAddRelu6Fp16},
    {PrimType_AddFusion, ActType_No, ElementAddFp16, ElementOptAddFp16},
    {PrimType_SubFusion, ActType_Relu, ElementSubReluFp16, ElementOptSubReluFp16},
    {PrimType_SubFusion, ActType_Relu6, ElementSubRelu6Fp16, ElementOptSubRelu6Fp16},
    {PrimType_SubFusion, ActType_No, ElementSubFp16, ElementOptSubFp16},
    {PrimType_DivFusion, ActType_Relu, ElementDivReluFp16, ElementOptDivReluFp16},
    {PrimType_DivFusion, ActType_Relu6, ElementDivRelu6Fp16, ElementOptDivRelu6Fp16},
    {PrimType_DivFusion, ActType_No, ElementDivFp16, ElementOptDivFp16},
    {PrimType_RealDiv, ActType_Relu, ElementDivReluFp16, ElementOptDivReluFp16},
    {PrimType_RealDiv, ActType_Relu6, ElementDivRelu6Fp16, ElementOptDivRelu6Fp16},
    {PrimType_RealDiv, ActType_No, ElementDivFp16, ElementOptDivFp16},
    {PrimType_FloorMod, ActType_No, ElementFloorModFp16, ElementOptFloorModFp16},
    {PrimType_FloorDiv, ActType_No, ElementFloorDivFp16, ElementOptFloorDivFp16},
    {PrimType_LogicalAnd, ActType_No, ElementLogicalAndFp16, ElementOptLogicalAndFp16},
    {PrimType_LogicalOr, ActType_No, ElementLogicalOrFp16, ElementOptLogicalOrFp16},
    {PrimType_SquaredDifference, ActType_No, ElementSquaredDifferenceFp16, ElementOptSquaredDifferenceFp16},
    {PrimType_Maximum, ActType_No, ElementMaximumFp16, ElementOptMaximumFp16},
    {PrimType_Minimum, ActType_No, ElementMinimumFp16, ElementOptMinimumFp16}};

  size_t length = sizeof(f16_fun_table) / sizeof(ArithmeticF16Funcions);
  for (size_t i = 0; i < length; i++) {
    if (f16_fun_table[i].primitive_type_ == arithmetic_f16->arithmetic_.primitive_type_ &&
        f16_fun_table[i].activation_type_ ==
          ((ArithmeticParameter *)(arithmetic_f16->arithmetic_.base_.param_))->activation_type_) {
      arithmetic_f16->functions_ = f16_fun_table[i];
      return;
    }
  }
}

int ArithmeticF16DoExecute(KernelBase *base, const void *input0, const void *input1, void *output, int64_t size) {
  ArithmeticF16Struct *arithmetic_f16 = (ArithmeticF16Struct *)base;

  if (arithmetic_f16->arithmetic_.scalar_opt_) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_f16->functions_.optimzie_);
    return arithmetic_f16->functions_.optimzie_((const float16_t *)input0, (const float16_t *)input1,
                                                (float16_t *)output, size,
                                                arithmetic_f16->arithmetic_.in_elements_num0_ == 1);
  }

  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_f16->functions_.compute_);
  return arithmetic_f16->functions_.compute_((const float16_t *)input0, (const float16_t *)input1, (float16_t *)output,
                                             size);
}

int ArithmeticF16Resize(KernelBase *self) {
  ArithmeticF16Struct *arithmetic_f16 = (ArithmeticF16Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_f16);
  ArithmeticStruct *arithmetic = (ArithmeticStruct *)self;

  arithmetic->in_data_size_ = sizeof(float16_t);
  arithmetic->out_data_size_ = sizeof(float16_t);
  if (arithmetic->in_elements_num1_ != 1 && arithmetic->in_elements_num0_ != 1) {
    if (arithmetic->a_matrix_.is_const_ && self->in_[FIRST_INPUT]->data_type_ == kNumberTypeFloat32) {
      TensorC *t = self->in_[FIRST_INPUT];
      NNACL_CHECK_NULL_RETURN_ERR(t->data_);
      void *f32_data = t->data_;
      t->data_type_ = kNumberTypeFloat16;
      t->data_ = self->env_->Alloc(self->env_->allocator_, GetSize(t));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]->data_);
      Float32ToFloat16((float *)(f32_data), (float16_t *)(t->data_), GetElementNum(t));
      self->env_->Free(self->env_->allocator_, f32_data);
    }
    if (arithmetic->b_matrix_.is_const_ && self->in_[SECOND_INPUT]->data_type_ == kNumberTypeFloat32) {
      TensorC *t = self->in_[SECOND_INPUT];
      NNACL_CHECK_NULL_RETURN_ERR(t->data_);
      void *f32_data = t->data_;
      t->data_type_ = kNumberTypeFloat16;
      t->data_ = self->env_->Alloc(self->env_->allocator_, GetSize(t));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]->data_);
      Float32ToFloat16((float *)(f32_data), (float16_t *)(t->data_), GetElementNum(t));
      self->env_->Free(self->env_->allocator_, f32_data);
    }
  }
  return ArithmeticResize(self);
}

void FreeArithmeticF16Buffers(ArithmeticF16Struct *arithmetic_f16) {
  for (int i = 0; i < THREE_TENSOR; i++) {
    if (arithmetic_f16->tmp_buffer_[i] != NULL) {
      arithmetic_f16->arithmetic_.base_.env_->Free(arithmetic_f16->arithmetic_.base_.env_->allocator_,
                                                   arithmetic_f16->tmp_buffer_[i]);
      arithmetic_f16->tmp_buffer_[i] = NULL;
    }
  }
}

int ArithmeticF16Compute(KernelBase *self) {
  ArithmeticF16Struct *arithmetic_f16 = (ArithmeticF16Struct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(arithmetic_f16);

  int in0_data_type = self->in_[FIRST_INPUT]->data_type_;
  int in1_data_type = self->in_[SECOND_INPUT]->data_type_;
  int out_data_type = self->out_[OUTPUT_INDEX]->data_type_;

  NNACL_CHECK_FALSE(in0_data_type != kNumberTypeFloat32 && in0_data_type != kNumberTypeFloat16,
                    NNACL_UNSUPPORTED_DATA_TYPE);
  NNACL_CHECK_FALSE(in1_data_type != kNumberTypeFloat16 && in1_data_type != kNumberTypeFloat32,
                    NNACL_UNSUPPORTED_DATA_TYPE);

  if (!arithmetic_f16->arithmetic_.a_matrix_.is_valid_) {
    arithmetic_f16->arithmetic_.a_matrix_.data_ = GetOrAllocFp16Data(self->in_[FIRST_INPUT], self->env_, true);
    arithmetic_f16->tmp_buffer_[FIRST_INPUT] =
      in0_data_type == kNumberTypeFloat16 ? NULL : arithmetic_f16->arithmetic_.a_matrix_.data_;
  }

  if (!arithmetic_f16->arithmetic_.b_matrix_.is_valid_) {
    arithmetic_f16->arithmetic_.b_matrix_.data_ = GetOrAllocFp16Data(self->in_[SECOND_INPUT], self->env_, true);
    arithmetic_f16->tmp_buffer_[SECOND_INPUT] =
      in1_data_type == kNumberTypeFloat16 ? NULL : arithmetic_f16->arithmetic_.b_matrix_.data_;
  }

  arithmetic_f16->arithmetic_.c_matrix_.data_ = GetOrAllocFp16Data(self->out_[OUTPUT_INDEX], self->env_, false);
  arithmetic_f16->tmp_buffer_[THIRD_INPUT] =
    out_data_type == kNumberTypeFloat16 ? NULL : arithmetic_f16->arithmetic_.c_matrix_.data_;

  int ret = ArithmeticCompute(self);
  if (ret == NNACL_OK && out_data_type == kNumberTypeFloat32) {
    NNACL_CHECK_NULL_RETURN_ERR(arithmetic_f16->arithmetic_.c_matrix_.data_);
    NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]->data_);
    Float16ToFloat32((float16_t *)(arithmetic_f16->arithmetic_.c_matrix_.data_),
                     (float *)(self->out_[OUTPUT_INDEX]->data_), GetElementNum(self->out_[OUTPUT_INDEX]));
  }

  FreeArithmeticF16Buffers(arithmetic_f16);
  return NNACL_OK;
}

KernelBase *CreateArithmeticF16(OpParameter *param, int data_type) {
  ArithmeticF16Struct *arithmetic_f16 = (ArithmeticF16Struct *)malloc(sizeof(ArithmeticF16Struct));
  NNACL_CHECK_NULL_RETURN_NULL(arithmetic_f16);
  memset(arithmetic_f16, 0, sizeof(ArithmeticF16Struct));

  ArithmeticStruct *arithmetic = &arithmetic_f16->arithmetic_;
  arithmetic->block_boundary_infos_size_ = 0;
  arithmetic->a_matrix_.batch_post_sum_ = NULL;
  arithmetic->b_matrix_.batch_post_sum_ = NULL;
  arithmetic->c_matrix_.batch_post_sum_ = NULL;
  arithmetic->broadcast_buffer_[FIRST_INPUT] = NULL;
  arithmetic->broadcast_buffer_[SECOND_INPUT] = NULL;
  arithmetic->base_.Prepare = ArithmeticPrepare;
  arithmetic->base_.Resize = ArithmeticF16Resize;
  arithmetic->base_.Release = ArithmeticRelease;
  arithmetic->base_.Compute = ArithmeticF16Compute;

  arithmetic->execute_ = ArithmeticF16DoExecute;
  arithmetic->tile_function_ = TileOneDimensionFp16;
  arithmetic->init_function_ = InitArithmeticF16RunFunction;

  return (KernelBase *)arithmetic_f16;
}

REG_KERNEL_CREATOR(PrimType_MulFusion, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_AddFusion, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_SubFusion, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_DivFusion, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_FloorMod, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_FloorDiv, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_LogicalAnd, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_LogicalOr, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_Maximum, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_Minimum, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_Eltwise, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_RealDiv, kNumberTypeFloat16, CreateArithmeticF16)
REG_KERNEL_CREATOR(PrimType_SquaredDifference, kNumberTypeFloat16, CreateArithmeticF16)
