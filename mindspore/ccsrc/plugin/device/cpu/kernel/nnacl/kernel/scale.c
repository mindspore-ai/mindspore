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

#include "nnacl/kernel/scale.h"
#include "nnacl/common_func.h"
#include "nnacl/scale_parameter.h"
#include "nnacl/fp32/scale_fp32.h"
#include "nnacl/tensor_c_utils.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/utils_fp16.h"
#include "nnacl/fp16/scale_fp16.h"
#endif

int ScaleRunF16(ScaleStruct *scale, int task_id, ActType act_type) {
#ifdef ENABLE_FP16
  switch (act_type) {
    case ActType_Relu6:
      DoScaleRelu6Fp16((const float16_t *)scale->input_, (float16_t *)scale->output_, (const float16_t *)scale->scale_,
                       (const float16_t *)scale->offset_, task_id, scale);
      break;
    case ActType_Relu:
      Fp16DoScaleRelu((const float16_t *)scale->input_, (float16_t *)scale->output_, (const float16_t *)scale->scale_,
                      (const float16_t *)scale->offset_, task_id, scale);
      break;
    case ActType_No:
      DoScaleFp16((const float16_t *)scale->input_, (float16_t *)scale->output_, (const float16_t *)scale->scale_,
                  (const float16_t *)scale->offset_, task_id, scale);
      break;
    default:
      return NNACL_ERR;
  }
  return NNACL_OK;
#endif
  return NNACL_DISABLE_FP16;
}

int ScaleInitInputDataType(ScaleStruct *scale) {
  if (scale->data_type_ == kNumberTypeFloat32) {
    return NNACL_OK;
  }

#ifdef ENABLE_FP16
  TensorC *scale_tensor = scale->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);
  if (scale_tensor->data_type_ != kNumberTypeFloat16 && scale->malloc_scale_ == false) {
    scale->malloc_scale_ = true;
    scale->scale_ = GetOrAllocFp16Data(scale_tensor, scale->base_.env_, true);
  } else {
    scale->malloc_scale_ = false;
    scale->scale_ = NULL;
  }

  if (scale->base_.in_size_ == TWO_TENSOR) {
    /* already done in prepare */
    return NNACL_OK;
  }

  TensorC *offset_tensor = scale->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);
  if (offset_tensor->data_type_ != kNumberTypeFloat16 && scale->malloc_scale_ == false) {
    scale->malloc_offset_ = true;
    scale->offset_ = GetOrAllocFp16Data(offset_tensor, scale->base_.env_, true);
  } else {
    scale->malloc_offset_ = false;
    scale->offset_ = NULL;
  }
  return NNACL_OK;
#endif
  return NNACL_DISABLE_FP16;
}

int ScaleRunF32(ScaleStruct *scale, int task_id, ActType act_type) {
  switch (act_type) {
    case ActType_Relu6:
      DoScaleRelu6((const float *)scale->input_, (float *)scale->output_, (const float *)scale->scale_,
                   (const float *)scale->offset_, task_id, scale);
      break;
    case ActType_Relu:
      DoScaleRelu((const float *)scale->input_, (float *)scale->output_, (const float *)scale->scale_,
                  (const float *)scale->offset_, task_id, scale);
      break;
    case ActType_No:
      DoScale((const float *)scale->input_, (float *)scale->output_, (const float *)scale->scale_,
              (const float *)scale->offset_, task_id, scale);
      break;
    default:
      return NNACL_SCALE_UNSUPPORT_ACT_TYPE;
  }
  return NNACL_OK;
}

int ScaleRun(void *cdata, int task_id, float l, float r) {
  ScaleStruct *scale = (ScaleStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(scale);
  ActType act_type = ((ScaleParameter *)scale->base_.param_)->activation_type_;
  if (scale->data_type_ == kNumberTypeFloat16) {
    return ScaleRunF16(scale, task_id, act_type);
  } else if (scale->data_type_ == kNumberTypeFloat32) {
    return ScaleRunF32(scale, task_id, act_type);
  }
  return NNACL_UNSUPPORTED_DATA_TYPE;
}

int ScaleCalculateParameter(ScaleStruct *scale) {
  TensorC *input_tensor = scale->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *scale_tensor = scale->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);
  TensorC *output_tensor = scale->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);

  scale->outer_size_ = 1;
  scale->axis_size_ = 1;
  scale->inner_size_ = 1;
  for (int i = 0; i < scale->axis_; i++) {
    scale->outer_size_ *= input_tensor->shape_[i];
  }
  for (size_t i = 0; i < scale_tensor->shape_size_; i++) {
    scale->axis_size_ *= input_tensor->shape_[i + scale->axis_];
  }
  for (size_t i = scale->axis_ + scale_tensor->shape_size_; i < input_tensor->shape_size_; i++) {
    scale->inner_size_ *= input_tensor->shape_[i];
  }

  scale->base_.thread_nr_ = MSMIN(scale->base_.thread_nr_, scale->outer_size_);
  NNACL_CHECK_ZERO_RETURN_ERR(scale->base_.thread_nr_);

  return NNACL_OK;
}

int ScaleInitScaleOffset(ScaleStruct *scale) {
  TensorC *scale_tensor = scale->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);
  int data_type_size = DataTypeCSize(scale->data_type_);

  if (scale->base_.in_size_ == TWO_TENSOR) {
    scale->malloc_offset_ = true;
    int malloc_size = GetElementNum(scale_tensor) * data_type_size;
    NNACL_CHECK_MALLOC_SIZE(malloc_size);
    scale->offset_ = scale->base_.env_->Alloc(scale->base_.env_->allocator_, malloc_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(scale->offset_);
    memset(scale->offset_, 0, malloc_size);
  }

  if (scale->data_type_ == kNumberTypeFloat16) {
    /*  handle fp16 scale and offset in compute */
    return NNACL_OK;
  }

  if (scale_tensor->data_ != NULL) {
    scale->malloc_scale_ = true;
    int malloc_size = GetElementNum(scale_tensor) * data_type_size;
    NNACL_CHECK_MALLOC_SIZE(malloc_size);
    scale->scale_ = scale->base_.env_->Alloc(scale->base_.env_->allocator_, malloc_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(scale->scale_);
    (void)memcpy(scale->scale_, scale_tensor->data_, malloc_size);
  } else {
    scale->malloc_scale_ = false;
    scale->scale_ = NULL;
  }

  if (scale->base_.in_size_ == TWO_TENSOR) {
    return NNACL_OK;
  }
  NNACL_CHECK_FALSE(scale->base_.in_size_ != THREE_TENSOR, NNACL_SCALE_INPUT_NUM_INVALID);

  TensorC *offset_tensor = scale->base_.in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(offset_tensor);
  if (offset_tensor->data_ != NULL) {
    scale->malloc_offset_ = true;
    int malloc_size = GetElementNum(offset_tensor) * data_type_size;
    NNACL_CHECK_MALLOC_SIZE(malloc_size);
    scale->offset_ = scale->base_.env_->Alloc(scale->base_.env_->allocator_, malloc_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(scale->scale_);
    (void)memcpy(scale->offset_, offset_tensor->data_, malloc_size);
  } else {
    scale->malloc_offset_ = false;
    scale->offset_ = NULL;
  }

  return NNACL_OK;
}

int ScaleCheckInputsOutputs(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);

  for (size_t i = 0; i < self->in_size_; i++) {
    TensorC *input_tensor = self->in_[i];
    NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
    if (input_tensor->data_type_ != kNumberTypeFloat32 && input_tensor->data_type_ != kNumberTypeFloat16) {
      return NNACL_UNSUPPORTED_DATA_TYPE;
    }
  }

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  if (output_tensor->data_type_ != kNumberTypeFloat32 && output_tensor->data_type_ != kNumberTypeFloat16) {
    return NNACL_UNSUPPORTED_DATA_TYPE;
  }
  return NNACL_OK;
}

int ScaleRelease(struct KernelBase *self) {
  ScaleStruct *scale = (ScaleStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(scale);

  if (scale->malloc_scale_ && scale->scale_ != NULL) {
    self->env_->Free(self->env_->allocator_, scale->scale_);
    scale->scale_ = NULL;
    scale->malloc_scale_ = false;
  }

  if (scale->malloc_offset_ && scale->offset_ != NULL) {
    self->env_->Free(self->env_->allocator_, scale->offset_);
    scale->offset_ = NULL;
    scale->malloc_offset_ = false;
  }
  return NNACL_OK;
}

int ScaleResize(struct KernelBase *self) {
  ScaleStruct *scale = (ScaleStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(scale);

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *scale_tensor = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);

  int origin_axis = ((ScaleParameter *)self->param_)->axis_;
  scale->axis_ = origin_axis < 0 ? origin_axis + input_tensor->shape_size_ : origin_axis;

  for (size_t i = 0; i < scale_tensor->shape_size_; i++) {
    if (i + scale->axis_ >= input_tensor->shape_size_) {
      return NNACL_SCALE_AXIS_AND_SHAPE_UNMATCH;
    }
    if (input_tensor->shape_[i + scale->axis_] != scale_tensor->shape_[i]) {
      return NNACL_SCALE_SCALE_SHAPE_UNMATCH;
    }
  }

  int ret = ScaleCalculateParameter(scale);
  if (ret != NNACL_OK) {
    return ret;
  }
  return NNACL_OK;
}

int ScaleCompute(struct KernelBase *self) {
  ScaleStruct *scale = (ScaleStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(scale);

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  scale->input_ = input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(scale->input_);

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  scale->output_ = output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(scale->output_);

  int ret = ScaleInitInputDataType(scale);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (!scale->malloc_scale_) {
    TensorC *scale_tensor = self->in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(scale_tensor);
    scale->scale_ = scale_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(scale->scale_);
  }

  if (!scale->malloc_offset_) {
    TensorC *offset_tensor = self->in_[THIRD_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(offset_tensor);
    scale->offset_ = offset_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(scale->offset_);
  }

  return self->env_->ParallelLaunch(self->env_->thread_pool_, ScaleRun, self, self->thread_nr_);
}

int ScalePrepare(struct KernelBase *self) {
  ScaleStruct *scale = (ScaleStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(scale);

  int ret = ScaleCheckInputsOutputs(self);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ScaleInitScaleOffset(scale);
  if (ret != NNACL_OK) {
    return ret;
  }

  return NNACL_OK;
}

KernelBase *CreateScale(OpParameter *param, int data_type) {
  ScaleStruct *scale = (ScaleStruct *)malloc(sizeof(ScaleStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(scale);
  memset(scale, 0, sizeof(ScaleStruct));
  scale->data_type_ = data_type;
  scale->scale_ = NULL;
  scale->offset_ = NULL;
  scale->malloc_scale_ = false;
  scale->malloc_offset_ = false;
  scale->base_.Prepare = ScalePrepare;
  scale->base_.Resize = ScaleResize;
  scale->base_.Compute = ScaleCompute;
  scale->base_.Release = ScaleRelease;
  return (KernelBase *)scale;
}

REG_KERNEL_CREATOR(PrimType_ScaleFusion, kNumberTypeFloat16, CreateScale)
REG_KERNEL_CREATOR(PrimType_ScaleFusion, kNumberTypeFloat32, CreateScale)
