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

#include "nnacl/kernel/one_hot.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/one_hot_parameter.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/fp32/one_hot_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/one_hot_fp16.h"
#endif

int OneHotRun(void *cdata, int task_id, float l, float r) {
  OneHotStruct *one_hot = (OneHotStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(one_hot);

  int *indices_data = (int *)one_hot->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(indices_data);

  TensorC *output_tensor = one_hot->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  void *output_data = one_hot->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  if (output_tensor->data_type_ == kNumberTypeFloat32) {
    return OneHotToFp32(indices_data, one_hot->on_value_, one_hot->off_value_, (float *)output_data, one_hot, task_id,
                        one_hot->base_.thread_nr_);
#ifdef ENABLE_FP16
  } else if (output_tensor->data_type_ == kNumberTypeFloat16) {
    return OneHotToFp16(indices_data, (float16_t)one_hot->on_value_, (float16_t)one_hot->off_value_,
                        (float16_t *)output_data, one_hot, task_id, one_hot->base_.thread_nr_);
#endif
  }

  return NNACL_UNSUPPORTED_DATA_TYPE;
}

int OneHotInitOnOffValueForFourInputs(OneHotStruct *one_hot) {
  TensorC *on_value_tensor = one_hot->base_.in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(on_value_tensor);
  void *on_value_data = on_value_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(on_value_data);
  if (on_value_tensor->data_type_ == kNumberTypeFloat32) {
    one_hot->on_value_ = *((float *)on_value_data);
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (on_value_tensor->data_type_ == kNumberTypeFloat16) {
    one_hot->on_value_ = *((float16_t *)on_value_data);
#endif
  } else {
    return NNACL_ONE_HOR_ON_VALUE_TENSOR_DATA_TYPE_INVALID;
  }

  TensorC *off_value_tensor = one_hot->base_.in_[FOURTH_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(off_value_tensor);
  void *off_value_data = off_value_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(off_value_data);
  if (on_value_tensor->data_type_ == kNumberTypeFloat32) {
    one_hot->off_value_ = *((float *)off_value_data);
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (on_value_tensor->data_type_ == kNumberTypeFloat16) {
    one_hot->off_value_ = *((float16_t *)off_value_data);
#endif
  } else {
    return NNACL_ONE_HOR_OFF_VALUE_TENSOR_DATA_TYPE_INVALID;
  }

  return NNACL_OK;
}

int OneHotInitOnOffValueForThreeInputs(OneHotStruct *one_hot) {
  TensorC *value_tensor = one_hot->base_.in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(value_tensor);
  void *value_data = value_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(value_data);

  if (value_tensor->data_type_ == kNumberTypeFloat32) {
    one_hot->off_value_ = ((float *)value_data)[Index0];
    one_hot->on_value_ = ((float *)value_data)[Index1];
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  } else if (value_tensor->data_type_ == kNumberTypeFloat16) {
    one_hot->off_value_ = ((float16_t *)value_data)[Index0];
    one_hot->on_value_ = ((float16_t *)value_data)[Index1];
#endif
  } else {
    return NNACL_ONE_HOR_ON_OFF_VALUE_TENSOR_DATA_TYPE_INVALID;
  }
  return NNACL_OK;
}

int OneHotInitParamsAndOnOffValue(OneHotStruct *one_hot) {
  TensorC *depth_tensor = one_hot->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(depth_tensor);

  if (depth_tensor->data_type_ == kNumberTypeInt32) {
    const int *depth = (int *)depth_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(depth);
    one_hot->depth_ = *depth;
  } else {
    return NNACL_ONE_HOR_DEPTH_TENSOR_DATA_TYPE_INVALID;
  }

  if (one_hot->base_.in_size_ == FOUR_TENSOR) {
    // 4 inputs: indices, depth, on_value, off_value
    one_hot->support_neg_index_ = false;
    int ret = OneHotInitOnOffValueForFourInputs(one_hot);
    if (ret != NNACL_OK) {
      return ret;
    }
  } else {
    // 3 inputs: indices, depth, off_on_value
    one_hot->support_neg_index_ = true;
    int ret = OneHotInitOnOffValueForThreeInputs(one_hot);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int OneHotCompute(KernelBase *self) {
  OneHotStruct *one_hot = (OneHotStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(one_hot);
  int ret = OneHotInitParamsAndOnOffValue(one_hot);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, OneHotRun, self, self->thread_nr_);
  if (ret != NNACL_OK) {
    return ret;
  }

  return NNACL_OK;
}

int OneHotPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ != FOUR_TENSOR && self->in_size_ != THREE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ != ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  TypeIdC data_type = self->in_[FIRST_INPUT]->data_type_;
  NNACL_CHECK_FALSE(data_type != kNumberTypeInt32 && data_type != kNumberTypeInt64, NNACL_OUTPUT_TENSOR_ERROR);
  return NNACL_OK;
}

int OneHotResize(KernelBase *self) {
  OneHotStruct *one_hot = (OneHotStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(one_hot);

  TensorC *indices = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(indices);

  int origin_axis = ((OneHotParameter *)self->param_)->axis_;
  one_hot->axis_ = origin_axis < 0 ? origin_axis + (int)indices->shape_size_ + 1 : origin_axis;
  NNACL_CHECK_FALSE(one_hot->axis_ < 0 && one_hot->axis_ > (int)indices->shape_size_, NNACL_ONE_HOT_AXIS_INVALID);

  one_hot->outer_size_ = 1;
  for (int i = 0; i < one_hot->axis_; i++) {
    one_hot->outer_size_ *= indices->shape_[i];
  }
  if (one_hot->outer_size_ == 0) {
    return NNACL_ONE_HOT_OUTER_SIZE_INVALID;
  }
  one_hot->inner_size_ = GetElementNum(indices) / one_hot->outer_size_;
  NNACL_CHECK_FALSE(one_hot->inner_size_ <= 0, NNACL_ONE_HOT_INNER_SIZE_INVALID);

  self->thread_nr_ = self->UpdateThread(TC_PTYPE(PrimType_OneHot), one_hot->inner_size_, one_hot->outer_size_,
                                        GetElementNum(self->out_[OUTPUT_INDEX]), self->thread_nr_);
  return NNACL_OK;
}

KernelBase *CreateOneHot(OpParameter *param, int data_type) {
  OneHotStruct *one_hot = (OneHotStruct *)malloc(sizeof(OneHotStruct));
  NNACL_CHECK_NULL_RETURN_NULL(one_hot);
  one_hot->base_.Release = DefaultRelease;
  one_hot->base_.Prepare = OneHotPrepare;
  one_hot->base_.Resize = OneHotResize;
  one_hot->base_.Compute = OneHotCompute;
  return (KernelBase *)one_hot;
}

REG_KERNEL_CREATOR(PrimType_OneHot, kNumberTypeInt32, CreateOneHot)
