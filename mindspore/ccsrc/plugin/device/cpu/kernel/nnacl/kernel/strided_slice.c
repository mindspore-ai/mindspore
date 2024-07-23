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

#include "nnacl/kernel/strided_slice.h"
#include "nnacl/strided_slice_parameter.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/strided_slice_fp32.h"
#include "nnacl/kernel/reshape.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"

#define MinStridedSlicePerThread 16384

int StridedSliceFaseRun(void *cdata, int task_id, float l, float r) {
  StridedSliceStruct *strided_slice = (StridedSliceStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice);

  uint8_t *input_data = strided_slice->base_.in_[FIRST_INPUT]->data_;
  uint8_t *output_data = strided_slice->base_.out_[OUTPUT_INDEX]->data_;
  int *in_shape = strided_slice->base_.in_[FIRST_INPUT]->shape_;
  int *out_shape = strided_slice->base_.out_[OUTPUT_INDEX]->shape_;
  int begin_index = strided_slice->begins_[strided_slice->split_axis_];
  int caled_num = task_id * strided_slice->cal_num_per_thread_;
  int64_t inner_size = (int64_t)strided_slice->inner_size_;

  if (strided_slice->parallel_on_outer_) {
    uint8_t *cur_in_ptr = input_data + (caled_num * in_shape[strided_slice->split_axis_] + begin_index) * inner_size;
    uint8_t *cur_out_ptr = output_data + caled_num * out_shape[strided_slice->split_axis_] * inner_size;
    int cur_outer = (int)strided_slice->outer_ - caled_num;
    if (cur_outer <= 0) {
      return NNACL_OK;
    }
    if (cur_outer > strided_slice->cal_num_per_thread_) {
      cur_outer = strided_slice->cal_num_per_thread_;
    }
    FastStride(cur_in_ptr, cur_out_ptr, out_shape[strided_slice->split_axis_],
               strided_slice->strides_[strided_slice->split_axis_], cur_outer, strided_slice->inner_size_,
               (size_t)in_shape[strided_slice->split_axis_] * strided_slice->inner_size_);
    return NNACL_OK;
  }

  if (strided_slice->parallel_on_split_axis_) {
    uint8_t *cur_in_ptr =
      input_data + (caled_num * strided_slice->strides_[strided_slice->split_axis_] + begin_index) * inner_size;
    uint8_t *cur_out_ptr = output_data + caled_num * inner_size;
    int cal_axis_num = out_shape[strided_slice->split_axis_] - caled_num;
    if (cal_axis_num <= 0) {
      return NNACL_OK;
    }
    if (cal_axis_num > strided_slice->cal_num_per_thread_) {
      cal_axis_num = strided_slice->cal_num_per_thread_;
    }
    FastStride(cur_in_ptr, cur_out_ptr, (uint32_t)cal_axis_num, strided_slice->strides_[strided_slice->split_axis_], 1,
               strided_slice->inner_size_, 0);
    return NNACL_OK;
  }

  return NNACL_STRIDED_SLICE_INVALID_PARALLEL_MOD;
}

int StridedSliceFastRun(StridedSliceStruct *strided_slice) {
  // Update length of inner size, because data type of tensor may be changed
  // from float32 to float16 during fp16 sub-graph partition process.
  size_t data_type_size = DataTypeCSize(strided_slice->base_.in_[FIRST_INPUT]->data_type_);
  NNACL_CHECK_FALSE(data_type_size == 0, NNACL_STRIDED_SLICE_UNSUPPORTED_DATA_TYPE);
  strided_slice->inner_size_ = strided_slice->inner_ * data_type_size;

  NNACL_CHECK_NULL_RETURN_ERR(strided_slice->base_.in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice->base_.in_[OUTPUT_INDEX]->data_);
  return strided_slice->base_.env_->ParallelLaunch(strided_slice->base_.env_->thread_pool_, StridedSliceFaseRun,
                                                   strided_slice, strided_slice->base_.thread_nr_);
}

bool StridedSliceMatchInOutShapeEqualPattern(StridedSliceStruct *strided_slice) {
  for (int i = 0; i < MAX_SHAPE_SIZE; i++) {
    if (strided_slice->strides_[i] < 0) {
      return false;
    }
  }

  TensorC *in_tensor = strided_slice->base_.in_[FIRST_INPUT];
  TensorC *out_tensor = strided_slice->base_.out_[OUTPUT_INDEX];

  if (in_tensor->data_type_ != out_tensor->data_type_) {
    return false;
  }

  if (in_tensor->shape_size_ != out_tensor->shape_size_) {
    return false;
  }

  if (in_tensor->shape_size_ < ONE_TENSOR) {
    return false;
  }

  for (size_t i = 0; i < in_tensor->shape_size_; ++i) {
    if (in_tensor->shape_[i] != out_tensor->shape_[i]) {
      return false;
    }
    if (in_tensor->shape_[i] == -1) {
      return false;
    }
  }
  return true;
}

int StridedSliceSoftCopyInputToOutput(StridedSliceStruct *strided_slice) {
  TensorC *in_tensor = strided_slice->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor->data_);
  TensorC *out_tensor = strided_slice->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor->data_);

  int total_num = GetElementNum(in_tensor);
  NNACL_CHECK_FALSE(total_num == 0, NNACL_STRIDED_SLICE_INVALID_DATA_SIZE);

  strided_slice->base_.thread_nr_ =
    NNACL_MIN(strided_slice->base_.thread_nr_, UP_DIV(total_num, MinStridedSlicePerThread));
  if (strided_slice->base_.thread_nr_ < 1) {
    strided_slice->base_.thread_nr_ = 1;
  }

  int block_num = UP_DIV(total_num, strided_slice->base_.thread_nr_);
  strided_slice->base_.thread_nr_ = UP_DIV(total_num, block_num);

  if (in_tensor->data_ != out_tensor->data_) {
    if (strided_slice->base_.thread_nr_ == 1) {
      (void)memcpy(out_tensor->data_, in_tensor->data_, total_num * (int)DataTypeCSize(in_tensor->data_type_));
      return NNACL_OK;
    }
    ReshapeStruct reshape;
    reshape.base_.in_ = strided_slice->base_.in_;
    reshape.base_.out_ = strided_slice->base_.out_;
    reshape.block_num_ = block_num;
    reshape.total_num_ = total_num;
    reshape.base_.thread_nr_ = strided_slice->base_.thread_nr_;
    return strided_slice->base_.env_->ParallelLaunch(strided_slice->base_.env_->thread_pool_, ParallelReshape, &reshape,
                                                     strided_slice->base_.thread_nr_);
  }
  return NNACL_OK;
}

bool StridedSliceMatchFastPattern(StridedSliceStruct *strided_slice) {
  // This function is seeking if that the number of only one dimension
  // is different between input and output. If so, we can do some trick.
  // Example 1:
  // input shape info:  [1, 80, 46, 40]
  // output shape info: [1, 80, 20, 40]
  // Example 2:
  // input shape info:  [1, 46, 40]
  // output shape info: [1, 20, 40]
  TensorC *in_tensor = strided_slice->base_.in_[FIRST_INPUT];
  TensorC *out_tensor = strided_slice->base_.out_[OUTPUT_INDEX];
  if (in_tensor->shape_size_ != out_tensor->shape_size_) {
    return false;
  }

  int axis_list[MAX_SHAPE_SIZE];
  int axis_list_size = 0;
  for (size_t i = 0; i < in_tensor->shape_size_; i++) {
    if (in_tensor->shape_[i] != out_tensor->shape_[i]) {
      axis_list[axis_list_size++] = (int)i;
    }
  }
  if (axis_list_size == 1) {
    strided_slice->split_axis_ = axis_list[Index0];
    return true;
  }
  return false;
}

void StridedSliceInitFastRunParam(StridedSliceStruct *strided_slice) {
  TensorC *input_tenspr = strided_slice->base_.in_[FIRST_INPUT];
  int *in_shape = input_tenspr->shape_;
  int *out_shape = strided_slice->base_.out_[OUTPUT_INDEX]->shape_;

  // reset && cal inner, outer
  strided_slice->outer_ = 1;
  strided_slice->inner_ = 1;
  for (int i = 0; i < strided_slice->split_axis_; ++i) {
    strided_slice->outer_ *= (size_t)in_shape[i];
  }
  for (size_t i = (size_t)strided_slice->split_axis_ + 1; i < input_tenspr->shape_size_; i++) {
    strided_slice->inner_ *= (size_t)in_shape[i];
  }

  if (strided_slice->outer_ == 1) {
    strided_slice->parallel_on_split_axis_ = true;
    strided_slice->parallel_on_outer_ = false;
  } else {
    strided_slice->parallel_on_split_axis_ = false;
    strided_slice->parallel_on_outer_ = true;
  }

  strided_slice->base_.thread_nr_ = strided_slice->base_.UpdateThread(
    TC_TYPE(PrimType_StridedSlice, strided_slice->parallel_on_outer_), 1, 1,
    GetElementNum(strided_slice->base_.out_[OUTPUT_INDEX]), strided_slice->base_.thread_nr_);

  strided_slice->cal_num_per_thread_ =
    strided_slice->parallel_on_split_axis_
      ? UP_DIV(out_shape[strided_slice->split_axis_], strided_slice->base_.thread_nr_)
      : UP_DIV((int)strided_slice->outer_, strided_slice->base_.thread_nr_);
}

int StridedSliceResize(KernelBase *self) {
  StridedSliceStruct *strided_slice = (StridedSliceStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice);

  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->shape_size_ > MAX_SHAPE_SIZE, NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE);

  StridedSliceParameter *param = (StridedSliceParameter *)self->param_;
  memcpy(strided_slice->begins_, param->begins_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice->ends_, param->ends_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice->in_shape_, param->in_shape_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice->strides_, param->strides_, MAX_SHAPE_SIZE * sizeof(int));
  strided_slice->in_shape_size_ = param->in_shape_length_;

  strided_slice->soft_copy_mode_ = StridedSliceMatchInOutShapeEqualPattern(strided_slice);
  strided_slice->fast_run_ = StridedSliceMatchFastPattern(strided_slice);
  if (strided_slice->fast_run_) {
    StridedSliceInitFastRunParam(strided_slice);
  }

  if (strided_slice->soft_copy_mode_ == false && strided_slice->fast_run_ == false) {
    return PadStridedSliceParameterTo8D(strided_slice);
  }

  return NNACL_OK;
}

int StridedSliceCompute(KernelBase *self) {
  StridedSliceStruct *strided_slice = (StridedSliceStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice);

  if (strided_slice->soft_copy_mode_) {
    return StridedSliceSoftCopyInputToOutput(strided_slice);
  }
  if (strided_slice->fast_run_) {
    return StridedSliceFastRun(strided_slice);
  }

  return DoStridedSliceIn8D(self->in_[FIRST_INPUT]->data_, self->out_[OUTPUT_INDEX]->data_, strided_slice);
}

KernelBase *CreateStridedSlice(OpParameter *param, int data_type) {
  StridedSliceStruct *strided_slice = (StridedSliceStruct *)malloc(sizeof(StridedSliceStruct));
  NNACL_CHECK_NULL_RETURN_NULL(strided_slice);
  strided_slice->data_type_ = data_type;
  strided_slice->base_.Release = DefaultRelease;
  strided_slice->base_.Prepare = DefaultPrepare1In1Out;
  strided_slice->base_.Resize = StridedSliceResize;
  strided_slice->base_.Compute = StridedSliceCompute;
  return (KernelBase *)strided_slice;
}

REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeFloat32, CreateStridedSlice)
REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeFloat16, CreateStridedSlice)
REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeInt64, CreateStridedSlice)
REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeInt32, CreateStridedSlice)
REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeInt8, CreateStridedSlice)
REG_KERNEL_CREATOR(PrimType_StridedSlice, kNumberTypeBool, CreateStridedSlice)
