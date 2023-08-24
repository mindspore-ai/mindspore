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

#include "nnacl/kernel/reduce.h"
#include <math.h>
#include "nnacl/fp32/reduce_fp32.h"
#include "nnacl/kernel/reshape.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/kernel/default_kernel_base.h"

void InitialReduceKernelList(KernelBase *base) {
  ReduceStruct *reduce = (ReduceStruct *)base;
  ReduceParameter *param = (ReduceParameter *)(base->param_);

  ReduceKernelList func_list[] = {{Reduce_Sum, ReduceSum, IntReduceSum, NULL, ReduceSumByLastAxis},
                                  {Reduce_Mean, ReduceMean, IntReduceMean, NULL, NULL},
                                  {Reduce_Max, ReduceMax, IntReduceMax, NULL, ReduceMaxByLastAxis},
                                  {Reduce_Min, ReduceMin, IntReduceMin, NULL, NULL},
                                  {Reduce_Prod, ReduceProd, IntReduceProd, NULL, NULL},
                                  {Reduce_SumSquare, ReduceSum, IntReduceSum, NULL, NULL},
                                  {Reduce_ASum, ReduceSum, IntReduceSum, NULL, NULL},
                                  {Reduce_All, NULL, NULL, ReduceAll, NULL},
                                  {Reduce_L2, ReduceL2Norm, NULL, NULL, NULL}};

  size_t list_len = sizeof(func_list) / sizeof(ReduceKernelList);
  for (size_t i = 0; i < list_len; ++i) {
    if (param->mode_ == func_list[i].type_) {
      reduce->compute_ = func_list[i];
      return;
    }
  }
}

int CallReduceUnit(KernelBase *base, int task_id) {
  ReduceStruct *reduce = (ReduceStruct *)base;
  NNACL_CHECK_NULL_RETURN_ERR(reduce->src_data_);
  NNACL_CHECK_NULL_RETURN_ERR(reduce->dst_data_);

  if (reduce->data_type_ == kNumberTypeFloat32) {
    if (reduce->inner_size_ == 1 && reduce->compute_.float_last_axis_func_ != NULL) {
      return reduce->compute_.float_last_axis_func_(reduce->outer_size_, reduce->inner_size_, reduce->axis_size_,
                                                    (float *)(reduce->src_data_), (float *)(reduce->dst_data_), task_id,
                                                    reduce->base_.thread_nr_);
    } else {
      NNACL_CHECK_NULL_RETURN_ERR(reduce->compute_.float_function_);
      return reduce->compute_.float_function_(reduce->outer_size_, reduce->inner_size_, reduce->axis_size_,
                                              (float *)(reduce->src_data_), (float *)(reduce->dst_data_), task_id,
                                              reduce->base_.thread_nr_);
    }
  }

  if (reduce->data_type_ == kNumberTypeBool) {
    NNACL_CHECK_NULL_RETURN_ERR(reduce->compute_.bool_function_);
    return reduce->compute_.bool_function_(reduce->outer_size_, reduce->inner_size_, reduce->axis_size_,
                                           (bool *)(reduce->src_data_), (bool *)(reduce->dst_data_), task_id,
                                           reduce->base_.thread_nr_);
  }

  if (reduce->data_type_ == kNumberTypeInt32) {
    NNACL_CHECK_NULL_RETURN_ERR(reduce->compute_.int_function_);
    return reduce->compute_.int_function_(reduce->outer_size_, reduce->inner_size_, reduce->axis_size_,
                                          (int *)(reduce->src_data_), (int *)(reduce->dst_data_), task_id,
                                          reduce->base_.thread_nr_);
  }

  return NNACL_REDUCE_UNSUPPORTED_DATA_TYPE;
}

int ReduceImpl(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  ReduceStruct *reduce = (ReduceStruct *)cdata;
  return reduce->call_uint_((KernelBase *)reduce, task_id);
}

int CopyReduceyInputToOutput(ReduceStruct *reduce) {
  int total_size = GetSize(reduce->base_.in_[FIRST_INPUT]);
  NNACL_CHECK_FALSE(total_size == 0, NNACL_REDUCE_INPUT_SHAPE_SIZE_INVALID);
  int block_size = UP_DIV(total_size, reduce->base_.thread_nr_);
  int tmp_thread_num = UP_DIV(total_size, block_size);
  NNACL_CHECK_FALSE(tmp_thread_num == 0, NNACL_REDUCE_INPUT_SHAPE_SIZE_INVALID);

  ReshapeStruct reshape_struct;
  reshape_struct.base_.in_ = reduce->base_.in_;
  reshape_struct.base_.out_ = reduce->base_.out_;
  reshape_struct.block_size_ = block_size;
  reshape_struct.total_size_ = total_size;
  reshape_struct.base_.thread_nr_ = tmp_thread_num;
  return reduce->base_.env_->ParallelLaunch(reduce->base_.env_->thread_pool_, ParallelReshape, &reshape_struct,
                                            tmp_thread_num);
}

int MallocReduceTmpBuffer(ReduceStruct *reduce) {
  // Clean pointers in data_buffer for free condition checking in FreeReduceTmpBuffer.
  memset(reduce->data_buffers_, 0, reduce->data_buffers_size_ * sizeof(void *));

  for (int i = 0; i < reduce->data_buffers_size_; i++) {
    reduce->data_buffers_[i] = reduce->base_.env_->Alloc(
      reduce->base_.env_->allocator_, reduce->data_buffer_sizes_[i] * DataTypeCSize(reduce->data_type_));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(reduce->data_buffers_[i]);
  }
  return NNACL_OK;
}

void FreeReduceTmpBuffer(ReduceStruct *reduce) {
  for (int i = 0; i < reduce->data_buffers_size_; i++) {
    if (reduce->data_buffers_[i] != NULL) {
      reduce->base_.env_->Free(reduce->base_.env_->allocator_, reduce->data_buffers_[i]);
    }
    reduce->data_buffers_[i] = NULL;
  }
}

int CalculateReduceCoeffOutput(KernelBase *base) {
  ReduceStruct *reduce = (ReduceStruct *)base;

  if (reduce->data_type_ != kNumberTypeFloat32) {
    return NNACL_REDUCE_COEFF_DATA_TYPE_INVALID;
  }
  TensorC *out_tensor = reduce->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor->data_);
  int num = GetElementNum(out_tensor);

  float *out_data = (float *)out_tensor->data_;
  for (int i = 0; i < num; ++i) {
    out_data[i] *= ((ReduceParameter *)reduce->base_.param_)->coeff;
  }
  return NNACL_OK;
}

void HandleReduceASumAndSumSquare(KernelBase *base) {
  ReduceStruct *reduce = (ReduceStruct *)base;
  if (reduce->data_type_ == kNumberTypeInt32 || reduce->data_type_ == kNumberTypeBool) {
    return;
  }

  TensorC *in_tensor = base->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(in_tensor);
  float *data = (float *)in_tensor->data_;
  NNACL_CHECK_NULL_RETURN_VOID(data);

  int num = GetElementNum(in_tensor);

  if (((ReduceParameter *)base->param_)->mode_ == Reduce_ASum) {
    for (int i = 0; i < num; ++i) {
      if (data[i] < 0.0f) {
        data[i] = 0.0f - data[i];
      }
    }
  }

  if (((ReduceParameter *)base->param_)->mode_ == Reduce_SumSquare) {
    for (int i = 0; i < num; ++i) {
      data[i] = data[i] * data[i];
    }
    return;
  }
}

int ReduceCheckInputsOutputs(ReduceStruct *reduce) {
  NNACL_CHECK_FALSE(reduce->base_.in_size_ < ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(reduce->base_.out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  for (size_t i = 0; i < reduce->base_.in_size_; i++) {
    NNACL_CHECK_NULL_RETURN_ERR(reduce->base_.in_[i]);
  }
  for (size_t i = 0; i < reduce->base_.out_size_; i++) {
    NNACL_CHECK_NULL_RETURN_ERR(reduce->base_.out_[i]);
  }
  TensorC *input_tensor = reduce->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  if (reduce->base_.in_size_ > ONE_TENSOR) {
    TensorC *axes_tensor = reduce->base_.in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(axes_tensor);
    NNACL_CHECK_FALSE(axes_tensor->data_type_ != kNumberTypeInt && axes_tensor->data_type_ != kNumberTypeInt32 &&
                        axes_tensor->data_type_ != kNumberTypeInt64,
                      NNACL_REDUCE_AXES_TENSOR_ERROR);
  }
  return NNACL_OK;
}

int ReduceCommonPrepare(ReduceStruct *reduce) {
  int ret = ReduceCheckInputsOutputs(reduce);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (reduce->base_.in_size_ == ONE_TENSOR) {
    reduce->num_axes_ = 0;
    return NNACL_OK;
  }

  TensorC *axes_tensor = reduce->base_.in_[SECOND_INPUT];
  reduce->num_axes_ = GetElementNum(axes_tensor);
  if (axes_tensor->data_ != NULL && (reduce->num_axes_ <= 0 || reduce->num_axes_ > MAX_SHAPE_SIZE)) {
    return NNACL_REDUCE_AXES_TENSOR_ERROR;
  }
  if (axes_tensor->data_ == NULL) {
    reduce->num_axes_ = reduce->base_.in_[FIRST_INPUT]->shape_size_;
    for (int i = 0; i < reduce->num_axes_; i++) {
      reduce->axes_[i] = i;
    }
  } else {
    if (axes_tensor->data_type_ == kNumberTypeInt32 || axes_tensor->data_type_ == kNumberTypeInt) {
      NNACL_CHECK_FALSE(GetSize(axes_tensor) == 0, NNACL_REDUCE_AXES_TENSOR_ERROR);
      (void)memcpy(reduce->axes_, axes_tensor->data_, GetSize(axes_tensor));
    } else {
      int64_t *axes_data = axes_tensor->data_;
      for (size_t i = 0; i < reduce->num_axes_; i++) {
        reduce->axes_[i] = (int32_t)axes_data[i];
      }
    }
  }

  return NNACL_OK;
}

int CheckReduceParameters(ReduceStruct *reduce) {
  int input_shape_size = reduce->base_.in_[FIRST_INPUT]->shape_size_;
  NNACL_CHECK_FALSE(reduce->num_axes_ > input_shape_size, NNACL_REDUCE_INPUT_SHAPE_SIZE_INVALID);

  for (int i = 0; i < reduce->num_axes_; i++) {
    NNACL_CHECK_FALSE(reduce->axes_[i] < -input_shape_size, NNACL_REDUCE_INPUT_SHAPE_SIZE_INVALID);
    NNACL_CHECK_FALSE(reduce->axes_[i] >= input_shape_size, NNACL_REDUCE_INPUT_SHAPE_SIZE_INVALID);

    if (reduce->axes_[i] < 0) {
      reduce->axes_[i] += input_shape_size;
    }
  }

  if (((ReduceParameter *)reduce->base_.param_)->reduce_to_end_) {
    // actual num of axes to reduce
    reduce->num_axes_ = (int)(input_shape_size)-reduce->axes_[0];
    for (int i = 1; i < reduce->num_axes_; ++i) {
      reduce->axes_[i] = reduce->axes_[0] + i;
    }
  }

  if (reduce->num_axes_ == 0) {
    for (int i = 0; i < input_shape_size; i++) {
      reduce->axes_[i] = i;
    }
    reduce->num_axes_ = input_shape_size;
  }
  return NNACL_OK;
}

void ReduceCalculateInnerOuterSize(ReduceStruct *reduce) {
  TensorC *input_tensor = reduce->base_.in_[FIRST_INPUT];
  int tmp_input_shape[MAX_SHAPE_SIZE];
  memcpy(tmp_input_shape, input_tensor->shape_, MAX_SHAPE_SIZE * sizeof(int));
  reduce->offset_size_ = 0;

  for (int i = 0; i < reduce->num_axes_; ++i) {
    int axis = reduce->axes_[i];
    int outer_size = 1;
    for (int j = 0; j < axis; j++) {
      outer_size *= tmp_input_shape[j];
    }
    reduce->outer_sizes_[reduce->offset_size_] = outer_size;

    int inner_size = 1;
    for (int k = axis + 1; k < input_tensor->shape_size_; k++) {
      inner_size *= tmp_input_shape[k];
    }
    reduce->inner_sizes_[reduce->offset_size_] = inner_size;
    reduce->axis_sizes_[reduce->offset_size_] = tmp_input_shape[axis];

    reduce->offset_size_++;
    tmp_input_shape[axis] = 1;
  }
}

void ReduceCalculateTmpBufferSize(ReduceStruct *reduce) {
  reduce->data_buffers_size_ = 0;

  TensorC *input_tensor = reduce->base_.in_[FIRST_INPUT];
  int tmp_input_shape[MAX_SHAPE_SIZE];
  memcpy(tmp_input_shape, input_tensor->shape_, MAX_SHAPE_SIZE * sizeof(int));
  // calculate size of buffer to malloc for each reducing axis
  for (int i = 0; i < reduce->num_axes_ - 1; i++) {
    int axis = reduce->axes_[i];
    size_t size = 1;
    for (size_t j = 0; j < input_tensor->shape_size_; j++) {
      if (axis != (int)(j)) {
        size *= (size_t)(tmp_input_shape[j]);
      }
    }
    reduce->data_buffer_sizes_[reduce->data_buffers_size_++] = size;
    tmp_input_shape[axis] = 1;
  }
}

void ReduceDecideIfOnlyCopy(ReduceStruct *reduce) {
  ReduceModeC can_not_copy[] = {Reduce_SumSquare, Reduce_ASum, Reduce_All, Reduce_L2};
  for (int i = 0; i < sizeof(can_not_copy) / sizeof(ReduceModeC); i++) {
    if (can_not_copy[i] == ((ReduceParameter *)reduce->base_.param_)->mode_) {
      reduce->only_copy_ = false;
      return;
    }
  }

  int *in_shape = reduce->base_.in_[FIRST_INPUT]->shape_;

  for (int i = 0; i < reduce->num_axes_; i++) {
    int axis = reduce->axes_[i];
    if (in_shape[axis] != 1) {
      reduce->only_copy_ = false;
      return;
    }
  }
  reduce->only_copy_ = true;
  return;
}

int ReducePrepare(struct KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  ReduceStruct *reduce = (ReduceStruct *)self;

  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, ONE_TENSOR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, ONE_TENSOR);

  int ret = ReduceCommonPrepare(reduce);
  if (ret != NNACL_OK) {
    return ret;
  }

  reduce->init_kernel_list_(self);
  return NNACL_OK;
}

int ReduceResize(struct KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  ReduceStruct *reduce = (ReduceStruct *)self;

  int ret = CheckReduceParameters(reduce);
  if (ret != NNACL_OK) {
    return ret;
  }

  ReduceDecideIfOnlyCopy(reduce);
  ReduceCalculateTmpBufferSize(reduce);
  ReduceCalculateInnerOuterSize(reduce);

  if (reduce->num_axes_ == 1) {
    self->thread_nr_ = self->UpdateThread(
      TC_TYPE(PrimType_ReduceFusion, ((ReduceParameter *)reduce->base_.param_)->mode_),
      reduce->inner_sizes_[Index0] * reduce->axis_sizes_[Index0],
      reduce->inner_sizes_[Index0] * reduce->axis_sizes_[Index0], reduce->outer_sizes_[Index0], self->thread_nr_);
  } else {
    self->thread_nr_ = self->UpdateThread(TC_TYPE(PrimType_ReduceFusion, Reduce_Max + 1), 0, 0,
                                          GetElementNum(self->out_[OUTPUT_INDEX]), self->thread_nr_);
  }
  return NNACL_OK;
}

int ReduceCompute(struct KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self);
  ReduceStruct *reduce = (ReduceStruct *)self;
  NNACL_CHECK_FALSE(self->in_[FIRST_INPUT]->data_type_ != reduce->data_type_, NNACL_ERR);

  if (reduce->only_copy_) {
    return CopyReduceyInputToOutput(reduce);
  }

  int ret = MallocReduceTmpBuffer(reduce);
  if (ret != NNACL_OK) {
    FreeReduceTmpBuffer(reduce);
    return ret;
  }

  reduce->src_data_ = self->in_[FIRST_INPUT]->data_;
  reduce->handle_sum_square_(self);
  for (int i = 0; i < reduce->num_axes_; i++) {
    if (i != (reduce->num_axes_ - 1)) {
      reduce->dst_data_ = reduce->data_buffers_[i];
    } else {
      reduce->dst_data_ = self->out_[FIRST_INPUT]->data_;
    }
    reduce->outer_size_ = reduce->outer_sizes_[i];
    reduce->inner_size_ = reduce->inner_sizes_[i];
    reduce->axis_size_ = reduce->axis_sizes_[i];
    NNACL_CHECK_FALSE(reduce->axis_size_ == 0, NNACL_REDUCE_AXIS_SIZE_ERROR);

    ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ReduceImpl, self, self->thread_nr_);
    if (ret != NNACL_OK) {
      FreeReduceTmpBuffer(reduce);
      return ret;
    }
    reduce->src_data_ = reduce->dst_data_;
  }

  ReduceParameter *param = (ReduceParameter *)reduce->base_.param_;
  if (param->reduce_to_end_ && fabsf(param->coeff) > 1e-5) {
    ret = reduce->calculate_coeff_(self);
  }

  FreeReduceTmpBuffer(reduce);
  return ret;
}

KernelBase *CreateReduce(OpParameter *param, int data_type) {
  ReduceStruct *reduce = (ReduceStruct *)malloc(sizeof(ReduceStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(reduce);
  memset(reduce, 0, sizeof(ReduceStruct));
  reduce->data_type_ = data_type;
  reduce->base_.Release = DefaultRelease;
  reduce->base_.Prepare = ReducePrepare;
  reduce->base_.Resize = ReduceResize;
  reduce->base_.Compute = ReduceCompute;
  reduce->handle_sum_square_ = HandleReduceASumAndSumSquare;
  reduce->calculate_coeff_ = CalculateReduceCoeffOutput;
  reduce->init_kernel_list_ = InitialReduceKernelList;
  reduce->call_uint_ = CallReduceUnit;
  return (KernelBase *)reduce;
}

REG_KERNEL_CREATOR(PrimType_ReduceFusion, kNumberTypeBool, CreateReduce)
REG_KERNEL_CREATOR(PrimType_ReduceFusion, kNumberTypeInt32, CreateReduce)
REG_KERNEL_CREATOR(PrimType_ReduceFusion, kNumberTypeFloat32, CreateReduce)
