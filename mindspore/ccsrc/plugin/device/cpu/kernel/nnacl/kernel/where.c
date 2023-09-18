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

#include "nnacl/kernel/where.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/common_func.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/fp32/where_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/where_fp16.h"
#endif
#include "nnacl/base/broadcast_to.h"

int WhereExcuteFp16(WhereStruct *where, int task_id) {
#ifdef ENABLE_FP16
  WhereWithTripleInputsFp16((float16_t *)where->x_, (float16_t *)where->y_, (float16_t *)where->output_, &where->args_,
                            task_id, where->base_.thread_nr_);
#endif
  return NNACL_OK;
}

int WhereExcute(WhereStruct *where, int task_id) {
  WhereWithTripleInputs((float *)where->x_, (float *)where->y_, (float *)where->output_, &where->args_, task_id,
                        where->base_.thread_nr_);
  return NNACL_OK;
}

int WhereRun(void *cdata, int task_id, float l, float r) {
  WhereStruct *where = (WhereStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(where);

  NNACL_CHECK_NULL_RETURN_ERR(where->x_);
  NNACL_CHECK_NULL_RETURN_ERR(where->y_);
  NNACL_CHECK_NULL_RETURN_ERR(where->output_);
  NNACL_CHECK_NULL_RETURN_ERR(where->args_.condition_);

  if (where->data_type_ == kNumberTypeFloat16) {
    return WhereExcuteFp16(where, task_id);
  }
  return WhereExcute(where, task_id);
}

int WhereRunWithSingleInput(WhereStruct *where) {
  TensorC *input = where->base_.in_[FIRST_INPUT];
  int32_t *int32_condition = NULL;
  float *fp32_condition = NULL;
  bool *bool_condition = NULL;
  switch (where->data_type_) {
    case kNumberTypeInt32:
      int32_condition = (int32_t *)input->data_;
      NNACL_CHECK_NULL_RETURN_ERR(int32_condition);
      break;
    case kNumberTypeFloat32:
      fp32_condition = (float *)input->data_;
      NNACL_CHECK_NULL_RETURN_ERR(fp32_condition);
      break;
    case kNumberTypeBool:
      bool_condition = (bool *)input->data_;
      NNACL_CHECK_NULL_RETURN_ERR(bool_condition);
      break;
    default:
      return NNACL_WHERE_CONDITION_DATA_TYPE_ERROR;
  }
  WhereArgs *where_args = &where->args_;
  where_args->condition_num_ = GetElementNum(input);
  where_args->rank_ = input->shape_size_;
  int strides[MAX_SHAPE_SIZE];
  ComputeStrides(input->shape_, strides, where_args->rank_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(where_args->condition_num_, where_args->rank_, NNACL_ERR);
  int data_num_int = where_args->condition_num_ * where_args->rank_;
  NNACL_CHECK_TRUE_RET(data_num_int >= 0, NNACL_ERR);
  size_t result_size = (size_t)data_num_int * sizeof(int32_t);
  int32_t *result = where->base_.env_->Alloc(where->base_.env_->allocator_, result_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(result);

  int result_index = 0;
  int true_num = 0;
  for (int index = 0; index < where_args->condition_num_; index++) {
    bool condition = false;
    switch (where->data_type_) {
      case kNumberTypeInt32:
        condition = (bool)int32_condition[index];
        break;
      case kNumberTypeFloat32:
        condition = (bool)fp32_condition[index];
        break;
      case kNumberTypeBool:
        condition = (bool)bool_condition[index];
        break;
      default:
        return NNACL_WHERE_CONDITION_DATA_TYPE_ERROR;
    }
    if (condition) {
      true_num++;
      int dim = index;
      for (int j = 0; j < where_args->rank_; j++) {
        NNACL_CHECK_ZERO_RETURN_ERR(strides[j]);
        result[result_index++] = dim / strides[j];
        dim %= strides[j];
      }
    }
  }

  TensorC *output = where->base_.out_[OUTPUT_INDEX];
  if (output->data_ != NULL) {
    /* the data should be nullptr */
    where->base_.env_->Free(where->base_.env_->allocator_, output->data_);
  }
  int output_shape[] = {true_num, where_args->rank_};
  output->shape_changed_ = ShapeEqual(output->shape_, output->shape_size_, output_shape, Num2);
  output->shape_size_ = Num2;
  memcpy(output->shape_, output_shape, Num2 * sizeof(int));

  if (true_num > 0) {
    output->data_ = result;
  }
  return NNACL_OK;
}

int WhereBroadCastForInput(WhereStruct *where, TensorC *condition, TensorC *x, TensorC *y,
                           void **condition_broadcast_buf, void **x_broadcast_buf, void **y_broadcast_buf,
                           TensorC *output) {
  size_t broad_cast_buf_size = GetElementNum(output);
  if (output->data_type_ == kNumberTypeFloat32) {
    broad_cast_buf_size *= sizeof(float);
  } else {
    return NNACL_WHERE_BROAD_CAST_FAILED;
  }
  BroadcastShapeInfo condition_info;
  condition_info.input_shape_size_ = condition->shape_size_;
  condition_info.output_shape_size_ = output->shape_size_;
  memcpy(condition_info.input_shape_, condition->shape_, condition->shape_size_ * sizeof(int));
  memcpy(condition_info.output_shape_, output->shape_, output->shape_size_ * sizeof(int));

  BroadcastShapeInfo x_info;
  x_info.input_shape_size_ = x->shape_size_;
  x_info.output_shape_size_ = output->shape_size_;
  memcpy(x_info.input_shape_, x->shape_, x->shape_size_ * sizeof(int));
  memcpy(x_info.output_shape_, output->shape_, output->shape_size_ * sizeof(int));

  BroadcastShapeInfo y_info;
  y_info.input_shape_size_ = y->shape_size_;
  y_info.output_shape_size_ = output->shape_size_;
  memcpy(y_info.input_shape_, y->shape_, y->shape_size_ * sizeof(int));
  memcpy(y_info.output_shape_, output->shape_, output->shape_size_ * sizeof(int));

  *condition_broadcast_buf = where->base_.env_->Alloc(where->base_.env_->allocator_, broad_cast_buf_size);
  if (*condition_broadcast_buf == NULL) {
    return NNACL_WHERE_BROAD_CAST_FAILED;
  }
  BroadcastToSize8(condition->data_, &condition_info, *condition_broadcast_buf);

  *x_broadcast_buf = where->base_.env_->Alloc(where->base_.env_->allocator_, broad_cast_buf_size);
  if (*x_broadcast_buf == NULL) {
    where->base_.env_->Free(where->base_.env_->allocator_, *condition_broadcast_buf);
    return NNACL_WHERE_BROAD_CAST_FAILED;
  }
  BroadcastToSize32(x->data_, &x_info, *x_broadcast_buf);

  *y_broadcast_buf = where->base_.env_->Alloc(where->base_.env_->allocator_, broad_cast_buf_size);
  if (*y_broadcast_buf == NULL) {
    where->base_.env_->Free(where->base_.env_->allocator_, *condition_broadcast_buf);
    where->base_.env_->Free(where->base_.env_->allocator_, *x_broadcast_buf);
    return NNACL_WHERE_BROAD_CAST_FAILED;
  }
  BroadcastToSize32(y->data_, &y_info, *y_broadcast_buf);
  return NNACL_OK;
}

int WhereRunWithTripleInputs(WhereStruct *where) {
  TensorC *condition = where->base_.in_[Index0];
  NNACL_CHECK_NULL_RETURN_ERR(condition);
  TensorC *x = where->base_.in_[Index1];
  NNACL_CHECK_NULL_RETURN_ERR(x);
  TensorC *y = where->base_.in_[Index2];
  NNACL_CHECK_NULL_RETURN_ERR(y);
  TensorC *output = where->base_.out_[Index0];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  int condition_nums = GetElementNum(condition);
  int x_num = GetElementNum(x);
  int y_num = GetElementNum(y);
  int out_num = GetElementNum(output);
  int num_max = condition_nums > x_num ? condition_nums : (x_num > y_num ? x_num : y_num);

  where->x_ = x->data_;
  where->y_ = y->data_;
  where->output_ = output->data_;

  WhereArgs *args = &where->args_;
  args->condition_ = (bool *)condition->data_;
  args->condition_num_ = condition_nums;
  args->x_num_ = x_num;
  args->y_num_ = y_num;
  args->max_num_ = num_max;

  void *condition_broadcast_buf = NULL;
  void *x_broadcast_buf = NULL;
  void *y_broadcast_buf = NULL;

  if (out_num < num_max) {
    return NNACL_WHERE_INVALID_OUT_NUM;
  }
  if (((condition_nums != 1) && (condition_nums != num_max)) || ((x_num != 1) && (x_num != num_max)) ||
      ((y_num != 1) && (y_num != num_max))) {
    if (condition_nums != GetElementNum(y) && condition->shape_size_ != y->shape_size_) {
      int ret = WhereBroadCastForInput(where, condition, x, y, &condition_broadcast_buf, &x_broadcast_buf,
                                       &y_broadcast_buf, output);
      if (ret != NNACL_OK) {
        return NNACL_WHERE_BROAD_CAST_FAILED;
      }
      int max_num = GetElementNum(output);
      args->condition_ = (bool *)condition_broadcast_buf;
      where->x_ = x_broadcast_buf;
      where->y_ = y_broadcast_buf;
      where->output_ = output->data_;
      args->condition_num_ = max_num;
      args->x_num_ = max_num;
      args->y_num_ = max_num;
      args->max_num_ = max_num;
    } else {
      /* The length of three inputs are not equal to 1 or length of output, which is unacceptable */
      return NNACL_WHERE_CONDITION_NUM_INVALID;
    }
  }
  if (num_max <= 0) {
    /* Error, inputs' length are zero */
    return NNACL_WHERE_NUM_MAX_INVALID;
  }
  int ret =
    where->base_.env_->ParallelLaunch(where->base_.env_->thread_pool_, WhereRun, where, where->base_.thread_nr_);
  if (condition_broadcast_buf != NULL) {
    where->base_.env_->Free(where->base_.env_->allocator_, condition_broadcast_buf);
    condition_broadcast_buf = NULL;
  }
  if (x_broadcast_buf != NULL) {
    where->base_.env_->Free(where->base_.env_->allocator_, x_broadcast_buf);
    x_broadcast_buf = NULL;
  }
  if (y_broadcast_buf != NULL) {
    where->base_.env_->Free(where->base_.env_->allocator_, y_broadcast_buf);
    y_broadcast_buf = NULL;
  }
  return ret;
}

int WhereCompute(KernelBase *self) {
  WhereStruct *where = (WhereStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(where);

  int ret = NNACL_ERR;
  if (self->in_size_ == Num1) {
    ret = WhereRunWithSingleInput(where);
  } else if (self->in_size_ == Num3) {
    ret = WhereRunWithTripleInputs(where);
  } else {
    ret = NNACL_WHERE_INPUT_NUM_INVALID;
  }
  return ret;
}

int WherePrepare(KernelBase *self) {
  NNACL_CHECK_TRUE_RET(self->in_size_ == Num1 || self->in_size_ == Num3, NNACL_WHERE_INPUT_NUM_INVALID);
  NNACL_CHECK_TRUE_RET(self->out_size_ == Num1, NNACL_OUTPUT_TENSOR_ERROR);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  return NNACL_OK;
}

KernelBase *CreateWhere(OpParameter *param, int data_type) {
  WhereStruct *where = (WhereStruct *)malloc(sizeof(WhereStruct));
  NNACL_CHECK_NULL_RETURN_NULL(where);
  memset(where, 0, sizeof(WhereStruct));
  where->data_type_ = data_type;
  where->base_.Prepare = WherePrepare;
  where->base_.Compute = WhereCompute;
  where->base_.Resize = DefaultResize;
  where->base_.Release = DefaultRelease;
  return (KernelBase *)where;
}

REG_KERNEL_CREATOR(PrimType_Where, kNumberTypeBool, CreateWhere)
REG_KERNEL_CREATOR(PrimType_Where, kNumberTypeInt32, CreateWhere)
REG_KERNEL_CREATOR(PrimType_Where, kNumberTypeFloat16, CreateWhere)
REG_KERNEL_CREATOR(PrimType_Where, kNumberTypeFloat32, CreateWhere)
