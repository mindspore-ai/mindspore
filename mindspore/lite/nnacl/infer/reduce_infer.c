/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/reduce_infer.h"
#include "nnacl/infer/infer_register.h"

int ReduceOnAllAxes(const TensorC *input, TensorC *output, int *out_shape, size_t out_shape_size, bool keep_dims) {
  if (keep_dims) {
    for (size_t i = 0; i < input->shape_size_; i++) {
      ShapePush(out_shape, &out_shape_size, 1);
    }
  }
  SetShapeArray(output, out_shape, out_shape_size);
  output->data_type_ = input->data_type_;
  return NNACL_OK;
}

int ReduceOnSelectedAxes(const TensorC *input, size_t num_axes, int *actual_axes, TensorC *output, int *out_shape,
                         size_t out_shape_size, bool keep_dims) {
  for (size_t i = 0; i < input->shape_size_; i++) {
    bool reduce_axis = false;
    for (size_t idx = 0; idx < num_axes; ++idx) {
      if ((size_t)(actual_axes[idx]) == i || (size_t)(actual_axes[idx] + input->shape_size_) == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        ShapePush(out_shape, &out_shape_size, 1);
      }
    } else {
      ShapePush(out_shape, &out_shape_size, input->shape_[i]);
    }
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

int ReduceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  ReduceParameter *param = (ReduceParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  bool keep_dims = param->keep_dims_;
  int out_shape[MAX_SHAPE_SIZE];
  const size_t out_shape_size = 0;
  // get axes from input tensor
  const TensorC *axes_input = inputs[1];
  if (axes_input->shape_size_ == 1 && axes_input->shape_[0] == 0) {
    return ReduceOnAllAxes(input, output, out_shape, out_shape_size, keep_dims);
  }
  int *axes = (int *)axes_input->data_;
  if (axes == NULL) {
    return NNACL_NULL_PTR;
  }
  size_t num_axes;
  if (axes_input->shape_size_ == 1) {
    num_axes = axes_input->shape_[0];
  } else if (axes_input->shape_size_ == 0) {
    num_axes = 1;
  } else {
    return NNACL_ERR;
  }

  int rank = (int)(input->shape_size_);
  int actual_axes[MAX_SHAPE_SIZE];
  size_t actual_axes_size = 0;
  ShapeSet(actual_axes, &actual_axes_size, axes, num_axes);

  if (param->reduce_to_end_) {
    if (num_axes != 1) {
      return NNACL_ERR;
    }

    int begin_axis;
    begin_axis = axes[0] < 0 ? axes[0] + rank : axes[0];
    for (size_t i = begin_axis + 1; i < rank; ++i) {
      ShapePush(actual_axes, &actual_axes_size, i);
    }
    num_axes = rank - begin_axis;
    keep_dims = false;
  }
  // reduce on all axes
  if (num_axes == 0) {
    return ReduceOnAllAxes(input, output, out_shape, out_shape_size, keep_dims);
  }
  // reduce on selected axes
  return ReduceOnSelectedAxes(input, num_axes, actual_axes, output, out_shape, out_shape_size, keep_dims);
}

REG_INFER(Reduce, PrimType_ReduceFusion, ReduceInferShape)
