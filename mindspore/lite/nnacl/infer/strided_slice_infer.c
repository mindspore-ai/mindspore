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

#include "nnacl/infer/strided_slice_infer.h"

const size_t kStridedSliceOutputNum = 1;
const size_t kStridedSliceInputNum = 1;
const size_t kStridedSliceMultiInputNumMin = 3;
const size_t kStridedSliceMultiInputNumMax = 5;

bool CheckInputs(const TensorC *const *inputs, size_t inputs_size) {
  for (size_t i = 1; i < inputs_size; ++i) {
    if (inputs[i]->data_ == NULL) {
      return false;
    }
  }
  return true;
}

int HandleAxesCheckNull(const TensorC *input_tensor, const TensorC *begin_tensor, int *begin_data,
                        const TensorC *end_tensor, int *end_data) {
  if (input_tensor == NULL || begin_tensor == NULL || end_tensor == NULL || begin_data == NULL || end_data == NULL) {
    return NNACL_NULL_PTR;
  }
  return NNACL_OK;
}

int HandleAxesInputExist(const TensorC *const *inputs, int *ndim_, int *in_shape_, int *begins_, int *strides_,
                         int *ends_) {
  const TensorC *input_tensor = inputs[0];
  const TensorC *begin_tensor = inputs[1];
  int *begin_data = (int *)(begin_tensor->data_);
  const TensorC *end_tensor = inputs[2];
  int *end_data = (int *)(end_tensor->data_);

  int handle_check_ret = HandleAxesCheckNull(input_tensor, begin_tensor, begin_data, end_tensor, end_data);
  if (handle_check_ret != NNACL_OK) {
    return handle_check_ret;
  }

  // when input contains axes, begins, ends, strides will be expand to the same length as input rank
  *ndim_ = (int)(input_tensor->shape_size_);
  int begin_ndim = GetElementNum(begin_tensor);

  int *axes_data = NULL;
  const TensorC *axes_tensor = inputs[3];
  if (GetElementNum(axes_tensor) != 0) {
    if (GetElementNum(axes_tensor) != begin_ndim) {
      return NNACL_ERR;
    }
    axes_data = (int *)(axes_tensor->data_);
    if (axes_data == NULL) {
      return NNACL_NULL_PTR;
    }
  }

  int *stride_data = NULL;
  const TensorC *stride_tensor = inputs[4];
  if (GetElementNum(stride_tensor) != 0) {
    if (GetElementNum(stride_tensor) != begin_ndim) {
      return NNACL_ERR;
    }
    stride_data = (int *)(stride_tensor->data_);
    if (stride_data == NULL) {
      return NNACL_ERR;
    }
  }

  int axes[MAX_SHAPE_SIZE];
  if (axes_data == NULL) {
    for (int i = 0; i < begin_ndim; ++i) {
      axes[i] = i;
    }
  } else {
    for (size_t i = 0; i < begin_ndim; i++) {
      axes[i] = axes_data[i];
    }
    for (int i = 0; i < begin_ndim; ++i) {
      if (axes[i] < 0) {
        axes[i] += *ndim_;
      }
    }
  }

  for (size_t i = 0; i < *ndim_; i++) {
    in_shape_[i] = 0;
    begins_[i] = 0;
    strides_[i] = 0;
  }
  for (size_t i = 0; i < *ndim_; ++i) {
    in_shape_[i] = input_tensor->shape_[i];
  }
  for (size_t i = 0; i < *ndim_; ++i) {
    int axes_it = 0;
    for (size_t j = 0; j < begin_ndim; j++) {
      if (axes[j] == i) {
        axes_it = j;
        break;
      } else {
        axes_it++;
      }
    }
    if (axes_it != begin_ndim) {
      int axis = axes_it;
      // begins or ends exceed limit will be set to limit
      begins_[i] = imax(imin(begin_data[axis], input_tensor->shape_[i] - 1), -input_tensor->shape_[i]);
      ends_[i] = imax(imin(end_data[axis], input_tensor->shape_[i]), -input_tensor->shape_[i] - 1);
      strides_[i] = stride_data[axis];
    } else {
      begins_[i] = 0;
      ends_[i] = input_tensor->shape_[i];
      strides_[i] = 1;
    }
  }
  return NNACL_OK;
}

// note: begin, end, stride length are equal, but may less than rank of input
int StridedSliceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  if (outputs_size != kStridedSliceOutputNum) {
    return NNACL_PARAM_INVALID;
  }
  if (inputs_size != kStridedSliceInputNum &&
      !(inputs_size <= kStridedSliceMultiInputNumMax && inputs_size >= kStridedSliceMultiInputNumMin)) {
    return NNACL_PARAM_INVALID;
  }
  if (parameter == NULL || outputs[0] == NULL || inputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], inputs[0]);

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int in_shape_[MAX_SHAPE_SIZE];
  int begins_[MAX_SHAPE_SIZE];
  int ends_[MAX_SHAPE_SIZE];
  size_t in_shape_size_ = 0;
  if (parameter->infer_flag_) {
    ShapeSet(in_shape_, &in_shape_size_, input->shape_, input->shape_size_);
  }
  size_t begins_size_ = 0;
  size_t ends_size_ = 0;
  int strides_[MAX_SHAPE_SIZE];
  size_t strides_size_ = 0;
  int begins_mask_[MAX_SHAPE_SIZE];
  int ends_mask_[MAX_SHAPE_SIZE];
  int ellipsis_mask_[MAX_SHAPE_SIZE];
  size_t ellipsis_mask_size_ = 0;
  int new_axis_mask_[MAX_SHAPE_SIZE];
  size_t new_axis_mask_size_ = 0;
  int shrink_axis_mask_[MAX_SHAPE_SIZE];
  size_t shrink_axis_mask_size_ = 0;

  StridedSliceParameter *param = (StridedSliceParameter *)parameter;
  param->num_axes_ = in_shape_size_;
  param->in_shape_length_ = in_shape_size_;

  int ndim_ = 0;
  if (inputs_size == kStridedSliceInputNum) {
    ndim_ = (int)(param->num_axes_);

    for (int i = 0; i < ndim_; i++) {
      ShapePush(begins_, &begins_size_, param->begins_[i]);
      ShapePush(ends_, &ends_size_, param->ends_[i]);
      ShapePush(strides_, &strides_size_, param->strides_[i]);
    }
  }
  if (!CheckInputs(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (inputs_size == 4) {
    const TensorC *begin_tensor = inputs[1];
    int *begin_data = (int *)(begin_tensor->data_);
    const TensorC *end_tensor = inputs[2];
    int *end_data = (int *)(end_tensor->data_);
    const TensorC *stride_tensor = inputs[3];
    int *stride_data = (int *)(stride_tensor->data_);
    if (begin_data == NULL || end_data == NULL || stride_data == NULL) {
      return NNACL_ERR;
    }
    ndim_ = GetElementNum(begin_tensor);
    for (int i = 0; i < ndim_; ++i) {
      ShapePush(begins_, &begins_size_, begin_data[i]);
      ShapePush(ends_, &ends_size_, end_data[i]);
      ShapePush(strides_, &strides_size_, stride_data[i]);
    }
  }
  if (inputs_size == 5) {
    int ret = HandleAxesInputExist(inputs, &ndim_, in_shape_, begins_, strides_, ends_);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  // set all mask to original input shape
  ellipsis_mask_size_ = ndim_;
  new_axis_mask_size_ = ndim_;
  shrink_axis_mask_size_ = ndim_;
  begins_size_ = ndim_;
  ends_size_ = ndim_;
  strides_size_ = ndim_;

  //   convert bit to vector
  for (int i = 0; i < ndim_; i++) {
    begins_mask_[i] = (uint32_t)(param->begins_mask_) & (1 << i);
    ends_mask_[i] = (uint32_t)(param->ends_mask_) & (1 << i);
    ellipsis_mask_[i] = (uint32_t)(param->ellipsisMask_) & (1 << i);
    new_axis_mask_[i] = (uint32_t)(param->newAxisMask_) & (1 << i);
    shrink_axis_mask_[i] = (uint32_t)(param->shrinkAxisMask_) & (1 << i);
  }

  // ApplyNewAxisMask();
  for (size_t i = 0; i < new_axis_mask_size_; i++) {
    if (new_axis_mask_[i]) {
      ndim_ += 1;
      ShapeInsert(in_shape_, &in_shape_size_, i, 1);
      begins_[i] = 0;
      ends_[i] = 1;
      strides_[i] = 1;

      ShapePush(begins_, &begins_size_, 0);
      ShapePush(ends_, &ends_size_, in_shape_[ndim_ - 1]);
      ShapePush(strides_, &strides_size_, 1);

      begins_mask_[i] = false;
      ends_mask_[i] = false;
      ellipsis_mask_[i] = false;
      shrink_axis_mask_[i] = false;
    }
  }
  // ApplyBeginMask();
  for (int i = 0; i < ndim_; i++) {
    if (begins_mask_[i]) {
      begins_[i] = 0;
    }
  }
  // ApplyEndMask();
  for (int i = 0; i < ndim_; i++) {
    if (ends_mask_[i]) {
      ends_[i] = in_shape_[i];
    }
  }
  // ApplyEllipsisMask();
  for (size_t i = 0; i < ellipsis_mask_size_; i++) {
    if (ellipsis_mask_[i]) {
      begins_[i] = 0;
      ends_[i] = in_shape_[i];
      break;
    }
  }

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, in_shape_, in_shape_size_);

  // TransIndexToPositive();
  for (int i = 0; i < (int)(begins_size_); ++i) {
    if (begins_[i] < 0) {
      begins_[i] += in_shape_[i];
    }
    if (ends_[i] < 0) {
      ends_[i] += in_shape_[i];
    }
  }

  for (int i = 0; i < ndim_; i++) {
    if (strides_[i] == 0) {
      return NNACL_ERR;
    }
    output_shape[i] = (ends_[i] - begins_[i] + strides_[i] + (strides_[i] < 0 ? 1 : -1)) / strides_[i];
  }

  // ApplyShrinkMask
  int old_out_shape[MAX_SHAPE_SIZE];
  size_t old_out_shape_size = 0;
  ShapeSet(old_out_shape, &old_out_shape_size, output_shape, output_shape_size);
  output_shape_size = 0;
  for (size_t i = 0; i < shrink_axis_mask_size_; i++) {
    if (shrink_axis_mask_[i]) {
      ends_[i] = begins_[i] + 1;
      strides_[i] = 1;
    } else {
      ShapePush(output_shape, &output_shape_size, old_out_shape[i]);
    }
  }
  for (size_t i = shrink_axis_mask_size_; i < old_out_shape_size; i++) {
    ShapePush(output_shape, &output_shape_size, old_out_shape[i]);
  }

  SetShapeArray(outputs[0], output_shape, output_shape_size);

  for (int i = 0; i < ndim_; i++) {
    param->begins_[i] = begins_[i];
    param->ends_[i] = ends_[i];
    param->in_shape_[i] = in_shape_[i];
    param->strides_[i] = strides_[i];
  }

  for (int i = ndim_; i < param->in_shape_length_; i++) {
    param->begins_[i] = 0;
    param->ends_[i] = in_shape_[i];
    param->in_shape_[i] = in_shape_[i];
    param->strides_[i] = 1;
  }

  return NNACL_OK;
}
