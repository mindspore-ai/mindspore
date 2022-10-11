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
#include "nnacl/infer/infer_register.h"
#include "nnacl/op_base.h"

const size_t kStridedSliceOutputNum = 1;
const size_t kStridedSliceInputNum = 1;
const size_t kStridedSliceMultiInputNumMin = 3;
const size_t kStridedSliceMultiInputNumMax = 5;

typedef struct StridedSliceTransferBuffer {
  int ndim_;

  int begins_[MAX_SHAPE_SIZE];
  int ends_[MAX_SHAPE_SIZE];
  int strides_[MAX_SHAPE_SIZE];
  int begins_mask_[MAX_SHAPE_SIZE];
  int ends_mask_[MAX_SHAPE_SIZE];
  int ellipsis_mask_[MAX_SHAPE_SIZE];
  int new_axis_mask_[MAX_SHAPE_SIZE];
  int shrink_axis_mask_[MAX_SHAPE_SIZE];

  size_t begins_size_;
  size_t ends_size_;
  size_t strides_size_;
  size_t ellipsis_mask_size_;
  size_t new_axis_mask_size_;
  size_t shrink_axis_mask_size_;
} StridedSliceTransferBuffer;

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

int HandleAxesInputNotExist(const TensorC *const *inputs, struct StridedSliceTransferBuffer *transfer_buffer) {
  const TensorC *begin_tensor = inputs[1];
  int *begin_data = (int *)(begin_tensor->data_);
  const TensorC *end_tensor = inputs[2];
  int *end_data = (int *)(end_tensor->data_);
  const TensorC *stride_tensor = inputs[3];
  int *stride_data = (int *)(stride_tensor->data_);
  if (begin_data == NULL || end_data == NULL || stride_data == NULL) {
    return NNACL_ERR;
  }
  transfer_buffer->ndim_ = GetElementNum(begin_tensor);
  for (int i = 0; i < transfer_buffer->ndim_; ++i) {
    ShapePush(transfer_buffer->begins_, &transfer_buffer->begins_size_, begin_data[i]);
    ShapePush(transfer_buffer->ends_, &transfer_buffer->ends_size_, end_data[i]);
    ShapePush(transfer_buffer->strides_, &transfer_buffer->strides_size_, stride_data[i]);
  }
  return NNACL_OK;
}

int GenerateAxes(const TensorC *axes_tensor, int *axes, int num, int ndim) {
  int *axes_data = NULL;
  if (GetElementNum(axes_tensor) != 0) {
    if (GetElementNum(axes_tensor) != num) {
      return NNACL_ERR;
    }
    axes_data = (int *)(axes_tensor->data_);
    if (axes_data == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  if (axes_data == NULL) {
    for (int i = 0; i < num; ++i) {
      axes[i] = i;
    }
  } else {
    for (int i = 0; i < num; i++) {
      axes[i] = axes_data[i];
    }
    for (int i = 0; i < num; ++i) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
    }
  }
  return NNACL_OK;
}

int HandleAxesInputExist(const TensorC *const *inputs, int *ndim, int *in_shape, int *begins, int *strides, int *ends) {
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
  *ndim = (int)(input_tensor->shape_size_);
  int begin_ndim = GetElementNum(begin_tensor);

  int *stride_data = NULL;
  const TensorC *stride_tensor = inputs[4];
  int stride_data_num = GetElementNum(stride_tensor);
  if (stride_data_num != 0) {
    MS_CHECK_TRUE_RET(stride_data_num == begin_ndim, NNACL_ERR);
    stride_data = (int *)(stride_tensor->data_);
  }

  const TensorC *axes_tensor = inputs[3];
  int axes[MAX_SHAPE_SIZE] = {0};
  int ret = GenerateAxes(axes_tensor, axes, begin_ndim, *ndim);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (*ndim > MAX_SHAPE_SIZE || *ndim < 0) {
    return NNACL_ERR;
  }
  for (int i = 0; i < *ndim; i++) {
    in_shape[i] = 0;
    begins[i] = 0;
    strides[i] = 0;
  }
  for (int i = 0; i < *ndim; ++i) {
    in_shape[i] = input_tensor->shape_[i];
  }
  for (int i = 0; i < *ndim; ++i) {
    int axes_it = 0;
    if (begin_ndim > MAX_SHAPE_SIZE || begin_ndim < 0) {
      return NNACL_ERR;
    }
    for (int j = 0; j < begin_ndim; j++) {
      if (axes[j] == i) {
        axes_it = j;
        break;
      } else {
        axes_it++;
      }
    }
    if (axes_it != begin_ndim) {
      int axis = axes_it;
      if (begin_data[axis] > input_tensor->shape_[i] - 1) {
        begins[i] = begin_data[axis];
      } else {
        begins[i] = imax(imin(begin_data[axis], input_tensor->shape_[i] - 1), -input_tensor->shape_[i]);
      }
      // ends exceed limit will be set to limit
      ends[i] = imax(imin(end_data[axis], input_tensor->shape_[i]), -input_tensor->shape_[i] - 1);
      if (stride_data == NULL) {
        return NNACL_ERR;
      }
      strides[i] = stride_data[axis];
    } else {
      begins[i] = 0;
      ends[i] = input_tensor->shape_[i];
      strides[i] = 1;
    }
  }
  return NNACL_OK;
}

int StrideSlicePreCheck(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
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
  if (inputs_size >= kStridedSliceMultiInputNumMin) {
    bool begins_type_ok =
      (inputs[C1NUM]->data_type_ == kNumberTypeInt32) || (inputs[C1NUM]->data_type_ == kNumberTypeInt64);
    bool ends_type_ok =
      (inputs[C2NUM]->data_type_ == kNumberTypeInt32) || (inputs[C2NUM]->data_type_ == kNumberTypeInt64);
    if (!(begins_type_ok && ends_type_ok)) {
      return NNACL_PARAM_INVALID;
    }
  }
  return NNACL_OK;
}

void Bit2Vector(StridedSliceTransferBuffer *transfer_buffer, const StridedSliceParameter *param) {
  for (unsigned i = 0; i < (unsigned)(size_t)(transfer_buffer->ndim_); i++) {
    transfer_buffer->begins_mask_[i] = (unsigned)(param->begins_mask_) & (1 << i);
    transfer_buffer->ends_mask_[i] = (unsigned)(param->ends_mask_) & (1 << i);
    transfer_buffer->ellipsis_mask_[i] = (unsigned)(param->ellipsisMask_) & (1 << i);
    transfer_buffer->new_axis_mask_[i] = (unsigned)(param->newAxisMask_) & (1 << i);
    transfer_buffer->shrink_axis_mask_[i] = (unsigned)(param->shrinkAxisMask_) & (1 << i);
  }
}

int ApplyNewAxisMask(StridedSliceTransferBuffer *transfer_buffer, StridedSliceParameter *param, int *in_shape,
                     size_t *out_shape_size) {
  for (size_t i = 0; i < transfer_buffer->new_axis_mask_size_; i++) {
    if (transfer_buffer->new_axis_mask_[i]) {
      if (*out_shape_size >= MAX_SHAPE_SIZE) {
        return NNACL_ERR;
      }
      int ret = ShapeInsert(in_shape, out_shape_size, i, 1);
      if (ret != NNACL_OK) {
        return NNACL_ERR;
      }
      transfer_buffer->begins_[i] = 0;
      transfer_buffer->ends_[i] = 1;
      transfer_buffer->strides_[i] = 1;

      ShapePush(transfer_buffer->begins_, &transfer_buffer->begins_size_, 0);
      ShapePush(transfer_buffer->ends_, &transfer_buffer->ends_size_, in_shape[(size_t)(transfer_buffer->ndim_) - 1]);
      ShapePush(transfer_buffer->strides_, &transfer_buffer->strides_size_, 1);

      transfer_buffer->begins_mask_[i] = false;
      transfer_buffer->ends_mask_[i] = false;
      transfer_buffer->ellipsis_mask_[i] = false;
      transfer_buffer->shrink_axis_mask_[i] = false;
    }
  }
  return NNACL_OK;
}

void ApplyBeginMask(StridedSliceTransferBuffer *transfer_buffer) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    if (transfer_buffer->begins_mask_[i]) {
      transfer_buffer->begins_[i] = 0;
    }
  }
}

int ApplyEndMask(StridedSliceTransferBuffer *transfer_buffer, const int *in_shape, size_t in_shape_size) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    if (transfer_buffer->ends_mask_[i]) {
      if ((size_t)i >= in_shape_size) {
        return NNACL_ERR;
      }
      transfer_buffer->ends_[i] = in_shape[i];
    }
  }
  return NNACL_OK;
}

int ApplyEllipsisMask(StridedSliceTransferBuffer *transfer_buffer, const int *in_shape, size_t in_shape_size) {
  for (size_t i = 0; i < transfer_buffer->ellipsis_mask_size_; i++) {
    if (transfer_buffer->ellipsis_mask_[i]) {
      if (i >= in_shape_size) {
        return NNACL_ERR;
      }
      transfer_buffer->begins_[i] = 0;
      transfer_buffer->ends_[i] = in_shape[i];
      break;
    }
  }
  return NNACL_OK;
}

int TransIndexToPositive(StridedSliceTransferBuffer *transfer_buffer, const int *in_shape, size_t max_shape_size,
                         size_t in_shape_size) {
  for (size_t i = 0; i < transfer_buffer->begins_size_; i++) {
    if (i >= max_shape_size) {
      return NNACL_ERR;
    }
    if (transfer_buffer->begins_[i] < 0) {
      transfer_buffer->begins_[i] += in_shape[i];
    }
    if (transfer_buffer->ends_[i] < 0) {
      transfer_buffer->ends_[i] += in_shape[i];
    }
    if (i < in_shape_size) {
      if (transfer_buffer->begins_[i] < 0 || transfer_buffer->begins_[i] > in_shape[i]) {
        return NNACL_ERR;
      }
      if ((transfer_buffer->ends_[i] < 0 && transfer_buffer->ends_[i] != -1) ||
          transfer_buffer->ends_[i] > in_shape[i]) {
        return NNACL_ERR;
      }
    }
  }
  return NNACL_OK;
}

void ApplyShrinkMask(StridedSliceTransferBuffer *transfer_buffer, int *output_shape, size_t *output_shape_size) {
  int old_out_shape[MAX_SHAPE_SIZE] = {0};
  size_t old_out_shape_size = 0;
  ShapeSet(old_out_shape, &old_out_shape_size, output_shape, *output_shape_size);
  *output_shape_size = 0;
  for (size_t i = 0; i < transfer_buffer->shrink_axis_mask_size_; i++) {
    if (transfer_buffer->shrink_axis_mask_[i]) {
      transfer_buffer->ends_[i] = transfer_buffer->begins_[i] + 1;
      transfer_buffer->strides_[i] = 1;
    } else {
      ShapePush(output_shape, output_shape_size, old_out_shape[i]);
    }
  }
  for (size_t i = transfer_buffer->shrink_axis_mask_size_; i < old_out_shape_size; i++) {
    ShapePush(output_shape, output_shape_size, old_out_shape[i]);
  }
}

int TransferBuffer2Param(const StridedSliceTransferBuffer *transfer_buffer, StridedSliceParameter *param,
                         const int *in_shape, size_t in_shape_size) {
  if (transfer_buffer->ndim_ >= (int)(in_shape_size) || param->in_shape_length_ >= (int)(in_shape_size)) {
    return NNACL_ERR;
  }
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    param->begins_[i] = transfer_buffer->begins_[i];
    param->ends_[i] = transfer_buffer->ends_[i];
    param->in_shape_[i] = in_shape[i];
    param->strides_[i] = transfer_buffer->strides_[i];
  }

  for (int i = transfer_buffer->ndim_; i < param->in_shape_length_; i++) {
    param->begins_[i] = 0;
    param->ends_[i] = in_shape[i];
    param->in_shape_[i] = in_shape[i];
    param->strides_[i] = 1;
  }
  return NNACL_OK;
}

void InitStridedSliceTransferBuffer(StridedSliceTransferBuffer *transfer_buffer) {
  transfer_buffer->begins_size_ = 0;
  transfer_buffer->ends_size_ = 0;
  transfer_buffer->strides_size_ = 0;
  transfer_buffer->ellipsis_mask_size_ = 0;
  transfer_buffer->new_axis_mask_size_ = 0;
  transfer_buffer->shrink_axis_mask_size_ = 0;
}

void SetMaskSize(StridedSliceTransferBuffer *transfer_buffer) {
  transfer_buffer->ellipsis_mask_size_ = (size_t)(transfer_buffer->ndim_);
  transfer_buffer->new_axis_mask_size_ = (size_t)(transfer_buffer->ndim_);
  transfer_buffer->shrink_axis_mask_size_ = (size_t)(transfer_buffer->ndim_);
  transfer_buffer->begins_size_ = (size_t)(transfer_buffer->ndim_);
  transfer_buffer->ends_size_ = (size_t)(transfer_buffer->ndim_);
  transfer_buffer->strides_size_ = (size_t)(transfer_buffer->ndim_);
}

// note: begin, end, stride length are equal, but may less than rank of input
int StridedSliceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  int check_ret = StrideSlicePreCheck(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], inputs[0]);

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  int in_shape[MAX_SHAPE_SIZE] = {0};
  size_t in_shape_size = 0;
  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  ShapeSet(in_shape, &in_shape_size, input->shape_, input->shape_size_);

  StridedSliceTransferBuffer transfer_buffer;
  InitStridedSliceTransferBuffer(&transfer_buffer);

  StridedSliceParameter *param = (StridedSliceParameter *)parameter;

  transfer_buffer.ndim_ = 0;
  if (inputs_size == kStridedSliceInputNum) {
    transfer_buffer.ndim_ = (int)(in_shape_size);
    if (transfer_buffer.ndim_ > MAX_SHAPE_SIZE) {
      return NNACL_ERR;
    }
    for (int i = 0; i < transfer_buffer.ndim_; i++) {
      ShapePush(transfer_buffer.begins_, &transfer_buffer.begins_size_, param->begins_[i]);
      ShapePush(transfer_buffer.ends_, &transfer_buffer.ends_size_, param->ends_[i]);
      ShapePush(transfer_buffer.strides_, &transfer_buffer.strides_size_, param->strides_[i]);
    }
  }
  if (!CheckInputs(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (inputs_size == 4) {
    int ret = HandleAxesInputNotExist(inputs, &transfer_buffer);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  if (inputs_size == 5) {
    int ret = HandleAxesInputExist(inputs, &transfer_buffer.ndim_, in_shape, transfer_buffer.begins_,
                                   transfer_buffer.strides_, transfer_buffer.ends_);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  // set all mask to original input shape
  SetMaskSize(&transfer_buffer);
  Bit2Vector(&transfer_buffer, param);
  int ret = ApplyNewAxisMask(&transfer_buffer, param, in_shape, &in_shape_size);
  if (ret != NNACL_OK) {
    return ret;
  }

  // update parameter with new input shape
  param->num_axes_ = (int)(in_shape_size);
  param->in_shape_length_ = (int)(in_shape_size);

  ApplyBeginMask(&transfer_buffer);
  ret = ApplyEndMask(&transfer_buffer, in_shape, MAX_SHAPE_SIZE);
  if (ret != NNACL_OK) {
    return ret;
  }
  ret = ApplyEllipsisMask(&transfer_buffer, in_shape, MAX_SHAPE_SIZE);
  if (ret != NNACL_OK) {
    return ret;
  }

  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, in_shape, in_shape_size);
  ret = TransIndexToPositive(&transfer_buffer, in_shape, MAX_SHAPE_SIZE, input->shape_size_);
  if (ret != NNACL_OK) {
    return ret;
  }
  for (int i = 0; i < transfer_buffer.ndim_; i++) {
    if (transfer_buffer.strides_[i] == 0 || in_shape[i] < transfer_buffer.ends_[i]) {
      return NNACL_ERR;
    }
    output_shape[i] = (transfer_buffer.ends_[i] - transfer_buffer.begins_[i] + transfer_buffer.strides_[i] +
                       (transfer_buffer.strides_[i] < 0 ? 1 : -1)) /
                      transfer_buffer.strides_[i];
  }
  ApplyShrinkMask(&transfer_buffer, output_shape, &output_shape_size);
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  ret = TransferBuffer2Param(&transfer_buffer, param, in_shape, MAX_SHAPE_SIZE);
  if (ret != NNACL_OK) {
    return ret;
  }
  return NNACL_OK;
}

REG_INFER(StridedSlice, PrimType_StridedSlice, StridedSliceInferShape)
