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
  return NNACL_OK;
}

void Bit2Vector(StridedSliceTransferBuffer *transfer_buffer, StridedSliceParameter *param) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    transfer_buffer->begins_mask_[i] = (uint32_t)(param->begins_mask_) & (1 << i);
    transfer_buffer->ends_mask_[i] = (uint32_t)(param->ends_mask_) & (1 << i);
    transfer_buffer->ellipsis_mask_[i] = (uint32_t)(param->ellipsisMask_) & (1 << i);
    transfer_buffer->new_axis_mask_[i] = (uint32_t)(param->newAxisMask_) & (1 << i);
    transfer_buffer->shrink_axis_mask_[i] = (uint32_t)(param->shrinkAxisMask_) & (1 << i);
  }
}

void ApplyNewAxisMask(StridedSliceTransferBuffer *transfer_buffer, StridedSliceParameter *param, int *in_shape_,
                      size_t *in_shape_size) {
  for (size_t i = 0; i < transfer_buffer->new_axis_mask_size_; i++) {
    if (transfer_buffer->new_axis_mask_[i]) {
      transfer_buffer->ndim_ += 1;
      ShapeInsert(in_shape_, in_shape_size, i, 1);
      transfer_buffer->begins_[i] = 0;
      transfer_buffer->ends_[i] = 1;
      transfer_buffer->strides_[i] = 1;

      ShapePush(transfer_buffer->begins_, &transfer_buffer->begins_size_, 0);
      ShapePush(transfer_buffer->ends_, &transfer_buffer->ends_size_, in_shape_[transfer_buffer->ndim_ - 1]);
      ShapePush(transfer_buffer->strides_, &transfer_buffer->strides_size_, 1);

      transfer_buffer->begins_mask_[i] = false;
      transfer_buffer->ends_mask_[i] = false;
      transfer_buffer->ellipsis_mask_[i] = false;
      transfer_buffer->shrink_axis_mask_[i] = false;
    }
  }
}

void ApplyBeginMask(StridedSliceTransferBuffer *transfer_buffer) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    if (transfer_buffer->begins_mask_[i]) {
      transfer_buffer->begins_[i] = 0;
    }
  }
}

void ApplyEndMask(StridedSliceTransferBuffer *transfer_buffer, int *in_shape_) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    if (transfer_buffer->ends_mask_[i]) {
      transfer_buffer->ends_[i] = in_shape_[i];
    }
  }
}

void ApplyEllipsisMask(StridedSliceTransferBuffer *transfer_buffer, int *in_shape_) {
  for (size_t i = 0; i < transfer_buffer->ellipsis_mask_size_; i++) {
    if (transfer_buffer->ellipsis_mask_[i]) {
      transfer_buffer->begins_[i] = 0;
      transfer_buffer->ends_[i] = in_shape_[i];
      break;
    }
  }
}

void TransIndexToPositive(StridedSliceTransferBuffer *transfer_buffer, int *in_shape_) {
  for (int i = 0; i < (int)(transfer_buffer->begins_size_); ++i) {
    if (transfer_buffer->begins_[i] < 0) {
      transfer_buffer->begins_[i] += in_shape_[i];
    }
    if (transfer_buffer->ends_[i] < 0) {
      transfer_buffer->ends_[i] += in_shape_[i];
    }
  }
}

void ApplyShrinkMask(StridedSliceTransferBuffer *transfer_buffer, int *output_shape, size_t *output_shape_size) {
  int old_out_shape[MAX_SHAPE_SIZE];
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

void TransferBuffer2Param(StridedSliceTransferBuffer *transfer_buffer, StridedSliceParameter *param, int *in_shape_) {
  for (int i = 0; i < transfer_buffer->ndim_; i++) {
    param->begins_[i] = transfer_buffer->begins_[i];
    param->ends_[i] = transfer_buffer->ends_[i];
    param->in_shape_[i] = in_shape_[i];
    param->strides_[i] = transfer_buffer->strides_[i];
  }

  for (int i = transfer_buffer->ndim_; i < param->in_shape_length_; i++) {
    param->begins_[i] = 0;
    param->ends_[i] = in_shape_[i];
    param->in_shape_[i] = in_shape_[i];
    param->strides_[i] = 1;
  }
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
  transfer_buffer->ellipsis_mask_size_ = transfer_buffer->ndim_;
  transfer_buffer->new_axis_mask_size_ = transfer_buffer->ndim_;
  transfer_buffer->shrink_axis_mask_size_ = transfer_buffer->ndim_;
  transfer_buffer->begins_size_ = transfer_buffer->ndim_;
  transfer_buffer->ends_size_ = transfer_buffer->ndim_;
  transfer_buffer->strides_size_ = transfer_buffer->ndim_;
}

// note: begin, end, stride length are equal, but may less than rank of input
int StridedSliceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
#ifdef Debug
  int check_ret = StrideSlicePreCheck(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], inputs[0]);

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int in_shape_[MAX_SHAPE_SIZE];
  size_t in_shape_size = 0;
  ShapeSet(in_shape_, &in_shape_size, input->shape_, input->shape_size_);

  StridedSliceTransferBuffer transfer_buffer;
  InitStridedSliceTransferBuffer(&transfer_buffer);

  StridedSliceParameter *param = (StridedSliceParameter *)parameter;
  param->num_axes_ = in_shape_size;
  param->in_shape_length_ = in_shape_size;

  transfer_buffer.ndim_ = 0;
  if (inputs_size == kStridedSliceInputNum) {
    transfer_buffer.ndim_ = (int)(param->num_axes_);
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
    const TensorC *begin_tensor = inputs[1];
    int *begin_data = (int *)(begin_tensor->data_);
    const TensorC *end_tensor = inputs[2];
    int *end_data = (int *)(end_tensor->data_);
    const TensorC *stride_tensor = inputs[3];
    int *stride_data = (int *)(stride_tensor->data_);
    if (begin_data == NULL || end_data == NULL || stride_data == NULL) {
      return NNACL_ERR;
    }
    transfer_buffer.ndim_ = GetElementNum(begin_tensor);
    for (int i = 0; i < transfer_buffer.ndim_; ++i) {
      ShapePush(transfer_buffer.begins_, &transfer_buffer.begins_size_, begin_data[i]);
      ShapePush(transfer_buffer.ends_, &transfer_buffer.ends_size_, end_data[i]);
      ShapePush(transfer_buffer.strides_, &transfer_buffer.strides_size_, stride_data[i]);
    }
  }

  if (inputs_size == 5) {
    int ret = HandleAxesInputExist(inputs, &transfer_buffer.ndim_, in_shape_, transfer_buffer.begins_,
                                   transfer_buffer.strides_, transfer_buffer.ends_);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  // set all mask to original input shape
  SetMaskSize(&transfer_buffer);
  Bit2Vector(&transfer_buffer, param);
  ApplyNewAxisMask(&transfer_buffer, param, in_shape_, &in_shape_size);
  ApplyBeginMask(&transfer_buffer);
  ApplyEndMask(&transfer_buffer, in_shape_);
  ApplyEllipsisMask(&transfer_buffer, in_shape_);

  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, in_shape_, in_shape_size);
  TransIndexToPositive(&transfer_buffer, in_shape_);
  for (int i = 0; i < transfer_buffer.ndim_; i++) {
    if (transfer_buffer.strides_[i] == 0) {
      return NNACL_ERR;
    }
    output_shape[i] = (transfer_buffer.ends_[i] - transfer_buffer.begins_[i] + transfer_buffer.strides_[i] +
                       (transfer_buffer.strides_[i] < 0 ? 1 : -1)) /
                      transfer_buffer.strides_[i];
  }
  ApplyShrinkMask(&transfer_buffer, output_shape, &output_shape_size);
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  TransferBuffer2Param(&transfer_buffer, param, in_shape_);

  return NNACL_OK;
}

REG_INFER(StridedSlice, PrimType_StridedSlice, StridedSliceInferShape)
