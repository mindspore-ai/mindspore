/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/control/tensorlist_getitem_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensorlist_c_utils.h"

int TensorListGetItemInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  if (inputs[0]->data_type_ != kObjectTypeTensorType) {
    return NNACL_ERR;
  }
  TensorListC *input0 = (TensorListC *)(inputs[0]);
  const TensorC *get_index = inputs[1];
  if (get_index->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (GetElementNum(get_index) != 1) {
    return NNACL_ERR;
  }
  TensorC *output = outputs[0];
  if (!InferFlag(inputs, inputs_size) || input0->element_num_ == 0) {
    return NNACL_INFER_INVALID;
  }
  int index = ((int *)(get_index->data_))[0];
  if (index < 0 || index > ((int)(input0->element_num_ - 1))) {
    return NNACL_ERR;
  }
  TensorC *tensor_index = input0->tensors_[index];
  NNACL_CHECK_NULL_RETURN_ERR(tensor_index);

  if (tensor_index->data_type_ != kTypeUnknown) {
    output->data_type_ = tensor_index->data_type_;
  } else {
    output->data_type_ = input0->tensors_data_type_;
  }
  output->format_ = input0->tensors_[index]->format_;

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (tensor_index->data_type_ != kTypeUnknown) {
    ShapeSet(output->shape_, &(output->shape_size_), tensor_index->shape_, tensor_index->shape_size_);
  } else {
    const TensorC *input2 = inputs[2];
    NNACL_CHECK_NULL_RETURN_ERR(input2);
    NNACL_CHECK_NULL_RETURN_ERR(input2->data_);
    int *ele_shape_data = (int *)(input2->data_);
    NNACL_CHECK_NULL_RETURN_ERR(ele_shape_data);
    int element_shape[MAX_SHAPE_SIZE] = {0};
    size_t element_shape_size = 0;
    for (int i = 0; i < GetElementNum(input2); ++i) {
      ShapePush(element_shape, &element_shape_size, ele_shape_data[i]);
    }
    int status =
      TensorListMergeShape(element_shape, &element_shape_size, input0->element_shape_, input0->element_shape_size_);
    if (status != NNACL_OK) {
      return NNACL_ERR;
    }
    if (!TensorListIsFullyDefined(element_shape, element_shape_size)) {
      for (size_t i = 0; i < input0->element_num_; ++i) {
        TensorC *input = input0->tensors_[i];
        NNACL_CHECK_NULL_RETURN_ERR(input);
        if (input->data_type_ != kTypeUnknown) {
          status = TensorListMergeShape(element_shape, &element_shape_size, input->shape_, input->shape_size_);
          if (status != NNACL_OK) {
            return NNACL_ERR;
          }
        }
      }
    }
    if (!TensorListIsFullyDefined(element_shape, element_shape_size)) {  // the pre is the same judge condition
      return NNACL_ERR;
    }

    SetShapeArray(output, element_shape, element_shape_size);
  }

  return NNACL_OK;
}

REG_INFER(TensorListGetItem, PrimType_TensorListGetItem, TensorListGetItemInferShape)
