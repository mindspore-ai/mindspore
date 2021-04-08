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

#include "nnacl/infer/tensorlist_getitem_infer.h"
#include "nnacl/infer/infer_register.h"

int TensorListGetItemInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  TensorListC *input0 = (TensorListC *)(inputs[0]);
  const TensorC *get_index = inputs[1];
  if (GetElementNum(get_index) != 1) {
    return NNACL_ERR;
  }
  if (get_index->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  int index = ((int *)(get_index->data_))[0];
  if (index < 0 || index > (input0->element_num_ - 1)) {
    return NNACL_ERR;
  }
  TensorC *tensor_index = &input0->tensors_[index];

  TensorC *output = outputs[0];
  if (tensor_index->data_type_ != kTypeUnknown) {
    output->data_type_ = tensor_index->data_type_;
  } else {
    output->data_type_ = input0->tensors_data_type_;
  }
  output->format_ = input0->tensors_[index].format_;

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  if (tensor_index->data_type_ != kTypeUnknown) {
    ShapeSet(output->shape_, &(output->shape_size_), tensor_index->shape_, tensor_index->shape_size_);
  } else {
    const TensorC *input2 = inputs[2];
    if (input2->data_ == NULL) {
      return NNACL_NULL_PTR;
    }
    int *ele_shape_data = (int *)(input2->data_);
    int element_shape[MAX_SHAPE_SIZE];
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
      for (int i = 0; i < input0->element_num_; ++i) {
        TensorC *input = &input0->tensors_[i];
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
