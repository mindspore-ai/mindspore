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

#include "nnacl/infer/tensorlist_stack_infer.h"
#include "nnacl/infer/infer_register.h"

int TensorListStackInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  TensorListC *input0 = (TensorListC *)(inputs[0]);
  if (input0->element_num_ == 0) {
    return NNACL_ERR;
  }
  const TensorC *ele_shape = inputs[1];  // element shape
  if (ele_shape->data_ == NULL) {
    return NNACL_NULL_PTR;
  }
  int *ele_shape_ptr = (int *)(ele_shape->data_);
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  for (int i = 0; i < GetElementNum(ele_shape); ++i) {
    ShapePush(output_shape, &output_shape_size, ele_shape_ptr[i]);
  }

  int status =
    TensorListMergeShape(output_shape, &output_shape_size, input0->element_shape_, input0->element_shape_size_);
  if (status == NNACL_ERR) {
    return NNACL_ERR;
  }
  if (!TensorListIsFullyDefined(output_shape, output_shape_size)) {
    return NNACL_ERR;
  }
  if (!TensorListIsFullyDefined(input0->element_shape_, input0->element_shape_size_)) {
    for (int i = 0; i < input0->element_num_; ++i) {
      TensorC *tensor_ele = &input0->tensors_[i];
      if (tensor_ele->data_type_ != kTypeUnknown) {
        status = TensorListMergeShape(output_shape, &output_shape_size, tensor_ele->shape_, tensor_ele->shape_size_);
        if (status == NNACL_ERR) {
          return NNACL_ERR;
        }
      }
    }
  }
  TensorC *output = outputs[0];
  output->data_type_ = input0->tensors_data_type_;
  output->format_ = input0->format_;
  ShapeInsert(output_shape, &output_shape_size, 0, input0->element_num_);
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(TensorListStack, PrimType_TensorListStack, TensorListStackInferShape)
