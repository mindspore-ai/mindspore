/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "src/common/tensor_util.h"
#include "mindspore/lite/nnacl/infer/tensorlist_getitem_infer.h"

namespace mindspore {

class TensorlistGetItemInferTest : public mindspore::CommonTest {
 public:
  TensorlistGetItemInferTest() {}
};

// [[1, 2], [3, 4, 5], [6, 7, 8, 9]] -> [6, 7, 8, 9]
TEST_F(TensorlistGetItemInferTest, TensorlistGetItemInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  TensorListC *input0 = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
  input0->element_num_ = 3;
  input0->tensors_ = reinterpret_cast<TensorC *>(malloc(input0->element_num_ * sizeof(TensorC)));
  input0->tensors_[0].shape_size_ = 2;
  input0->tensors_[0].shape_[0] = 1;
  input0->tensors_[0].shape_[1] = 2;
  input0->tensors_[0].data_type_ = kNumberTypeInt32;
  input0->tensors_[1].shape_size_ = 3;
  input0->tensors_[1].shape_[0] = 3;
  input0->tensors_[1].shape_[1] = 4;
  input0->tensors_[1].shape_[2] = 5;
  input0->tensors_[1].data_type_ = kNumberTypeInt32;
  input0->tensors_[2].shape_size_ = 4;
  input0->tensors_[2].shape_[0] = 6;
  input0->tensors_[2].shape_[1] = 7;
  input0->tensors_[2].shape_[2] = 8;
  input0->tensors_[2].shape_[3] = 9;
  input0->tensors_[2].data_type_ = kNumberTypeInt32;
  // input0->tensors_[2]->format_ = Format_NHWC;
  inputs[0] = reinterpret_cast<TensorC *>(input0);
  inputs[0]->data_type_ = kObjectTypeTensorType;

  inputs[1] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<int> inputs1_data = {2};
  inputs[1]->data_ = inputs1_data.data();

  inputs[2] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = TensorListGetItemInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                        reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 9);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i]->data_type_ == kObjectTypeTensorType) {
      TensorListC *tensorListC = reinterpret_cast<TensorListC *>(inputs[i]);
      lite::FreeTensorListC(tensorListC);
    } else {
      free(inputs[i]);
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    free(outputs[i]);
  }
}

// retest mergeshape

}  // namespace mindspore
