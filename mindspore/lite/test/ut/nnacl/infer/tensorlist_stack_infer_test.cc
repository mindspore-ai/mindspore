/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/control/tensorlist_stack_infer.h"

namespace mindspore {

class TensorlistStackInferTest : public mindspore::CommonTest {
 public:
  TensorlistStackInferTest() {}
};

// TensorList[[2, 4], [2, 4], [2, 4]] -> size(3, 2, 4)
TEST_F(TensorlistStackInferTest, TensorlistStackInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  auto *input0 = new TensorListC;
  input0->element_num_ = 3;
  auto in_tensors_c = reinterpret_cast<TensorC *>(malloc(input0->element_num_ * sizeof(TensorC)));
  input0->tensors_ = &in_tensors_c;
  input0->element_shape_size_ = 2;
  input0->element_shape_[0] = 2;
  input0->element_shape_[1] = 4;
  input0->tensors_data_type_ = kNumberTypeInt32;

  in_tensors_c[0].shape_size_ = 2;
  in_tensors_c[0].shape_[0] = 2;
  in_tensors_c[0].shape_[1] = 4;
  in_tensors_c[0].data_type_ = kNumberTypeInt32;

  in_tensors_c[1].shape_size_ = 2;
  in_tensors_c[1].shape_[0] = 2;
  in_tensors_c[1].shape_[1] = 4;
  in_tensors_c[1].data_type_ = kNumberTypeInt32;

  in_tensors_c[2].shape_size_ = 2;
  in_tensors_c[2].shape_[0] = 2;
  in_tensors_c[2].shape_[1] = 4;
  in_tensors_c[2].data_type_ = kNumberTypeInt32;
  // input0->tensors_[2]->format_ = Format_NHWC;
  inputs[0] = reinterpret_cast<TensorC *>(input0);
  inputs[0]->data_type_ = kObjectTypeTensorType;

  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  std::vector<int> inputs1_data = {-1, 4};
  inputs[1]->data_ = inputs1_data.data();

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  auto *parameter = new OpParameter;
  int ret = TensorListStackInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i]->data_type_ == kObjectTypeTensorType) {
      auto *tensorList_c = reinterpret_cast<TensorListC *>(inputs[i]);
      free(*tensorList_c->tensors_);
    }
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

// retest mergeshape

}  // namespace mindspore
