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
#include "mindspore/lite/nnacl/infer/select_infer.h"

namespace mindspore {

class SelectInferTest : public mindspore::CommonTest {
 public:
  SelectInferTest() {}
};

/*
 * inputs_size: 3
 * outputs_size: 1
 * inputs[1].shape: [4, 5, 6, 7]
 * outputs[0].shape: [4, 5, 6 ,7]
 */
TEST_F(SelectInferTest, SelectInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 4;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 6;
  inputs[1]->shape_[3] = 7;
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->format_ = Format_NHWC;
  inputs[2] = new TensorC;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = SelectInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 7);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

/*
 * inputs_size: 5
 * outputs_size: 2
 * inputs[1].shape: [4, 5, 6, 7]
 * outputs[0].shape: [4, 5, 6 ,7]
 * inputs[2].shape: [8, 9, 10, 11]
 * outputs[1].shape: [8, 9, 10, 11]
 */
TEST_F(SelectInferTest, SelectInferTest1) {
  size_t inputs_size = 5;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 4;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 6;
  inputs[1]->shape_[3] = 7;
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->format_ = Format_NHWC;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 4;
  inputs[2]->shape_[0] = 8;
  inputs[2]->shape_[1] = 9;
  inputs[2]->shape_[2] = 10;
  inputs[2]->shape_[3] = 11;
  inputs[2]->data_type_ = kNumberTypeInt32;
  inputs[2]->format_ = Format_NHWC;
  inputs[3] = new TensorC;
  inputs[4] = new TensorC;
  inputs[5] = new TensorC;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = SelectInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 7);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  ASSERT_EQ(outputs[1]->shape_size_, 4);
  ASSERT_EQ(outputs[1]->shape_[0], 8);
  ASSERT_EQ(outputs[1]->shape_[1], 9);
  ASSERT_EQ(outputs[1]->shape_[2], 10);
  ASSERT_EQ(outputs[1]->shape_[3], 11);
  ASSERT_EQ(outputs[1]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[1]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SelectInferTest, SelectInferTest2) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  TensorListC *inputs1 = new TensorListC;
  inputs1->data_type_ = kObjectTypeTensorType;
  inputs1->format_ = Format_NHWC;
  inputs1->max_elements_num_ = 8;
  inputs1->tensors_data_type_ = kNumberTypeInt32;
  inputs1->element_shape_size_ = 4;
  inputs1->element_shape_[0] = 4;
  inputs1->element_shape_[1] = 5;
  inputs1->element_shape_[2] = 6;
  inputs1->element_shape_[3] = 7;
  inputs[1] = reinterpret_cast<TensorC *>(inputs1);
  inputs[2] = new TensorC;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = reinterpret_cast<TensorC *>(new TensorListC);
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = SelectInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  TensorListC *outputs0 = reinterpret_cast<TensorListC *>(outputs[0]);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs0->element_shape_size_, 4);
  ASSERT_EQ(outputs0->element_shape_[0], 4);
  ASSERT_EQ(outputs0->element_shape_[1], 5);
  ASSERT_EQ(outputs0->element_shape_[2], 6);
  ASSERT_EQ(outputs0->element_shape_[3], 7);
  ASSERT_EQ(outputs0->tensors_data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs0->max_elements_num_, 8);
  ASSERT_EQ(outputs0->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SelectInferTest, SelectInferTest3) {
  size_t inputs_size = 5;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  TensorListC *inputs1 = new TensorListC;
  inputs1->data_type_ = kObjectTypeTensorType;
  inputs1->format_ = Format_NHWC;
  inputs1->max_elements_num_ = 8;
  inputs1->tensors_data_type_ = kNumberTypeInt32;
  inputs1->element_shape_size_ = 4;
  inputs1->element_shape_[0] = 4;
  inputs1->element_shape_[1] = 5;
  inputs1->element_shape_[2] = 6;
  inputs1->element_shape_[3] = 7;
  inputs[1] = reinterpret_cast<TensorC *>(inputs1);
  // inputs[2] = new TensorC;
  TensorListC *inputs2 = new TensorListC;
  inputs2->data_type_ = kObjectTypeTensorType;
  inputs2->format_ = Format_NHWC;
  inputs2->max_elements_num_ = 8;
  inputs2->tensors_data_type_ = kNumberTypeInt32;
  inputs2->element_shape_size_ = 4;
  inputs2->element_shape_[0] = 8;
  inputs2->element_shape_[1] = 9;
  inputs2->element_shape_[2] = 10;
  inputs2->element_shape_[3] = 11;
  inputs[2] = reinterpret_cast<TensorC *>(inputs2);
  inputs[3] = new TensorC;
  inputs[4] = new TensorC;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = reinterpret_cast<TensorC *>(new TensorListC);
  outputs[1] = reinterpret_cast<TensorC *>(new TensorListC);
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = SelectInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  TensorListC *outputs0 = reinterpret_cast<TensorListC *>(outputs[0]);
  TensorListC *outputs1 = reinterpret_cast<TensorListC *>(outputs[1]);

  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs0->element_shape_size_, 4);
  ASSERT_EQ(outputs0->element_shape_[0], 4);
  ASSERT_EQ(outputs0->element_shape_[1], 5);
  ASSERT_EQ(outputs0->element_shape_[2], 6);
  ASSERT_EQ(outputs0->element_shape_[3], 7);
  ASSERT_EQ(outputs0->tensors_data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs0->max_elements_num_, 8);
  ASSERT_EQ(outputs0->format_, Format_NHWC);

  ASSERT_EQ(outputs1->element_shape_size_, 4);
  ASSERT_EQ(outputs1->element_shape_[0], 8);
  ASSERT_EQ(outputs1->element_shape_[1], 9);
  ASSERT_EQ(outputs1->element_shape_[2], 10);
  ASSERT_EQ(outputs1->element_shape_[3], 11);
  ASSERT_EQ(outputs1->tensors_data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs1->max_elements_num_, 8);
  ASSERT_EQ(outputs1->format_, Format_NHWC);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
