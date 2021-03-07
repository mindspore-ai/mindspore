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
#include "mindspore/lite/nnacl/infer/slice_infer.h"

namespace mindspore {

class SliceInferTest : public mindspore::CommonTest {
 public:
  SliceInferTest() {}
};

TEST_F(SliceInferTest, SliceInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SliceParameter *parameter = new SliceParameter;
  parameter->begin_[0] = 1;
  parameter->begin_[1] = 1;
  parameter->size_[0] = 1;
  parameter->size_[1] = 3;
  parameter->axis_[0] = 0;
  parameter->axis_[1] = 1;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SliceInferTest, SliceInferTest1) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SliceParameter *parameter = new SliceParameter;
  parameter->begin_[0] = 1;
  parameter->begin_[1] = 0;
  parameter->begin_[2] = 0;
  parameter->size_[0] = 1;
  parameter->size_[1] = 1;
  parameter->size_[2] = 3;
  parameter->axis_[0] = 0;
  parameter->axis_[1] = 1;
  parameter->axis_[2] = 2;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SliceInferTest, SliceInferTest2) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SliceParameter *parameter = new SliceParameter;
  parameter->begin_[0] = 1;
  parameter->begin_[1] = 0;
  parameter->begin_[2] = 0;
  parameter->size_[0] = 1;
  parameter->size_[1] = 2;
  parameter->size_[2] = 3;
  parameter->axis_[0] = 0;
  parameter->axis_[1] = 1;
  parameter->axis_[2] = 2;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SliceInferTest, SliceInferTest3) {
  size_t inputs_size = 5;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 4;
  inputs[1] = new TensorC;
  std::vector<int> inputs1 = {1, 0, 0};
  inputs[1]->data_ = inputs1.data();
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 3;
  inputs[2] = new TensorC;
  std::vector<int> inputs2 = {2, 2, 3};
  inputs[2]->data_ = inputs2.data();
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 3;
  inputs[3] = new TensorC;
  std::vector<int> inputs3 = {0, 1, 2};
  inputs[3]->data_ = inputs3.data();
  inputs[3]->shape_size_ = 1;
  inputs[3]->shape_[0] = 3;
  inputs[4] = new TensorC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SliceParameter *parameter = new SliceParameter;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SliceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
