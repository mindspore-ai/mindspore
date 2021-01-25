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
#include "mindspore/lite/nnacl/infer/split_infer.h"

namespace mindspore {

class SplitInferTest : public mindspore::CommonTest {
 public:
  SplitInferTest() {}
};

TEST_F(SplitInferTest, SplitInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 40;
  std::vector<TensorC *> outputs(3, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  outputs[2] = new TensorC;
  SplitParameter *parameter = new SplitParameter;
  parameter->num_split_ = 3;
  // parameter->split_count_ = 3;
  std::vector<int> split_sizes = {4, 15, 11};
  parameter->split_sizes_ = split_sizes.data();
  parameter->split_dim_ = 1;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SplitInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[1]->shape_size_, 2);
  ASSERT_EQ(outputs[1]->shape_[0], 5);
  ASSERT_EQ(outputs[1]->shape_[1], 15);
  ASSERT_EQ(outputs[2]->shape_size_, 2);
  ASSERT_EQ(outputs[2]->shape_[0], 5);
  ASSERT_EQ(outputs[2]->shape_[1], 11);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SplitInferTest, SplitInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 8;
  inputs[0]->shape_[2] = 6;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  SplitParameter *parameter = new SplitParameter;
  parameter->num_split_ = 0;
  // parameter->num_split_ = 2;
  // parameter->split_count_ = 0;
  parameter->split_dim_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SplitInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 8);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 2);
  ASSERT_EQ(outputs[1]->shape_[1], 8);
  ASSERT_EQ(outputs[1]->shape_[2], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SplitInferTest, SplitInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(3, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  outputs[2] = new TensorC;
  SplitParameter *parameter = new SplitParameter;
  parameter->num_split_ = 3;
  parameter->split_count_ = 3;
  parameter->split_sizes_ = reinterpret_cast<int *>(malloc(sizeof(int) * 3));
  parameter->split_sizes_[0] = 1;
  parameter->split_sizes_[1] = 4;
  parameter->split_sizes_[2] = 2;
  parameter->split_dim_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SplitInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  ASSERT_EQ(outputs[1]->shape_size_, 4);
  ASSERT_EQ(outputs[1]->shape_[0], 4);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
  ASSERT_EQ(outputs[1]->shape_[2], 6);
  ASSERT_EQ(outputs[1]->shape_[3], 4);
  ASSERT_EQ(outputs[2]->shape_size_, 4);
  ASSERT_EQ(outputs[2]->shape_[0], 4);
  ASSERT_EQ(outputs[2]->shape_[1], 5);
  ASSERT_EQ(outputs[2]->shape_[2], 6);
  ASSERT_EQ(outputs[2]->shape_[3], 2);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
  free(parameter->split_sizes_);
}

TEST_F(SplitInferTest, SplitInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  SplitParameter *parameter = new SplitParameter;
  parameter->num_split_ = 0;
  // parameter->num_split_ = 2;
  // parameter->split_count_ = 0;
  parameter->split_dim_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SplitInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 7);
  ASSERT_EQ(outputs[1]->shape_size_, 4);
  ASSERT_EQ(outputs[1]->shape_[0], 2);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
  ASSERT_EQ(outputs[1]->shape_[2], 6);
  ASSERT_EQ(outputs[1]->shape_[3], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SplitInferTest, SplitInferTest4) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1200;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(100, NULL);
  for (size_t i = 0; i < 100; i++) {
    outputs[i] = new TensorC;
  }
  SplitParameter *parameter = new SplitParameter;
  parameter->num_split_ = 0;
  // parameter->num_split_ = 2;
  // parameter->split_count_ = 0;
  parameter->split_dim_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SplitInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  for (size_t i = 0; i < 100; i++) {
    ASSERT_EQ(outputs[i]->shape_size_, 4);
    ASSERT_EQ(outputs[i]->shape_[0], 12);
    ASSERT_EQ(outputs[i]->shape_[1], 5);
    ASSERT_EQ(outputs[i]->shape_[2], 6);
    ASSERT_EQ(outputs[i]->shape_[3], 7);
  }

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
