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
#include "mindspore/lite/nnacl/infer/range_infer.h"

namespace mindspore {

class RangeInferTest : public mindspore::CommonTest {
 public:
  RangeInferTest() {}
};

// https://tensorflow.google.cn/api_docs/python/tf/range?hl=en
TEST_F(RangeInferTest, RangeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  RangeParameter *parameter = new RangeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->limit_ = 18;
  parameter->start_ = 3;
  parameter->delta_ = 3;  // delta must be decimal
  int ret = RangeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(RangeInferTest, RangeInferTest1) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  std::vector<int> input0_data = {3};
  std::vector<int> input1_data = {18};
  std::vector<int> input2_data = {3};
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 1;
  inputs[0]->data_ = input0_data.data();
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  inputs[1]->data_ = input1_data.data();
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 1;
  inputs[2]->data_ = input2_data.data();
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  RangeParameter *parameter = new RangeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->limit_ = 18;
  // parameter->start_ = 3;
  // parameter->delta_ = 3;
  int ret = RangeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(RangeInferTest, RangeInferTest2) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  std::vector<float> input0_data = {3.0};
  std::vector<float> input1_data = {18.0};
  std::vector<float> input2_data = {3.0};
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 1;
  inputs[0]->data_ = input0_data.data();
  inputs[0]->data_type_ = kNumberTypeFloat32;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  inputs[1]->data_ = input1_data.data();
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 1;
  inputs[2]->data_ = input2_data.data();
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  RangeParameter *parameter = new RangeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->limit_ = 18;
  // parameter->start_ = 3;
  // parameter->delta_ = 3;
  int ret = RangeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
