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
#include "mindspore/lite/nnacl/infer/pad_infer.h"

namespace mindspore {

class PadInferTest : public mindspore::CommonTest {
 public:
  PadInferTest() {}
};

TEST_F(PadInferTest, PadInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  std::vector<int> padding_tensor = {1, 1, 2, 2};
  inputs[1]->data_ = padding_tensor.data();
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PadParameter *parameter = new PadParameter;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = PadInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PadInferTest, PadInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PadParameter *parameter = new PadParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->padding_length = 4;
  parameter->paddings_[0] = 1;
  parameter->paddings_[1] = 1;
  parameter->paddings_[2] = 2;
  parameter->paddings_[3] = 2;
  int ret = PadInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PadInferTest, PadInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PadParameter *parameter = new PadParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->padding_length = 6;
  parameter->paddings_[0] = 0;
  parameter->paddings_[1] = 0;
  parameter->paddings_[2] = 1;
  parameter->paddings_[3] = 2;
  parameter->paddings_[4] = 3;
  parameter->paddings_[5] = 4;
  int ret = PadInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 6);
  ASSERT_EQ(outputs[0]->shape_[2], 11);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PadInferTest, PadInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 4;
  inputs[1] = new TensorC;
  std::vector<int> padding_tensor = {0, 0, 1, 2, 3, 4};
  inputs[1]->data_ = padding_tensor.data();
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = 6;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PadParameter *parameter = new PadParameter;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = PadInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 6);
  ASSERT_EQ(outputs[0]->shape_[2], 11);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PadInferTest, PadInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 4;
  inputs[0]->shape_[3] = 5;
  inputs[1] = new TensorC;
  std::vector<int> padding_tensor = {1, 2, 3, 4, 5, 6, 7, 8};
  inputs[1]->data_ = padding_tensor.data();
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = 8;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PadParameter *parameter = new PadParameter;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = PadInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 10);
  ASSERT_EQ(outputs[0]->shape_[2], 15);
  ASSERT_EQ(outputs[0]->shape_[3], 20);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
