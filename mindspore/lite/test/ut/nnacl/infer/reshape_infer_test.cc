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
#include "mindspore/lite/nnacl/infer/reshape_infer.h"
#include "nnacl/reshape_parameter.h"

namespace mindspore {

class ReshapeInferTest : public mindspore::CommonTest {
 public:
  ReshapeInferTest() {}
};

TEST_F(ReshapeInferTest, ReshapeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->shape_dim_ = 1;
  parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  std::vector<int32_t> shape_tensor = {6};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  std::vector<int8_t> shape_tensor = {6};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt8;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  std::vector<uint32_t> shape_tensor = {6};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeUInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 12;
  inputs[1] = new TensorC;
  std::vector<float> shape_tensor = {3.0, 4.0};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeFloat;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest5) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 12;
  inputs[1] = new TensorC;
  std::vector<int64_t> shape_tensor = {3, 4};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt64;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest6) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 3;
  inputs[1] = new TensorC;
  std::vector<int64_t> shape_tensor = {3, 6};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt64;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest7) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 3;
  inputs[1] = new TensorC;
  std::vector<int64_t> shape_tensor = {3, -1};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt64;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest8) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 8;
  inputs[1] = new TensorC;
  std::vector<int64_t> shape_tensor = {1, 2, 5, 4};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt64;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ReshapeInferTest, ReshapeInferTest9) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 8;
  inputs[1] = new TensorC;
  std::vector<int64_t> shape_tensor = {8, 5, -1, 1};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt64;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ReshapeParameter *parameter = new ReshapeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->shape_size_ = 1;
  // parameter->shape_[0] = 6;
  int ret = ReshapeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 8);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 1);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
