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
#include "mindspore/lite/nnacl/infer/concat_infer.h"

namespace mindspore {

class ConcatInferTest : public mindspore::CommonTest {
 public:
  ConcatInferTest() {}
};

TEST_F(ConcatInferTest, ConcatInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 4;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 3;
  inputs[1]->shape_[1] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = 0;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ConcatInferTest, ConcatInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 4;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 3;
  inputs[1]->shape_[1] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = 1;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 8);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ConcatInferTest, ConcatInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 2;
  inputs[1]->shape_[2] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = 0;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 10);
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

TEST_F(ConcatInferTest, ConcatInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 2;
  inputs[1]->shape_[3] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = 0;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 10);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ConcatInferTest, ConcatInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 6;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 4;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 2;
  inputs[1]->shape_[3] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = -1;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ConcatInferTest, ConcatInferTest5) {
  size_t inputs_size = 4;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 14;
  inputs[0]->shape_[2] = 14;
  inputs[0]->shape_[3] = 192;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = 14;
  inputs[1]->shape_[2] = 14;
  inputs[1]->shape_[3] = 192;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 4;
  inputs[2]->shape_[0] = 1;
  inputs[2]->shape_[1] = 14;
  inputs[2]->shape_[2] = 14;
  inputs[2]->shape_[3] = 192;
  inputs[3] = new TensorC;
  inputs[3]->shape_size_ = 4;
  inputs[3]->shape_[0] = 1;
  inputs[3]->shape_[1] = 14;
  inputs[3]->shape_[2] = 14;
  inputs[3]->shape_[3] = 192;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConcatParameter *parameter = new ConcatParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = 3;
  int ret = ConcatInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 14);
  ASSERT_EQ(outputs[0]->shape_[2], 14);
  ASSERT_EQ(outputs[0]->shape_[3], 768);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
