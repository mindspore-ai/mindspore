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
#include "mindspore/lite/nnacl/infer/matmul_infer.h"

namespace mindspore {

class MatmulInferTest : public mindspore::CommonTest {
 public:
  MatmulInferTest() {}
};

TEST_F(MatmulInferTest, MatmulInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 4;
  inputs[1]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  MatMulParameter *parameter = new MatMulParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->a_transpose_ = false;
  parameter->b_transpose_ = true;
  int ret = MatmulInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(MatmulInferTest, MatmulInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 4;
  inputs[0]->shape_[2] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 2;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 5;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  MatMulParameter *parameter = new MatMulParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->a_transpose_ = false;
  parameter->b_transpose_ = false;
  int ret = MatmulInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(MatmulInferTest, MatmulInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 128;
  inputs[0]->shape_[2] = 1;
  inputs[0]->shape_[3] = 1;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 96;
  inputs[1]->shape_[1] = 128;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  MatMulParameter *parameter = new MatMulParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->a_transpose_ = false;
  parameter->b_transpose_ = true;
  int ret = MatmulInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 1);
  ASSERT_EQ(inputs[0]->shape_[1], 128);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 96);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(MatmulInferTest, MatmulInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 1288;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 256;
  inputs[1]->shape_[1] = 1280;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  MatMulParameter *parameter = new MatMulParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->a_transpose_ = false;
  parameter->b_transpose_ = true;
  int ret = MatmulInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 256);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
