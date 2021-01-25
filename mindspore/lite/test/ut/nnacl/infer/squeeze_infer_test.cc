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
#include "mindspore/lite/nnacl/infer/squeeze_infer.h"

namespace mindspore {

class SqueezeInferTest : public mindspore::CommonTest {
 public:
  SqueezeInferTest() {}
};

TEST_F(SqueezeInferTest, SqueezeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  inputs[0]->shape_[4] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SqueezeParameter *parameter = new SqueezeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_size_ = 0;
  int ret = SqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SqueezeInferTest, SqueezeInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  inputs[0]->shape_[4] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SqueezeParameter *parameter = new SqueezeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_size_ = 1;
  parameter->axis_[0] = 1;
  int ret = SqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 1);
  ASSERT_EQ(outputs[0]->shape_[3], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SqueezeInferTest, SqueezeInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  inputs[0]->shape_[4] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SqueezeParameter *parameter = new SqueezeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_size_ = 2;
  parameter->axis_[0] = 1;
  parameter->axis_[1] = 3;
  int ret = SqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SqueezeInferTest, SqueezeInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  inputs[0]->shape_[4] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SqueezeParameter *parameter = new SqueezeParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_size_ = 1;
  parameter->axis_[0] = 0;
  int ret = SqueezeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_PARAM_INVALID);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
