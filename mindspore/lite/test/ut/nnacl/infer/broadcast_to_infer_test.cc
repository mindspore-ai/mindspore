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
#include "mindspore/lite/nnacl/infer/broadcast_to_infer.h"

namespace mindspore {

class BroadcastToInferTest : public mindspore::CommonTest {
 public:
  BroadcastToInferTest() {}
};

TEST_F(BroadcastToInferTest, BroadcastToInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 4;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  BroadcastToParameter *param = new BroadcastToParameter;
  param->op_parameter_.infer_flag_ = true;
  param->shape_size_ = 2;
  param->shape_[0] = 5;
  param->shape_[1] = 4;
  int ret = BroadcastToInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(BroadcastToInferTest, BroadcastToInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  BroadcastToParameter *param = new BroadcastToParameter;
  param->op_parameter_.infer_flag_ = true;
  param->shape_size_ = 3;
  param->shape_[0] = 3;
  param->shape_[1] = 3;
  param->shape_[2] = 3;
  int ret = BroadcastToInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(BroadcastToInferTest, BroadcastToInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  BroadcastToParameter *param = new BroadcastToParameter;
  param->op_parameter_.infer_flag_ = true;
  param->shape_size_ = 4;
  param->shape_[0] = 4;
  param->shape_[1] = 5;
  param->shape_[2] = 3;
  param->shape_[3] = 2;
  int ret = BroadcastToInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  ASSERT_EQ(outputs[0]->shape_[3], 2);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(BroadcastToInferTest, BroadcastToInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 4;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  BroadcastToParameter *param = new BroadcastToParameter;
  param->op_parameter_.infer_flag_ = true;
  param->shape_size_ = 4;
  param->shape_[0] = 4;
  param->shape_[1] = 5;
  param->shape_[2] = 3;
  param->shape_[3] = 2;
  int ret = BroadcastToInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_ERR);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
