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
#include "mindspore/lite/nnacl/infer/depth_to_space_infer.h"
#include "src/tensor.h"

namespace mindspore {

class DepthToSpaceInferTest : public mindspore::CommonTest {
 public:
  DepthToSpaceInferTest() {}
};

TEST_F(DepthToSpaceInferTest, DepthToSpaceInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->format_ = Format_NHWC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 1;
  inputs[0]->shape_[3] = 12;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  DepthToSpaceParameter *param = new DepthToSpaceParameter;
  param->op_parameter_.infer_flag_ = true;
  param->block_size_ = 2;
  int ret = DepthToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthToSpaceInferTest, DepthToSpaceInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->format_ = Format_NHWC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 4;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  DepthToSpaceParameter *param = new DepthToSpaceParameter;
  param->op_parameter_.infer_flag_ = true;
  param->block_size_ = 2;
  int ret = DepthToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthToSpaceInferTest, DepthToSpaceInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->format_ = Format_NHWC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 1;
  inputs[0]->shape_[3] = 4;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  DepthToSpaceParameter *param = new DepthToSpaceParameter;
  param->op_parameter_.infer_flag_ = true;
  param->block_size_ = 2;
  int ret = DepthToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthToSpaceInferTest, DepthToSpaceInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->format_ = Format_NHWC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 7;
  inputs[0]->shape_[3] = 32;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  DepthToSpaceParameter *param = new DepthToSpaceParameter;
  param->op_parameter_.infer_flag_ = true;
  param->block_size_ = 4;
  int ret = DepthToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(param));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 20);
  ASSERT_EQ(outputs[0]->shape_[2], 28);
  ASSERT_EQ(outputs[0]->shape_[3], 2);
  delete param;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthToSpaceInferTest, DepthToSpaceInferTest4) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  // inputs[0]->format_ = Format_NHWC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 7;
  inputs[0]->shape_[2] = 32;
  std::vector<TensorC *> outputs(inputs_size, NULL);
  outputs[0] = new TensorC;
  DepthToSpaceParameter *param = new DepthToSpaceParameter;
  param->op_parameter_.infer_flag_ = true;
  param->block_size_ = 4;
  int ret = DepthToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
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
