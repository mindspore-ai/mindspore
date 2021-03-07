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
#include "mindspore/lite/nnacl/infer/lsh_projection_infer.h"

namespace mindspore {

class LshProjectionInferTest : public mindspore::CommonTest {
 public:
  LshProjectionInferTest() {}
};

TEST_F(LshProjectionInferTest, LshProjectionInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LshProjectionParameter *parameter = new LshProjectionParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->lsh_type_ = LshProjectionType_SPARSE;
  int ret = LshProjectionInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(LshProjectionInferTest, LshProjectionInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LshProjectionParameter *parameter = new LshProjectionParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->lsh_type_ = LshProjectionType_DENSE;
  int ret = LshProjectionInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 4 * 3);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(LshProjectionInferTest, LshProjectionInferTest2) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 5;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 5;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LshProjectionParameter *parameter = new LshProjectionParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->lsh_type_ = LshProjectionType_DENSE;
  int ret = LshProjectionInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 4 * 3);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}  // note: may be error

}  // namespace mindspore
