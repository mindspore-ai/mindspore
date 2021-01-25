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
#include "mindspore/lite/nnacl/infer/batch_to_space_infer.h"

namespace mindspore {

class BatchToSpaceInferTest : public mindspore::CommonTest {
 public:
  BatchToSpaceInferTest() {}
};

// https://tensorflow.google.cn/api_docs/python/tf/batch_to_space?hl=en
TEST_F(BatchToSpaceInferTest, BatchToSpaceInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 1;
  inputs[0]->shape_[3] = 1;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  BatchToSpaceParameter *parameter = new BatchToSpaceParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->block_shape_[0] = 2;
  parameter->block_shape_[1] = 2;
  parameter->crops_[0] = 0;
  parameter->crops_[1] = 0;
  parameter->crops_[2] = 0;
  parameter->crops_[3] = 0;
  int ret = BatchToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
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

TEST_F(BatchToSpaceInferTest, BatchToSpaceInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 1;
  inputs[0]->shape_[3] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  BatchToSpaceParameter *parameter = new BatchToSpaceParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->block_shape_[0] = 2;
  parameter->block_shape_[1] = 2;
  parameter->crops_[0] = 0;
  parameter->crops_[1] = 0;
  parameter->crops_[2] = 0;
  parameter->crops_[3] = 0;
  int ret = BatchToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
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

TEST_F(BatchToSpaceInferTest, BatchToSpaceInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 1;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  BatchToSpaceParameter *parameter = new BatchToSpaceParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->block_shape_[0] = 2;
  parameter->block_shape_[1] = 2;
  parameter->crops_[0] = 0;
  parameter->crops_[1] = 0;
  parameter->crops_[2] = 0;
  parameter->crops_[3] = 0;
  int ret = BatchToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
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

TEST_F(BatchToSpaceInferTest, BatchToSpaceInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 8;
  inputs[0]->shape_[1] = 1;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 1;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  BatchToSpaceParameter *parameter = new BatchToSpaceParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->block_shape_[0] = 2;
  parameter->block_shape_[1] = 2;
  parameter->crops_[0] = 0;
  parameter->crops_[1] = 0;
  parameter->crops_[2] = 2;
  parameter->crops_[3] = 0;
  int ret = BatchToSpaceInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
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

}  // namespace mindspore
