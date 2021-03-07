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
#include "mindspore/lite/nnacl/infer/space_to_batch_infer.h"

namespace mindspore {

class SpaceToBatchInferTest : public mindspore::CommonTest {
 public:
  SpaceToBatchInferTest() {}
};

TEST_F(SpaceToBatchInferTest, SpaceToBatchInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 1;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SpaceToBatchParameter *parameter = new SpaceToBatchParameter;
  parameter->m_ = 2;
  parameter->block_sizes_[0] = 2;
  parameter->block_sizes_[1] = 2;
  parameter->paddings_[0] = 0;
  parameter->paddings_[1] = 0;
  parameter->paddings_[2] = 0;
  parameter->paddings_[3] = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SpaceToBatchInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
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

TEST_F(SpaceToBatchInferTest, SpaceToBatchInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 3;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SpaceToBatchParameter *parameter = new SpaceToBatchParameter;
  parameter->m_ = 2;
  parameter->block_sizes_[0] = 2;
  parameter->block_sizes_[1] = 2;
  parameter->paddings_[0] = 0;
  parameter->paddings_[1] = 0;
  parameter->paddings_[2] = 0;
  parameter->paddings_[3] = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SpaceToBatchInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 1);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SpaceToBatchInferTest, SpaceToBatchInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 4;
  inputs[0]->shape_[2] = 4;
  inputs[0]->shape_[3] = 1;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SpaceToBatchParameter *parameter = new SpaceToBatchParameter;
  parameter->m_ = 2;
  parameter->block_sizes_[0] = 2;
  parameter->block_sizes_[1] = 2;
  parameter->paddings_[0] = 0;
  parameter->paddings_[1] = 0;
  parameter->paddings_[2] = 0;
  parameter->paddings_[3] = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SpaceToBatchInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 1);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(SpaceToBatchInferTest, SpaceToBatchInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 4;
  inputs[0]->shape_[3] = 1;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  SpaceToBatchParameter *parameter = new SpaceToBatchParameter;
  parameter->m_ = 2;
  parameter->block_sizes_[0] = 2;
  parameter->block_sizes_[1] = 2;
  parameter->paddings_[0] = 0;
  parameter->paddings_[1] = 0;
  parameter->paddings_[2] = 2;
  parameter->paddings_[3] = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = SpaceToBatchInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                   reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 8);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
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
