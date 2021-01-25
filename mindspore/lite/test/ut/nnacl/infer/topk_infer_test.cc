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
#include "mindspore/lite/nnacl/infer/topk_infer.h"

namespace mindspore {

class TopKInferTest : public mindspore::CommonTest {
 public:
  TopKInferTest() {}
};

TEST_F(TopKInferTest, TopKInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 5;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  TopkParameter *parameter = new TopkParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->k_ = 6;
  int ret = TopKInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 4);
  ASSERT_EQ(outputs[1]->shape_[1], 3);
  ASSERT_EQ(outputs[1]->shape_[2], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(TopKInferTest, TopKInferInputsSize2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 5;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<int> tmp = {7};
  inputs[1]->data_ = tmp.data();
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  TopkParameter *parameter = new TopkParameter;
  parameter->op_parameter_.infer_flag_ = true;
  // parameter->k_ = 6;
  int ret = TopKInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 7);
  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 4);
  ASSERT_EQ(outputs[1]->shape_[1], 3);
  ASSERT_EQ(outputs[1]->shape_[2], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
