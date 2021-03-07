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
#include "mindspore/lite/nnacl/infer/while_infer.h"

namespace mindspore {

class WhileInferTest : public mindspore::CommonTest {
 public:
  WhileInferTest() {}
};

TEST_F(WhileInferTest, WhileInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 7;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = WhileInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                            reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 6);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
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
