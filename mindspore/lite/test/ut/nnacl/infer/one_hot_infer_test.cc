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
#include "mindspore/lite/nnacl/infer/one_hot_infer.h"

namespace mindspore {

class OneHotInferTest : public mindspore::CommonTest {
 public:
  OneHotInferTest() {}
};

TEST_F(OneHotInferTest, OneHotInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 4;
  inputs[1] = new TensorC;
  std::vector<int> input1_data = {3};
  inputs[1]->data_ = input1_data.data();
  inputs[2] = new TensorC;
  inputs[2]->data_type_ = kNumberTypeFloat32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OneHotParameter *parameter = new OneHotParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->axis_ = -2;
  int ret = OneHotInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeFloat32);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
