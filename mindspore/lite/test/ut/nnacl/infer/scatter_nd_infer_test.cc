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
#include "mindspore/lite/nnacl/infer/scatter_nd_infer.h"

namespace mindspore {

class ScatterNdInferTest : public mindspore::CommonTest {
 public:
  ScatterNdInferTest() {}
};

TEST_F(ScatterNdInferTest, ScatterNdInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 4;
  std::vector<int> input_data = {1, 2, 3, 4};
  inputs[0]->data_ = input_data.data();
  inputs[1] = new TensorC;
  inputs[2] = new TensorC;
  inputs[2]->data_type_ = kNumberTypeInt8;
  inputs[2]->format_ = kNCHW_H;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = ScatterNdInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  ASSERT_EQ(outputs[0]->shape_[3], 4);
  ASSERT_EQ(outputs[0]->format_, kNCHW_H);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt8);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
