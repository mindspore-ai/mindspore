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
#include "mindspore/lite/nnacl/infer/sparse_to_dense_infer.h"

namespace mindspore {

class SparseToDenseInferTest : public mindspore::CommonTest {
 public:
  SparseToDenseInferTest() {}
};

TEST_F(SparseToDenseInferTest, SparseToDenseInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<int> data_tmp = {2, 3, 4, 5};
  inputs[1]->data_ = data_tmp.data();
  inputs[2] = new TensorC;
  inputs[2]->data_type_ = kNumberTypeInt32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = SparseToDenseInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->shape_[3], 5);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
