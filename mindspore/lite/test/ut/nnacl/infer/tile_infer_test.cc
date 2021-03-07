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
#include "mindspore/lite/nnacl/infer/tile_infer.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {

class TileInferTest : public mindspore::CommonTest {
 public:
  TileInferTest() {}
};

TEST_F(TileInferTest, TileInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  TileParameter *parameter = new TileParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->multiples_size_ = 2;
  parameter->multiples_[0] = 4;
  parameter->multiples_[1] = 5;
  parameter->dims_size_ = 2;
  parameter->dims_[0] = 0;
  parameter->dims_[1] = 1;
  int ret = TileInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 2 * 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3 * 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(TileInferTest, TileInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  TileParameter *parameter = new TileParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->multiples_size_ = 2;
  parameter->multiples_[0] = 4;
  parameter->multiples_[1] = 5;
  parameter->dims_size_ = 2;
  parameter->dims_[0] = 1;
  parameter->dims_[1] = 2;
  int ret = TileInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 3 * 4);
  ASSERT_EQ(outputs[0]->shape_[2], 6 * 5);
  ASSERT_EQ(outputs[0]->shape_[3], 7);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
