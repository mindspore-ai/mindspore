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
#include "mindspore/lite/nnacl/infer/lstm_infer.h"

namespace mindspore {

class LstmInferTest : public mindspore::CommonTest {
 public:
  LstmInferTest() {}
};

TEST_F(LstmInferTest, LstmInferTest0) {
  size_t inputs_size = 6;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  int seq_len = 2;
  int batch = 4;
  int input_size = 5;
  int hidden_size = 2;
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = seq_len;
  inputs[0]->shape_[1] = batch;
  inputs[0]->shape_[2] = input_size;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = hidden_size * 4;
  inputs[1]->shape_[2] = input_size;
  inputs[2] = new TensorC;
  inputs[3] = new TensorC;
  inputs[4] = new TensorC;
  inputs[5] = new TensorC;
  std::vector<TensorC *> outputs(3, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  outputs[2] = new TensorC;
  LstmParameter *parameter = new LstmParameter;
  parameter->bidirectional_ = false;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = LstmInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], seq_len);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], batch);
  ASSERT_EQ(outputs[0]->shape_[3], hidden_size);
  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 1);
  ASSERT_EQ(outputs[1]->shape_[1], batch);
  ASSERT_EQ(outputs[1]->shape_[2], hidden_size);
  ASSERT_EQ(outputs[2]->shape_size_, 3);
  ASSERT_EQ(outputs[2]->shape_[0], 1);
  ASSERT_EQ(outputs[2]->shape_[1], batch);
  ASSERT_EQ(outputs[2]->shape_[2], hidden_size);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
