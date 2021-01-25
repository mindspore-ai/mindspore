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
#include "mindspore/lite/nnacl/infer/pooling_grad_infer.h"

namespace mindspore {

class PoolingGradInferTest : public mindspore::CommonTest {
 public:
  PoolingGradInferTest() {}
};

TEST_F(PoolingGradInferTest, PoolingGradInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 14;
  inputs[0]->shape_[2] = 14;
  inputs[0]->shape_[3] = 3;
  inputs[1] = new TensorC;
  inputs[2] = new TensorC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  PoolingParameter *parameter = new PoolingParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->window_w_ = 3;
  parameter->window_h_ = 3;
  parameter->stride_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_same;
  parameter->round_mode_ = RoundMode_Floor;
  int ret = PoolingGradInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                  reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 14);
  ASSERT_EQ(outputs[0]->shape_[2], 14);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
