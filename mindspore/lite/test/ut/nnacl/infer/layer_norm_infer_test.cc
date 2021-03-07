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
#include "mindspore/lite/nnacl/infer/layer_norm_infer.h"

namespace mindspore {

class LayerNormInferTest : public mindspore::CommonTest {
 public:
  LayerNormInferTest() {}
};

TEST_F(LayerNormInferTest, LayerNormInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LayerNormParameter *parameter = new LayerNormParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->normalized_dims_ = 1;
  parameter->elementwise_affine_ = false;
  parameter->normalized_shape_[0] = 3;
  int ret = LayerNormInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(LayerNormInferTest, LayerNormInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LayerNormParameter *parameter = new LayerNormParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->normalized_dims_ = 3;
  parameter->elementwise_affine_ = false;
  parameter->normalized_shape_[0] = 3;
  int ret = LayerNormInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_PARAM_INVALID);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(LayerNormInferTest, LayerNormInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  LayerNormParameter *parameter = new LayerNormParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->normalized_dims_ = 2;
  parameter->elementwise_affine_ = false;
  parameter->normalized_shape_[0] = 3;
  int ret = LayerNormInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_PARAM_INVALID);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
