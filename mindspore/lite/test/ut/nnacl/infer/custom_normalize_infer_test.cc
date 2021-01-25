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
#include "mindspore/lite/nnacl/infer/custom_normalize_infer.h"

namespace mindspore {

class CustomNormalizeInferTest : public mindspore::CommonTest {
 public:
  CustomNormalizeInferTest() {}
};

TEST_F(CustomNormalizeInferTest, CustomNormalizeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 1;
  std::vector<int> input_data = {2};
  inputs[0]->data_ = input_data.data();
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = CustomNormalizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(CustomNormalizeInferTest, CustomNormalizeInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 1;
  std::vector<int> input_data = {0};
  inputs[0]->data_ = input_data.data();
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = CustomNormalizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
