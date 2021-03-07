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
#include "mindspore/lite/nnacl/infer/custom_predict_infer.h"

namespace mindspore {

class CustomPredictInferTest : public mindspore::CommonTest {
 public:
  CustomPredictInferTest() {}
};

TEST_F(CustomPredictInferTest, CustomPredictInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  CustomPredictParameter *parameter = new CustomPredictParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->output_num = 5;
  int ret = CustomPredictInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  ASSERT_EQ(outputs[1]->shape_size_, 1);
  ASSERT_EQ(outputs[1]->shape_[0], 5);
  ASSERT_EQ(outputs[1]->data_type_, kNumberTypeFloat32);
  ASSERT_EQ(outputs[1]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
