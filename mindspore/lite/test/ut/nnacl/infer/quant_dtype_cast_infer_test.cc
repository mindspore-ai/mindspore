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
#include "mindspore/lite/nnacl/infer/quant_dtype_cast_infer.h"

namespace mindspore {

class QuantDtypeCastInferTest : public mindspore::CommonTest {
 public:
  QuantDtypeCastInferTest() {}
};

TEST_F(QuantDtypeCastInferTest, QuantDtypeCastInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4.0;
  inputs[0]->shape_[1] = 3.0;
  inputs[0]->data_type_ = kNumberTypeFloat32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  QuantDtypeCastParameter *parameter = new QuantDtypeCastParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->srcT_ = kNumberTypeFloat32;
  parameter->dstT_ = kNumberTypeInt;
  int ret = QuantDtypeCastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                     reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
