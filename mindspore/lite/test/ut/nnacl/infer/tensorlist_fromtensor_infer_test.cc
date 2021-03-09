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
#include "src/common/tensor_util.h"
#include "mindspore/lite/nnacl/infer/tensorlist_fromtensor_infer.h"

namespace mindspore {

class TensorlistFromtensorInferTest : public mindspore::CommonTest {
 public:
  TensorlistFromtensorInferTest() {}
};

TEST_F(TensorlistFromtensorInferTest, TensorlistFromtensorInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 6;
  inputs[0]->shape_[2] = 5;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<int> tmp = {-1, 5};
  inputs[1]->data_ = tmp.data();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 1;
  inputs[1]->shape_[1] = 2;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorListC)));
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = TensorListFromTensorInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(),
                                           outputs.size(), reinterpret_cast<OpParameter *>(parameter));
  TensorListC *out = reinterpret_cast<TensorListC *>(outputs[0]);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(out->element_num_, 4);
  ASSERT_EQ(out->data_type_, kObjectTypeTensorType);
  ASSERT_EQ(out->element_shape_size_, 2);
  ASSERT_EQ(out->element_shape_[0], -1);
  ASSERT_EQ(out->element_shape_[1], 5);
  ASSERT_EQ(out->tensors_data_type_, kNumberTypeInt32);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  for (size_t i = 0; i < out->element_num_; i++) {
    ASSERT_EQ(out->tensors_[i].shape_size_, 2);
    ASSERT_EQ(out->tensors_[i].shape_[0], 6);
    ASSERT_EQ(out->tensors_[i].shape_[1], 5);
  }
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->data_type_ == kObjectTypeTensorType) {
      TensorListC *tensorListC = reinterpret_cast<TensorListC *>(outputs[i]);
      lite::FreeTensorListC(tensorListC);
    } else {
      delete outputs[i];
    }
  }
}

}  // namespace mindspore
