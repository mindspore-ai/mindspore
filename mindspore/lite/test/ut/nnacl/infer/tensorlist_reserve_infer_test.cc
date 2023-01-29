/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/control/tensorlist_reserve_infer.h"

namespace mindspore {

class TensorlistReserveInferTest : public mindspore::CommonTest {
 public:
  TensorlistReserveInferTest() {}
};

TEST_F(TensorlistReserveInferTest, TensorlistReserveInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 1;
  inputs[0]->shape_[0] = 3;
  std::vector<int> inputs0 = {2, 3, 4};
  inputs[0]->data_ = inputs0.data();
  inputs[0]->data_type_ = kNumberTypeInt32;
  // inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<int> inputs1 = {5};
  inputs[1]->data_ = inputs1.data();
  inputs[1]->data_type_ = kNumberTypeInt32;

  std::vector<TensorC *> outputs(1, NULL);
  auto out = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
  out->tensors_ = nullptr;
  outputs[0] = reinterpret_cast<TensorC *>(out);
  auto *parameter = new OpParameter;
  int ret = TensorListReserveInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                        reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(out->element_num_, 5);
  ASSERT_EQ(out->data_type_, kObjectTypeTensorType);
  ASSERT_EQ(out->element_shape_size_, 3);
  ASSERT_EQ(out->element_shape_[0], 2);
  ASSERT_EQ(out->element_shape_[1], 3);
  ASSERT_EQ(out->element_shape_[2], 4);
  ASSERT_EQ(out->tensors_data_type_, kTypeUnknown);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  for (size_t i = 0; i < out->element_num_; i++) {
    ASSERT_EQ(out->tensors_[i]->shape_size_, 0);
  }
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  lite::FreeOutTensorC(&outputs);
  delete out;
}

}  // namespace mindspore
