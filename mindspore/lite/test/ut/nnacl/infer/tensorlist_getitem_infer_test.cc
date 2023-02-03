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
#include "nnacl/infer/control/tensorlist_getitem_infer.h"

namespace mindspore {

class TensorlistGetItemInferTest : public mindspore::CommonTest {
 public:
  TensorlistGetItemInferTest() {}
};

// [[1, 2], [3, 4, 5], [6, 7, 8, 9]] -> [6, 7, 8, 9]
TEST_F(TensorlistGetItemInferTest, TensorlistGetItemInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  auto *input0 = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
  input0->element_num_ = 3;
  auto in_tensors_c = reinterpret_cast<TensorC *>(malloc(input0->element_num_ * sizeof(TensorC)));
  input0->tensors_ = &in_tensors_c;
  in_tensors_c[0].shape_size_ = 2;
  in_tensors_c[0].shape_[0] = 1;
  in_tensors_c[0].shape_[1] = 2;
  in_tensors_c[0].data_type_ = kNumberTypeInt32;
  in_tensors_c[1].shape_size_ = 3;
  in_tensors_c[1].shape_[0] = 3;
  in_tensors_c[1].shape_[1] = 4;
  in_tensors_c[1].shape_[2] = 5;
  in_tensors_c[1].data_type_ = kNumberTypeInt32;
  in_tensors_c[2].shape_size_ = 4;
  in_tensors_c[2].shape_[0] = 6;
  in_tensors_c[2].shape_[1] = 7;
  in_tensors_c[2].shape_[2] = 8;
  in_tensors_c[2].shape_[3] = 9;
  in_tensors_c[2].data_type_ = kNumberTypeInt32;
  // input0->tensors_[2]->format_ = Format_NHWC;
  inputs[0] = reinterpret_cast<TensorC *>(input0);
  inputs[0]->data_type_ = kObjectTypeTensorType;

  inputs[1] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  std::vector<int> inputs1_data = {2};
  inputs[1]->data_ = inputs1_data.data();

  inputs[2] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  auto *parameter = new OpParameter;
  int ret = TensorListGetItemInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                        reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 9);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i]->data_type_ == kObjectTypeTensorType) {
      auto *tensorList_c = reinterpret_cast<TensorListC *>(inputs[i]);
      free(*tensorList_c->tensors_);
    }
    free(inputs[i]);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    free(outputs[i]);
  }
}

// retest mergeshape

}  // namespace mindspore
