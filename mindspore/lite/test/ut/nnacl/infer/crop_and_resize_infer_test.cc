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
#include "mindspore/lite/nnacl/infer/crop_and_resize_infer.h"

namespace mindspore {

class CropAndResizeInferTest : public mindspore::CommonTest {
 public:
  CropAndResizeInferTest() {}
};

/*
 * inputs[0].shape: [3, 4, 5, 6]
 * inputs[1].data: null
 * inputs[3].data: 7, 8
 * output[0].shape: [3, 7, 8, 6]
 */
TEST_F(CropAndResizeInferTest, CropAndResizeInferTest0) {
  size_t inputs_size = 4;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 4;
  inputs[0]->shape_[2] = 5;
  inputs[0]->shape_[3] = 6;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->data_ = nullptr;
  inputs[2] = new TensorC;
  inputs[3] = new TensorC;
  inputs[3]->shape_size_ = 1;
  inputs[3]->shape_[0] = 2;
  std::vector<int> input3 = {7, 8};
  inputs[3]->data_ = input3.data();
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = CropAndResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 3);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
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

/*
 * inputs[0].shape: [3, 4, 5, 6]
 * inputs[1].data: not null
 * inputs[1].shape: [9]
 * inputs[3].data: 7, 8
 * output[0].shape: [9, 7, 8, 6]
 */
TEST_F(CropAndResizeInferTest, CropAndResizeInferTest1) {
  size_t inputs_size = 4;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 3;
  inputs[0]->shape_[1] = 4;
  inputs[0]->shape_[2] = 5;
  inputs[0]->shape_[3] = 6;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<int> inputs1 = {11};
  inputs[1]->data_ = inputs1.data();
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 9;
  inputs[2] = new TensorC;
  inputs[3] = new TensorC;
  inputs[3]->shape_size_ = 1;
  inputs[3]->shape_[0] = 2;
  std::vector<int> input3 = {7, 8};
  inputs[3]->data_ = input3.data();
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  int ret = CropAndResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                    reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 9);
  ASSERT_EQ(outputs[0]->shape_[1], 7);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
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
