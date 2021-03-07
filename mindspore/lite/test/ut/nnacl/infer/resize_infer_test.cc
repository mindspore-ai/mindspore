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
#include "mindspore/lite/nnacl/infer/resize_infer.h"

namespace mindspore {

class ResizeInferTest : public mindspore::CommonTest {
 public:
  ResizeInferTest() {}
};

TEST_F(ResizeInferTest, ResizeInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 5;
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ResizeParameter *parameter = new ResizeParameter;
  parameter->new_width_ = 2;
  parameter->new_height_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = ResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 2);
  ASSERT_EQ(outputs[0]->shape_[3], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ResizeInferTest, ResizeInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 5;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<int32_t> shape_tensor = {4, 3, 2, 5};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->format_ = Format_NHWC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ResizeParameter *parameter = new ResizeParameter;
  // parameter->new_width_ = 2;
  // parameter->new_height_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = ResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 15);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ResizeInferTest, ResizeInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 5;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<float> shape_tensor = {4.0, 3.0, 2.0, 5.0};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeFloat32;
  inputs[1]->format_ = Format_NHWC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ResizeParameter *parameter = new ResizeParameter;
  // parameter->new_width_ = 2;
  // parameter->new_height_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = ResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 15);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(ResizeInferTest, ResizeInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 5;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  std::vector<int32_t> shape_tensor = {4, 3, 2, 5};
  inputs[1]->data_ = shape_tensor.data();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->format_ = Format_NHWC;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ResizeParameter *parameter = new ResizeParameter;
  // parameter->new_width_ = 2;
  // parameter->new_height_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = ResizeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                             reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 15);
  ASSERT_EQ(outputs[0]->shape_[2], 6);
  ASSERT_EQ(outputs[0]->shape_[3], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
