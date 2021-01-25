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
#include "mindspore/lite/nnacl/infer/detection_post_process_infer.h"

namespace mindspore {

class DetectionPostProcessInferTest : public mindspore::CommonTest {
 public:
  DetectionPostProcessInferTest() {}
};

TEST_F(DetectionPostProcessInferTest, DetectionPostProcessInferTest0) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 10;
  inputs[2] = new TensorC;
  inputs[2]->shape_[0] = 5;
  std::vector<TensorC *> outputs(4, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  outputs[2] = new TensorC;
  outputs[3] = new TensorC;
  DetectionPostProcessParameter *parameter = new DetectionPostProcessParameter;
  parameter->op_parameter_.infer_flag_ = true;
  parameter->max_detections_ = 20;
  parameter->max_classes_per_detection_ = 3;
  parameter->num_classes_ = 10;
  int ret = DetectionPostProcessInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(),
                                           outputs.size(), reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 60);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeFloat32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  ASSERT_EQ(outputs[1]->shape_size_, 2);
  ASSERT_EQ(outputs[1]->shape_[0], 1);
  ASSERT_EQ(outputs[1]->shape_[1], 60);
  ASSERT_EQ(outputs[1]->data_type_, kNumberTypeFloat32);
  ASSERT_EQ(outputs[1]->format_, Format_NHWC);
  ASSERT_EQ(outputs[2]->shape_size_, 2);
  ASSERT_EQ(outputs[2]->shape_[0], 1);
  ASSERT_EQ(outputs[2]->shape_[1], 60);
  ASSERT_EQ(outputs[2]->data_type_, kNumberTypeFloat32);
  ASSERT_EQ(outputs[2]->format_, Format_NHWC);
  ASSERT_EQ(outputs[3]->shape_size_, 1);
  ASSERT_EQ(outputs[3]->shape_[0], 1);
  ASSERT_EQ(outputs[3]->data_type_, kNumberTypeFloat32);
  ASSERT_EQ(outputs[3]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
