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
#include "mindspore/lite/nnacl/infer/depthwise_conv2d_infer.h"

namespace mindspore {

class DepthwiseConv2dInferTest : public mindspore::CommonTest {
 public:
  DepthwiseConv2dInferTest() {}
};

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 4;
  inputs[0]->shape_[2] = 4;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;  // in channel
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;  // channel_multiplier
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 3;
  parameter->kernel_w_ = 3;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_l_ = 1;
  parameter->pad_r_ = 1;
  parameter->pad_d_ = 1;
  parameter->pad_u_ = 1;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 4);
  ASSERT_EQ(outputs[0]->shape_[2], 4);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 120;
  inputs[0]->shape_[2] = 120;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 7;
  parameter->kernel_w_ = 7;
  parameter->stride_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_l_ = 3;
  parameter->pad_r_ = 3;
  parameter->pad_d_ = 3;
  parameter->pad_u_ = 3;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 60);
  ASSERT_EQ(outputs[0]->shape_[2], 60);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 30;
  inputs[0]->shape_[2] = 30;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 3;
  parameter->kernel_w_ = 3;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 1;
  parameter->pad_r_ = 1;
  parameter->pad_d_ = 1;
  parameter->pad_u_ = 1;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 30);
  ASSERT_EQ(outputs[0]->shape_[2], 30);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 30;
  inputs[0]->shape_[2] = 30;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 3;
  parameter->kernel_w_ = 3;
  parameter->stride_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 1;
  parameter->pad_r_ = 1;
  parameter->pad_d_ = 1;
  parameter->pad_u_ = 1;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 15);
  ASSERT_EQ(outputs[0]->shape_[2], 15);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 120;
  inputs[0]->shape_[2] = 120;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 6;
  inputs[1]->shape_[0] = 20;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 5;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 5;
  parameter->kernel_w_ = 5;
  parameter->stride_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 58);
  ASSERT_EQ(outputs[0]->shape_[2], 58);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest5) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 27;
  inputs[0]->shape_[2] = 27;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 3;
  parameter->kernel_w_ = 3;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 25);
  ASSERT_EQ(outputs[0]->shape_[2], 25);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest6) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 88;
  inputs[0]->shape_[2] = 88;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 1;
  inputs[1]->shape_[2] = 1;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 1;
  parameter->kernel_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 88);
  ASSERT_EQ(outputs[0]->shape_[2], 88);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest7) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 11;
  inputs[0]->shape_[2] = 11;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 9;
  inputs[1]->shape_[2] = 1;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 9;
  parameter->kernel_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 4;
  parameter->pad_u_ = 4;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 11);
  ASSERT_EQ(outputs[0]->shape_[2], 11);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest8) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 29;
  inputs[0]->shape_[2] = 29;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 3;
  inputs[1]->shape_[2] = 3;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 3;
  parameter->kernel_w_ = 3;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_same;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 29);
  ASSERT_EQ(outputs[0]->shape_[2], 29);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest9) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 14;
  inputs[0]->shape_[2] = 14;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 1;
  inputs[1]->shape_[2] = 1;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 1;
  parameter->kernel_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 14);
  ASSERT_EQ(outputs[0]->shape_[2], 14);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(DepthwiseConv2dInferTest, DepthwiseConv2dInferTest10) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 448;
  inputs[0]->shape_[2] = 448;
  inputs[0]->shape_[3] = 6;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 4;
  inputs[1]->shape_[0] = 6;
  inputs[1]->shape_[1] = 5;
  inputs[1]->shape_[2] = 5;
  inputs[1]->shape_[3] = 1;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  ConvParameter *parameter = new ConvParameter;
  parameter->channel_multiplie_ = 1;
  parameter->kernel_h_ = 5;
  parameter->kernel_w_ = 5;
  parameter->stride_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->dilation_h_ = 1;
  parameter->dilation_w_ = 1;
  parameter->pad_mode_ = Pad_same;
  parameter->pad_l_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_u_ = 0;
  parameter->op_parameter_.infer_flag_ = true;
  int ret = DepthwiseConv2dInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                      reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 224);
  ASSERT_EQ(outputs[0]->shape_[2], 224);
  ASSERT_EQ(outputs[0]->shape_[3], 6);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
