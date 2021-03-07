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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "nnacl/common_func.h"
#include "mindspore/lite/src/runtime/kernel/arm/int8/convolution_1x1_int8.h"

namespace mindspore {
using lite::Tensor;
class TestConv1x1Int8 : public mindspore::CommonTest {
 public:
  TestConv1x1Int8() {}
};

TEST_F(TestConv1x1Int8, Input1x1PrePack1) {
  auto conv_param = new ConvParameter();
  conv_param->input_channel_ = 6;
  conv_param->input_h_ = conv_param->input_w_ = 3;
  conv_param->output_h_ = conv_param->output_w_ = 3;
  conv_param->stride_h_ = conv_param->stride_w_ = 2;
  conv_param->pad_u_ = conv_param->pad_l_ = 1;
  int8_t in[] = {4,  13,  -3, 16, 19, 8,  19, -6, -2, -9, 9,  18, 23, 8,  47, -14, 15, 4,
                 -0, 37,  -0, 6,  0,  -1, 37, 13, 11, 1,  -1, 41, 9,  14, 3,  0,   8,  9,
                 14, -14, -8, -8, -8, 7,  19, 17, 13, 3,  9,  18, -1, -0, 18, 0,   4,  -2};
  int8_t correct[] = {0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 13, 11,
                      1, -1, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0};
  int8_t out[54] = {0};
  Conv1x1InputPack(in, out, conv_param, sizeof(int8_t));
  ASSERT_EQ(0, CompareOutputData(out, correct, 54, 0));
  delete conv_param;
}

TEST_F(TestConv1x1Int8, Input1x1PrePack2) {
  auto conv_param = new ConvParameter();
  int8_t in[] = {-0, -0, -7, -0, -6, 4,  9,  9,  12, -0, 6,  2,  13, 15, 16, -7, 9,  1,  10, 13, 17, 17, 4,  13,
                 -6, 5,  7,  -7, 15, 0,  1,  -5, -7, 18, 15, 19, -7, 13, 7,  -0, 16, -5, 16, -7, 6,  10, -5, 10,
                 9,  12, -9, -8, -4, 18, -5, 0,  7,  12, 13, 16, -9, -4, 18, -0, 8,  6,  2,  10, 16, 1,  -1, 2,
                 9,  8,  9,  13, 7,  -0, 15, -7, 0,  -0, 17, 19, 9,  17, -6, -2, 7,  -0, 10, -6, -6, 18, -0, 9,
                 9,  6,  3,  -1, -8, 10, 17, -9, 17, 6,  -3, 7,  -2, -0, -9, 1,  -3, 15, 13, 4,  18};
  int8_t correct[] = {0, 0, 0, 0, 0, 0, 15, -7, -7, 0, 0, 0, 9, 7, 0, 0, 0, 0, 0, 0};

  conv_param->input_h_ = 9;
  conv_param->input_w_ = 13;
  conv_param->input_channel_ = 1;
  conv_param->output_h_ = 4;
  conv_param->output_w_ = 5;
  conv_param->stride_h_ = conv_param->stride_w_ = 4;
  conv_param->pad_u_ = conv_param->pad_l_ = 2;

  int8_t out[20] = {0};
  Conv1x1InputPack(in, out, conv_param, sizeof(int8_t));
  ASSERT_EQ(0, CompareOutputData(out, correct, 20, 0));
  delete conv_param;
}

int Conv1x1Int8TestInit1_perchannel(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                                    ConvParameter *conv_param, int8_t **correct) {
  Tensor *in_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto in_quant_arg = new mindspore::lite::QuantArg();
  in_quant_arg->zeroPoint = -42, in_quant_arg->scale = 0.117647;
  in_t->AddQuantParam(*in_quant_arg);
  in_t->MallocData();
  int8_t in[] = {62,  -14, 88, 2,   -35, 43,  83,  -111, 75,  26, 14,  -121,
                 -78, 56,  37, -31, 15,  -75, -10, -115, -71, 74, -65, -15};
  memcpy(in_t->MutableData(), in, in_t->ElementsNum() * sizeof(int8_t));
  inputs_->push_back(in_t);

  Tensor *weight_t =
    new Tensor(kNumberTypeInt8, {3, 1, 1, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  auto weight_quant_arg1 = new mindspore::lite::QuantArg();
  weight_quant_arg1->zeroPoint = 66, weight_quant_arg1->scale = 0.96439215686275;
  auto weight_quant_arg2 = new mindspore::lite::QuantArg();
  weight_quant_arg2->zeroPoint = 33, weight_quant_arg2->scale = 0.76439215686275;
  auto weight_quant_arg3 = new mindspore::lite::QuantArg();
  weight_quant_arg3->zeroPoint = -20, weight_quant_arg3->scale = 0.99117647;
  weight_t->AddQuantParam(*weight_quant_arg1);
  weight_t->AddQuantParam(*weight_quant_arg2);
  weight_t->AddQuantParam(*weight_quant_arg3);
  int8_t weight[] = {65, 67, 65, 65, 32, 33, 34, 33, -19, -20, -19, -20};
  memcpy(weight_t->MutableData(), weight, weight_t->ElementsNum() * sizeof(int8_t));
  inputs_->push_back(weight_t);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 3}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  auto output_quant_arg = new mindspore::lite::QuantArg();
  output_quant_arg->zeroPoint = 7, output_quant_arg->scale = 0.294321233;
  out_t->AddQuantParam(*output_quant_arg);
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  int8_t nchw_co[] = {-83, 34, 100, 10, 113, 55, 3, 16, 63, 6, 93, 20, 5, 6, 42, 35, 28, -24};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(int8_t));

  conv_param->kernel_h_ = conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 1;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  conv_param->pad_u_ = conv_param->pad_l_ = 0;
  conv_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestConv1x1Int8, Conv1x1TestPerChannel) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto conv_param = new ConvParameter();
  int8_t *correct;
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int total_size = Conv1x1Int8TestInit1_perchannel(&inputs_, &outputs_, conv_param, &correct);
  kernel::Convolution1x1Int8CPUKernel *conv1x1 =
    new kernel::Convolution1x1Int8CPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs_, outputs_, ctx);

  conv1x1->Init();
  conv1x1->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int8_t *>(outputs_[0]->MutableData()), correct, total_size, 70));

  delete conv1x1;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}

int Conv1x1Int8TestInit1(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                         ConvParameter *conv_param, int8_t **correct) {
  Tensor *in_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto in_quant_arg = new mindspore::lite::QuantArg();
  in_quant_arg->zeroPoint = -42, in_quant_arg->scale = 0.117647;
  in_t->AddQuantParam(*in_quant_arg);
  in_t->MallocData();
  float in[] = {12.216284, 3.3466918,  15.327419, 5.234958,  0.804376,   9.952188,  14.727955,  -8.080715,
                13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  Quantize(in, in_t->ElementsNum(), in_quant_arg->scale, in_quant_arg->zeroPoint,
           reinterpret_cast<int8_t *>(in_t->MutableData()));
  inputs_->push_back(in_t);

  Tensor *weight_t =
    new Tensor(kNumberTypeInt8, {3, 1, 1, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto weight_quant_arg = new mindspore::lite::QuantArg();
  weight_quant_arg->zeroPoint = 66, weight_quant_arg->scale = 0.036439215686275;
  weight_t->AddQuantParam(*weight_quant_arg);
  weight_t->MallocData();
  float weight[] = {-0.7308652, 0.5257509,  -0.87825793, -1.123181,   -1.2206168, 0.562695,
                    1.5382664,  -0.5020635, 0.8591602,   -0.26410004, 1.1262615,  0.073132955};
  Quantize(weight, weight_t->ElementsNum(), weight_quant_arg->scale, weight_quant_arg->zeroPoint,
           reinterpret_cast<int8_t *>(weight_t->MutableData()));
  inputs_->push_back(weight_t);

  Tensor *out_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 3}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  auto output_quant_arg = new mindspore::lite::QuantArg();
  output_quant_arg->zeroPoint = 7, output_quant_arg->scale = 0.234321233;
  out_t->AddQuantParam(*output_quant_arg);
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  float nchw_co[] = {-26.51016327, 7.92113757, 27.25741343, 0.785643655,  31.3307619, 14.05927672,
                     -1.178490666, 2.5676252,  16.39408946, -0.394793726, 25.2866881, 3.827249175,
                     -0.626854507, -0.3122176, 10.42769169, 8.362184085,  6.04617807, -9.252362384};
  Quantize(nchw_co, out_t->ElementsNum(), output_quant_arg->scale, output_quant_arg->zeroPoint, *correct);

  conv_param->kernel_h_ = conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 1;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  conv_param->pad_u_ = conv_param->pad_l_ = 0;
  conv_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestConv1x1Int8, Conv1x1Int8Test1) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto conv_param = new ConvParameter();
  int8_t *correct;
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int total_size = Conv1x1Int8TestInit1(&inputs_, &outputs_, conv_param, &correct);
  kernel::Convolution1x1Int8CPUKernel *conv1x1 =
    new kernel::Convolution1x1Int8CPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs_, outputs_, ctx);

  conv1x1->Init();
  conv1x1->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int8_t *>(outputs_[0]->MutableData()), correct, total_size, 2));

  delete conv1x1;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}

int Conv1x1Int8TestInit2(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                         ConvParameter *conv_param, int8_t **correct) {
  size_t buffer_size;
  Tensor *in_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto in_quant_arg = new mindspore::lite::QuantArg();
  in_quant_arg->zeroPoint = -42, in_quant_arg->scale = 0.117647;
  in_t->AddQuantParam(*in_quant_arg);
  in_t->MallocData();
  std::string input_path = "./input";
  auto input = mindspore::lite::ReadFile(input_path.c_str(), &buffer_size);
  memcpy(in_t->MutableData(), input, buffer_size);
  inputs_->push_back(in_t);
  delete[] input;

  Tensor *weight_t =
    new Tensor(kNumberTypeInt8, {3, 1, 1, 4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  auto weight_quant_arg = new mindspore::lite::QuantArg();
  weight_quant_arg->zeroPoint = 66, weight_quant_arg->scale = 0.036439215686275;
  weight_t->AddQuantParam(*weight_quant_arg);
  weight_t->MallocData();
  std::string weight_path = "./weight";
  auto weight = mindspore::lite::ReadFile(weight_path.c_str(), &buffer_size);
  memcpy(weight_t->MutableData(), weight, buffer_size);
  inputs_->push_back(weight_t);
  delete[] weight;

  Tensor *bias_t = new Tensor(kNumberTypeInt32, {4}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  std::string bias_path = "./bias";
  auto bias = mindspore::lite::ReadFile(bias_path.c_str(), &buffer_size);
  memcpy(bias_t->MutableData(), bias, buffer_size);
  inputs_->push_back(bias_t);
  delete[] bias;

  Tensor *out_t = new Tensor(kNumberTypeInt8, {1, 2, 3, 3}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  auto output_quant_arg = new mindspore::lite::QuantArg();
  output_quant_arg->zeroPoint = 7, output_quant_arg->scale = 0.234321233;
  out_t->AddQuantParam(*output_quant_arg);
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  std::string output_path = "./output";
  auto output = mindspore::lite::ReadFile(output_path.c_str(), &buffer_size);
  memcpy(*correct, output, buffer_size);
  delete[] output;

  conv_param->kernel_h_ = conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 1;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  conv_param->pad_u_ = conv_param->pad_l_ = 0;
  conv_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestConv1x1Int8, Conv1x1Int8Test2) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto conv_param = new ConvParameter();
  int8_t *correct;
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int total_size = Conv1x1Int8TestInit2(&inputs_, &outputs_, conv_param, &correct);
  auto *conv1x1 =
    new kernel::Convolution1x1Int8CPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs_, outputs_, ctx);

  conv1x1->Init();
  conv1x1->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int8_t *>(outputs_[0]->MutableData()), correct, total_size, 2));

  delete conv1x1;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}
}  // namespace mindspore
