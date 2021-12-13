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

#include <sys/time.h>
#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/common/file_utils.h"
#include "src/common/log_adapter.h"
#include "src/runtime/kernel/arm/fp32/fullconnection_fp32.h"
#include "src/runtime/infer_manager.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestFcFp32 : public mindspore::CommonTest {
 public:
  TestFcFp32() {}
};

int FcTestInit1(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                MatMulParameter *matmal_param, float **correct) {
  auto *in_t = new Tensor(kNumberTypeFloat, {2, 2, 2, 2}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  float in[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
                17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  memcpy(in_t->MutableData(), in, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto *weight_t = new Tensor(kNumberTypeFloat, {3, 8}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  float weight[] = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
                    0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
                    -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
                    0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  memcpy(weight_t->MutableData(), weight, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  auto *bias_t = new Tensor(kNumberTypeFloat, {3}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  bias_t->MallocData();
  float bias[] = {1.6103756, -0.9872417, 0.546849};
  memcpy(bias_t->MutableData(), bias, sizeof(float) * bias_t->ElementsNum());
  inputs_->push_back(bias_t);

  auto *out_t = new Tensor(kNumberTypeFloat, {2, 3}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  float nchw_co[] = {1.6157111, -0.98469573, 0.6098231, 1.1649342, -1.2334653, 0.404779};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(float));

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = true;
  matmal_param->act_type_ = ActType_No;
  matmal_param->op_parameter_.type_ = 67;
  matmal_param->op_parameter_.is_train_session_ = false;
  KernelInferShape(*inputs_, *outputs_, reinterpret_cast<OpParameter *>(matmal_param));
  return out_t->ElementsNum();
}

TEST_F(TestFcFp32, FcTest1) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  int total_size = FcTestInit1(&inputs_, &outputs_, matmul_param, &correct);
  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  matmul_param->op_parameter_.thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *fc = new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx);
  fc->Init();
#ifdef SUPPORT_TRAIN
  fc->AllocWorkspace();
#endif
  fc->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete fc;
  delete ctx;
  for (unsigned int i = 0; i < inputs_.size(); i++) {
    delete inputs_[i];
  }
  for (unsigned int i = 0; i < outputs_.size(); i++) {
    delete outputs_[i];
  }
}

int FcTestInit2(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                MatMulParameter *matmal_param, float **correct) {
  size_t buffer_size;

  auto *in_t = new Tensor(kNumberTypeFloat, {20, 4, 2, 10}, mindspore::NCHW, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  std::string in_path = "./matmul/FcFp32_input1.bin";
  auto in_data = mindspore::lite::ReadFile(in_path.c_str(), &buffer_size);
  memcpy(in_t->MutableData(), in_data, buffer_size);
  inputs_->push_back(in_t);

  auto *weight_t = new Tensor(kNumberTypeFloat, {30, 80}, mindspore::NCHW, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  std::string weight_path = "./matmul/FcFp32_weight1.bin";
  auto w_data = mindspore::lite::ReadFile(weight_path.c_str(), &buffer_size);
  memcpy(weight_t->MutableData(), w_data, buffer_size);
  inputs_->push_back(weight_t);

  auto *bias_t = new Tensor(kNumberTypeFloat, {30}, mindspore::NCHW, lite::Tensor::Category::CONST_TENSOR);
  bias_t->MallocData();
  std::string bias_path = "./matmul/FcFp32_bias1.bin";
  auto bias_data = mindspore::lite::ReadFile(bias_path.c_str(), &buffer_size);
  memcpy(bias_t->MutableData(), bias_data, buffer_size);
  inputs_->push_back(bias_t);

  auto *out_t = new Tensor(kNumberTypeFloat, {20, 30}, mindspore::NCHW, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  std::string out_path = "./matmul/FcFp32_output1.bin";
  auto out_data = mindspore::lite::ReadFile(out_path.c_str(), &buffer_size);
  memcpy(*correct, out_data, out_t->ElementsNum() * sizeof(float));

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = true;
  matmal_param->act_type_ = ActType_No;
  matmal_param->op_parameter_.type_ = 67;
  KernelInferShape(*inputs_, *outputs_, reinterpret_cast<OpParameter *>(matmal_param));
  return out_t->ElementsNum();
}

TEST_F(TestFcFp32, FcTest2) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  int total_size = FcTestInit2(&inputs_, &outputs_, matmul_param, &correct);
  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  matmul_param->op_parameter_.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *fc = new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx);
  fc->Init();
#ifdef SUPPORT_TRAIN
  fc->AllocWorkspace();
#endif
  fc->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  for (unsigned int i = 0; i < inputs_.size(); i++) {
    delete inputs_[i];
  }
  for (unsigned int i = 0; i < outputs_.size(); i++) {
    delete outputs_[i];
  }
  delete fc;
  delete ctx;
}

void FcTestInit3(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                 MatMulParameter *matmal_param, float **correct) {
  auto *in_t = new Tensor(kNumberTypeFloat, {1, 1, 1, 20}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  float in[] = {1, 0, 3, 0, 4, 5, 2, 5, 2, 5, 1, 5, 0, 1, 2, 0, 2, 1, 0, 5};
  memcpy(in_t->MutableData(), in, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto *weight_t = new Tensor(kNumberTypeFloat, {16, 20}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  float weight[] = {0, 5, 5, 3, 0, 5, 3, 1, 0, 1, 3, 0, 5, 5, 2, 4, 0, 1, 1, 2, 3, 0, 5, 5, 4, 4, 1, 4, 1, 1, 5, 3,
                    3, 1, 0, 3, 1, 2, 4, 5, 3, 4, 4, 0, 3, 5, 0, 3, 4, 1, 0, 1, 3, 4, 0, 5, 2, 5, 0, 4, 2, 2, 2, 2,
                    4, 4, 5, 2, 1, 1, 5, 1, 4, 4, 5, 1, 2, 4, 0, 3, 1, 1, 0, 2, 1, 5, 2, 0, 1, 1, 5, 5, 4, 0, 0, 4,
                    2, 3, 2, 1, 4, 0, 5, 0, 2, 3, 1, 2, 1, 2, 1, 4, 2, 3, 5, 5, 4, 5, 2, 0, 3, 0, 2, 0, 1, 3, 0, 4,
                    1, 5, 2, 5, 4, 2, 5, 1, 4, 5, 3, 1, 0, 4, 4, 4, 1, 3, 4, 2, 2, 4, 1, 4, 0, 1, 0, 2, 4, 5, 2, 1,
                    0, 3, 5, 2, 4, 2, 1, 4, 2, 0, 1, 0, 2, 3, 0, 3, 2, 5, 5, 4, 3, 0, 0, 2, 0, 1, 5, 2, 2, 1, 3, 0,
                    3, 0, 5, 3, 3, 3, 5, 5, 3, 4, 0, 1, 2, 1, 2, 4, 3, 5, 4, 3, 0, 0, 4, 4, 2, 3, 5, 4, 3, 5, 1, 2,
                    1, 5, 0, 5, 1, 1, 5, 5, 0, 0, 1, 3, 2, 2, 2, 3, 4, 2, 2, 3, 2, 4, 3, 0, 2, 0, 3, 2, 1, 5, 2, 4,
                    4, 5, 2, 5, 0, 5, 3, 3, 0, 3, 2, 5, 5, 1, 1, 0, 2, 3, 0, 1, 1, 2, 4, 1, 3, 3, 5, 5, 0, 1, 0, 0,
                    1, 2, 3, 3, 5, 2, 2, 5, 1, 4, 3, 3, 0, 2, 5, 4, 3, 1, 2, 4, 0, 2, 1, 3, 1, 2, 1, 0, 5, 5, 4, 5};
  memcpy(weight_t->MutableData(), weight, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  auto *out_t = new Tensor(kNumberTypeFloat, {1, 16}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = false;
  matmal_param->act_type_ = ActType_No;
}

TEST_F(TestFcFp32, FcTest3) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  FcTestInit3(&inputs_, &outputs_, matmul_param, &correct);
  auto *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  matmul_param->op_parameter_.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *fc = new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx);
  fc->Init();
#ifdef SUPPORT_TRAIN
  fc->AllocWorkspace();
#endif
  struct timeval start, end;
  gettimeofday(&start, nullptr);
  for (int i = 0; i < 100000; ++i) fc->Run();
  gettimeofday(&end, nullptr);
  // printf("## elapsed: %llu\n", 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - end.tv_usec);
  for (unsigned int i = 0; i < inputs_.size(); i++) {
    delete inputs_[i];
  }
  for (unsigned int i = 0; i < outputs_.size(); i++) {
    delete outputs_[i];
  }
  delete fc;
  delete ctx;
}

int FcTest4_Init(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs, lite::InnerContext *context,
                 MatMulParameter *param, float **correct) {
  auto *in_t = new Tensor(kNumberTypeFloat32, {1, 4}, mindspore::NHWC);
  in_t->MallocData();
  float in[] = {1, 2, 3, 4};
  memcpy(in_t->MutableData(), in, sizeof(float) * in_t->ElementsNum());
  inputs->push_back(in_t);

  auto *weight_t = new Tensor(kNumberTypeFloat32, {10, 4}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  float weight[] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                    6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 1, 2, 3, 4};
  memcpy(weight_t->MutableData(), weight, sizeof(float) * weight_t->ElementsNum());
  inputs->push_back(weight_t);

  auto bias_t = new Tensor(kNumberTypeFloat32, {10}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  bias_t->MallocData();
  float bias_data[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  memcpy(bias_t->MutableData(), bias_data, sizeof(float) * bias_t->ElementsNum());
  inputs->push_back(bias_t);

  auto *out_t = new Tensor(kNumberTypeFloat32, {1, 10}, mindspore::NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs->push_back(out_t);

  *correct = new float[out_t->ElementsNum()];
  float nchw_co[] = {11, 21, 31, 41, 51, 62, 72, 82, 92, 32};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(float));
#ifdef SUPPORT_TRAIN
  param->op_parameter_.is_train_session_ = true;
#else
  param->op_parameter_.is_train_session_ = false;
#endif
  param->a_transpose_ = false;
  param->b_transpose_ = true;
  param->has_bias_ = false;
  param->act_type_ = ActType_No;

  param->op_parameter_.thread_num_ = 1;
  context->thread_num_ = 1;
  EXPECT_EQ(lite::RET_OK, context->Init());
  return out_t->ElementsNum();
}

int FcTest4_Resize(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs, float **correct) {
  auto &in_tensor = inputs->at(0);
  in_tensor->FreeData();
  in_tensor->set_shape({2, 4});
  float in[] = {1, 2, 3, 4, 2, 3, 1, 2};
  memcpy(in_tensor->MutableData(), in, in_tensor->Size());

  auto &out_tensor = outputs->at(0);
  out_tensor->FreeData();
  out_tensor->set_shape({2, 10});
  out_tensor->MallocData();

  *correct = new float[out_tensor->ElementsNum()];
  float nchw_co[] = {11, 21, 31, 41, 51, 62, 72, 82, 92, 32, 9, 17, 25, 33, 41, 50, 58, 66, 74, 21};
  memcpy(*correct, nchw_co, sizeof(nchw_co));
  return out_tensor->ElementsNum();
}

TEST_F(TestFcFp32, FcTest4_Vec2Batch) {
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  lite::InnerContext context;
  MatMulParameter *param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  memset(param, 0, sizeof(MatMulParameter));
  float *correct;
  int total_size = FcTest4_Init(&inputs, &outputs, &context, param, &correct);
  auto *kernel = new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, &context);
  kernel->Init();
#ifdef SUPPORT_TRAIN
  kernel->AllocWorkspace();
#endif
  kernel->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), correct, total_size, 0.0001));
  delete[] correct;

  total_size = FcTest4_Resize(&inputs, &outputs, &correct);
  kernel->ReSize();
#ifdef SUPPORT_TRAIN
  kernel->FreeWorkspace();
  kernel->AllocWorkspace();
#endif
  kernel->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), correct, total_size, 0.0001));
#ifdef SUPPORT_TRAIN
  kernel->FreeWorkspace();
#endif
  delete[] correct;
  for (auto &input : inputs) {
    delete input;
  }
  for (auto &output : outputs) {
    delete output;
  }
  delete kernel;
}
}  // namespace mindspore
