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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/runtime/kernel/arm/fp32/fullconnection.h"
#include "src/runtime/kernel/arm/nnacl/fp32/matmul.h"

namespace mindspore {
using mindspore::lite::tensor::Tensor;

class TestFcFp32 : public mindspore::CommonTest {
 public:
  TestFcFp32() {}
};

int FcTestInit1(std::vector<lite::tensor::Tensor *> *inputs_, std::vector<lite::tensor::Tensor *> *outputs_,
                MatMulParameter *matmal_param, float **correct) {
  Tensor *in_t = new Tensor(kNumberTypeFloat, {2, 2, 2, 2}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  in_t->MallocData();
  float in[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
                17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  memcpy(in_t->Data(), in, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  Tensor *weight_t = new Tensor(kNumberTypeFloat, {3, 8}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  weight_t->MallocData();
  float weight[] = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
                    0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
                    -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
                    0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  memcpy(weight_t->Data(), weight, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  Tensor *bias_t = new Tensor(kNumberTypeFloat, {3}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  bias_t->MallocData();
  float bias[] = {1.6103756, -0.9872417, 0.546849};
  memcpy(bias_t->Data(), bias, sizeof(float) * bias_t->ElementsNum());
  inputs_->push_back(bias_t);

  Tensor *out_t = new Tensor(kNumberTypeFloat, {2, 3}, schema::Format_NHWC, static_cast<schema::NodeType>(1));
  out_t->MallocData();
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  float nchw_co[] = {1.6157111, -0.98469573, 0.6098231, 1.1649342, -1.2334653, 0.404779};
  memcpy(*correct, nchw_co, out_t->ElementsNum() * sizeof(float));

  matmal_param->b_transpose_ = true;
  matmal_param->a_transpose_ = false;
  matmal_param->has_bias_ = true;
  matmal_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestFcFp32, FcTest1) {
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  int total_size = FcTestInit1(&inputs_, &outputs_, matmul_param, &correct);
  lite::Context *ctx = new lite::Context;
  ctx->thread_num_ = 2;
  kernel::FullconnectionCPUKernel *fc =
    new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, nullptr);

  fc->Init();
  fc->Run();
  CompareOutputData(reinterpret_cast<float *>(outputs_[0]->Data()), correct, total_size, 0.0001);
}

int FcTestInit2(std::vector<lite::tensor::Tensor *> *inputs_, std::vector<lite::tensor::Tensor *> *outputs_,
                MatMulParameter *matmal_param, float **correct) {
  size_t buffer_size;

  Tensor *in_t = new Tensor(kNumberTypeFloat, {20, 4, 2, 10}, schema::Format_NCHW, static_cast<schema::NodeType>(1));
  in_t->MallocData();
  std::string in_path = "./matmul/FcFp32_input1.bin";
  auto in_data = mindspore::lite::ReadFile(in_path.c_str(), &buffer_size);
  memcpy(in_t->Data(), in_data, buffer_size);
  inputs_->push_back(in_t);

  Tensor *weight_t = new Tensor(kNumberTypeFloat, {30, 80}, schema::Format_NCHW, static_cast<schema::NodeType>(1));
  weight_t->MallocData();
  std::string weight_path = "./matmul/FcFp32_weight1.bin";
  auto w_data = mindspore::lite::ReadFile(weight_path.c_str(), &buffer_size);
  memcpy(weight_t->Data(), w_data, buffer_size);
  inputs_->push_back(weight_t);

  Tensor *bias_t = new Tensor(kNumberTypeFloat, {30}, schema::Format_NCHW, static_cast<schema::NodeType>(1));
  bias_t->MallocData();
  std::string bias_path = "./matmul/FcFp32_bias1.bin";
  auto bias_data = mindspore::lite::ReadFile(bias_path.c_str(), &buffer_size);
  memcpy(bias_t->Data(), bias_data, buffer_size);
  inputs_->push_back(bias_t);

  Tensor *out_t = new Tensor(kNumberTypeFloat, {20, 30}, schema::Format_NCHW, static_cast<schema::NodeType>(1));
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
  return out_t->ElementsNum();
}

TEST_F(TestFcFp32, FcTest2) {
  std::vector<lite::tensor::Tensor *> inputs_;
  std::vector<lite::tensor::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  float *correct;
  int total_size = FcTestInit2(&inputs_, &outputs_, matmul_param, &correct);
  lite::Context *ctx = new lite::Context;
  ctx->thread_num_ = 1;
  kernel::FullconnectionCPUKernel *fc =
    new kernel::FullconnectionCPUKernel(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, nullptr);

  fc->Init();
  fc->Run();
  CompareOutputData(reinterpret_cast<float *>(outputs_[0]->Data()), correct, total_size, 0.0001);
}
}  // namespace mindspore
