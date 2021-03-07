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
#include "mindspore/lite/src/runtime/kernel/arm/fp32/power_fp32.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"

namespace mindspore {
class TestPowerFp32 : public mindspore::CommonTest {
 public:
  TestPowerFp32() {}
};

int PowerTestInit(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_, float *a_ptr,
                  float *b_ptr, const std::vector<int> &a_shape, const std::vector<int> &b_shape,
                  const std::vector<int> &c_shape) {
  auto in_t = new lite::Tensor(kNumberTypeFloat, a_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), a_ptr, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto weight_t =
    new lite::Tensor(kNumberTypeFloat, b_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  weight_t->MallocData();
  memcpy(weight_t->MutableData(), b_ptr, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  auto out_t = new lite::Tensor(kNumberTypeFloat, c_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  return out_t->ElementsNum();
}

int PowerTestInit2(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_, float *a_ptr,
                   const std::vector<int> &a_shape, const std::vector<int> &c_shape) {
  auto in_t = new lite::Tensor(kNumberTypeFloat, a_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), a_ptr, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto out_t = new lite::Tensor(kNumberTypeFloat, c_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  return out_t->ElementsNum();
}

TEST_F(TestPowerFp32, Simple) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto param = new PowerParameter();
  param->scale_ = 1;
  param->shift_ = 0;
  float a[] = {1, 2, 3, 4};
  float b[] = {5, 6, 7, 8};
  std::vector<int> a_shape = {2, 2};
  std::vector<int> b_shape = {2, 2};
  std::vector<int> c_shape = {2, 2};
  int total_size = PowerTestInit(&inputs_, &outputs_, a, b, a_shape, b_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *op = new kernel::PowerCPUKernel(reinterpret_cast<OpParameter *>(param), inputs_, outputs_, ctx);
  op->Init();
  op->Run();
  float correct[] = {1, 64, 2187, 65536};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete op;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}

TEST_F(TestPowerFp32, Broadcast) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto param = new PowerParameter();
  param->power_ = 2;
  param->scale_ = 1;
  param->shift_ = 0;
  float a[] = {1, 2, 3, 4};
  std::vector<int> a_shape = {2, 2};
  std::vector<int> c_shape = {2, 2};
  int total_size = PowerTestInit2(&inputs_, &outputs_, a, a_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *op = new kernel::PowerCPUKernel(reinterpret_cast<OpParameter *>(param), inputs_, outputs_, ctx);
  op->Init();
  op->Run();
  float correct[] = {1, 4, 9, 16};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete op;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}
}  // namespace mindspore
