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

#include <iostream>
#include "src/runtime/kernel/arm/fp32/elu_fp32.h"
#include "nnacl/fp32/elu_fp32.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"
#include "src/common/log_adapter.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestEluFp32 : public mindspore::CommonTest {
 public:
  TestEluFp32() {}
};

void EluTestInit(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_, EluParameter *elu_param) {
  Tensor *in_t_first =
    new Tensor(kNumberTypeFloat32, {6, 2}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t_first->MallocData();
  float in_first[] = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 0};
  memcpy(in_t_first->MutableData(), in_first, sizeof(float) * in_t_first->ElementsNum());
  inputs_->push_back(in_t_first);

  Tensor *outputs_t = new Tensor(kNumberTypeFloat32, {6, 2}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  outputs_t->MallocData();
  outputs_->push_back(outputs_t);

  elu_param->alpha_ = 2.0;
}

TEST_F(TestEluFp32, EluTest) {
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  auto elu_param_ = new EluParameter();
  EluTestInit(&inputs_, &outputs_, elu_param_);

  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::EluCPUKernel *elu =
    new kernel::EluCPUKernel(reinterpret_cast<OpParameter *>(elu_param_), inputs_, outputs_, ctx);

  elu->Init();
  elu->Run();

  std::cout << "output shape:" << std::endl;
  for (unsigned int i = 0; i < outputs_.front()->shape().size(); ++i) {
    std::cout << outputs_.front()->shape()[i] << ' ';
  }
  std::cout << std::endl;
  float *out = reinterpret_cast<float *>(outputs_.front()->MutableData());
  for (int i = 0; i < outputs_.front()->ElementsNum(); ++i) {
    std::cout << out[i] << ' ';
  }
  std::cout << std::endl;
}

};  // namespace mindspore
