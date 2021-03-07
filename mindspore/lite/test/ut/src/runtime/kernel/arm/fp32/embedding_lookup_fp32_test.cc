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
#include "src/runtime/kernel/arm/fp32/embedding_lookup_fp32.h"
#include "nnacl/fp32/embedding_lookup_fp32.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"
#include "src/common/log_adapter.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestEmbeddingLookupFp32 : public mindspore::CommonTest {
 public:
  TestEmbeddingLookupFp32() {}
};

void ElTestInit(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_,
                EmbeddingLookupParameter *embedding_lookup_param) {
  Tensor *in_t_first =
    new Tensor(kNumberTypeFloat32, {6, 2}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t_first->MallocData();
  float in_first[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  memcpy(in_t_first->MutableData(), in_first, sizeof(float) * in_t_first->ElementsNum());
  inputs_->push_back(in_t_first);

  Tensor *in_t_second =
    new Tensor(kNumberTypeFloat32, {4, 2}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t_second->MallocData();
  float in_second[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};
  memcpy(in_t_second->MutableData(), in_second, sizeof(float) * in_t_second->ElementsNum());
  inputs_->push_back(in_t_second);

  Tensor *ids_t = new Tensor(kNumberTypeFloat32, {2, 3}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  ids_t->MallocData();
  int ids[] = {1, 9, 2, 4, 6, 7};
  memcpy(ids_t->MutableData(), ids, sizeof(int) * ids_t->ElementsNum());
  inputs_->push_back(ids_t);

  Tensor *outputs_t =
    new Tensor(kNumberTypeInt32, {2, 3, 2}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  outputs_t->MallocData();
  outputs_->push_back(outputs_t);

  embedding_lookup_param->max_norm_ = 1;
}

TEST_F(TestEmbeddingLookupFp32, ElTest) {
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
  auto embedding_lookup_param_ = new EmbeddingLookupParameter();
  ElTestInit(&inputs_, &outputs_, embedding_lookup_param_);

  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::EmbeddingLookupCPUKernel *el = new kernel::EmbeddingLookupCPUKernel(
    reinterpret_cast<OpParameter *>(embedding_lookup_param_), inputs_, outputs_, ctx);

  el->Init();
  el->Run();

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

}  // namespace mindspore
