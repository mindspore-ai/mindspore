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
#include "src/runtime/kernel/arm/fp32/skip_gram_fp32.h"
#include "mindspore/lite/nnacl/skip_gram_parameter.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"
#include "src/common/log_adapter.h"
#include "src/common/string_util.h"

namespace mindspore {
using mindspore::lite::StringPack;
using mindspore::lite::Tensor;

class TestSkipGramFp32 : public mindspore::CommonTest {
 public:
  TestSkipGramFp32() {}
};

void SkipGramTestInit(std::vector<Tensor *> *inputs_, std::vector<Tensor *> *outputs_,
                      SkipGramParameter *skip_gram_param) {
  Tensor *in_t_first = new Tensor(kObjectTypeString, {}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  char sentence[] = "The quick brown fox jumps over the lazy dog";
  std::vector<StringPack> str;
  str.push_back({43, sentence});
  mindspore::lite::WriteStringsToTensor(in_t_first, str);
  inputs_->push_back(in_t_first);

  Tensor *output = new Tensor(kObjectTypeString, {}, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  outputs_->push_back(output);

  skip_gram_param->ngram_size = 3;
  skip_gram_param->max_skip_size = 2;
  skip_gram_param->include_all_ngrams = true;
  skip_gram_param->op_parameter_.type_ = mindspore::schema::PrimitiveType_SkipGram;
  skip_gram_param->op_parameter_.thread_num_ = 2;
}

TEST_F(TestSkipGramFp32, ElTest) {
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;

  auto skip_gram_param_ = new SkipGramParameter();
  SkipGramTestInit(&inputs_, &outputs_, skip_gram_param_);

  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::SkipGramCPUKernel *el =
    new kernel::SkipGramCPUKernel(reinterpret_cast<OpParameter *>(skip_gram_param_), inputs_, outputs_, ctx);

  el->Init();
  el->Run();

  std::vector<StringPack> output = mindspore::lite::ParseTensorBuffer(outputs_[0]);
  for (unsigned int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].len; j++) {
      printf("%c", output[i].data[j]);
    }
    printf("\n");
  }
}

}  // namespace mindspore
