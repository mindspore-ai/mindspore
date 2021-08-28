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

#include "src/runtime/kernel/arm/string/skip_gram.h"

#include "include/errorcode.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::StringPack;
using mindspore::schema::PrimitiveType_SkipGram;

namespace mindspore::kernel {
int SkipGramCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SkipGramCPUKernel::ReSize() { return RET_OK; }

void ParseSentenceToWords(const StringPack &sentence, std::vector<StringPack> *words) {
  MS_ASSERT(words != nullptr);
  int pre = 0;
  int i;
  for (i = 0; i < sentence.len; i++) {
    if (sentence.data[i] != ' ') {
      pre = i;
      break;
    }
  }
  for (; i < sentence.len; i++) {
    if (sentence.data[i] == ' ') {
      if (sentence.data[pre] != ' ') {
        words->push_back({i - pre, sentence.data + pre});
      }
      pre = i + 1;
    }
  }
  if (sentence.data[sentence.len - 1] != ' ') {
    words->push_back({sentence.len - pre, sentence.data + pre});
  }
}

int SkipGramCPUKernel::Run() {
  skip_gram_parameter_ = reinterpret_cast<SkipGramParameter *>(op_parameter_);
  MS_ASSERT(skip_gram_parameter_ != nullptr);
  if (skip_gram_parameter_->ngram_size < 1) {
    MS_LOG(ERROR) << "Skip Gram Parameter Error, NgramSize should be at least 1, get "
                  << skip_gram_parameter_->ngram_size;
    return RET_ERROR;
  }

  StringPack sentence = mindspore::lite::ParseTensorBuffer(in_tensors_.at(0)).at(0);
  std::vector<StringPack> words;
  ParseSentenceToWords(sentence, &words);

  std::vector<std::vector<StringPack>> result;
  std::vector<int> stack(skip_gram_parameter_->ngram_size, 0);

  int index = 1;
  int size = words.size();
  while (index >= 0) {
    if (index < skip_gram_parameter_->ngram_size && stack.at(index) + 1 < size &&
        (index == 0 || stack.at(index) - stack.at(index - 1) <= skip_gram_parameter_->max_skip_size)) {
      stack.at(index)++;
      index++;
      if (index < skip_gram_parameter_->ngram_size) {
        stack.at(index) = stack.at(index - 1);
      }
    } else {
      if (index > 0 && ((skip_gram_parameter_->include_all_ngrams && index <= skip_gram_parameter_->ngram_size) ||
                        (!skip_gram_parameter_->include_all_ngrams && index == skip_gram_parameter_->ngram_size))) {
        std::vector<StringPack> gram(2 * index - 1);
        char blank[1] = {' '};
        StringPack blank_str = {1, blank};
        for (int i = 0; i < 2 * index - 2; i += 2) {
          gram.at(i) = words.at(stack.at(i / 2));
          gram.at(i + 1) = blank_str;
        }
        gram.at(2 * index - 2) = words.at(stack.at(index - 1));
        result.push_back(gram);
      }
      index--;
    }
  }
  auto ret = mindspore::lite::WriteSeperatedStringsToTensor(out_tensors_.at(0), result);
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SkipGram, LiteKernelCreator<SkipGramCPUKernel>)
}  // namespace mindspore::kernel
