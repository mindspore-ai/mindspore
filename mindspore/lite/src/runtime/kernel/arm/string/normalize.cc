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
#include "src/runtime/kernel/arm/string/normalize.h"
#include <string>
#include <map>
#include <regex>
#include <algorithm>
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CustomNormalize;

namespace mindspore::kernel {
namespace {
const char kPunctuationsRegex[] = "[.*()\"]";
const std::map<std::string, std::string> *kRegexTransforms = new (std::nothrow) std::map<std::string, std::string>({
  {"([\\S]+)n't", "$1 not"},
  {"([\\S]+)'nt", "$1 not"},
  {"([\\S]+)'ll", "$1 will"},
  {"([\\S]+)'re", "$1 are"},
  {"([\\S]+)'ve", "$1 have"},
  {"i'm", "i am"},
});
const int32_t kMaxStringLength = 300;

}  // namespace

int NormalizeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int NormalizeCPUKernel::ReSize() { return RET_OK; }

std::string NormalizeCPUKernel::Trim(const std::string &str, const std::string &pattern /*= " \t\n\v\f\r"*/) {
  auto begin = str.find_first_not_of(pattern);
  if (begin == std::string::npos) {
    MS_LOG(WARNING) << "Meaningless input string!";
    return "";
  }
  auto end = str.find_last_not_of(pattern);
  const auto range = end - begin + 1;
  return str.substr(begin, range);
}

std::string NormalizeCPUKernel::GlobalReplace(const std::string &str, const std::string &reg,
                                              const std::string &replace) {
  std::regex e(reg);
  return std::regex_replace(str, e, replace);
}

std::string NormalizeCPUKernel::Normalize(const std::string &str) {
  std::string result;
  std::transform(str.begin(), str.end(), back_inserter(result), [](unsigned char c) { return std::tolower(c); });
  result = Trim(result);
  result = GlobalReplace(result, kPunctuationsRegex, "");
  result = GlobalReplace(result, "\\s('t|'nt|n't|'d|'ll|'s|'m|'ve|'re)([\\s,;:/])", "$1$2");
  result = GlobalReplace(result, "\\s('t|'nt|n't|'d|'ll|'s|'m|'ve|'re)$", "$1");
  // transform shortening to full
  MS_ASSERT(kRegexTransforms != nullptr);
  for (auto iter = kRegexTransforms->begin(); iter != kRegexTransforms->end(); iter++) {
    result = GlobalReplace(result, iter->first, iter->second);
  }
  result = GlobalReplace(result, "([?])+", "$1");
  result = GlobalReplace(result, "([!])+", "$1");
  result = GlobalReplace(result, "([^?!]+)([?!])", "$1 $2 ");
  result = GlobalReplace(result, "([?!])([?!])", "$1 $2");

  result = GlobalReplace(result, "[\\s,:;\\-&'\"]+$", "");
  result = GlobalReplace(result, "^[\\s,:;\\-&'\"]+", "");

  result = Trim(result);
  if (result.size() > kMaxStringLength) {
    result = result.substr(0, kMaxStringLength);
  }
  result = "<S> " + result + " <E>";
  return result;
}

void NormalizeCPUKernel::FreeBuffer() {
  for (size_t j = 0; j < normalized_strs.size(); ++j) {
    if (normalized_strs[j] != nullptr) {
      context_->allocator->Free(normalized_strs[j]);
      normalized_strs[j] = nullptr;
    }
  }
}

int NormalizeCPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  int string_num = lite::GetStringCount(input_tensor);
  std::vector<lite::StringPack> all_string_pack = ParseTensorBuffer(input_tensor);

  std::vector<lite::StringPack> out_string_pack;
  normalized_strs.resize(string_num, nullptr);

  for (int i = 0; i < string_num; ++i) {
    auto chars = all_string_pack[i];
    std::string str(chars.data, chars.len);
    std::string result = Normalize(str);
    int str_length = result.size();

    char *normalized_str = nullptr;
    normalized_str = reinterpret_cast<char *>(context_->allocator->Malloc(sizeof(char) * str_length));
    if (normalized_str == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed!";
      FreeBuffer();
      return RET_ERROR;
    }
    normalized_strs[i] = normalized_str;

    memcpy(normalized_str, result.data(), str_length);
    out_string_pack.push_back({str_length, normalized_str});
  }
  if (string_num == 0) {
    out_string_pack.push_back({1, ""});
  }
  auto out_tensor = out_tensors_.at(0);
  WriteStringsToTensor(out_tensor, out_string_pack);
  FreeBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuNormalizeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) NormalizeCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new NormalizeCPUKernel fail!";
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_CustomNormalize, CpuNormalizeKernelCreator)
}  // namespace mindspore::kernel
