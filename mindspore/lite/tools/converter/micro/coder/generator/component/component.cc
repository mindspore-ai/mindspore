/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/micro/coder/generator/component/component.h"
#include <string>
#include "coder/log.h"

namespace mindspore::lite::micro {
const char *kInputPrefixName = nullptr;
const char *kOutputPrefixName = nullptr;
const char *kWeightPrefixName = nullptr;
const char *kBufferPrefixName = nullptr;
const char *kBufferPrefixNameAdd = nullptr;

char *ModifyPrefixName(char *name, int model_index, const std::string &prefix) {
  if (name != nullptr) {
    free(name);
    name = nullptr;
  }
  std::string variable_prefix_name = "m" + std::to_string(model_index) + prefix;
  name = static_cast<char *>(malloc((variable_prefix_name.size() + 1) * sizeof(char)));
  if (name == nullptr) {
    MS_LOG(ERROR) << "malloc failed";
    return nullptr;
  }
  int ret = memcpy_s(name, (variable_prefix_name.size() + 1) * sizeof(char), variable_prefix_name.c_str(),
                     (variable_prefix_name.size() + 1) * sizeof(char));
  if (ret == RET_ERROR) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return nullptr;
  }
  return name;
}

#define Free(name)                  \
  if (name != nullptr) {            \
    free(const_cast<char *>(name)); \
    name = nullptr;                 \
  }

void FreeGlobalVariable() {
  Free(kInputPrefixName);
  Free(kOutputPrefixName);
  Free(kWeightPrefixName);
  Free(kBufferPrefixName);
  Free(kBufferPrefixNameAdd);
}

void InitGlobalVariable(int model_index) {
  kInputPrefixName = ModifyPrefixName(const_cast<char *>(kInputPrefixName), model_index, "_input");
  kOutputPrefixName = ModifyPrefixName(const_cast<char *>(kOutputPrefixName), model_index, "_output");
  kWeightPrefixName = ModifyPrefixName(const_cast<char *>(kWeightPrefixName), model_index, "_weight");
  kBufferPrefixName = ModifyPrefixName(const_cast<char *>(kBufferPrefixName), model_index, "_buffer");
  kBufferPrefixNameAdd = ModifyPrefixName(const_cast<char *>(kBufferPrefixNameAdd), model_index, "_buffer + ");
}
}  // namespace mindspore::lite::micro
