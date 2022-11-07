/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/converter_context.h"
#include <vector>
#include "include/registry/converter_context.h"

namespace mindspore {
namespace converter {
constexpr int kMaxInput = 255;

void ConverterContext::SetGraphOutputTensorNames(const std::vector<std::vector<char>> &&output_names) {
  auto converter_context = lite::ConverterInnerContext::GetInstance();
  if (converter_context == nullptr) {
    MS_LOG(ERROR) << "Set graph output's names failed.";
    return;
  }
  converter_context->SetGraphOutputTensorNames(VectorCharToString(output_names));
}

std::vector<std::vector<char>> ConverterContext::GetGraphOutputTensorNamesInChar() {
  auto converter_context = lite::ConverterInnerContext::GetInstance();
  if (converter_context == nullptr) {
    MS_LOG(ERROR) << "Get graph output's names failed.";
    return {};
  }
  return VectorStringToChar(converter_context->GetGraphOutputTensorNames());
}

std::map<std::vector<char>, std::vector<char>> ConverterContext::GetConfigInfo(const std::vector<char> &&section) {
  if (section.empty()) {
    MS_LOG(ERROR) << "Get config information parameter is empty.";
    return {};
  }
  if (section.size() > kMaxInput) {
    MS_LOG(ERROR) << "Config information parameter is too long";
    return {};
  }
  auto converter_context = lite::ConverterInnerContext::GetInstance();
  if (converter_context == nullptr) {
    MS_LOG(ERROR) << "Get config information only used by external extension failed.";
    return {};
  }
  auto &external_used_config_infos = converter_context->GetExternalUsedConfigInfos();
  if (external_used_config_infos.find(CharToString(section)) == external_used_config_infos.end()) {
    MS_LOG(ERROR) << "This section " << section << " config info is not existed.";
    return {};
  }
  return MapStringToVectorChar(external_used_config_infos.at(CharToString(section)));
}
}  // namespace converter
}  // namespace mindspore
