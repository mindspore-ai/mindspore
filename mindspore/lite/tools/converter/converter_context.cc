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
#include <string>
#include <vector>
#include "include/registry/converter_context.h"

namespace mindspore {
namespace converter {
void ConverterContext::SetGraphOutputTensorNames(const std::vector<std::string> &output_names) {
  auto converter_context = lite::ConverterInnerContext::GetInstance();
  MS_ASSERT(converter_context != nullptr);
  converter_context->SetGraphOutputTensorNames(output_names);
}

std::vector<std::string> ConverterContext::GetGraphOutputTensorNames() {
  auto converter_context = lite::ConverterInnerContext::GetInstance();
  MS_ASSERT(converter_context != nullptr);
  return converter_context->GetGraphOutputTensorNames();
}
}  // namespace converter
}  // namespace mindspore
