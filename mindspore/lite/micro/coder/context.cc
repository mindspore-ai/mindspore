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

#include "coder/context.h"
#include "coder/allocator/allocator.h"

namespace mindspore::lite::micro {
CoderContext::CoderContext() {
  this->input_name_ = "g_Input";
  this->output_name_ = "g_Output";
  this->buffer_name_ = "g_Buffer";
  this->weight_name_ = "g_Weight";
}

void CoderContext::AppendCode(const std::string &codeBlock) { this->code_blocks_.emplace_back(codeBlock); }

void CoderContext::AppendInitCode(const std::string &codeBlock) { this->initialContent_.push_back(codeBlock); }

}  // namespace mindspore::lite::micro
