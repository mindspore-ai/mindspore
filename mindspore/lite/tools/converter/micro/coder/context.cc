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
#include "coder/generator/component/component.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro {
CoderContext::CoderContext(int model_index) {
  this->input_name_ = kInputPrefixName;
  this->output_name_ = kOutputPrefixName;
  this->buffer_name_ = kBufferPrefixName;
  this->weight_name_ = kWeightPrefixName;
  this->pack_weight_offset_name_ = kPackWeightOffsetName;
  this->pack_weight_size_name_ = kPackWeightSizeName;
  this->model_index_ = model_index;
}

void CoderContext::AppendCode(const std::string &codeBlock) { this->code_blocks_.emplace_back(codeBlock); }

void CoderContext::AppendInitCode(const std::string &codeBlock) { this->initialContent_.push_back(codeBlock); }

void CoderContext::AppendInitWeightSizeCode(const std::string &codeBlock) {
  this->weight_buffer_size_code_blocks_.push_back(codeBlock);
}

std::vector<std::string> CoderContext::GetInitWeightSizeCode() const {
  std::vector<std::string> ret(weight_buffer_size_code_blocks_);
  if (weight_buffer_size_ > 0) {
    nnacl::NNaclFp32Serializer w_init_size_code;
    w_init_size_code.CodeAddAssignExpression(pack_weight_size_name_, weight_buffer_size_);
    ret.push_back(w_init_size_code.str());
  }
  return ret;
}

void CoderContext::AppendInitWeightSizeCode(size_t w_buf_size) { weight_buffer_size_ += w_buf_size; }
}  // namespace mindspore::lite::micro
