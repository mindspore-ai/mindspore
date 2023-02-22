/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/format_transpose.h"

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(FormatTranspose, BaseOperator);
void FormatTranspose::Init(const Format &src_format, const Format &dst_format) {
  this->set_src_format(src_format);
  this->set_dst_format(dst_format);
}

void FormatTranspose::set_src_format(const Format &src_format) {
  (void)this->AddAttr(kSrcFormat, api::MakeValue(static_cast<int>(src_format)));
}

void FormatTranspose::set_dst_format(const Format &dst_format) {
  (void)this->AddAttr(kDstFormat, api::MakeValue(static_cast<int>(dst_format)));
}

Format FormatTranspose::get_src_format() const {
  auto value_ptr = GetAttr(kSrcFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

Format FormatTranspose::get_dst_format() const {
  auto value_ptr = GetAttr(kDstFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
REGISTER_PRIMITIVE_C(kNameFormatTranspose, FormatTranspose);
}  // namespace ops
}  // namespace mindspore
