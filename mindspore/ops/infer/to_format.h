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

#ifndef MINDSPORE_CORE_OPS_TOFORMAT_H_
#define MINDSPORE_CORE_OPS_TOFORMAT_H_
#include <map>
#include <memory>
#include <string>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameToFormat = "ToFormat";
class OPS_API ToFormat : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ToFormat);
  ToFormat() : BaseOperator(kNameToFormat) {}
  void Init(const int64_t src_t, const int64_t dst_t);
  void set_src_t(const int64_t src_t);
  void set_dst_t(const int64_t dst_t);
  int64_t get_src_t() const;
  int64_t get_dst_t() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TOFORMAT_H_
