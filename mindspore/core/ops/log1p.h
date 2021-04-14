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

#ifndef MINDSPORE_CORE_OPS_LOG1P_H_
#define MINDSPORE_CORE_OPS_LOG1P_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLog1p = "Log1p";
class Log1p : public PrimitiveC {
 public:
  Log1p() : PrimitiveC(kNameLog1p) { InitIOName({"x"}, {"y"}); }
  ~Log1p() = default;
  MS_DECLARE_PARENT(Log1p, PrimitiveC);
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LOG1P_H_
