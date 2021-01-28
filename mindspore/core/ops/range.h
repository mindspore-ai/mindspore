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

#ifndef MINDSPORE_CORE_OPS_RANGE_H_
#define MINDSPORE_CORE_OPS_RANGE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRange = "Range";
class Range : public PrimitiveC {
 public:
  Range() : PrimitiveC(kNameRange) {}
  ~Range() = default;
  MS_DECLARE_PARENT(Range, PrimitiveC);
  void Init(const int64_t d_type, const int64_t start, const int64_t limit, const int64_t delta);
  void set_d_type(const int64_t d_type);
  void set_start(const int64_t start);
  void set_limit(const int64_t limit);
  void set_delta(const int64_t delta);
  int64_t get_d_type() const;
  int64_t get_start() const;
  int64_t get_limit() const;
  int64_t get_delta() const;
};

AbstractBasePtr RangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimRangePtr = std::shared_ptr<Range>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANGE_H_
