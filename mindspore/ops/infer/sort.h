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

#ifndef MINDSPORE_CORE_OPS_SORT_H_
#define MINDSPORE_CORE_OPS_SORT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSort = "Sort";
class OPS_API Sort : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Sort);
  Sort() : BaseOperator(kNameSort) { InitIOName({"x"}, {"y1", "y2"}); }
  void Init(int64_t axis = -1, bool descending = false);
  void set_axis(int64_t axis);
  int64_t get_axis() const;
  void set_descending(bool descending);
  bool get_descending() const;
};
OPS_API abstract::AbstractBasePtr SortInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SORT_H_
