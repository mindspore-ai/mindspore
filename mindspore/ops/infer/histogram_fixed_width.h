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

#ifndef MINDSPORE_CORE_OPS_HISTOGRAM_FIXED_WIDTH_H_
#define MINDSPORE_CORE_OPS_HISTOGRAM_FIXED_WIDTH_H_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/type_id.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameHistogramFixedWidth = "HistogramFixedWidth";
/// Refer to Python API @ref mindspore.ops.HistogramFixedWidth for more details.
class OPS_API HistogramFixedWidth : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(HistogramFixedWidth);

  HistogramFixedWidth() : BaseOperator(kNameHistogramFixedWidth) { InitIOName({"x", "range"}, {"y"}); }

  void Init(const int32_t nbins, const TypeId dtype = kNumberTypeInt32);

  void set_nbins(const int32_t nbins);

  void set_dtype(const TypeId dtype);

  int32_t get_nbins() const;

  TypeId get_dtype() const;
};
OPS_API abstract::AbstractBasePtr HistogramFixedWidthInfer(const abstract::AnalysisEnginePtr &,
                                                           const PrimitivePtr &primitive,
                                                           const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_HISTOGRAM_FIXED_WIDTH_H_
