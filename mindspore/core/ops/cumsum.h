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

#ifndef MINDSPORE_CORE_OPS_CUMSUM_H_
#define MINDSPORE_CORE_OPS_CUMSUM_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCumSum = "CumSum";
class CumSum : public PrimitiveC {
 public:
  CumSum() : PrimitiveC(kNameCumSum) {}
  ~CumSum() = default;
  MS_DECLARE_PARENT(CumSum, PrimitiveC);
  void Init(const bool exclusive, const bool reverse);
  void set_exclusive(const bool exclusive);
  void set_reverse(const bool reverse);
  bool get_exclusive() const;
  bool get_reverse() const;
};
AbstractBasePtr CumSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimCumSum = std::shared_ptr<CumSum>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CUMSUM_H_
