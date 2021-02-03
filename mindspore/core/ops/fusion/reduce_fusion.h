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

#ifndef MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
#define MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/reduce.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceFusion = "ReduceFusion";
class ReduceFusion : public Reduce {
 public:
  ReduceFusion() : Reduce(kNameReduceFusion) {}
  ~ReduceFusion() = default;
  MS_DECLARE_PARENT(ReduceFusion, PrimitiveC);
  void Init(const bool keep_dims = false, const ReduceMode mode = ReduceMode::Reduce_Mean,
            const bool reduce_to_end = false, const float coeff = 1.0);
  void set_keep_dims(const bool keep_dims);
  void set_mode(const ReduceMode mode);
  void set_reduce_to_end(const bool reduce_to_end);
  void set_coeff(const float coeff);
  bool get_keep_dims() const;
  ReduceMode get_mode() const;
  bool get_reduce_to_end() const;
  float get_coeff() const;
};
AbstractBasePtr ReduceFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);
using PrimReduceFusiuonPtr = std::shared_ptr<ReduceFusion>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_FUSION_H_
