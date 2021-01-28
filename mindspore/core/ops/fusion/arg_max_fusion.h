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

#ifndef MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
#define MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
#include <memory>
#include <vector>

#include "ops/arg_max.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgMaxFusion = "ArgMaxFusion";
class ArgMaxFusion : public ArgMax {
 public:
  ArgMaxFusion() : ArgMax(kNameArgMaxFusion) { InitIOName({"x"}, {"output"}); }
  ~ArgMaxFusion() = default;
  MS_DECLARE_PARENT(ArgMaxFusion, ArgMax);
  void Init(const bool keep_dims, const bool out_max_value, const int64_t top_k, const int64_t axis = -1);

  void set_keep_dims(const bool keep_dims);
  void set_out_max_value(const bool out_max_value);
  void set_top_k(const int64_t top_k);

  bool get_keep_dims() const;
  bool get_out_max_value() const;
  int64_t get_top_k() const;
};
AbstractBasePtr ArgMaxFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);
using PrimArgMaxFusion = std::shared_ptr<ArgMaxFusion>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARGMAX_FUSION_H_
