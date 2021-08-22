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

#ifndef MINDSPORE_CORE_OPS_LRN_H_
#define MINDSPORE_CORE_OPS_LRN_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLRN = "LRN";
class LRN : public PrimitiveC {
 public:
  LRN() : PrimitiveC(kNameLRN) { InitIOName({"x"}, {"y"}); }
  ~LRN() = default;
  MS_DECLARE_PARENT(LRN, PrimitiveC);
  void Init(const int64_t depth_radius = 5, const float bias = 1.0, const float alpha = 1.0, const float beta = 0.5,
            const std::string &norm_region = "ACROSS_CHANNELS");
  void set_depth_radius(const int64_t depth_radius);
  void set_bias(const float bias);
  void set_alpha(const float alpha);
  void set_beta(const float beta);
  void set_norm_region(const std::string &norm_region);
  int64_t get_depth_radius() const;
  float get_bias() const;
  float get_alpha() const;
  float get_beta() const;
  std::string get_norm_region() const;
};
AbstractBasePtr LrnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimLrn = std::shared_ptr<LRN>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_LRN_H_
