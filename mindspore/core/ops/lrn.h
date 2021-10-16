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
/// \brief Local Response Normalization. Refer to Python API @ref mindspore.ops.LRN for more details.
class MS_CORE_API LRN : public PrimitiveC {
 public:
  /// \brief Constructor.
  LRN() : PrimitiveC(kNameLRN) { InitIOName({"x"}, {"y"}); }
  /// \brief Destructor.
  ~LRN() = default;
  MS_DECLARE_PARENT(LRN, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.LRN for the inputs.
  void Init(const int64_t depth_radius = 5, const float bias = 1.0, const float alpha = 1.0, const float beta = 0.5,
            const std::string &norm_region = "ACROSS_CHANNELS");
  /// \brief Set depth_radius.
  void set_depth_radius(const int64_t depth_radius);
  /// \brief Set bias.
  void set_bias(const float bias);
  /// \brief Set alpha.
  void set_alpha(const float alpha);
  /// \brief Set beta.
  void set_beta(const float beta);
  /// \brief Set norm_region.
  void set_norm_region(const std::string &norm_region);
  /// \brief Get depth_radius.
  ///
  /// \return depth_radius.
  int64_t get_depth_radius() const;
  /// \brief Get bias.
  ///
  /// \return bias.
  float get_bias() const;
  /// \brief Get alpha.
  ///
  /// \return alpha.
  float get_alpha() const;
  /// \brief Get beta.
  ///
  /// \return beta.
  float get_beta() const;
  /// \brief Get norm_region.
  ///
  /// \return norm_region.
  std::string get_norm_region() const;
};
AbstractBasePtr LrnInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimLrn = std::shared_ptr<LRN>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_LRN_H_
