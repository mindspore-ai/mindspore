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

#ifndef MINDSPORE_CORE_OPS_REDUCE_H_
#define MINDSPORE_CORE_OPS_REDUCE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduce = "Reduce";
/// \brief Reduce defined Reduce operator prototype of lite.
class MS_CORE_API Reduce : public PrimitiveC {
 public:
  /// \brief Constructor.
  Reduce() : PrimitiveC(kNameReduce) { InitIOName({"input_x", "axis"}, {"y"}); }

  /// \brief Constructor.
  explicit Reduce(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"input_x", "axis"}, {"y"}); }

  /// \brief Destructor.
  ~Reduce() = default;

  MS_DECLARE_PARENT(Reduce, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] keep_dims Define whether keep the dims reduced, default false.
  void Init(const bool keep_dims = false);

  /// \brief Method to set keep_dims attribute.
  ///
  /// \param[in] keep_dims Define whether keep the dims reduced, default false.
  void set_keep_dims(const bool keep_dims);

  /// \brief Method to get keep_dims attribute.
  ///
  /// \return keep_dims attribute.
  bool get_keep_dims() const;
};
AbstractBasePtr ReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimReducePtr = std::shared_ptr<Reduce>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_H_
