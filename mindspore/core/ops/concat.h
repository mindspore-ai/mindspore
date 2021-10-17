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

#ifndef MINDSPORE_CORE_OPS_CONCAT_H_
#define MINDSPORE_CORE_OPS_CONCAT_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConcat = "Concat";
/// \brief Connect tensor in the specified axis.
/// Refer to Python API @ref mindspore.ops.Concat for more details.
class MS_CORE_API Concat : public PrimitiveC {
 public:
  /// \brief Constructor.
  Concat() : PrimitiveC(kNameConcat) {}
  /// \brief Destructor.
  ~Concat() = default;
  MS_DECLARE_PARENT(Concat, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Concat for the inputs.
  void Init(const int64_t axis = 0);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
};
AbstractBasePtr ConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimConcatPtr = std::shared_ptr<Concat>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CONCAT_H_
