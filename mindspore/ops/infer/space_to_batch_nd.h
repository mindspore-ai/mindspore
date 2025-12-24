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

#ifndef MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_
#define MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSpaceToBatchND = "SpaceToBatchND";
/// \brief Divides spatial dimensions into blocks and combines the block size with the original batch.
/// Refer to Python API @ref mindspore.ops.SpaceToBatchND for more details.
class OPS_API SpaceToBatchND : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SpaceToBatchND);
  /// \brief Constructor.
  SpaceToBatchND() : BaseOperator(kNameSpaceToBatchND) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SpaceToBatchND for the inputs.
  void Init(const std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> paddings);
  /// \brief Set paddings.
  void set_paddings(const std::vector<std::vector<int64_t>> paddings);
  /// \brief Set block_shape.
  void set_block_shape(std::vector<int64_t> block_shape);
  /// \brief Get block_shape.
  ///
  /// \return block_shape.
  std::vector<int64_t> get_block_shape() const;
  /// \brief Get paddings.
  ///
  /// \return paddings.
  std::vector<std::vector<int64_t>> get_paddings() const;
};
OPS_API abstract::AbstractBasePtr SpaceToBatchNDInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_
