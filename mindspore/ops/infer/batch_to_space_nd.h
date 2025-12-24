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

#ifndef MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_
#define MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchToSpaceND = "BatchToSpaceND";
/// \brief Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.
/// Refer to Python API @ref mindspore.ops.BatchToSpaceND for more details.
class OPS_API BatchToSpaceND : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BatchToSpaceND);
  /// \brief Constructor.
  BatchToSpaceND() : BaseOperator(kNameBatchToSpaceND) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BatchToSpaceND for the inputs.
  void Init(const std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> crops);
  /// \brief Set crops.
  void set_crops(std::vector<std::vector<int64_t>> crops);
  /// \brief Set block_shape.
  void set_block_shape(std::vector<int64_t> block_shape);
  /// \brief Get block_shape.
  ///
  /// \return block_shape.
  std::vector<int64_t> get_block_shape() const;
  /// \brief Get crops.
  ///
  /// \return crops.
  std::vector<std::vector<int64_t>> get_crops() const;
};
OPS_API abstract::AbstractBasePtr BatchToSpaceNDInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimBatchToSpaceNDPtr = std::shared_ptr<BatchToSpaceND>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_
