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

#ifndef MINDSPORE_CORE_OPS_SPACE_TO_DEPTH_H_
#define MINDSPORE_CORE_OPS_SPACE_TO_DEPTH_H_
#include <memory>
#include <vector>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSpaceToDepth = "SpaceToDepth";
/// \brief Rearranges blocks of spatial data into depth.
/// Refer to Python API @ref mindspore.ops.SpaceToDepth for more details.
class OPS_API SpaceToDepth : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SpaceToDepth);
  /// \brief Constructor.
  SpaceToDepth() : BaseOperator(kNameSpaceToDepth) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SpaceToDepth for the inputs.
  void Init(const int64_t block_size, const Format &format = Format::NCHW);
  /// \brief Set block_size.
  void set_block_size(const int64_t block_size);
  /// \brief Get block_size.
  ///
  /// \return block_size.
  int64_t get_block_size() const;
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
};
OPS_API abstract::AbstractBasePtr SpaceToDepthInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SpaceToDepth_H_
