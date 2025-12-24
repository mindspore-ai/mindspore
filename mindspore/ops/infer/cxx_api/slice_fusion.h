/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_SLICE_FUSION_H_
#define MINDSPORE_CORE_OPS_SLICE_FUSION_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSliceFusion = "SliceFusion";
/// \brief SliceFusion defined Slice operator prototype of lite.
class OPS_API SliceFusion : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SliceFusion);
  /// \brief Constructor.
  SliceFusion() : BaseOperator(kNameSliceFusion) { InitIOName({"x", "begin", "size"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axes Define this operation will be performed on which axes.
  void Init(const std::vector<int64_t> &axes);

  /// \brief Method to set axes attribute.
  ///
  /// \param[in] axes Define this operation will be performed on which axes.
  void set_axes(const std::vector<int64_t> &axes);

  /// \brief Method to get axes attribute.
  ///
  /// \return axes.
  std::vector<int64_t> get_axes() const;
};
abstract::AbstractBasePtr SliceFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SLICE_FUSION_H_
