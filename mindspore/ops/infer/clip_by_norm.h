/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CLIP_BY_NORM_H_
#define MINDSPORE_CORE_OPS_CLIP_BY_NORM_H_
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameClipByNorm = "ClipByNorm";
/// \brief Clips tensor value to a maximum `L_2`-norm.
/// Refer to Python API @ref mindspore.ops.ClipByNorm for more details.
class OPS_API ClipByNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ClipByNorm);
  /// \brief Constructor.
  ClipByNorm() : BaseOperator(kNameClipByNorm) { InitIOName({"x", "clip_norm"}, {"output"}); }
  explicit ClipByNorm(const std::string &op_name) : BaseOperator(op_name) {
    InitIOName({"x", "clip_norm"}, {"output"});
  }
  /// \brief Init single axis. Refer to the initialization of Python API @ref mindspore.ops.ClipByNorm.
  void Init(const int64_t axis = -1);
  /// \brief Init multiple axis. Refer to the initialization of Python API @ref mindspore.ops.ClipByNorm.
  void Init(const std::vector<int64_t> &axis);
  /// \brief Get the attribute `axis`. The return obj is a vector, which includes the `axis`.
  std::vector<int64_t> GetAxis() const;
};

OPS_API abstract::AbstractBasePtr ClipByNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args_abs);

using ClipByNormPtr = std::shared_ptr<ClipByNorm>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CLIP_BY_NORM_H_
