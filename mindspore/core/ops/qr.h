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
#ifndef MINDSPORE_CORE_OPS_QR_H_
#define MINDSPORE_CORE_OPS_QR_H_

#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameQr = "Qr";
/// \brief Qr operator prototype.
class MIND_API Qr : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Qr);
  Qr() : BaseOperator(kNameQr) { InitIOName({"x"}, {"q", "r"}); }
  void Init(const bool full_matrices = false);

  /// \brief Set full_matrices.
  void set_full_matrices(const bool full_matrices);

  /// \brief Get full_matrices.
  ///
  /// \return full_matrices.
  bool get_full_matrices() const;
};
abstract::AbstractBasePtr QrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimQrPtr = std::shared_ptr<Qr>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QR_H_
