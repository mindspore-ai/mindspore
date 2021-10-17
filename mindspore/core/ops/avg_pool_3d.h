/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_3D_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_3D_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
/// \brief 3D Average pooling operation. Refer to Python API @ref mindspore.ops.AvgPool3D for more details.
class MS_CORE_API AvgPool3D : public PrimitiveC {
 public:
  /// \brief Constructor.
  AvgPool3D() : PrimitiveC(prim::kPrimAvgPool3D->name()) { InitIOName({"input"}, {"output"}); }
  /// \brief Destructor.
  ~AvgPool3D() = default;
  MS_DECLARE_PARENT(AvgPool3D, PrimitiveC);
};

AbstractBasePtr AvgPool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimAvgPool3DPtr = std::shared_ptr<AvgPool3D>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_3D_H_
