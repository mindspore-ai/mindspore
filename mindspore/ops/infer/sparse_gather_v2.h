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

#ifndef MINDSPORE_CORE_OPS_SPARSE_GATHER_V2_H_
#define MINDSPORE_CORE_OPS_SPARSE_GATHER_V2_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "abstract/ops/op_infer.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseGatherV2 = "SparseGatherV2";

abstract::ShapePtr GatherInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

TypePtr GatherInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);

OPS_API abstract::AbstractBasePtr GatherInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);

/// \brief Returns a slice of the input tensor based on the specified indices and axis.
/// Refer to Python API @ref mindspore.ops.SparseGatherV2 for more details.
class OPS_API SparseGatherV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseGatherV2);
  /// \brief Constructor.
  SparseGatherV2() : BaseOperator(kNameSparseGatherV2) { InitIOName({"param", "indices", "axis"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.SparseGatherV2 for the inputs.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_GATHER_V2_H_
