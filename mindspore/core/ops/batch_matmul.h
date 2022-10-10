/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
#define MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/mat_mul.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchMatMul = "BatchMatMul";

/// \brief Computes matrix multiplication between two tensors by batch.
/// Refer to Python API @ref mindspore.ops.BatchMatmul for more details.
class MIND_API BatchMatMul : public MatMul {
 public:
  MIND_API_BASE_MEMBER(BatchMatMul);
  /// \brief Constructor.
  BatchMatMul() : MatMul(kNameBatchMatMul) { InitIOName({"x1", "x2"}, {"output"}); }
};
abstract::AbstractBasePtr BatchMatmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
