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

#ifndef MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
#define MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
/// \brief Computes matrix multiplication between two tensors by batch.
/// Refer to Python API @ref mindspore.ops.BatchMatmul for more details.
class MS_CORE_API BatchMatmul : public PrimitiveC {
 public:
  /// \brief Constructor.
  BatchMatmul() : PrimitiveC(prim::kPrimBatchMatMul->name()) { InitIOName({"x1", "x2"}, {"output"}); }
  /// \brief Destructor.
  ~BatchMatmul() = default;
  MS_DECLARE_PARENT(BatchMatmul, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.BatchMatmul for the inputs.
  void Init(bool transpose_a = false, bool transpose_b = false);
  /// \brief Set transpose_a.
  void set_transpose_a(bool transpose_a);
  /// \brief Set transpose_b.
  void set_transpose_b(bool transpose_b);
  /// \brief Get transpose_a.
  ///
  /// \return transpose_a.
  bool get_transpose_a() const;
  /// \brief Get transpose_b.
  ///
  /// \return transpose_b.
  bool get_transpose_b() const;
};
AbstractBasePtr BatchMatmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args);
using PrimBatchMatmulPtr = std::shared_ptr<BatchMatmul>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_MATMUL_H_
