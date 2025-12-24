/*
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

#ifndef MINDSPORE_CORE_OPS_SPARSE_TENSOR_DENSE_MATMUL_H_
#define MINDSPORE_CORE_OPS_SPARSE_TENSOR_DENSE_MATMUL_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseTensorDenseMatmul = "SparseTensorDenseMatmul";
class OPS_API SparseTensorDenseMatmul : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseTensorDenseMatmul);
  SparseTensorDenseMatmul() : BaseOperator(kNameSparseTensorDenseMatmul) {
    InitIOName({"indices", "values", "sparse_shape", "dense"}, {"output"});
  }
  void Init(const bool adjoint_st = false, bool adjoint_dt = false);
  void set_adjoint_st(const bool adjoint_st);
  bool get_adjoint_st() const;
  void set_adjoint_dt(const bool adjoint_dt);
  bool get_adjoint_dt() const;
};
OPS_API abstract::AbstractBasePtr SparseTensorDenseMatmulInfer(
  const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
  const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_TENSOR_DENSE_MATMUL_H_
