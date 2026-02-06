/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <memory>
#include <vector>

#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "infer/ops_func_impl/linalg_vector_norm.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"

namespace mindspore {
namespace ops {
namespace {
InferInfoPtr MakeInferInfo(const AbstractBasePtr &abs, const std::string &op_type, const std::string &arg_name) {
  return std::make_unique<AbstractInferInfoAdapter>(abs, op_type, arg_name);
}

InferInfoPtrList BuildInputInfos(const AbstractBasePtr &x_abs) {
  auto ord_abs = std::make_shared<abstract::AbstractNone>();
  auto dim_abs = std::make_shared<abstract::AbstractNone>();
  auto keepdim_abs = std::make_shared<abstract::AbstractScalar>(static_cast<bool>(false));
  auto dtype_abs = std::make_shared<abstract::AbstractNone>();

  InferInfoPtrList input_infos;
  input_infos.push_back(MakeInferInfo(x_abs, "LinalgVectorNorm", "x"));
  input_infos.push_back(MakeInferInfo(ord_abs, "LinalgVectorNorm", "ord"));
  input_infos.push_back(MakeInferInfo(dim_abs, "LinalgVectorNorm", "dim"));
  input_infos.push_back(MakeInferInfo(keepdim_abs, "LinalgVectorNorm", "keepdim"));
  input_infos.push_back(MakeInferInfo(dtype_abs, "LinalgVectorNorm", "dtype"));
  return input_infos;
}
}  // namespace

TEST(TestLinalgVectorNormInferType, Complex64OutputFloat32WhenDtypeNone) {
  auto primitive = std::make_shared<Primitive>("LinalgVectorNorm");
  auto x_abs = std::make_shared<abstract::AbstractTensor>(kComplex64, ShapeVector{2, 3});
  auto input_infos = BuildInputInfos(x_abs);

  auto infer_impl = std::make_shared<LinalgVectorNormFuncImpl>();
  auto infer_types = infer_impl->InferType(primitive, input_infos);
  ASSERT_EQ(infer_types.size(), 1);
  ASSERT_EQ(infer_types[0], kNumberTypeFloat32);
}

TEST(TestLinalgVectorNormInferType, Complex128OutputFloat64WhenDtypeNone) {
  auto primitive = std::make_shared<Primitive>("LinalgVectorNorm");
  auto x_abs = std::make_shared<abstract::AbstractTensor>(kComplex128, ShapeVector{2, 3});
  auto input_infos = BuildInputInfos(x_abs);

  auto infer_impl = std::make_shared<LinalgVectorNormFuncImpl>();
  auto infer_types = infer_impl->InferType(primitive, input_infos);
  ASSERT_EQ(infer_types.size(), 1);
  ASSERT_EQ(infer_types[0], kNumberTypeFloat64);
}
}  // namespace ops
}  // namespace mindspore
