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

#include "ops/matrix_diag.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto matrixdiag_prim = primitive->cast<PrimMatrixDiagPtr>();
  MS_EXCEPTION_IF_NULL(matrixdiag_prim);
  auto prim_name = matrixdiag_prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto assist_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("assist_shape", input_args[1]->BuildShape(), prim_name);

  CheckAndConvertUtils::CheckInteger("assist rank", (int64_t)assist_shape.size(), kGreaterEqual, 2, prim_name);
  CheckAndConvertUtils::Check("x_shape rank", (int64_t)x_shape.size() + 1, kLessEqual, "assist rank",
                              (int64_t)assist_shape.size(), prim_name);
  CheckAndConvertUtils::Check("assist's penultimate dimension", assist_shape[(int64_t)assist_shape.size() - 2], kEqual,
                              "assist's last dimension", assist_shape[(int64_t)assist_shape.size() - 1], prim_name);

  int64_t x_end_dim = x_shape.size() - 1;
  int64_t assist_end_dim = assist_shape.size() - 1;
  while (x_end_dim >= 0) {
    if (x_shape[x_end_dim] != 1) {
      CheckAndConvertUtils::Check("reverse x dim", x_shape[x_end_dim], kEqual, "reverse assist dim",
                                  assist_shape[assist_end_dim - 1], prim_name);
    }
    x_end_dim--;
    assist_end_dim--;
  }
  return std::make_shared<abstract::Shape>(assist_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeFloat16,
                                        kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->BuildType());
  types.emplace("assist", input_args[1]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  auto type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return data_type;
}
}  // namespace

AbstractBasePtr MatrixDiagInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameMatrixDiag, MatrixDiag);
}  // namespace ops
}  // namespace mindspore
