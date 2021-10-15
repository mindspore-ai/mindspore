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
 * distributed under the License is distributed on an "ASF IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/batch_matmul.h"
#include <map>
#include <string>
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
// batchmatmul
namespace {
abstract::ShapePtr BatchMatmulInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto x_shp = x_shape_map[kShape];
  auto y_shp = y_shape_map[kShape];
  constexpr size_t dim_limit = 3;
  if (x_shp.size() != y_shp.size() || x_shp.size() < dim_limit) {
    MS_EXCEPTION(ValueError) << "For BatchMatMul, input x, y should have the same dimension size and should be greater"
                             << "or equal to 3, while x size = " << x_shp.size() << ", y size = " << y_shp.size();
  }
  constexpr size_t offset = 2;
  for (size_t i = 0; i < x_shp.size() - offset; ++i) {
    if (x_shp[i] != y_shp[i]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " shapes in dim[" << i << "] are not the same"
                               << "while x1 is " << x_shp[i] << ", x2 is " << y_shp[i];
    }
  }
  std::vector<int> x_last(x_shp.end() - SizeToInt(offset), x_shp.end());
  std::vector<int> y_last(y_shp.end() - SizeToInt(offset), y_shp.end());
  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);
  int64_t x_col = x_last[static_cast<size_t>(!transpose_a)];
  int64_t y_row = y_last[static_cast<size_t>(transpose_b)];
  if (std::find(x_shp.begin(), x_shp.end(), -1) == x_shp.end() &&
      std::find(y_shp.begin(), y_shp.end(), -1) == y_shp.end()) {
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << " evaluator shapes of inputs can not do this operator, "
                               << "got " << x_col << " and " << y_row << " , with x1 shape " << x_shp
                               << "(transpose_a=" << transpose_a << "})"
                               << ", x2 shape " << y_shp << "(transpose_b=" << transpose_b << "})";
    }
  }
  (void)primitive->AddAttr("transpose_x1", transpose_a_ptr);
  (void)primitive->AddAttr("transpose_x2", transpose_b_ptr);
  ShapeVector x_min_shape = x_shape_map[kMinShape];
  ShapeVector x_max_shape = x_shape_map[kMaxShape];
  ShapeVector y_min_shape = y_shape_map[kMinShape];
  ShapeVector y_max_shape = y_shape_map[kMaxShape];
  CheckAndConvertUtils::CheckMinMaxShape(x_shp, &x_min_shape, &x_max_shape);
  CheckAndConvertUtils::CheckMinMaxShape(y_shp, &y_min_shape, &y_max_shape);
  // Additional check for dynamic shape
  // Last infer will be real shape values
  bool x_not_dyn =
    std::all_of(x_shp.begin(), x_shp.end(), [](int64_t value) { return value != abstract::Shape::SHP_ANY; });
  bool y_not_dyn =
    std::all_of(y_shp.begin(), y_shp.end(), [](int64_t value) { return value != abstract::Shape::SHP_ANY; });
  if (x_not_dyn && y_not_dyn) {
    size_t x_offset = x_shp.size() - offset;
    size_t y_offset = y_shp.size() - offset;
    auto x_c = x_shp[x_offset + (transpose_a ? 0 : 1)];
    auto y_r = y_shp[y_offset + (transpose_b ? 1 : 0)];
    if (x_c != y_r) {
      MS_LOG(EXCEPTION) << "BatchMatMul shape error, got x_col: " << x_c << ", y_row: " << y_r
                        << ". In BatchMatMul x_col and y_row should be equal.";
    }
  }
  ShapeVector ret_shape;
  ShapeVector ret_min_shape;
  ShapeVector ret_max_shape;
  auto make_shape = [&transpose_a, &transpose_b, &offset](ShapeVector &output, const ShapeVector xshp,
                                                          const ShapeVector yshp) -> void {
    for (size_t i = 0; i < xshp.size() - offset; i++) {
      if (xshp[i] != yshp[i]) {
        if (xshp[i] > 0 && yshp[i] > 0) {
          MS_LOG(EXCEPTION) << "BatchMatMul input x, y are different at index " << i << ".";
        }
        output.push_back(abstract::Shape::SHP_ANY);
      } else {
        output.push_back(xshp[i]);
      }
    }
    size_t x_offset = xshp.size() - offset;
    size_t y_offset = yshp.size() - offset;
    output.push_back(xshp[x_offset + (transpose_a ? 1 : 0)]);
    output.push_back(yshp[y_offset + (transpose_b ? 0 : 1)]);
    return;
  };
  make_shape(ret_shape, x_shp, y_shp);
  make_shape(ret_min_shape, x_min_shape, y_min_shape);
  make_shape(ret_max_shape, x_max_shape, y_max_shape);
  return std::make_shared<abstract::Shape>(ret_shape, ret_min_shape, ret_max_shape);
}

TypePtr BatchMatmulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("w", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

void BatchMatmul::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}

void BatchMatmul::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, MakeValue(transpose_a)); }

void BatchMatmul::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, MakeValue(transpose_b)); }

bool BatchMatmul::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool BatchMatmul::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr BatchMatmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("BatchMatmul infer", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           primitive->name());
  return abstract::MakeAbstract(BatchMatmulInferShape(primitive, input_args),
                                BatchMatmulInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BatchMatmul, prim::kPrimBatchMatMul, BatchMatmulInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
