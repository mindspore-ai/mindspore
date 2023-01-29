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
 * distributed under the License is distributed on an "ASF IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/batch_matmul.h"
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/mat_mul.h"

namespace mindspore {
namespace ops {
// batchmatmul
namespace {
constexpr size_t kMatSize = 2;

void BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp, bool transpose_a,
                          bool transpose_b, size_t offset) {
  if (xshp.empty() || yshp.empty()) {
    return;
  }
  ShapeVector long_input = xshp.size() > yshp.size() ? xshp : yshp;
  ShapeVector short_input = xshp.size() > yshp.size() ? yshp : xshp;
  size_t size_diff = long_input.size() - short_input.size();
  for (size_t i = 0; i < long_input.size() - offset; i++) {
    if (long_input[i] < 0) {
      output->push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      output->push_back(long_input[i] > short_input[i - size_diff] ? long_input[i] : short_input[i - size_diff]);
    } else {
      output->push_back(long_input[i]);
    }
  }
  size_t x_offset = xshp.size() - offset;
  size_t y_offset = yshp.size() - offset;
  output->push_back(xshp[x_offset + (transpose_a ? 1 : 0)]);
  output->push_back(yshp[y_offset + (transpose_b ? 0 : 1)]);
  return;
}

void CheckBatchMatmulInputSize(const std::string &op_name, const std::string &input_name, const ShapeVector &shape) {
  constexpr size_t dim_limit = 2;
  if (shape.size() < dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D or higher dimensional Tensor, but got " << shape.size() << "D shape "
                             << shape;
  }
}

void CheckBatchMatmulInputWhetherCanBeMul(const std::string &name, const ShapeVector &x_shape,
                                          const ShapeVector &y_shape, bool transpose_a, bool transpose_b) {
  ShapeVector x_mat_shape(x_shape.end() - kMatSize, x_shape.end());
  ShapeVector y_mat_shape(y_shape.end() - kMatSize, y_shape.end());
  int64_t x_col = x_mat_shape[static_cast<size_t>(!transpose_a)];
  int64_t y_row = y_mat_shape[static_cast<size_t>(transpose_b)];
  if (std::find(x_shape.begin(), x_shape.end(), -1) == x_shape.end() &&
      std::find(y_shape.begin(), y_shape.end(), -1) == y_shape.end()) {
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                               << x_shape << "(transpose_a=" << transpose_a << "), y shape " << y_shape
                               << "(transpose_b=" << transpose_b << ")";
    }
  }
}

void CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name, const ShapeVector &x_shape,
                                                const ShapeVector &y_shape) {
  ShapeVector x_batch(x_shape.begin(), x_shape.end() - kMatSize);
  ShapeVector y_batch(y_shape.begin(), y_shape.end() - kMatSize);
  if (x_batch == y_batch) {
    return;
  }

  size_t min_size = std::min(x_batch.size(), y_batch.size());
  for (size_t i = 0; i < min_size; ++i) {
    auto x = *(x_batch.rbegin() + i);
    auto y = *(y_batch.rbegin() + i);
    if (x != 1 && y != 1 && x != y) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", one of the input's batch dim must be equal to another input's peer batch dim, or "
                                  "be equal to 1, or be empty, but are "
                               << x << " and " << y << ", with x shape " << x_shape << ", y shape " << y_shape;
    }
  }
}

abstract::ShapePtr BatchMatmulInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto constexpr kBatchMatmulInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, kBatchMatmulInputNum,
                                           primitive->name());
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a Tensor type, but got:" << input_args[0]->ToString();
  }
  if (y_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'y' must be a Tensor type, but got:" << input_args[1]->ToString();
  }
  auto x_shp = x_shape_map[kShape];
  auto y_shp = y_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
  ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
  bool transpose_a = GetValue<bool>(transpose_a_ptr);
  bool transpose_b = GetValue<bool>(transpose_b_ptr);
  (void)primitive->AddAttr("transpose_x1", transpose_a_ptr);
  (void)primitive->AddAttr("transpose_x2", transpose_b_ptr);

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(y_shp);
  if (!dynamic_shape) {
    CheckBatchMatmulInputSize(prim_name, "x", x_shp);
    CheckBatchMatmulInputSize(prim_name, "y", y_shp);
    CheckBatchMatmulInputWhetherCanBeMul(prim_name, x_shp, y_shp, transpose_a, transpose_b);
    CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x_shp, y_shp);
  }

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, y_shp, transpose_a, transpose_b, kMatSize);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr BatchMatmulInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("w", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  TypePtr x_type = input_args[0]->BuildType();
  if (x_type->type_id() == TypeId::kNumberTypeInt8 || x_type->ToString() == "Tensor[Int8]") {
    x_type = kInt32;
  }
  if (prim->HasAttr("cast_type")) {
    auto out_type = prim->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', MatMul cast_type must be a 'Type', but got: '"
                               << out_type << "'.";
    }
    x_type = out_type->cast<TypePtr>();
  }
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(BatchMatMul, MatMul);

AbstractBasePtr BatchMatmulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("BatchMatmul infer", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           primitive->name());
  auto infer_type = BatchMatmulInferType(primitive, input_args);
  auto infer_shape = BatchMatmulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBatchMatmulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchMatmulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchMatmulInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchMatmulInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchMatMul, prim::kPrimBatchMatMul, AGBatchMatmulInfer, false);
}  // namespace ops
}  // namespace mindspore
