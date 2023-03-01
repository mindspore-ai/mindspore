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

#include <string>
#include <memory>
#include <vector>

#include "ops/mat_mul.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MatMul::Init(bool transpose_a, bool transpose_b) {
  set_transpose_a(transpose_a);
  set_transpose_b(transpose_b);
}

void MatMul::set_transpose_a(bool transpose_a) { (void)AddAttr(kTransposeA, api::MakeValue(transpose_a)); }

void MatMul::set_transpose_b(bool transpose_b) { (void)AddAttr(kTransposeB, api::MakeValue(transpose_b)); }

bool MatMul::get_transpose_a() const {
  auto value_ptr = GetAttr(kTransposeA);
  return GetValue<bool>(value_ptr);
}

bool MatMul::get_transpose_b() const {
  auto value_ptr = GetAttr(kTransposeB);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(MatMul, BaseOperator);
class MatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    constexpr auto kMatMulInputNum = 2;
    const std::string op_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                             kMatMulInputNum, op_name);
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);

    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(x->shape());
    auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
    MS_EXCEPTION_IF_NULL(y);
    MS_EXCEPTION_IF_NULL(y->shape());
    auto x_shp = x->shape()->shape();
    auto y_shp = y->shape()->shape();

    ValuePtr transpose_a_ptr = primitive->GetAttr("transpose_a");
    ValuePtr transpose_b_ptr = primitive->GetAttr("transpose_b");
    bool transpose_a = GetValue<bool>(transpose_a_ptr);
    bool transpose_b = GetValue<bool>(transpose_b_ptr);

    if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
      ShapeVector ret_shape{abstract::Shape::kShapeRankAny};
      return std::make_shared<abstract::Shape>(ret_shape);
    }

    const size_t SHAPE_SIZE = 2;
    if (x_shp.size() != SHAPE_SIZE || y_shp.size() != SHAPE_SIZE) {
      MS_EXCEPTION(ValueError) << "MatMul inputs should have the same dimension size and equal to 2.";
    }
    auto x_col = x_shp[(transpose_a ? 0 : 1)];
    auto y_row = y_shp[(transpose_b ? 1 : 0)];
    if (x_col != y_row && x_col >= 0 && y_row >= 0) {
      MS_EXCEPTION(ValueError) << "MatMul shape error, got x_col: " << x_col << ", y_row: " << y_row
                               << ". In MatMul x_col and y_row should be equal.";
    }

    ShapeVector ret_shape;
    auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output, const ShapeVector xshp,
                                                   const ShapeVector yshp) -> void {
      if (!xshp.empty() && !yshp.empty()) {
        output.push_back(xshp[(transpose_a ? 1 : 0)]);
        output.push_back(yshp[(transpose_b ? 0 : 1)]);
      }
      return;
    };
    make_shape(ret_shape, x_shp, y_shp);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
    TypePtr x_type = x->element()->GetTypeTrack();
    if (x_type->type_id() == TypeId::kNumberTypeInt8) {
      x_type = kInt32;
    }
    if (primitive->HasAttr("cast_type")) {
      auto out_type = primitive->GetAttr("cast_type");
      MS_EXCEPTION_IF_NULL(out_type);
      if (!out_type->isa<Type>()) {
        MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
      }
      x_type = out_type->cast<TypePtr>();
    }
    return x_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MatMul, prim::kPrimMatMul, MatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
