/**
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
#include "ops/quantile.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "mindapi/base/shared_ptr.h"
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
namespace {
constexpr int kQuantileDefaultDim = 10000;

abstract::ShapePtr QuantileInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto input = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto q_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto q_dim = q_shape.size();
  if (IsDynamicRank(input_shape) || IsDynamicRank(q_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  std::vector<int64_t> out_shape;
  auto dim_ptr = primitive->GetAttr("dim");
  MS_EXCEPTION_IF_NULL(dim_ptr);

  auto dim = GetValue<int64_t>(dim_ptr);
  int64_t input_dim = SizeToLong(input_shape.size());
  int64_t wrapped_input_dim = input_dim;

  if (wrapped_input_dim == 0) {
    wrapped_input_dim = 1;
  }

  if (dim != kQuantileDefaultDim && (dim < -wrapped_input_dim || dim >= wrapped_input_dim)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the attr dim must be range of [" << -wrapped_input_dim
                             << "," << (wrapped_input_dim - 1) << "]";
  }

  if (q_dim > 1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the input q must be a scalar or 1D tensor,but got dimension = " << q_dim << ".";
  }

  if (dim < 0) {
    dim = dim + wrapped_input_dim;
  }
  auto keep_dims_ptr = primitive->GetAttr("keep_dims");
  MS_EXCEPTION_IF_NULL(keep_dims_ptr);
  auto keep_dims = GetValue<bool>(keep_dims_ptr);
  int q_size = 1;
  for (uint64_t i = 0; i < q_shape.size(); i++) {
    q_size *= q_shape[i];
  }

  if (dim != kQuantileDefaultDim && input_dim > 0) {
    out_shape = input_shape;
    if (keep_dims) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  } else if (keep_dims) {
    out_shape = std::vector<int64_t>(input_dim, 1);
  }
  if (q_dim > 0) {
    out_shape.insert(out_shape.begin(), q_size);
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr QuantileInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto q = input_args[1];
  MS_EXCEPTION_IF_NULL(q);
  auto q_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(q_type);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> dict_type;
  (void)dict_type.insert(std::make_pair("q", q_type));
  (void)dict_type.insert(std::make_pair("input", input_type));
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, prim_name);

  auto q_value = q->BuildValue();
  MS_EXCEPTION_IF_NULL(q_value);
  if (q->isa<abstract::AbstractTensor>()) {
    CheckAndConvertUtils::CheckTensorTypeSame(dict_type, valid_types, prim_name);
  } else if (q->isa<abstract::AbstractScalar>()) {
    if (q_value != nullptr) {
      if (!q_value->isa<FloatImm>()) {
        MS_EXCEPTION(TypeError) << "For '" << prim_name
                                << "', the type of 'q' must be float or tensor, but got: " << q_type->ToString() << ".";
      }
      auto value = GetValue<float>(q_value);
      if (value < 0 || value > 1) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the 'q' must in the range [0, 1], but got: " << value
                                 << ".";
      }
    }
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the type of 'q' must be float or tensor, but got: " << q_type->ToString() << ".";
  }
  return input_type;
}
}  // namespace

void Quantile::set_dim(int64_t dim) { (void)AddAttr(kDim, api::MakeValue(dim)); }

void Quantile::set_keepdim(bool keepdim) { (void)AddAttr(kKeepdim, api::MakeValue(keepdim)); }

void Quantile::set_ignorenan(bool ignorenan) { (void)AddAttr(kIgnoreNan, api::MakeValue(ignorenan)); }

int64_t Quantile::get_dim() const {
  auto value_ptr = GetAttr(kDim);
  return GetValue<int64_t>(value_ptr);
}

bool Quantile::get_keepdim() const {
  auto value_ptr = GetAttr(kKeepdim);
  return GetValue<bool>(value_ptr);
}

bool Quantile::get_ignorenan() const {
  auto value_ptr = GetAttr(kIgnoreNan);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(Quantile, BaseOperator);
AbstractBasePtr QuantileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = QuantileInferType(primitive, input_args);
  auto infer_shape = QuantileInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGQuantileInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantileInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantileInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return QuantileInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Quantile, prim::kPrimQuantile, AGQuantileInfer, false);
}  // namespace ops
}  // namespace mindspore
