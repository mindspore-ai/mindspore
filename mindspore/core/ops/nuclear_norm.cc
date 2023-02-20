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
#include <algorithm>
#include <set>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/nuclear_norm.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
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
constexpr int64_t kDimIsNone = 1000;
abstract::ShapePtr NuclearNormInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto input_rank = SizeToLong(input_shape.size());
  const int64_t min_dim = 2;
  (void)CheckAndConvertUtils::CheckInteger("input_size", input_rank, kGreaterEqual, min_dim, prim_name);

  std::vector<int64_t> dim_list;
  auto dim_ptr = primitive->GetAttr("dim");
  MS_EXCEPTION_IF_NULL(dim_ptr);
  dim_list = GetValue<std::vector<int64_t>>(dim_ptr);
  if (dim_list.size() == 1 && dim_list[0] == kDimIsNone) {
    if (input_rank != min_dim) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', when dim is not set, or dim is None, the input_size must be equal to 2, but got "
                               << input_rank << ".";
    }
    dim_list.clear();
    dim_list.push_back(0);
    dim_list.push_back(1);
  }
  auto keepdim = GetValue<bool>(primitive->GetAttr("keepdim"));
  auto temp_shape = input_shape;
  for (size_t i = 0; i < dim_list.size(); ++i) {
    CheckAndConvertUtils::CheckInRange("dim value", dim_list[i], kIncludeLeft, {-input_rank, input_rank}, prim_name);
    if (dim_list[i] < 0) {
      dim_list[i] += input_rank;
    }
  }
  if (dim_list[0] == dim_list[1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the elements in attribute dim must be different.";
  }
  for (size_t i = 0; i < dim_list.size(); ++i) {
    if (!keepdim) {
      temp_shape[LongToSize(dim_list[i])] = -1;
    } else {
      temp_shape[LongToSize(dim_list[i])] = 1;
    }
  }
  if (!keepdim) {
    for (std::vector<int64_t>::iterator iter = temp_shape.begin(); iter != temp_shape.end(); ++iter) {
      if (*iter == -1) {
        iter = temp_shape.erase(iter);
        iter -= 1;
      }
    }
  }
  return std::make_shared<abstract::Shape>(temp_shape);
}

TypePtr NuclearNormInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  const int64_t DIMSIZE1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("The input number", SizeToLong(input_args.size()), kEqual, DIMSIZE1,
                                           op_name);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), valid_types, op_name);
}
}  // namespace

AbstractBasePtr NuclearNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = NuclearNormInferType(primitive, input_args);
  auto infer_shape = NuclearNormInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(NuclearNorm, BaseOperator);

// AG means auto generated
class MIND_API AGNuclearNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NuclearNormInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NuclearNormInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NuclearNormInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NuclearNorm, prim::kPrimNuclearNorm, AGNuclearNormInfer, false);
}  // namespace ops
}  // namespace mindspore
