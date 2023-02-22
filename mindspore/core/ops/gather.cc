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

#include "ops/gather.h"

#include <set>
#include <memory>
#include <algorithm>
#include <iterator>

#include "utils/check_convert_utils.h"
#include "ops/gather_comm.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr GatherInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  auto params_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  // Dynamic rank.
  if (params_shape_ptr->IsDimUnknown() || indices_shape_ptr->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  abstract::AbstractTensorPtr indices =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  abstract::AbstractTensorPtr params =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  int64_t axis_val = 0;
  bool is_axis_dyn = false;
  // 3rd input is a Tensor when Gather is a dynamic shape operator
  if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
    auto axis = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    if (axis_value_ptr->isa<tensor::Tensor>()) {
      auto axis_vec = CheckAndConvertUtils::CheckTensorIntValue("axis", axis_value_ptr, op_name);
      if (axis_vec.size() != 1) {
        MS_LOG(EXCEPTION) << " The input number of Gather axis must be int, but got " << axis_vec;
      }
      axis_val = axis_vec[0];
    } else {
      is_axis_dyn = true;
    }
  } else if (input_args[kInputIndex2]->isa<abstract::AbstractScalar>()) {
    auto axis_value = input_args[kInputIndex2]->cast<abstract::AbstractScalarPtr>()->BuildValue();
    if (axis_value->isa<AnyValue>()) {
      is_axis_dyn = true;
    } else {
      axis_val = GetValue<int64_t>(axis_value);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', the third input type should be tensor or scalar, but got invalid abstract type:"
                      << input_args[kInputIndex2]->type_name() << ".";
  }
  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();
  ShapeVector out_shape = {};
  constexpr int dynamic_rank_value = -2;
  if (IsDynamicRank(params_shp) || IsDynamicRank(indices_shp) || is_axis_dyn) {
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  auto params_rank = static_cast<int64_t>(params_shp.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis_val, kIncludeLeft, {-params_rank, params_rank}, op_name);
  // check axis_val within interval: [0, params_rank)
  if (!(-params_rank <= axis_val) || !(axis_val < params_rank)) {
    MS_LOG(EXCEPTION) << "For 'Gather', axis value must be within range [" << -params_rank << ", " << params_rank
                      << "], but got: " << axis_val << ".";
  }
  if (axis_val < 0) {
    axis_val += params_rank;
  }
  auto calc_shape = [axis_val](const ShapeVector &ind_vec, const ShapeVector &params_vec) -> ShapeVector {
    ShapeVector out_vec;
    (void)std::copy(params_vec.begin(), params_vec.begin() + axis_val, std::back_inserter(out_vec));
    (void)copy(ind_vec.begin(), ind_vec.end(), std::back_inserter(out_vec));
    (void)copy(params_vec.begin() + axis_val + 1, params_vec.end(), std::back_inserter(out_vec));
    return out_vec;
  };
  out_shape = calc_shape(indices_shp, params_shp);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr GatherInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  std::set<TypePtr> valid_params_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("params", input_args[kInputIndex0]->BuildType(), valid_params_types,
                                            op_name);
  std::set<TypePtr> int_types = {kInt8, kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[kInputIndex1]->BuildType(), int_types,
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTypeValid("axis", input_args[kInputIndex2]->BuildType(), int_types, op_name);

  abstract::AbstractTensorPtr params =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  return params->BuildType();
}

AbstractBasePtr GatherInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = GatherInferType(primitive, input_args);
  auto infer_shape = GatherInferShape(primitive, input_args);
  (void)primitive->AddAttr("batch_dims", MakeValue(static_cast<int64_t>(0)));  // Add temporarily for gatherv2 on ascend
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Gather, BaseOperator);

// AG means auto generated
class MIND_API AGGatherInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Gather, prim::kPrimGather, AGGatherInfer, false);
}  // namespace ops
}  // namespace mindspore
