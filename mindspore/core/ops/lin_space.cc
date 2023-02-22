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

#include "ops/lin_space.h"

#include <memory>
#include <map>
#include <string>
#include <set>
#include <vector>

#include "ops/primitive_c.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->isa<AnyValue>()))
TypePtr LinSpaceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);

  auto start_dtype = input_args[kInputIndex0]->BuildType();
  auto stop_dtype = input_args[kInputIndex1]->BuildType();

  std::map<std::string, TypePtr> type_dict = {
    {"start type", start_dtype},
    {"stop type", stop_dtype},
  };
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, {kFloat32, kFloat64}, prim_name);
}
abstract::ShapePtr LinSpaceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  auto start_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(start_shape_ptr);
  auto stop_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(stop_shape_ptr);

  auto num_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(num_value);

  bool is_compile = IsNoneOrAnyValue(num_value);
  // Do it later
  if (start_shape_ptr->IsDynamic() || stop_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }

  int64_t num = 0;
  if (!is_compile) {
    if (input_args[kInputIndex2]->isa<abstract::AbstractTensor>()) {
      if (num_value->isa<tensor::Tensor>()) {
        auto num_shape_ptr = input_args[kInputIndex2]->BuildShape();
        const auto num_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(num_shape_ptr)[kShape];
        if (num_shape.size() != 0) {
          MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                                  << "], the 'num' must be int or 0D int32/int64 Tensor, but got " << num_shape.size()
                                  << "D Tensor.";
        }
        auto num_input = CheckAndConvertUtils::CheckTensorIntValue("num", num_value, prim_name);
        num = num_input[0];
      } else {
        MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                                << "], the 'num' must be int or 0D int32/int64 Tensor, but got "
                                << num_value->ToString() << ".";
      }
    } else if (input_args[kInputIndex2]->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION_IF_NULL(num_value);
      if (!num_value->isa<Int64Imm>()) {
        MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                                << "], the 'num' must be int or 0D int32/int64 Tensor, but got "
                                << num_value->ToString() << ".";
      }
      num = num_value->cast<Int64ImmPtr>()->value();
    } else {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                              << "], the 'num' must be int or 0D int32/int64 Tensor, but got " << num_value->ToString()
                              << ".";
    }
  } else {
    ShapeVector out_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  (void)CheckAndConvertUtils::CheckValue<int64_t>("num", num, kGreaterThan, 0, prim_name);

  const auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(start_shape_ptr)[kShape];
  const auto stop_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(stop_shape_ptr)[kShape];

  size_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = LongToSize(GetValue<int64_t>(value_ptr));
  }

  (void)CheckAndConvertUtils::CheckValue("rank of 'start'", start_shape.size(), kEqual, batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckValue("rank of 'stop'", stop_shape.size(), kEqual, batch_rank, prim_name);

  if (batch_rank == 0) {
    ShapeVector out_shape = {num};
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    CheckAndConvertUtils::Check("shape of 'start'", start_shape, kEqual, stop_shape, prim_name);
    ShapeVector out_shape(start_shape.begin(), start_shape.end());
    out_shape.push_back(num);
    return std::make_shared<abstract::Shape>(out_shape);
  }
}
}  // namespace
MIND_API_OPERATOR_IMPL(LinSpace, BaseOperator);
AbstractBasePtr LinSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = LinSpaceInferType(primitive, input_args);
  auto infer_shape = LinSpaceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLinSpaceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LinSpaceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LinSpaceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LinSpaceInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LinSpace, prim::kPrimLinSpace, AGLinSpaceInfer, false);
}  // namespace ops
}  // namespace mindspore
