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

#include "ops/fill_v2.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
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
MIND_API_OPERATOR_IMPL(FillV2, BaseOperator);

// AG means auto generated
class MIND_API AGFillV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();

    const int64_t kDimOne = 1;
    const int64_t kDimZero = 0;

    auto input2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
    if (!IsDynamic(input2_shape)) {
      if (input2_shape.size() > 1 || (input2_shape.size() == 1 && input2_shape[0] > 1)) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', the value input only takes scalar or scalar within a tensor!";
      }
    }

    auto input1_type = input_args[kInputIndex0]->BuildType();
    auto value_ptr = input_args[kInputIndex0]->BuildValue();
    MS_EXCEPTION_IF_NULL(value_ptr);

    if (!IsValueKnown(value_ptr)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }

    ShapeVector output_shape{};
    if (input1_type->isa<TensorType>()) {
      auto input1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
      CheckAndConvertUtils::CheckInteger("rank of shape", SizeToLong(input1_shape.size()), kEqual, kDimOne, prim_name);
      output_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", value_ptr, prim_name);
    } else if (IsIdentidityOrSubclass(input1_type, kTuple)) {
      output_shape = CheckAndConvertUtils::CheckTupleInt("shape", value_ptr, prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the `shape` "
                              << " must be a tuple or tensor with all Int elements, but got " << value_ptr->type_name()
                              << ".";
    }

    for (size_t i = 0; i < output_shape.size(); ++i) {
      CheckAndConvertUtils::CheckInteger("the " + std::to_string(i) + "th dimension of input shape", output_shape[i],
                                         kGreaterThan, kDimZero, prim_name);
    }

    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto input1_type = input_args[kInputIndex0]->BuildType();
    auto input2_type = input_args[kInputIndex1]->BuildType();

    // Check the data type of the first input
    if (input1_type->isa<TensorType>()) {
      const std::set<TypePtr> input1_valid_types = {kInt32, kInt64};
      (void)CheckAndConvertUtils::CheckTensorTypeValid("input1 datatype", input1_type, input1_valid_types, prim_name);
    }
    // Check the data type of the second input and infer the data type of the output from the second input
    (void)CheckAndConvertUtils::CheckTensorTypeValid("output datatype", input2_type,
                                                     common_valid_types_with_complex_and_bool, prim_name);

    return input2_type;
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    for (auto &input : input_args) {
      MS_EXCEPTION_IF_NULL(input);
    }
    const int64_t input_num = 2;
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto infer_type = InferType(primitive, input_args);
    auto infer_shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FillV2, prim::kPrimFillV2, AGFillV2Infer, false);
}  // namespace ops
}  // namespace mindspore
