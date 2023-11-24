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
#include "ops/bn_training_reduce.h"

#include <map>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/format.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
void BNTrainingReduceCheckFormat(const PrimitivePtr &primitive, const mindspore::Format format) {
  static std::vector<mindspore::Format> valid_formats{Format::NHWC, Format::NCHW, Format::NCDHW};
  auto CheckFormat = [format](const mindspore::Format other) { return format == other; };
  bool is_valid = std::any_of(valid_formats.begin(), valid_formats.end(), CheckFormat);
  if (MS_UNLIKELY(!is_valid)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', data format must be NCHW, NHWC and NCDHW, but got "
                      << FormatEnumToString(format) << ".";
  }
}

abstract::TupleShapePtr BNTrainingReduceInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto shape = input_args[0]->GetShape()->GetShapeVector();
  constexpr auto kMinInputDim = 1;
  (void)CheckAndConvertUtils::CheckInteger("x_dim", SizeToLong(shape.size()), kGreaterThan, kMinInputDim, prim_name);

  ShapeVector sum_shape{abstract::TensorShape::kShapeDimAny};
  ShapeVector square_sum_shape{abstract::TensorShape::kShapeDimAny};
  if (!IsDynamicRank(shape)) {
    // get format
    auto format_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (MS_UNLIKELY(!format_opt.has_value())) {
      MS_LOG(EXCEPTION) << "For " << prim_name << ", failed to get format's value.";
    }
    auto format = static_cast<mindspore::Format>(format_opt.value());
    BNTrainingReduceCheckFormat(primitive, format);

    auto c_axis = kInputIndex1;
    if (format == Format::NHWC && !IsDynamicRank(shape)) {
      c_axis = shape.size() - kInputIndex1;
    }
    sum_shape[0] = shape[c_axis];
    square_sum_shape[0] = shape[c_axis];
  }

  std::vector<abstract::BaseShapePtr> out_shape_list{
    std::make_shared<abstract::TensorShape>(std::move(sum_shape)),
    std::make_shared<abstract::TensorShape>(std::move(square_sum_shape))};

  return std::make_shared<abstract::TupleShape>(std::move(out_shape_list));
}

TypePtr BNTrainingReduceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[0]->GetType();
  std::set<TypePtr> check_list = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, check_list, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, input_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(BNTrainingReduce, BaseOperator);
AbstractBasePtr BNTrainingReduceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = BNTrainingReduceInferType(primitive, input_args);
  auto infer_shape = BNTrainingReduceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBNTrainingReduceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingReduceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingReduceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingReduceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BNTrainingReduce, prim::kPrimBNTrainingReduce, AGBNTrainingReduceInfer, false);
}  // namespace ops
}  // namespace mindspore
