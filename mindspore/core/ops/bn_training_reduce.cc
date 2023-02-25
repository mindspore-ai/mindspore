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

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/format.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t BNTrainingReduceGetAndCheckFormat(const PrimitivePtr &primitive, const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result ||
      (data_format != static_cast<int64_t>(Format::NHWC) && data_format != static_cast<int64_t>(Format::NCHW) &&
       data_format != static_cast<int64_t>(Format::NCDHW))) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', data format must be NCHW, NHWC or NCDHW, but got "
                      << data_format << ".";
  }
  return data_format;
}
abstract::TupleShapePtr BNTrainingReduceInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_shape[kShape];

  constexpr auto kMinInputDim = 1;
  (void)CheckAndConvertUtils::CheckInteger("x_dim", SizeToLong(shape.size()), kGreaterThan, kMinInputDim,
                                           primitive->name());
  auto data_format_ptr = primitive->GetAttr("format");
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  int64_t data_format = BNTrainingReduceGetAndCheckFormat(primitive, data_format_ptr);
  size_t c_axis = kInputIndex1;
  constexpr auto kNHWCInputDim = 4;
  if (data_format == static_cast<int64_t>(Format::NHWC) && shape.size() == kNHWCInputDim) {
    c_axis = kInputIndex3;
  }
  ShapeVector batch = {shape[c_axis]};
  abstract::ShapePtr sum_shape;
  abstract::ShapePtr square_sum_shape;

  sum_shape = std::make_shared<abstract::Shape>(batch);
  square_sum_shape = std::make_shared<abstract::Shape>(batch);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{sum_shape, square_sum_shape});
}

TypePtr BNTrainingReduceInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_type = input_args[0]->BuildType();
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
