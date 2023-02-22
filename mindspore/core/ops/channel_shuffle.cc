
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

#include "ops/channel_shuffle.h"

#include <memory>
#include <set>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr ChannelShuffleInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  ShapeVector out_shape;
  const int64_t min_dims = 3;
  int64_t group = GetValue<int64_t>(primitive->GetAttr("group"));
  auto input_shape_ = shape_map[kShape];
  auto dims = input_shape_.size();

  if (IsDynamic(input_shape_)) {
    return std::make_shared<abstract::Shape>(input_shape_);
  }

  if (dims <= min_dims) {
    MS_EXCEPTION(TypeError) << "For ChannelShuffle, expect input with > 3 dims, "
                            << "but got " << input_shape_.size() << ".";
  }
  int64_t c = input_shape_[1];
  if (group <= 0) {
    MS_EXCEPTION(ValueError) << "For ChannelShuffle, number of groups "
                             << "to divide channels in must be positive, but got " << group << ".";
  } else if ((c % group) != 0) {
    MS_EXCEPTION(ValueError) << "For ChannelShuffle, number of channels must be divisible by groups.";
  }
  auto out_base_shape = std::make_shared<abstract::Shape>(input_shape_);
  return out_base_shape->cast<abstract::ShapePtr>();
}

TypePtr ChannelShuffleInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16, kInt32,
                                         kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, prim->name());
  return x_dtype;
}
AbstractBasePtr ChannelShuffleInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto type = ChannelShuffleInferType(primitive, input_args);
  auto shape = ChannelShuffleInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

MIND_API_OPERATOR_IMPL(ChannelShuffle, BaseOperator);

// AG means auto generated
class MIND_API AGChannelShuffleInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ChannelShuffleInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ChannelShuffleInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ChannelShuffleInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ChannelShuffle, prim::kPrimChannelShuffle, AGChannelShuffleInfer, false);
}  // namespace ops
}  // namespace mindspore
