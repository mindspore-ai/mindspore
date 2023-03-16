/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include <memory>
#include <vector>

#include "ops/one_hot.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
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
void OneHot::Init(const int64_t axis) { this->set_axis(axis); }

void OneHot::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t OneHot::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

MIND_API_OPERATOR_IMPL(OneHot, BaseOperator);
namespace {
const int64_t kOneHotInputsNum = 4;
}
class OneHotInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = kOneHotInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto op_name = primitive->name();
    const size_t depth_index = 1;
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto in_shape = shape_map[kShape];

    if (IsDynamicRank(in_shape)) {
      return input_args[0]->BuildShape();
    }

    int64_t axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
    CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-1, SizeToLong(in_shape.size())}, op_name);
    auto depth = input_args[depth_index]->BuildValue();
    MS_EXCEPTION_IF_NULL(depth);
    int64_t depth_value;
    if (depth->isa<tensor::Tensor>()) {
      auto depth_data = CheckAndConvertUtils::CheckTensorIntValue("depth", depth, op_name);
      if (depth_data.size() != 1) {
        MS_LOG_EXCEPTION << "For " << op_name << ", size of depth shouble be 1, but got " << depth_data.size();
      }
      depth_value = depth_data[0];
      (void)CheckAndConvertUtils::CheckInteger("depth value", depth_value, kGreaterEqual, 0, op_name);
    } else if (depth->isa<Int64Imm>()) {
      depth_value = GetValue<int64_t>(depth);
      (void)CheckAndConvertUtils::CheckInteger("depth value", depth_value, kGreaterEqual, 0, op_name);
    } else if (depth->isa<ValueAny>()) {
      depth_value = abstract::Shape::kShapeDimAny;
    } else {
      MS_EXCEPTION(TypeError) << "For '" << op_name
                              << "', 'depth' must be a tensor or number of int64, but got an invalid type.";
    }

    if (axis >= 0) {
      (void)in_shape.insert(in_shape.begin() + axis, depth_value);
    } else {
      in_shape.push_back(depth_value);
    }

    return std::make_shared<abstract::Shape>(in_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto op_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kOneHotInputsNum,
                                             op_name);

    (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[kInputIndex0]->BuildType(),
                                                     {kUInt8, kInt32, kInt64}, op_name);
    (void)CheckAndConvertUtils::CheckTypeValid("depth", input_args[kInputIndex1]->BuildType(),
                                               {kInt8, kInt16, kInt32, kInt64}, op_name);
    std::map<std::string, TypePtr> args = {{"on_value", input_args[kInputIndex2]->BuildType()},
                                           {"off_dtype", input_args[kInputIndex3]->BuildType()}};
    return CheckAndConvertUtils::CheckTensorTypeSame(
      args,
      {kBool, kInt, kInt8, kInt16, kInt32, kInt64, kUInt, kUInt8, kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32,
       kFloat64, kComplex64, kComplex128},
      op_name);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(OneHot, prim::kPrimOneHot, OneHotInfer, false);
}  // namespace ops
}  // namespace mindspore
