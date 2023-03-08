/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/transpose.h"
#include <vector>
#include <string>
#include <set>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputNum = 2;
}  // namespace
std::vector<int64_t> Transpose::get_perm() {
  PrimitivePtr prim = this->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<int64_t> perm = {};
  if (prim->HasAttr(kAttrPerm)) {
    auto value_ptr = prim->GetAttr(kAttrPerm);
    if (value_ptr->isa<tensor::Tensor>()) {
      perm = CheckAndConvertUtils::CheckTensorIntValue(kAttrPerm, value_ptr, prim->name());
    } else {
      perm = CheckAndConvertUtils::CheckIntOrTupleInt(kAttrPerm, value_ptr, prim->name());
    }
  }
  return perm;
}

MIND_API_OPERATOR_IMPL(Transpose, BaseOperator);

ShapeVector CheckAndGetPermValue(const std::vector<AbstractBasePtr> &input_args, const PrimitivePtr &primitive) {
  const std::string &op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, op_name);
  auto input_value = input_args[kInputIndex1]->BuildValue();
  if (input_args[kInputIndex1]->isa<abstract::AbstractTuple>()) {
    if (IsValueKnown(input_value)) {
      return CheckAndConvertUtils::CheckTupleInt("perm", input_value, op_name);
    }
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    if (input_value->isa<tensor::Tensor>()) {
      return CheckAndConvertUtils::CheckTensorIntValue("perm", input_value, op_name);
    }
    auto perm_shape = CheckAndConvertUtils::GetTensorInputShape("perm", input_args, 1);
    if (perm_shape->shape().size() != 1) {
      MS_EXCEPTION(ValueError) << "For 'transpose perm', " << op_name << " must be 1-D, but got"
                               << perm_shape->shape().size() << "-D.";
    }
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << op_name
                            << "], the perm must be a tuple or a tensor with all Int elements, but got "
                            << input_args[kInputIndex1]->type_name() << ".";
  }

  return {};
}

class TransposeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const std::string &op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, op_name);
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("input_x size", SizeToLong(x_shape.size()), kGreaterThan, 0, op_name);

    auto for_format_change_value = primitive->GetAttr(kAttrForFormatChange);
    if (for_format_change_value != nullptr && GetValue<bool>(for_format_change_value)) {
      return std::make_shared<abstract::Shape>(x_shape);
    }
    ShapeVector p_value;
    if (x_shape[0] == 0) {
      MS_EXCEPTION(ValueError) << "For 'Transpose', first dim of input_x's shape can not be 0, but got 0.";
    }
    if (IsDynamicRank(x_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }

    auto p_value_raw = CheckAndGetPermValue(input_args, primitive);
    if (p_value_raw.empty()) {
      ShapeVector out_shape;
      (void)out_shape.insert(out_shape.end(), x_shape.size(), abstract::Shape::kShapeDimAny);
      return std::make_shared<abstract::Shape>(out_shape);
    }

    for (auto p : p_value_raw) {
      p = (p >= 0) ? p : (p_value_raw.size() + p);
      if (std::abs(p) >= SizeToLong(x_shape.size())) {
        MS_EXCEPTION(ValueError) << "Perm value can not exceed shape range, must be in [0, " << (x_shape.size() - 1)
                                 << "], but got perm:" << p;
      }
      p_value.emplace_back(p);
    }
    if (x_shape.size() != p_value.size()) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the dim of 'input_x' and 'perm' must be equal, but got "
                               << x_shape.size() << " and " << p_value.size() << " respectively.";
    }
    for (auto i : p_value) {
      (void)CheckAndConvertUtils::CheckInteger("perm element", i, kLessThan, SizeToLong(p_value.size()), op_name);
    }
    std::vector<int64_t> tmp(p_value);
    for (auto it = tmp.begin(); it != tmp.end();) {
      auto dim = *it;
      if (!tmp.empty()) {
        it = tmp.erase(it);
      }
      if (std::find(tmp.begin(), tmp.end(), dim) != tmp.end()) {
        MS_EXCEPTION(ValueError) << "For '" << op_name << "', the value of perm is wrong.";
      }
    }
    std::vector<int64_t> in_shape(p_value);
    (void)std::transform(in_shape.begin(), in_shape.end(), in_shape.begin(),
                         [x_shape](size_t i) { return x_shape[i]; });
    return std::make_shared<abstract::Shape>(in_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim->name());
    return CheckAndConvertUtils::CheckSubClass("input_x", input_args[0]->BuildType(), {kTensorType}, prim->name());
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Transpose, prim::kPrimTranspose, TransposeInfer, false);
}  // namespace ops
}  // namespace mindspore
