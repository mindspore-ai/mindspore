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
#include "infer/grad/pad_v3_grad.h"
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "mindapi/helper.h"
#include "primitive/math_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore {
namespace ops {
namespace {
void PaddingsValueCheck(const PrimitivePtr &primitive, const ShapeVector &x_shape,
                        const std::vector<int64_t> &paddings_val, const std::string &prim_name) {
  const int64_t max_x_dim = 5;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    (void)CheckAndConvertUtils::CheckInteger("x_dim", SizeToLong(x_shape.size()), kLessThan, max_x_dim, prim_name);
    // For Ascend, ge::PadV3Grad only support paddings has positive value, and this node is called when mode
    // is not 'constant'
    auto mode = GetValue<std::string>(primitive->GetAttr("mode"));
    if (mode != "constant") {
      (void)CheckAndConvertUtils::CheckPositiveVector("paddings", paddings_val, prim_name);
    }
  }
}

std::optional<std::vector<int64_t>> PadV3GradFetchPaddingsValFromArg(const PrimitivePtr &primitive,
                                                                     const AbstractBasePtr &paddings_arg) {
  std::vector<int64_t> paddings_val;
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  auto padding_type = paddings_arg->GetType();
  if (padding_type->isa<TensorType>()) {
    auto paddings_shape_ptr = paddings_arg->GetShape();
    MS_EXCEPTION_IF_NULL(paddings_shape_ptr);
    auto paddings_value_ptr = paddings_arg->GetValue();
    MS_EXCEPTION_IF_NULL(paddings_value_ptr);
    if (paddings_shape_ptr->IsDynamic() || paddings_value_ptr->isa<ValueAny>()) {
      return std::nullopt;
    }
    paddings_val =
      CheckAndConvertUtils::CheckTensorIntValue("paddings value", paddings_value_ptr, prim_name, padding_type);
  } else if (padding_type->isa<Tuple>() || padding_type->isa<List>()) {
    auto value_ptr = paddings_arg->GetValue();
    if (IsValueKnown(value_ptr)) {
      paddings_val = CheckAndConvertUtils::CheckIntOrTupleInt("paddings value", paddings_arg, prim_name);
    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  return paddings_val;
}

std::vector<int64_t> PadV3GradDealWithPaddingValue(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape,
                                                   size_t paddings_size, const std::vector<int64_t> &ori_paddings_val) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> paddings_val(ori_paddings_val);
  if (paddings_size == x_shape.size() * kInputIndex2) {
    MS_LOG(INFO) << "For " << prim_name
                 << ", the paddings' val has been changed in ascend backend pass, which causes paddings' size expanded "
                    "to 2 times x_rank(8, or 10)";
    // (0, 1, 2, 3, 4, 5, 6, 7) -> (6, 7, 4, 5, 2, 3, 0, 1)
    std::reverse(paddings_val.begin(), paddings_val.end());
    for (size_t i = 1; i < paddings_val.size(); i += kInputIndex2) {
      std::swap(paddings_val[i - 1], paddings_val[i]);
    }
  } else {
    PaddingsValueCheck(primitive, x_shape, paddings_val, prim_name);
    auto paddings_contiguous = GetValue<bool>(primitive->GetAttr("paddings_contiguous"));
    if (!paddings_contiguous) {
      for (size_t i = 0; i < paddings_size; ++i) {
        if (i % kInputIndex2 == 0) {
          paddings_val[i] = ori_paddings_val[i / kInputIndex2];
        } else {
          paddings_val[i] = ori_paddings_val[(i + paddings_size) / kInputIndex2];
        }
      }
    }
  }
  return paddings_val;
}

std::vector<int64_t> PadV3GradInferOutputShapeWithPaddings(const PrimitivePtr &primitive,
                                                           const std::vector<int64_t> &x_shape, size_t paddings_size,
                                                           const std::vector<int64_t> &paddings_val) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> out_shape;
  if (paddings_size == kInputIndex2) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 2", SizeToLong(kInputIndex3), kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kInputIndex4) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 4", SizeToLong(kInputIndex4), kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[kInputIndex2] - paddings_val[kInputIndex3]);
    (void)out_shape.emplace_back(x_shape[kInputIndex3] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kInputIndex6) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 6", SizeToLong(kInputIndex5), kEqual,
                                             SizeToLong(x_shape.size()), prim_name);
    (void)out_shape.emplace_back(x_shape[0]);
    (void)out_shape.emplace_back(x_shape[1]);
    (void)out_shape.emplace_back(x_shape[kInputIndex2] - paddings_val[kInputIndex4] - paddings_val[kInputIndex5]);
    (void)out_shape.emplace_back(x_shape[kInputIndex3] - paddings_val[kInputIndex2] - paddings_val[kInputIndex3]);
    (void)out_shape.emplace_back(x_shape[kInputIndex4] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kInputIndex2 * x_shape.size()) {
    MS_LOG(INFO) << "For " << prim_name
                 << ", the paddings' val has been changed in ascend backend pass, which causes paddings' size expanded "
                    "to 2 times x_rank(8, or 10)";
    MS_ASSERT(paddings_val.size() == kInputIndex8 || paddings_val.size() == kInputIndex10);
    for (size_t i = 0; i < x_shape.size(); i++) {
      (void)out_shape.push_back(x_shape[i] - paddings_val[paddings_size - kInputIndex2 * i - kInputIndex1] -
                                paddings_val[paddings_size - kInputIndex2 * (i + kInputIndex1)]);
    }
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the length of paddings must be 2, 4 or 6, but got "
                             << paddings_size;
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("out_shape", out_shape, prim_name);
  return out_shape;
}

abstract::ShapePtr PadV3GradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  // support dynamic rank
  if (x_shape_ptr->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  const auto &x_shape = x_shape_ptr->GetShapeVector();
  auto ori_paddings_val_opt = PadV3GradFetchPaddingsValFromArg(primitive, input_args[kInputIndex1]);
  if (x_shape_ptr->IsDynamic() || !ori_paddings_val_opt.has_value()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(x_shape.size(), abstract::Shape::kShapeDimAny));
  }

  auto ori_paddings_val = ori_paddings_val_opt.value();
  auto paddings_size = ori_paddings_val.size();
  auto paddings_val = PadV3GradDealWithPaddingValue(primitive, x_shape, paddings_size, ori_paddings_val);
  auto out_shape = PadV3GradInferOutputShapeWithPaddings(primitive, x_shape, paddings_size, paddings_val);

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadV3GradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> args = {{"x", input_args[0]->GetType()}};
  auto mode = GetValue<string>(prim->GetAttr("mode"));
  if (mode == kConstant) {
    return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                     {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool},
                                                     prim->name());
  } else {
    return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                     {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
                                                     prim->name());
  }
}
}  // namespace

AbstractBasePtr PadV3GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = PadV3GradInferType(primitive, input_args);
  auto infer_shape = PadV3GradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool PadV3Grad::get_paddings_contiguous() const { return GetValue<bool>(GetAttr("paddings_contiguous")); }
std::string PadV3Grad::get_mode() const { return GetValue<string>(GetAttr("mode")); }

MIND_API_OPERATOR_NAME_IMPL(PadV3Grad, kNamePadV3Grad, BaseOperator);

// AG means auto generated
class OPS_API AGPadV3GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3GradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {kInputIndex1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PadV3Grad, prim::kPrimPadV3Grad, AGPadV3GradInfer, false);
}  // namespace ops
}  // namespace mindspore
