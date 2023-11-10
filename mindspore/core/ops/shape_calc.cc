/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "ops/shape_calc.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "ir/kernel_tensor_value.h"
#include "ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/hash_set.h"
#include "utils/log_adapter.h"
#include "utils/anf_utils.h"

namespace mindspore::ops {
namespace {
ShapeVector GetShapeFromScalarOrTensor(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::TensorShape>()) {
    return base_shape->GetShapeVector();
  } else if (!base_shape->isa<abstract::NoShape>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[ShapeCalc], only support tuple of scalar or tensor now, but got "
                            << base_shape;
  }

  return ShapeVector{};
}

bool TryGetValueArg(const AbstractBasePtr &abs, ShapeArray *args, std::vector<std::vector<size_t>> *pos_idx) {
  size_t offset_base = args->size();
  pos_idx->push_back(std::vector<size_t>{offset_base});
  auto value_ptr = abs->GetValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!ops::IsValueKnown(value_ptr)) {
    args->push_back(ShapeVector{});
    return false;
  }
  if (value_ptr->isa<Int64Imm>()) {
    auto scalar_optional = ops::GetScalarValue<int64_t>(value_ptr);
    if (scalar_optional.has_value()) {
      args->push_back(ShapeVector{scalar_optional.value()});
      return true;
    }
  } else if (value_ptr->isa<tensor::Tensor>() || value_ptr->isa<KernelTensorValue>() ||
             value_ptr->isa<ValueSequeue>()) {
    auto shape_value_optional = ops::GetArrayValue<int64_t>(abs);
    if (shape_value_optional.has_value()) {
      auto shape_array_value = shape_value_optional.value();
      if (!shape_array_value.HasUnknownValue()) {
        args->push_back(shape_array_value.ToVector());
        return true;
      }
    }
  } else {
    MS_EXCEPTION(TypeError) << "For ShapeCalc, the shape input type must be Tensor/Scalar/Tuple/List, but got "
                            << value_ptr->ToString() << ".";
  }

  return false;
}
}  // namespace

bool TryGetShapeArg(const AbstractBasePtr &abs, ShapeArray *args, std::vector<std::vector<size_t>> *pos_idx) {
  MS_EXCEPTION_IF_NULL(args);
  MS_EXCEPTION_IF_NULL(pos_idx);

  size_t offset_base = args->size();
  std::vector<size_t> pos;
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::NoShape>() || base_shape->isa<abstract::TensorShape>()) {
    args->push_back(GetShapeFromScalarOrTensor(base_shape));
    pos.push_back(offset_base);
  } else if (base_shape->isa<abstract::SequenceShape>()) {
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    for (size_t i = 0; i < sequence_shape->size(); ++i) {
      args->push_back(GetShapeFromScalarOrTensor((*sequence_shape)[i]));
      pos.push_back(offset_base + i);
    }
  } else {
    if (base_shape->isa<abstract::DynamicSequenceShape>()) {
      auto dynamic_sequence = base_shape->cast<abstract::DynamicSequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(dynamic_sequence);
      auto element_base_shape = dynamic_sequence->element_shape();
      args->push_back(GetShapeFromScalarOrTensor(element_base_shape));
      pos.push_back(offset_base);
    }
    pos_idx->push_back(pos);
    return false;
  }

  pos_idx->push_back(pos);
  return true;
}

ShapeCalcBaseFunctorPtr ShapeCalc::get_functor() const {
  auto attr = api::ToRef<mindspore::Primitive>(impl_).GetAttr(kAttrFunctor);
  MS_EXCEPTION_IF_NULL(attr);
  return attr->cast<ShapeCalcBaseFunctorPtr>();
}

std::vector<bool> ShapeCalc::get_value_depend() const { return GetValue<std::vector<bool>>(GetAttr(kAttrValueDepend)); }
ShapeArray ShapeCalc::get_calc_result() const { return GetValue<ShapeArray>(GetAttr(kAttrCalcResult)); }

class MIND_API ShapeCalcInfer : public abstract::OpInferBase {
 public:
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto value_depend = GetValue<std::vector<bool>>(primitive->GetAttr(kAttrValueDepend));
    ShapeArray args;
    HashSet<size_t> unknown_inputs;
    bool is_any_dynamic_shape = false;
    std::vector<std::vector<size_t>> pos_idx;
    pos_idx.reserve(input_args.size());
    (void)primitive->AddAttr(kInputRealTuple, MakeValue(true));
    for (size_t i = 0; i < input_args.size(); ++i) {
      const auto &abs = input_args[i];
      MS_EXCEPTION_IF_NULL(abs);
      if (!value_depend[i]) {
        // If it is not value depend, use shape as arg.
        size_t offset_base = args.size();
        if (!TryGetShapeArg(abs, &args, &pos_idx)) {
          (void)unknown_inputs.insert(i);
        } else {
          auto is_new_dynamic = std::any_of(args.begin() + offset_base, args.end(), IsDynamic);
          is_any_dynamic_shape = is_any_dynamic_shape || is_new_dynamic;
        }
      } else {
        // Value depended, try to get value from input abstract.
        if (!TryGetValueArg(abs, &args, &pos_idx)) {
          (void)unknown_inputs.insert(i);
        }
      }
    }

    auto functor_attr = primitive->GetAttr(kAttrFunctor);
    MS_EXCEPTION_IF_NULL(functor_attr);
    auto functor = functor_attr->cast<ShapeCalcBaseFunctorPtr>();
    MS_EXCEPTION_IF_NULL(functor);

    ShapeVector out;
    bool is_dynamic_sequence = false;
    if (!unknown_inputs.empty() || is_any_dynamic_shape) {
      auto infer_res = functor->Infer(args, unknown_inputs, pos_idx);
      out = infer_res.first;
      is_dynamic_sequence = infer_res.second;
    } else {
      auto ans = functor->Calc(args, pos_idx);
      primitive->set_attr(kAttrCalcResult, MakeValue(ans));
      out.reserve(ans.size());
      std::transform(ans.cbegin(), ans.cend(), std::back_inserter(out),
                     [](const ShapeVector &shape) { return SizeToLong(shape.size()); });
    }
    if (!is_dynamic_sequence && out.size() == 1) {
      // single output does not use AbstractTuple to avoid TupleGetItem
      return std::make_shared<abstract::AbstractTensor>(kInt64, out);
    }

    // multiple outputs
    if (!is_dynamic_sequence && primitive->HasAttr(kOutputRealTuple) && !out.empty()) {
      auto first_len = out[0];
      if (std::any_of(out.begin() + 1, out.end(), [first_len](int64_t len) { return first_len != len; })) {
        MS_LOG(EXCEPTION) << "For 'ShapeCalc', each output should have same size in dynamic length case.";
      }
    }

    AbstractBasePtrList abs_list;
    abs_list.reserve(out.size());
    (void)std::transform(out.begin(), out.end(), std::back_inserter(abs_list),
                         [](int64_t s) { return std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector{s}); });
    auto output_abstract = std::make_shared<abstract::AbstractTuple>(abs_list);
    if (is_dynamic_sequence) {
      (void)primitive->AddAttr(kOutputRealTuple, MakeValue(true));
      output_abstract->CheckAndConvertToDynamicLenSequence();
    }
    return output_abstract;
  }

  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InferShapeAndType(nullptr, primitive, input_args)->GetShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InferShapeAndType(nullptr, primitive, input_args)->GetType();
  }
};

MIND_API_OPERATOR_IMPL(ShapeCalc, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ShapeCalc, prim::kPrimShapeCalc, ShapeCalcInfer, false);
}  // namespace mindspore::ops
