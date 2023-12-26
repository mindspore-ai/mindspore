/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "mindspore/core/symbolic_shape/utils.h"
#include <algorithm>
#include <utility>
#include <memory>
#include "ir/kernel_tensor_value.h"
#include "mindspore/core/utils/check_convert_utils.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace symshape {
namespace {
SymbolPtr GenValueByTensorShape(const abstract::TensorShapePtr &shape, const TypePtr &type_ptr) {
  MS_EXCEPTION_IF_NULL(shape);
  auto &shape_vec = shape->shape();
  if (IsDynamic(shape_vec)) {
    return ListSymbol::Make();
  }
  if (shape_vec.size() > 1) {
    MS_LOG(WARNING) << "Symbolic value only support 0-D or 1-D value, but got the shape: " << shape_vec;
    return ListSymbol::Make();
  }
  if (shape_vec.size() == 0) {
    if (type_ptr->generic_type_id() == kNumberTypeBool) {
      return BoolSymbol::Make();
    }
    if (type_ptr->generic_type_id() == kNumberTypeFloat) {
      return FloatSymbol::Make();
    }
    return IntSymbol::Make();
  }
  SymbolPtrList list(LongToSize(shape_vec[0]));
  if (type_ptr->generic_type_id() == kNumberTypeBool) {
    std::generate(list.begin(), list.end(), []() { return BoolSymbol::Make(); });
  } else if (type_ptr->generic_type_id() == kNumberTypeFloat) {
    std::generate(list.begin(), list.end(), []() { return FloatSymbol::Make(); });
  } else {
    std::generate(list.begin(), list.end(), []() { return IntSymbol::Make(); });
  }
  return ListSymbol::Make(std::move(list));
}

SymbolPtr GenValueByShape(const BaseShapePtr &baseshape, const TypePtr &type_ptr) {
  if (baseshape->isa<abstract::NoShape>()) {
    return IntSymbol::Make();
  }
  if (baseshape->isa<abstract::TensorShape>()) {
    return GenValueByTensorShape(baseshape->cast<abstract::TensorShapePtr>(), type_ptr);
  }
  if (baseshape->isa<abstract::DynamicSequenceShape>()) {
    return ListSymbol::Make();
  }
  auto seq_shape = baseshape->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(seq_shape);
  SymbolPtrList result(seq_shape->size());
  auto seq_type = type_ptr->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(seq_type);
  if (seq_shape->size() != seq_type->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size of seq_shape and seq_type should equal, but got " << seq_shape->size()
                               << " vs " << seq_type->size();
  }
  (void)std::transform(
    seq_shape->shape().begin(), seq_shape->shape().end(), seq_type->elements().begin(), result.begin(),
    [](const BaseShapePtr &shape_elm, const TypePtr &type_elm) { return GenValueByShape(shape_elm, type_elm); });
  return ListSymbol::Make(std::move(result));
}
}  // namespace

SymbolPtr ConstValueToSymbol(const ValuePtr &v) {
  if (v->isa<ValueSequence>()) {
    auto seq = v->cast_ptr<ValueSequence>();
    MS_EXCEPTION_IF_NULL(seq);
    SymbolPtrList result(seq->size());
    (void)std::transform(seq->value().begin(), seq->value().end(), result.begin(),
                         [](const ValuePtr &v) { return ConstValueToSymbol(v); });
    return ListSymbol::Make(std::move(result));
  }
  if (v->isa<tensor::Tensor>()) {
    auto tensor_value = CheckAndConvertUtils::CheckTensorIntValue(v->ToString(), v, "ConstSymbolicValue");
    auto tensor = v->cast_ptr<tensor::Tensor>();
    return tensor->shape().empty() ? IntSymbol::Make(tensor_value[0]) : IntValues2Symbol(tensor_value);
  }
  if (v->isa<IntegerImm>()) {
    return IntSymbol::Make(v->isa<Int64Imm>() ? GetValue<int64_t>(v) : static_cast<int64_t>(GetValue<int32_t>(v)));
  }
  if (v->isa<BoolImm>()) {
    return BoolSymbol::Make(GetValue<bool>(v));
  }
  if (v->isa<FloatImm>()) {
    return FloatSymbol::Make(v->isa<FP64Imm>() ? GetValue<double>(v) : static_cast<double>(GetValue<float>(v)));
  }
  if (v->isa<StringImm>()) {
    return StrSymbol::Make(GetValue<std::string>(v));
  }
  if (v->isa<KernelTensorValue>()) {
    auto type_ptr = v->type();
    if (type_ptr == nullptr) {
      MS_LOG(WARNING) << "type of KernelTensorPtr is null! trying getting Tuple Int";
      auto value = CheckAndConvertUtils::CheckTupleInt(v->ToString(), v, "ConstSymbolicValue");
      return IntValues2Symbol(value);
    }
    if (type_ptr->type_id() == kNumberTypeBool) {
      return BoolSymbol::Make(ops::GetScalarValue<bool>(v).value());
    }
    if (type_ptr->type_id() == kObjectTypeString) {
      return StrSymbol::Make(ops::GetScalarValue<std::string>(v).value());
    }
    auto value = CheckAndConvertUtils::CheckTensorIntValue(v->ToString(), v, "ConstSymbolicValue", type_ptr);
    return IntValues2Symbol(value);
  }
  MS_LOG(EXCEPTION)
    << "Value should be one of {ValueSequence, Tensor, IntegerImm, BoolImm, FloatImm, StringImm}, but got "
    << v->ToString();
  return nullptr;
}

SymbolPtr BuildSymbolicValue(const AbstractBasePtr &abstract) {
  auto value_ptr = abstract->GetValue();
  if (value_ptr->isa<ValueAny>()) {
    return GenValueByShape(abstract->GetShape(), abstract->GetType());
  }
  return ConstValueToSymbol(value_ptr);
}

ShapeVector ToShape(const Symbol *symbol) {
  if (!symbol->HasData()) {
    return {abstract::Shape::kShapeRankAny};
  }
  auto *list = symbol->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(list);
  ShapeVector shape(list->size());
  (void)std::transform(list->symbols().cbegin(), list->symbols().cend(), shape.begin(), [](const SymbolPtr &s) {
    auto int_smbl = s->as<IntSymbol>();
    MS_EXCEPTION_IF_NULL(int_smbl);
    if (!int_smbl->HasData()) {
      return abstract::Shape::kShapeDimAny;
    }
    return int_smbl->value();
  });
  return shape;
}

ValuePtr SymbolToValue(const Symbol *symbol) {
  if (!symbol->HasData()) {
    return kValueAny;
  }
  if (symbol->is<IntSymbol>()) {
    return MakeValue<int64_t>(symbol->as<IntSymbol>()->value());
  }
  if (symbol->is<BoolSymbol>()) {
    return MakeValue<bool>(symbol->as<BoolSymbol>()->value());
  }
  if (symbol->is<FloatSymbol>()) {
    return MakeValue<double>(symbol->as<FloatSymbol>()->value());
  }
  if (symbol->is<StrSymbol>()) {
    return MakeValue<std::string>(symbol->as<StrSymbol>()->value());
  }
  if (symbol->is<ListSymbol>()) {
    auto list_shape = symbol->as<ListSymbol>();
    if (!list_shape->AllHaveData()) {
      return kValueAny;
    }
    ValuePtrList res;
    res.reserve(list_shape->size());
    (void)std::transform(list_shape->symbols().cbegin(), list_shape->symbols().cend(), std::back_inserter(res),
                         [](const SymbolPtr &s) { return SymbolToValue(s.get()); });
    return std::make_shared<ValueTuple>(std::move(res));
  }
  return kValueAny;
}

SymbolPtr ShapeVector2Symbol(const ShapeVector &shape, const OpPtr &op) {
  if (IsDynamicRank(shape)) {
    return ListSymbol::Make(op);
  }
  SymbolPtrList result(shape.size());
  (void)std::transform(shape.begin(), shape.end(), result.begin(), [op](int64_t s) {
    if (s == abstract::Shape::kShapeDimAny) {
      return IntSymbol::Make(op);
    } else {
      return IntSymbol::Make(s, op);
    }
  });
  return ListSymbol::Make(std::move(result), op);
}

SymbolPtr IntValues2Symbol(const std::vector<int64_t> &shape, const OpPtr &op) {
  SymbolPtrList result(shape.size());
  (void)std::transform(shape.begin(), shape.end(), result.begin(), [op](int64_t s) { return IntSymbol::Make(s, op); });
  return ListSymbol::Make(std::move(result), op);
}

std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post, bool raw_str) {
  std::ostringstream oss;
  oss << pre;
  bool first = true;
  for (auto &s : slist) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << (raw_str ? s->ToRawString() : s->ToString());
  }
  oss << post;
  return oss.str();
}
}  // namespace symshape
}  // namespace mindspore
