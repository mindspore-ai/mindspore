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
#include "mindspore/core/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace symshape {
namespace {
SymbolPtr GenValueByTensorShape(const ShapeVector &shape, const TypePtr &type_ptr) {
  if (IsDynamic(shape)) {
    return ListSymbol::Make();
  }
  if (shape.size() > 1) {
    MS_LOG(WARNING) << "Symbolic value only support 0-D or 1-D value, but got the shape: " << shape;
    return ListSymbol::Make();
  }
  if (shape.size() == 0) {
    if (type_ptr->generic_type_id() == kNumberTypeBool) {
      return BoolSymbol::Make();
    }
    if (type_ptr->generic_type_id() == kNumberTypeFloat) {
      return FloatSymbol::Make();
    }
    return IntSymbol::Make();
  }
  SymbolPtrList list(LongToSize(shape[0]));
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
    return GenValueByTensorShape({}, type_ptr);
  }
  if (baseshape->isa<abstract::TensorShape>()) {
    auto tensor_type = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    return GenValueByTensorShape(baseshape->cast<abstract::TensorShapePtr>()->shape(), tensor_type->element());
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

SymbolPtr KernelTensorValueToSymbol(const ValuePtr &v, bool to_scalar) {
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
  if (type_ptr->type_id() == kNumberTypeInt64) {
    return IntSymbol::Make(ops::GetScalarValue<int64_t>(v).value());
  }
  if (type_ptr->type_id() == kNumberTypeInt32) {
    return IntSymbol::Make(static_cast<int64_t>(ops::GetScalarValue<int32_t>(v).value()));
  }
  auto value_opt = ops::GetArrayValue<int64_t>(v);
  if (value_opt.has_value()) {
    auto vec = value_opt.value().ToVector();
    if (to_scalar && !vec.empty()) {
      return IntSymbol::Make(vec[0]);
    }
    return IntValues2Symbol(vec);
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Unsupported KernelTensorValue to Symbol: " << type_ptr->ToString();
}

SymbolPtr ConstValueToSymbol(const ValuePtr &v, bool to_scalar) {
  if (v->isa<KernelTensorValue>()) {
    return KernelTensorValueToSymbol(v, to_scalar);
  }
  if (v->isa<ValueSequence>()) {
    auto seq = v->cast_ptr<ValueSequence>();
    MS_EXCEPTION_IF_NULL(seq);
    SymbolPtrList result(seq->size());
    (void)std::transform(seq->value().begin(), seq->value().end(), result.begin(),
                         [to_scalar](const ValuePtr &v) { return ConstValueToSymbol(v, to_scalar); });
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
  auto shape = abstract->GetShape();
  if (shape->isa<abstract::TensorShape>() && shape->cast_ptr<abstract::TensorShape>()->shape().empty()) {
    return ConstValueToSymbol(value_ptr, true);
  }
  return ConstValueToSymbol(value_ptr, false);
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

int64_t AsInt(const Symbol *s) { return s->as<IntSymbol>()->value(); }

std::set<int64_t> NormAxis(const ListSymbol *axis, size_t rank) {
  std::set<int64_t> result;
  for (auto &item : axis->symbols()) {
    result.insert(NormAxis(AsInt(item), rank));
  }
  return result;
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

BaseShapePtr QueryShape(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto symbolic_shape = abs->GetSymbolicShape();
  if (symbolic_shape == nullptr) {
    return nullptr;
  }
  auto digital_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(digital_shape);
  if (!symbolic_shape->HasData()) {
    return digital_shape;
  }
  if (digital_shape->isa<abstract::NoShape>()) {
    return digital_shape;
  }
  if (digital_shape->isa<abstract::TensorShape>()) {
    return std::make_shared<abstract::TensorShape>(ToShape(symbolic_shape));
  }
  abstract::BaseShapePtrList shape_arr;
  shape_arr.reserve(symbolic_shape->size());
  (void)std::transform(
    symbolic_shape->symbols().begin(), symbolic_shape->symbols().end(), std::back_inserter(shape_arr),
    [](const SymbolPtr &s) { return std::make_shared<abstract::TensorShape>(ToShape(s->as<ListSymbol>())); });
  return std::make_shared<abstract::TupleShape>(std::move(shape_arr));
}

ValuePtr QueryValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto symbolic_value = abs->GetSymbolicValue();
  if (symbolic_value == nullptr) {
    auto value = abs->GetValue();
    return value != nullptr ? value : kValueAny;
  }
  return symbolic_value->ToValue();
}
}  // namespace symshape
}  // namespace mindspore
