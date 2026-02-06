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
#include "abstract/symbolic_shape/utils.h"
#include <algorithm>
#include <utility>
#include <memory>
#include <map>
#include "abstract/symbolic_shape/symbol_info.h"
#include "ir/kernel_tensor_value.h"
#include "ir/dtype/tensor_type.h"
#include "utils/check_convert_utils.h"
#include "utils/value_utils.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace symshape {
namespace {
SymbolPtr GenValueByTensorShape(const ShapeVector &shape, const TypePtr &type_ptr) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (IsDynamic(shape)) {
    return ListSymbol::Make();
  }
  if (shape.size() > 1) {
    MS_LOG(INFO) << "Symbolic value only support 0-D or 1-D value, but got the shape: " << shape;
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
  MS_EXCEPTION_IF_NULL(baseshape);
  MS_EXCEPTION_IF_NULL(type_ptr);
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
    return BoolSymbol::Make(GetScalarValue<bool>(v).value());
  }
  if (type_ptr->type_id() == kObjectTypeString) {
    return StrSymbol::Make(GetScalarValue<std::string>(v).value());
  }
  if (type_ptr->type_id() == kNumberTypeInt64) {
    return IntSymbol::Make(GetScalarValue<int64_t>(v).value());
  }
  if (type_ptr->type_id() == kNumberTypeInt32) {
    return IntSymbol::Make(static_cast<int64_t>(GetScalarValue<int32_t>(v).value()));
  }
  if (type_ptr->type_id() == kNumberTypeFloat64) {
    return FloatSymbol::Make(GetScalarValue<double>(v).value());
  }
  if (type_ptr->type_id() == kNumberTypeFloat32) {
    return FloatSymbol::Make(static_cast<double>(GetScalarValue<float>(v).value()));
  }
  auto value_opt = GetArrayValue<int64_t>(v);
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
  MS_EXCEPTION_IF_NULL(v);
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
  MS_EXCEPTION_IF_NULL(abstract);
  auto value_ptr = abstract->GetValue();
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (value_ptr->isa<ValueAny>()) {
    return GenValueByShape(abstract->GetShape(), abstract->GetType());
  }
  if (value_ptr->isa<None>()) {
    return kNoneSymbol;
  }
  auto shape = abstract->GetShape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::TensorShape>() && shape->cast_ptr<abstract::TensorShape>()->shape().empty()) {
    return ConstValueToSymbol(value_ptr, true);
  }
  return ConstValueToSymbol(value_ptr, false);
}

ShapeVector ToShape(const Symbol *symbol) {
  MS_EXCEPTION_IF_NULL(symbol);
  if (!symbol->HasData()) {
    return {abstract::Shape::kShapeRankAny};
  }
  auto *list = symbol->as<ListSymbol>();
  ShapeVector shape(list->size());
  (void)std::transform(list->symbols().cbegin(), list->symbols().cend(), shape.begin(), [](const SymbolPtr &s) {
    auto int_smbl = s->as<IntSymbol>();
    if (!int_smbl->HasData()) {
      return abstract::Shape::kShapeDimAny;
    }
    return int_smbl->value();
  });
  return shape;
}

SymbolPtr IntValues2Symbol(const std::vector<int64_t> &shape, const OpPtr &op) {
  SymbolPtrList result(shape.size());
  (void)std::transform(shape.begin(), shape.end(), result.begin(), [op](int64_t s) { return IntSymbol::Make(s, op); });
  return ListSymbol::Make(std::move(result), op);
}

int64_t AsInt(const Symbol *s) {
  MS_EXCEPTION_IF_NULL(s);
  return s->as<IntSymbol>()->value();
}

std::set<int64_t> NormAxis(const ListSymbol *axis, size_t rank) {
  MS_EXCEPTION_IF_NULL(axis);
  std::set<int64_t> result;
  for (auto &item : axis->symbols()) {
    result.insert(NormAxis(AsInt(item), rank));
  }
  return result;
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

std::vector<ListSymbolPtr> BuildSymbolicShapeBySymbolInfo(const AbstractBasePtrList &args_abs,
                                                          const std::vector<SymbolInfoList> &symbol_infos) {
  auto n = std::min(symbol_infos.size(), args_abs.size());
  std::vector<ListSymbolPtr> result(n);
  std::map<int64_t, SymbolPtr> id_sym_map;
  for (size_t i = 0; i < n; i++) {
    if (args_abs[i] == nullptr) {
      continue;
    }
    result[i] = args_abs[i]->GetShape()->BuildSymbolicShape();
    if (symbol_infos[i].empty()) {
      continue;
    }
    auto shape = result[i]->symbols();
    if (symbol_infos[i].size() != shape.size()) {
      MS_LOG(WARNING) << "The symbol_infos[i].size() should be equals to shape.size(), but got "
                      << symbol_infos[i].size() << " vs " << shape.size();
      return {};
    }
    bool has_uniq_symbol = false;
    for (size_t j = 0; j < symbol_infos[i].size(); j++) {
      auto info = symbol_infos[i][j];
      if (info.id.has_value()) {
        auto iter = id_sym_map.find(info.id.value());
        if (iter != id_sym_map.end()) {
          shape[j] = iter->second;
          has_uniq_symbol = true;
          continue;
        } else {
          id_sym_map[info.id.value()] = shape[j];
        }
      }
      auto shape_item = shape[j]->as<IntSymbol>();
      MS_EXCEPTION_IF_NULL(shape_item);
      shape_item->SetDivisorRemainder(info.divisor, info.remainder);
      // the s = d * N + r, for N >= 1
      info.min = std::max(info.min, info.divisor + info.remainder);
      if (info.min <= 0) {
        info.min = 1;
      } else if (info.min > 1) {
        shape_item->SetRangeMin(info.min);
      }
      if (info.max > info.min) {
        shape_item->SetRangeMax(info.max);
      }
      MS_LOG(INFO) << "SymbolInfo for input[" << i << "].shape[" << j << "]: max=" << info.max << ", min=" << info.min
                   << ", divisor=" << info.divisor << ", remainder=" << info.remainder;
    }
    if (has_uniq_symbol) {
      result[i] = ListSymbol::Make(std::move(shape));
    }
  }
  return result;
}
}  // namespace symshape
}  // namespace mindspore
