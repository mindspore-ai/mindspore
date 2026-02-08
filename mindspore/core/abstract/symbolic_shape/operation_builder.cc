/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "abstract/symbolic_shape/operation_builder.h"
#include <functional>
#include "abstract/symbolic_shape/utils.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace symshape {
SymbolPtr OperationBuilder::BuildShape(const PrimitivePtr &prim, const AbstractBasePtrList &input_args,
                                       const AbstractBasePtr &out) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(out);
  is_building_shape_ = true;
  prim_ = prim;
  input_args_ = &input_args;
  out_ = out;
  if (symbol_builder_info_.build_shape_func == nullptr) {
    return nullptr;
  }
  return symbol_builder_info_.build_shape_func(this);
}

SymbolPtr OperationBuilder::BuildValue(const PrimitivePtr &prim, const AbstractBasePtrList &input_args,
                                       const AbstractBasePtr &out) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(out);
  is_building_shape_ = false;
  prim_ = prim;
  input_args_ = &input_args;
  out_ = out;
  if (symbol_builder_info_.build_value_func == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(out);
  auto sym = symbol_builder_info_.build_value_func(this);
  if (sym != nullptr && out_abstract() != nullptr && !CheckOutputValue(sym)) {
    MS_LOG(INFO) << "The dtype of output symbol does not match the abstract, " << sym->ToString() << " vs "
                 << out->ToString();
    return nullptr;
  }
  return sym;
}

SymbolPtr OperationBuilder::GetShape(const AbstractBasePtr &abs) const {
  MS_EXCEPTION_IF_NULL(abs);
  auto real_shape = abs->GetSymbolicShape();
  if (real_shape != nullptr) {
    return real_shape;
  }
  auto baseshape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(baseshape);
  real_shape = baseshape->BuildSymbolicShape();
  MS_EXCEPTION_IF_NULL(real_shape);
  MS_LOG(DEBUG) << "The input symbolic shape is empty, create new symbol: " << real_shape->ToString();
  abs->SetSymbolicShape(real_shape);
  return real_shape;
}

SymbolPtr OperationBuilder::GetValue(const AbstractBasePtr &abs) const {
  MS_EXCEPTION_IF_NULL(abs);
  SymbolPtr smbl = abs->GetSymbolicValue();
  if (smbl != nullptr) {
    return smbl;
  }
  smbl = BuildSymbolicValue(abs);
  MS_EXCEPTION_IF_NULL(smbl);
  MS_LOG(DEBUG) << "The input symbolic value is empty, create new symbol: " << smbl->ToString();
  abs->SetSymbolicValue(smbl);
  return smbl;
}

SymbolPtr OperationBuilder::GetAttr(const std::string &attr_name) const {
  auto attr = prim_->GetAttr(attr_name);
  if (attr == nullptr) {
    return nullptr;
  }
  return ConstValueToSymbol(attr);
}

SymbolPtr OperationBuilder::GetInputOrAttr(size_t index, const std::string &attr_name) const {
  if (input_args_->size() > index) {
    return GetInputValue(index);
  }
  return GetAttr(attr_name);
}

SymbolPtrList OperationBuilder::GetSymbolsOfDepend() const {
  bool build_value = !this->is_building_shape();
  auto depends = symbol_builder_info_.GetDepends(this->prim(), this->input_num(), build_value);
  if (depends.empty()) {
    MS_LOG(WARNING) << "For " << this->prim()->name() << ", the depends list is empty.";
    return {};
  }
  if (this->input_num() < depends.size()) {
    MS_LOG(WARNING) << "For " << this->prim()->name() << ", the input args num is less than the depends size. "
                    << this->input_num() << " vs " << depends.size();
    return {};
  }
  SymbolPtrList symbols;
  symbols.reserve(depends.size());
  for (size_t i = 0; i < depends.size(); i++) {
    if (depends[i] == DependOn::kShape) {
      (void)symbols.emplace_back(this->GetInputShape(i));
    } else if (depends[i] == DependOn::kValue) {
      (void)symbols.emplace_back(this->GetInputValue(i));
    }
  }
  return symbols;
}

SymbolPtr OperationBuilder::Emit(const OpPtr &op) const {
  op->SetOutAbstract(this->out_abstract());
  auto ret = emitter_->Emit(op);
  op->SetOutAbstract(nullptr);
  return ret;
}

bool OperationBuilder::CheckOutputValue(const SymbolPtr &v) const {
  auto type = this->out_abstract()->GetType();
  auto type_id = type->generic_type_id();
  if (type->isa<TensorType>()) {
    type_id = type->cast<TensorTypePtr>()->element()->generic_type_id();
  }
  std::function<bool(const SymbolPtr &)> check;
  check = [&check, type_id](const SymbolPtr &s) -> bool {
    if (s->is<ListSymbol>()) {
      auto list = s->as<ListSymbol>();
      return std::all_of(list->symbols().begin(), list->symbols().end(), check);
    }
    switch (type_id) {
      case kNumberTypeInt:
      case kNumberTypeUInt:
        return s->is<IntSymbol>();
      case kNumberTypeFloat:
        return s->is<FloatSymbol>();
      case kNumberTypeBool:
        return s->is<BoolSymbol>();
      default:
        break;
    }
    return true;
  };
  return check(v);
}

SymbolPtr TransparentInput(OperationBuilder *b) {
  auto symbols = b->GetSymbolsOfDepend();
  return symbols.size() == 1 ? symbols[0] : ListSymbol::Make(std::move(symbols));
}

const OperationBuilderInfo *OperationBuilderInfoRegistry::GetBuildInfo(const std::string &name) {
  const auto &builders = OperationBuilderInfoRegistry::Instance().builders_;
  auto iter = builders.find(name);
  return (iter == builders.end() ? nullptr : &(iter->second));
}

OperationBuilderPtr OperationBuilderInfoRegistry::GetBuilder(const std::string &name, OperationEmitter *e) {
  auto *build_info = GetBuildInfo(name);
  if (build_info == nullptr) {
    return nullptr;
  }
  return std::make_unique<OperationBuilder>(e, *build_info);
}

std::vector<DependOn> GetShapeDepends(const PrimitivePtr &prim, size_t input_num) {
  MS_EXCEPTION_IF_NULL(prim);
  auto build_info = OperationBuilderInfoRegistry::GetBuildInfo(prim->name());
  if (build_info == nullptr) {
    return std::vector<DependOn>();
  }
  auto ret = build_info->GetDepends(prim, input_num, false);
  if (!ret.empty()) {
    ret.resize(input_num, DependOn::kNone);
  }
  return ret;
}

std::vector<DependOn> GetValueDepends(const PrimitivePtr &prim, size_t input_num) {
  MS_EXCEPTION_IF_NULL(prim);
  auto build_info = OperationBuilderInfoRegistry::GetBuildInfo(prim->name());
  if (build_info == nullptr) {
    return std::vector<DependOn>();
  }
  auto ret = build_info->GetDepends(prim, input_num, true);
  if (!ret.empty()) {
    ret.resize(input_num, DependOn::kNone);
  }
  return ret;
}
}  // namespace symshape
}  // namespace mindspore
