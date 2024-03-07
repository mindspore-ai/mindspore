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
#include "mindspore/core/symbolic_shape/operation_builder.h"
#include "mindspore/core/symbolic_shape/utils.h"

namespace mindspore {
namespace symshape {
SymbolPtr OperationBuilder::TransparentInput(const PrimitivePtr &prim, bool build_value) const {
  auto depends = symbol_builder_info_.GetDepends(prim, build_value);
  if (depends.empty()) {
    return nullptr;
  }
  auto iter1 = std::find_if(depends.begin(), depends.end(), [](DependOn d) { return d != DependOn::kNone; });
  if (iter1 == depends.end()) {
    return nullptr;
  }
  auto iter2 = std::find_if(iter1 + 1, depends.end(), [](DependOn d) { return d != DependOn::kNone; });
  if (iter2 != depends.end()) {
    return nullptr;
  }
  size_t idx = iter1 - depends.begin();
  return (*iter1 == DependOn::kShape) ? GetInputShape(idx) : GetInputValue(idx);
}

SymbolPtr OperationBuilder::BuildShape(const PrimitivePtr &prim, const AbstractBasePtrList &input_args,
                                       const AbstractBasePtr &out) {
  is_building_shape_ = true;
  prim_ = prim;
  input_args_ = &input_args;
  out_ = out;
  if (symbol_builder_info_.build_shape_func == nullptr) {
    return TransparentInput(prim, false);
  }
  return symbol_builder_info_.build_shape_func(this);
}

SymbolPtr OperationBuilder::BuildValue(const PrimitivePtr &prim, const AbstractBasePtrList &input_args,
                                       const AbstractBasePtr &out) {
  is_building_shape_ = false;
  prim_ = prim;
  input_args_ = &input_args;
  out_ = out;
  if (symbol_builder_info_.build_value_func == nullptr) {
    return TransparentInput(prim, true);
  }
  return symbol_builder_info_.build_value_func(this);
}

SymbolPtr OperationBuilder::GetShape(const AbstractBasePtr &abs) const {
  auto real_shape = abs->GetSymbolicShape();
  if (real_shape != nullptr) {
    return real_shape;
  }
  auto baseshape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(baseshape);
  real_shape = baseshape->BuildSymbolicShape();
  MS_EXCEPTION_IF_NULL(real_shape);
  abs->SetSymbolicShape(real_shape);
  return real_shape;
}

SymbolPtr OperationBuilder::GetValue(const AbstractBasePtr &abs) const {
  SymbolPtr smbl = abs->GetSymbolicValue();
  if (smbl != nullptr) {
    return smbl;
  }
  smbl = BuildSymbolicValue(abs);
  MS_EXCEPTION_IF_NULL(smbl);
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
}  // namespace symshape
}  // namespace mindspore
