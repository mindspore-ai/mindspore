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
#include "mindspore/ccsrc/utils/symbol_engine/symbol_engine_impl_pi.h"
#include <algorithm>
#include <ostream>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "abstract/abstract_function.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_ops.h"
#include "mindspore/ops/infer/symbol_ops_impl/switch.h"
#include "mindspore/ops/infer/symbol_ops_impl/j_op.h"
#include "utils/check_convert_utils.h"
#include "utils/anf_utils.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace symshape {
SymbolEnginePIJITPtr SymbolEnginePIJIT::Build(const FuncGraphPtr &func_graph) {
  if (func_graph->symbol_engine() != nullptr) {
    CleanSymbols(func_graph);
  }
  auto engine = std::make_shared<SymbolEnginePIJIT>(func_graph);
  func_graph->set_symbol_engine(engine);
  engine->BuildImpl();
  return engine;
}

void SymbolEnginePIJIT::BuildImpl() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Build " << ToString() << " with graph " << func_graph->ToString();
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
}

bool SymbolEnginePIJIT::Infer(const AbstractBasePtrList &inputs) {
  if (!support_infer_) {
    MS_LOG(WARNING) << "The " << ToString() << " does not support infer";
    return false;
  }
  MS_LOG(DEBUG) << "Infer " << ToString() << " with inputs: " << inputs;
  if (inputs_abs_.size() < inputs.size()) {
    MS_LOG(EXCEPTION) << "The parameter size should be equal to or larger than inputs size, but got "
                      << inputs_abs_.size() << " vs " << inputs.size();
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto shape = inputs_abs_[i]->GetSymbolicShape(); shape != nullptr) {
      auto cur_shape = inputs[i]->GetShape()->BuildSymbolicShape();
      MS_EXCEPTION_IF_NULL(cur_shape);
      MS_LOG(DEBUG) << "Update shape for input[" << i << "]: " << cur_shape->ToRawString();
      shape->Update(cur_shape);
    }
    if (auto value = inputs_abs_[i]->GetSymbolicValue(); value != nullptr && value->CanUpdate()) {
      auto cur_value = BuildSymbolicValue(inputs[i]);
      MS_EXCEPTION_IF_NULL(cur_value);
      MS_LOG(DEBUG) << "Update value for input[" << i << "]: " << cur_value->ToRawString();
      value->Update(cur_value);
    }
  }
  for (auto &op : ops_) {
    op->Run();
  }
  return true;
}

SymbolPtr SymbolEnginePIJIT::BuildCNodeSymbolicShape(OperationBuilder *builder, const PrimitivePtr &prim,
                                                     const AbstractBasePtrList &inputs,
                                                     const AbstractBasePtr &abstract) {
  auto digital_shape = abstract->GetShape();
  MS_EXCEPTION_IF_NULL(digital_shape);
  if (common::GetEnv("MS_DEV_FORCE_BUILD_SYMBOL") != "on" && !digital_shape->IsDynamic()) {
    auto static_shape = digital_shape->BuildSymbolicShape();
    MS_LOG(DEBUG) << "Node " << prim->ToString() << " is static shape: " << digital_shape->ToString();
    return static_shape;
  }
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << prim->ToString() << " does not support BuildShape, builder not found.";
    return nullptr;
  }
  return builder->BuildShape(prim, inputs, abstract);
}

SymbolPtr SymbolEnginePIJIT::BuildCNodeSymbolicValue(OperationBuilder *builder, const PrimitivePtr &prim,
                                                     const AbstractBasePtrList &inputs,
                                                     const AbstractBasePtr &abstract) {
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << prim->ToString() << " does not support BuildValue, builder not found.";
    return nullptr;
  }
  return builder->BuildValue(prim, inputs, abstract);
}

DependStatus SymbolEnginePIJIT::GetDependStatus(const AbstractBasePtrList &inputs, const PrimitivePtr &prim) {
  auto inputs_num = inputs.size();
  auto shape_depends = GetShapeDepends(prim, inputs_num);
  auto value_depends = GetValueDepends(prim, inputs_num);
  DependStatus depend_status{true, true};
  for (size_t i = 0; i < inputs_num; i++) {
    if (shape_depends.empty() || (shape_depends[i] == DependOn::kShape && inputs[i]->GetSymbolicShape() == nullptr)) {
      depend_status.shape = false;
    }
    if (value_depends.empty() || (value_depends[i] == DependOn::kValue && inputs[i]->GetValue()->isa<ValueAny>() &&
                                  inputs[i]->GetSymbolicValue() == nullptr)) {
      depend_status.value = false;
    }
  }
  MS_LOG(DEBUG) << "The depend status of " << prim->ToString() << "): shape-depend=" << depend_status.shape
                << ", value-depend=" << depend_status.value;
  return depend_status;
}

bool SymbolEnginePIJIT::CheckCondition(const AbstractBasePtrList &inputs, const BoolSymbolPtr condition) {
  if (!Infer(inputs)) {
    MS_LOG(ERROR) << "infer failed.";
    return false;
  }

  if (condition->HasData()) {
    MS_LOG(DEBUG) << "check condition is: " << condition->value();
    return condition->value();
  }
  return false;
}

AbstractBasePtrList SymbolEnginePIJIT::ExtractInputsAbstractHint(const AbstractBasePtrList &inputs) {
  AbstractBasePtrList hint_inputs;
  (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(hint_inputs),
                       [this](const AbstractBasePtr &abs) {
                         MS_EXCEPTION_IF_NULL(abs);
                         auto hint_abs = abs->Clone();
                         auto hint = this->GetHint(abs->GetSymbolicShape());
                         auto hint_sym = hint ? hint->as_sptr<ListSymbol>() : nullptr;
                         hint_abs->SetSymbolicShape(hint_sym);
                         hint_abs->SetSymbolicValue(this->GetHint(abs->GetSymbolicValue()));
                         return hint_abs;
                       });
  return hint_inputs;
}

AbstractBasePtr SymbolEnginePIJIT::EvalOnePrimSymbol(const PrimitivePtr &prim, const AbstractBasePtrList &inputs_abs,
                                                     const AbstractBasePtr &output_abs) {
  AbstractBasePtrList hint_inputs = ExtractInputsAbstractHint(inputs_abs);
  auto abstract = CloneAbstractIfSymbolExists(output_abs);
  MS_EXCEPTION_IF_NULL(abstract);
  if (HasAbstractAny(inputs_abs, abstract)) {
    MS_LOG(DEBUG) << "The input or output has AbstractAny, which is not supported by symbol engine. ";
    return abstract;
  }

  auto builder = OperationBuilderInfoRegistry::GetBuilder(prim->name(), emitter_.get());

  auto depend_status = GetDependStatus(inputs_abs, prim);
  if (depend_status.value) {
    MS_LOG(DEBUG) << "Build value for node " << prim->ToString();
    auto sym_value = BuildCNodeSymbolicValue(builder.get(), prim, inputs_abs, abstract);
    auto real_sym_value = BuildCNodeSymbolicValue(builder.get(), prim, hint_inputs, abstract);
    MS_LOG(DEBUG) << "Set value for node: " << prim->ToString() << ". symbol: " << sym_value->ToString();
    MS_LOG(DEBUG) << "Set real value for node: " << prim->ToString() << ". symbol: " << real_sym_value->ToString();
    abstract->SetSymbolicValue(sym_value);
    SetHintMap(sym_value, real_sym_value);
  }

  if (depend_status.shape) {
    MS_LOG(DEBUG) << "Build shape for node " << prim->ToString();
    auto sym_shape = BuildCNodeSymbolicShape(builder.get(), prim, inputs_abs, abstract);
    auto real_sym_shape = BuildCNodeSymbolicShape(builder.get(), prim, hint_inputs, abstract);
    MS_EXCEPTION_IF_NULL(sym_shape);
    MS_LOG(DEBUG) << "Set shape for node: " << prim->ToString() << ". symbol: " << sym_shape->ToString();
    MS_LOG(DEBUG) << "Set real shape for node: " << prim->ToString() << ". real symbol: " << real_sym_shape->ToString();
    abstract->SetSymbolicShape(sym_shape->as_sptr<ListSymbol>());
    SetHintMap(sym_shape, real_sym_shape);
  }
  return abstract;
}

void SymbolEnginePIJIT::AddInputAbs(const AbstractBasePtr &abs, const AbstractBasePtr &hint_abs) {
  if (abs->GetSymbolicShape() == nullptr) {
    abs->SetSymbolicShape(abs->GetShape()->BuildSymbolicShape());
  }
  inputs_abs_.push_back(abs);
  if (hint_abs != nullptr) {
    SetHintMap(abs->GetSymbolicShape(), hint_abs->GetShape()->BuildSymbolicShape());
  }
}

void SymbolEnginePIJIT::BuildCNodeSymbol(const CNodePtr &cnode) {
  PrimitivePtr prim;
  prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    return;
  }
  AbstractBasePtrList inputs = ExtractInputsAbstract(cnode);
  cnode->set_abstract(EvalOnePrimSymbol(prim, inputs, cnode->abstract()));
}

std::string SymbolEnginePIJIT::DumpText() const {
  std::ostringstream oss;
  oss << ToString() << " {\n";
  for (auto op : ops_) {
    oss << op->DumpText();
  }
  oss << "}\n";
  return oss.str();
}
}  // namespace symshape
}  // namespace mindspore
