/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/eliminate_func_data_type.h"
#include <vector>
#include <memory>
#include <utility>
#include "ir/anf.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
namespace {
bool FuncDataTypeExistsInAbstractTuple(const AbstractBasePtr &node_abs) {
  MS_EXCEPTION_IF_NULL(node_abs);
  auto abs_tuple = dyn_cast<abstract::AbstractTuple>(node_abs);
  MS_EXCEPTION_IF_NULL(abs_tuple);
  for (const auto &abs : abs_tuple->elements()) {
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractTuple>() && FuncDataTypeExistsInAbstractTuple(abs)) {
      return true;
    }
    auto type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() == kObjectTypeFunction) {
      return true;
    }
  }
  return false;
}

void RemoveInputFuncNodeForKernelGraph(const KernelGraphPtr &kernel_graph, const AnfNodePtr &func_input_node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(func_input_node);
  if (kernel_graph->MutableInputs() != nullptr) {
    std::vector<AnfNodePtr> original_inputs = *(kernel_graph->MutableInputs());
    kernel_graph->MutableInputs()->clear();
    std::for_each(original_inputs.begin(), original_inputs.end(),
                  [&kernel_graph, &func_input_node](const AnfNodePtr &node) {
                    MS_EXCEPTION_IF_NULL(node);
                    if (node != func_input_node) {
                      kernel_graph->MutableInputs()->emplace_back(node);
                    }
                  });
    kernel_graph->SetInputNodes();
  }
}

abstract::AbstractBasePtrList EliminateFuncDataTypeForAbstractTuple(const abstract::AbstractTuplePtr &abs_tuple) {
  MS_EXCEPTION_IF_NULL(abs_tuple);
  AbstractBasePtrList new_abs;
  const auto &elements = abs_tuple->elements();
  for (const auto &abs : elements) {
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractTuple>()) {
      if (dyn_cast<abstract::AbstractTuple>(abs)->dynamic_len()) {
        new_abs.emplace_back(abs);
        continue;
      }
      new_abs.emplace_back(std::make_shared<abstract::AbstractTuple>(
        EliminateFuncDataTypeForAbstractTuple(dyn_cast<abstract::AbstractTuple>(abs))));
      continue;
    }
    auto type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() == kObjectTypeFunction) {
      new_abs.emplace_back(std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1)));
    } else {
      new_abs.emplace_back(abs);
    }
  }
  return new_abs;
}
}  // namespace

void EliminateFuncDataType::Init() {
  constant_ = NewValueNode(MakeValue<int32_t>(1));
  constant_abs_ = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int32Imm>(1));
  constant_->set_abstract(constant_abs_);
}

const AnfNodePtr EliminateFuncDataType::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  static uint32_t parameter_already_processed_graph_id = UINT32_MAX;
  // Case 1: for parameter node which has func data type, replace it with constant.
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->graph_id() != parameter_already_processed_graph_id) {
    auto manage = kernel_graph->manager();
    MS_EXCEPTION_IF_NULL(manage);
    auto tr = manage->Transact();
    std::vector<AnfNodePtr> new_params;
    const auto &original_params = kernel_graph->parameters();
    for (const auto &param : original_params) {
      MS_EXCEPTION_IF_NULL(param);
      auto abs = param->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractTuple>() && FuncDataTypeExistsInAbstractTuple(abs)) {
        RemoveInputFuncNodeForKernelGraph(kernel_graph, param);
        (void)tr.Replace(param, constant_);
      } else if (common::AnfAlgo::GetOutputInferDataType(param, 0) == kObjectTypeFunction) {
        RemoveInputFuncNodeForKernelGraph(kernel_graph, param);
        (void)tr.Replace(param, constant_);
      } else {
        new_params.emplace_back(param);
      }
    }
    tr.Commit();
    kernel_graph->set_parameters(std::move(new_params));
    parameter_already_processed_graph_id = kernel_graph->graph_id();
  }
  // Case 2: for non-parameter node which has func data type, replace its abstract with constant.
  MS_EXCEPTION_IF_NULL(node);
  const auto &abs = node->abstract();
  if (abs != nullptr) {
    if (abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
      if (abs_tuple->dynamic_len()) {
        return nullptr;
      }
      node->set_abstract(std::make_shared<abstract::AbstractTuple>(EliminateFuncDataTypeForAbstractTuple(abs_tuple)));
    } else if (common::AnfAlgo::GetOutputInferDataType(node, 0) == kObjectTypeFunction) {
      node->set_abstract(constant_abs_);
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
