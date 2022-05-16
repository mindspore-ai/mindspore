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

#include "common/graph_kernel/adapter/expander.h"

#include <map>
#include <set>
#include <string>
#include <memory>
#include "include/common/utils/python_adapter.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "common/graph_kernel/split_umonad.h"
#include "common/graph_kernel/substitute_dropout.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/adapter/callback_impl.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "ops/primitive_c.h"

namespace mindspore::graphkernel {
ExpanderPtr GetExpander(const AnfNodePtr &node, bool abstract) {
  auto expander =
    abstract
      ? std::make_shared<PyExpander>(std::static_pointer_cast<Callback>(std::make_shared<CallbackImplWithInferShape>()))
      : std::make_shared<PyExpander>(Callback::Instance());
  if (IsComplexOp(node)) return ComplexOpDecorator::Creator(expander);

  constexpr size_t kAssignInputIdx = 1;
  constexpr size_t kLambOptimizerInputIdx = 12;
  constexpr size_t kLambWeightInputIdx = 4;
  constexpr size_t kRandomInputIdx = 1;
  constexpr size_t kAdamInputIdx = 10;
  constexpr size_t kAdamWeightDecayInputIdx = 9;
  std::map<std::string, ExpanderCreatorFuncList> creators = {
    {prim::kPrimAssignAdd->name(), {OpUMonadExpanderDeco::GetCreator(kAssignInputIdx)}},
    {prim::kLambApplyOptimizerAssign->name(), {OpUMonadExpanderDeco::GetCreator(kLambOptimizerInputIdx)}},
    {prim::kLambApplyWeightAssign->name(), {OpUMonadExpanderDeco::GetCreator(kLambWeightInputIdx)}},
    {prim::kPrimStandardNormal->name(), {OpUMonadExpanderDeco::GetCreator(kRandomInputIdx)}},
    {prim::kPrimAdam->name(), {OpUMonadExpanderDeco::GetCreator(kAdamInputIdx)}},
    {prim::kPrimAdamWeightDecay->name(), {OpUMonadExpanderDeco::GetCreator(kAdamWeightDecayInputIdx)}},
    {prim::kPrimDropout->name(), {DropoutExpanderDeco::Creator}},
  };
  auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    return WrapExpander(expander, iter->second);
  }
  return expander;
}

FuncGraphPtr TryExpandCNode(const AnfNodePtr &node, const std::function<bool(const CNodePtr &kernel_node)> &func) {
  if (common::AnfAlgo::IsDynamicShape(node)) return nullptr;
  if (common::GetEnv("MS_DEV_EXPANDER_FALLBACK") == "off") return nullptr;
  auto expand_fg = GetCNodeFuncGraph(graphkernel::GetExpander(node)->Run(node));
  if (expand_fg != nullptr) {
    auto todos = TopoSort(expand_fg->get_return());
    for (const auto &n : todos) {
      auto cnode = n->cast<CNodePtr>();
      if (cnode == nullptr || !AnfUtils::IsRealKernel(cnode)) continue;
      auto suc = func(cnode);
      if (!suc) {
        MS_LOG(DEBUG) << "Expanding core ops [" << cnode->fullname_with_scope() << "] failed.";
        expand_fg = nullptr;
        break;
      }
    }
  }
  return expand_fg;
}

PrimitivePtr GetOpsPrim(const std::string &name) {
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  auto iter = op_primc_fns.find(name);
  if (iter == op_primc_fns.end()) return nullptr;
  return iter->second();
}

void ConvertAttrToInput(const FuncGraphPtr &graph) {
  auto todos = TopoSort(graph->get_return());
  for (const auto &node : todos) {
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto primitive = GetCNodePrimitive(node);
    if (!primitive) {
      continue;
    }
    primitive = primitive->Clone();
    std::set<std::string> attr2input_map = {prim::kPrimCast->name(), prim::kPrimReshape->name(),
                                            prim::kPrimReduceMax->name(), prim::kPrimReduceSum->name(),
                                            prim::kPrimTranspose->name()};
    if (attr2input_map.count(primitive->name())) {
      auto op = GetOpsPrim(primitive->name());
      MS_EXCEPTION_IF_NULL(op);
      auto input_names = op->GetAttr(kAttrInputNames);
      primitive->AddAttr(kAttrInputNames, input_names);
      primitive->AddAttr(kAttrOutputNames, op->GetAttr(kAttrOutputNames));
      auto cnode = dyn_cast<CNode>(node);
      AnfNodePtrList inputs = cnode->inputs();
      AnfNodePtrList new_inputs{inputs[0]};
      auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
      size_t j = 1;
      for (size_t i = 0; i < input_names_vec.size(); ++i) {
        if (primitive->HasAttr(input_names_vec[i])) {
          auto value = primitive->GetAttr(input_names_vec[i]);
          auto value_node = std::make_shared<ValueNode>(value);
          value_node->set_abstract(value->ToAbstract());
          new_inputs.push_back(value_node);
        } else {
          if (j >= inputs.size()) {
            MS_LOG(EXCEPTION) << "Index " << j << " is larger than input size [" << inputs.size() << "]";
          }
          new_inputs.push_back(inputs[j]);
          j++;
        }
      }
      new_inputs[0] = NewValueNode(primitive);
      cnode->set_inputs(new_inputs);
    }
  }
}

AnfNodePtr AttrToInputDeco::Run(const AnfNodePtr &node) {
  auto new_node = decorated_->Run(node);
  if (new_node == nullptr) return nullptr;
  auto new_cnode = dyn_cast<CNode>(new_node);
  auto expand_fg = GetCNodeFuncGraph(new_cnode);
  ConvertAttrToInput(expand_fg);
  new_cnode->set_input(0, NewValueNode(expand_fg));
  return new_cnode;
}

bool PyExpander::CreateJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  DumpOption dump_option;
  dump_option.extract_opinfo_from_anfnode = true;
  AkgKernelJsonGenerator json_generator(dump_option, cb_);
  return json_generator.CollectJson(node, kernel_json);
}

FuncGraphPtr PyExpander::ExpandToGraphByCallPyFn(const CNodePtr &node) {
  MS_LOG(DEBUG) << "CallPyFn: [" << kGetGraphKernelExpanderOpList << "].";
  auto res = python_adapter::CallPyFn(kGraphKernelModule, kGetGraphKernelExpanderOpList);
  // parse result.
  if (py::isinstance<py::none>(res)) {
    MS_LOG(ERROR) << "CallPyFn: [" << kGetGraphKernelExpanderOpList << "] failed.";
    return nullptr;
  }

  std::string expander_op_list = py::cast<std::string>(res);
  auto op_name = AnfUtils::GetCNodeName(node);
  if (expander_op_list.find(op_name) == std::string::npos) {
    MS_LOG(DEBUG) << "Do not support to expand: " << op_name;
    return nullptr;
  }

  nlohmann::json kernel_json;
  if (!CreateJsonInfo(node, &kernel_json)) {
    constexpr int recursive_level = 2;
    MS_LOG(ERROR) << "Expand json info to: " << node->DebugString(recursive_level) << " failed, ori_json:\n"
                  << kernel_json.dump();
    return nullptr;
  }
  auto node_desc_str = kernel_json.dump();
  // call graph kernel ops generator.
  MS_LOG(DEBUG) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] with input json:\n" << node_desc_str;
  auto ret = python_adapter::CallPyFn(kGraphKernelModule, kGetGraphKernelOpExpander, node_desc_str);
  // parse result.
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(ERROR) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] return invalid result, input json:\n"
                  << node_desc_str;
    return nullptr;
  }
  std::string kernel_desc_str = py::cast<std::string>(ret);
  if (kernel_desc_str.empty()) {
    return nullptr;
  }
  // decode json to func_graph.
  return JsonDescToAnf(kernel_desc_str);
}

FuncGraphPtr PyExpander::ExpandToGraph(const CNodePtr &node) {
  auto op_name = AnfUtils::GetCNodeName(node);
  // use cpp OpDesc in priority
  auto use_py = common::GetEnv("MS_DEV_PYEXPANDER");
  if (use_py.empty()) {
    if (expanders::OpDescFactory::Instance().HasOp(op_name)) {
      return DefaultExpander::ExpandToGraph(node);
    }
    // expander fallback do not use python
    if (std::dynamic_pointer_cast<CallbackImplWithInferShape>(cb_) != nullptr) return nullptr;
  }
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (pynative_mode && PyGILState_Check() == 0) {
    // Acquire Python GIL
    py::gil_scoped_acquire gil;
    if (PyGILState_Check() == 0) {
      MS_LOG(ERROR) << "Can not acquire python GIL.";
      return nullptr;
    }
    auto fg = ExpandToGraphByCallPyFn(node);
    py::gil_scoped_release rel;
    return fg;
  }
  return ExpandToGraphByCallPyFn(node);
}

AnfNodePtr ComplexOpDecorator::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  cnode->set_input(0, NewValueNode(std::make_shared<Primitive>("C" + prim->name(), prim->attrs())));
  return decorated_->Run(cnode);
}

void InlineExpandFuncGraph(const AnfNodePtr &expanding_node, const FuncGraphPtr &expanded_graph) {
  auto main_graph = expanding_node->func_graph();
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  auto cnode = expanding_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList inp(cnode->inputs().begin() + 1, cnode->inputs().end());
  auto out = InlineClone(expanded_graph, main_graph, inp, cnode->input(0)->scope());
  (void)mng->Replace(expanding_node, out);
}

bool IsComplexOp(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto input = cnode->input(i);
    TypePtr input_type = input->Type();
    if (input_type == nullptr || !input_type->isa<TensorType>()) {
      return false;
    }
    input_type = input_type->cast<TensorTypePtr>()->element();
    if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
      return true;
    }
  }
  return false;
}
}  // namespace mindspore::graphkernel
