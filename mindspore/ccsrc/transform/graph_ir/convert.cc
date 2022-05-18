/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "include/transform/graph_ir/convert.h"

#include <cinttypes>
#include <algorithm>
#include <queue>
#include <stack>
#include "include/common/utils/utils.h"

#include "mindspore/core/ops/core_ops.h"
#include "frontend/operator/ops.h"
#include "utils/anf_utils.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "utils/symbolic.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "include/transform/graph_ir/op_adapter_map.h"
#include "ops/state_ops.h"
#include "ops/array_ops.h"
#include "ops/elewise_calculation_ops.h"
#include "ops/math_ops.h"
#ifdef ENABLE_D
#include "ops/save_ops.h"
#endif
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_desc.h"

namespace mindspore {
namespace transform {
using ge::Operator;
using mindspore::kAnyValue;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using Variable = ge::op::Variable;
using Constant = ge::op::Constant;
using Assign = ge::op::Assign;
using Data = ge::op::Data;
using std::endl;

constexpr size_t kInputOffset = 2;
constexpr size_t kSwitchInputSize = 4;
constexpr size_t kSwitchBodyIndex = 2;
constexpr size_t kSwitchAfterIndex = 3;
constexpr size_t kAfterIndexInCache = 2;

namespace {
std::vector<AnfNodePtr> GetOrderedCNodes(const FuncGraphPtr fg, const AnfNodePtr node = nullptr) {
  MS_EXCEPTION_IF_NULL(fg);
  auto BelongSameGraph = std::bind(IncludeBelongGraph, fg, std::placeholders::_1);
  auto succ_include_fv = [&fg](const AnfNodePtr &node) -> std::vector<AnfNodePtr> {
    std::vector<AnfNodePtr> vecs;
    if (node == nullptr) {
      return vecs;
    }
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto &inputs = cnode->inputs();
      // Check if free variables used.
      for (const auto &input : inputs) {
        auto input_fg = GetValueNode<FuncGraphPtr>(input);
        if (input_fg) {
          for (auto &fv : input_fg->free_variables_nodes()) {
            if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
              vecs.push_back(fv);
            }
          }
        }
      }
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    return vecs;
  };

  return (node == nullptr) ? TopoSort(fg->get_return(), succ_include_fv, BelongSameGraph)
                           : TopoSort(node, succ_include_fv, BelongSameGraph);
}
}  // namespace

// ---------------implement of DfGraphConvertor-------------
bool IsCaseNode(const CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->inputs().empty() && node->input(0)->isa<CNode>() &&
      GetCNodeFuncName(node->input(0)->cast<CNodePtr>()) == "switch_layer") {
    return true;
  }
  return false;
}

bool IsPartialCNode(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (GetCNodeFuncName(cnode) == prim::kPrimPartial->name()) {
    return true;
  }
  return false;
}

bool IsPartialSuccNode(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!cnode->inputs().empty()) {
    for (size_t i = 0; i < cnode->inputs().size(); i++) {
      if (IsPartialCNode(cnode->input(i))) {
        return true;
      }
    }
  }
  return false;
}

bool IsWhileNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  if (!IsPartialSuccNode(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!IsPartialCNode(cnode->input(0))) {
    return false;
  }
  auto partial_node = cnode->input(0);
  MS_EXCEPTION_IF_NULL(partial_node);

  auto c_partial_node = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_partial_node);

  auto graph_node_input = c_partial_node->input(1);
  MS_EXCEPTION_IF_NULL(graph_node_input);
  auto graph_node = graph_node_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  auto graph_node_value = graph_node->value();
  MS_EXCEPTION_IF_NULL(graph_node_value);
  auto cond_graph = graph_node_value->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(cond_graph);
  if (!cond_graph->recursive()) {
    return false;
  }
  const auto &cond_set = cond_graph->nodes();
  for (auto beg = cond_set.begin(); beg != cond_set.end(); beg++) {
    if (!((*beg)->isa<CNode>())) {
      continue;
    }
    auto c_beg = (*beg)->cast<CNodePtr>();
    if (IsPartialSuccNode(c_beg) && c_beg->inputs().size() == kSwitchInputSize &&
        IsPartialCNode(c_beg->input(kSwitchBodyIndex)) && IsPartialCNode(c_beg->input(kSwitchAfterIndex)) &&
        GetCNodeFuncName(c_beg) == prim::kPrimSwitch->name()) {
      auto func_graph = node->func_graph();
      MS_LOG(DEBUG) << "there is while node: " << node->ToString() << " in graph: " << func_graph->ToString();
      return true;
    }
  }
  return false;
}

std::string GetCNodeTargetFuncName(const CNodePtr cnode) {
  if (IsCaseNode(cnode)) {
    return string(kNameCase);
  }
  if (IsWhileNode(cnode)) {
    return string(kNameWhile);
  }
  auto name = GetCNodeFuncName(cnode);
  if (name == "switch_layer") {
    name = "";
  }
  return name;
}

bool IsDynamicShapeNode(const AnfNodePtr node) {
  auto shape = node->Shape();
  if (shape == nullptr) {
    return false;
  }
  if (!shape->isa<abstract::Shape>()) {  // do not accept tuple shape as call node input
    return false;
  }
  if (AnfUtils::IsShapeDynamic(shape->cast<abstract::ShapePtr>())) {
    return true;
  }
  return false;
}

OpAdapterPtr DfGraphConvertor::FindAdapter(const AnfNodePtr node, bool train) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();

    std::string name = kNameCustomOp;
    if (!IsCustomCNode(cnode)) {
      name = GetCNodeTargetFuncName(cnode);
    }

    auto it_adpt = OpAdapterMap::get().find(name);
    if (it_adpt != OpAdapterMap::get().end()) {
      return it_adpt->second->Get(train);
    }
    MS_LOG(EXCEPTION) << "Can't find OpAdapter for " << name;
  }

  if (node->isa<ValueNode>()) {
    return OpAdapterMap::get()[kNameConst]->Get(train);
  }
  if (node->isa<Parameter>()) {
    return OpAdapterMap::get()[kNameParam]->Get(train);
  }
  return OpAdapterPtr(nullptr);
}

void DfGraphConvertor::InitLoopVar(std::vector<ge::Operator> *init_input) {
  MS_EXCEPTION_IF_NULL(init_input);
  if (this->training_) {
    GeTensorDesc desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
    auto var_iter_num = std::make_shared<Variable>("npu_runconfig/iterations_per_loop");
    auto var_loop_cond = std::make_shared<Variable>("npu_runconfig/loop_cond");
    auto var_one = std::make_shared<Variable>("npu_runconfig/one");
    auto var_zero = std::make_shared<Variable>("npu_runconfig/zero");
    (void)var_iter_num->update_output_desc_y(desc);
    (void)var_loop_cond->update_output_desc_y(desc);
    (void)var_one->update_output_desc_y(desc);
    (void)var_zero->update_output_desc_y(desc);
    vars_["npu_runconfig/iterations_per_loop"] = var_iter_num;
    vars_["npu_runconfig/loop_cond"] = var_loop_cond;
    vars_["npu_runconfig/one"] = var_one;
    vars_["npu_runconfig/zero"] = var_zero;

    int64_t value = 0;
    auto const_iter_num = std::make_shared<Constant>("const/npu_runconfig/iterations_per_loop");
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
      value = ConfigManager::GetInstance().iter_num();
    } else {
      MS_LOG(INFO) << "Run with normal(non-sink) mode, the iterator number will always be 1";
      ConfigManager::GetInstance().ResetIterNum();
    }
    value -= 1;  // iteration start from 0, the max iteration number for n loop should be n-1
    (void)const_iter_num->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_loop_cond = std::make_shared<Constant>("const/npu_runconfig/loop_cond");
    value = 0;
    (void)const_loop_cond->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_one = std::make_shared<Constant>("const/npu_runconfig/one");
    value = 1;
    (void)const_one->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    auto const_zero = std::make_shared<Constant>("const/npu_runconfig/zero");
    value = 0;
    (void)const_zero->set_attr_value(GeTensor(desc, reinterpret_cast<uint8_t *>(&value), sizeof(int64_t)));

    (void)const_iter_num->update_output_desc_y(desc);
    (void)const_loop_cond->update_output_desc_y(desc);
    (void)const_one->update_output_desc_y(desc);
    (void)const_zero->update_output_desc_y(desc);

    auto assign_iter_num = std::make_shared<Assign>("assign/npu_runconfig/iterations_per_loop");
    (void)assign_iter_num->set_input_ref(*var_iter_num).set_input_value(*const_iter_num);
    auto assign_loop_cond = std::make_shared<Assign>("assign/npu_runconfig/loop_cond");
    (void)assign_loop_cond->set_input_ref(*var_loop_cond).set_input_value(*const_loop_cond);
    auto assign_one = std::make_shared<Assign>("assign/npu_runconfig/one");
    (void)assign_one->set_input_ref(*var_one).set_input_value(*const_one);
    auto assign_zero = std::make_shared<Assign>("assign/npu_runconfig/zero");
    (void)assign_zero->set_input_ref(*var_zero).set_input_value(*const_zero);

    init_input->push_back(*var_iter_num);
    init_input->push_back(*var_loop_cond);
    init_input->push_back(*var_one);
    init_input->push_back(*var_zero);
    init_ops_.push_back(var_iter_num);
    init_ops_.push_back(var_loop_cond);
    init_ops_.push_back(var_one);
    init_ops_.push_back(var_zero);
    init_ops_.push_back(const_iter_num);
    init_ops_.push_back(const_loop_cond);
    init_ops_.push_back(const_one);
    init_ops_.push_back(const_zero);
    init_ops_.push_back(assign_iter_num);
    init_ops_.push_back(assign_loop_cond);
    init_ops_.push_back(assign_one);
    init_ops_.push_back(assign_zero);
  }
}

OpAdapterPtr DfGraphConvertor::FindAdapter(const std::string &name, bool train) {
  auto it = OpAdapterMap::get().find(name);
  if (it != OpAdapterMap::get().end()) {
    return it->second->Get(train);
  }
  MS_LOG(EXCEPTION) << "Can't find OpAdapter for " << name;
}

void DfGraphConvertor::DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it) {
  // draw init subgraph
  init_sout_ << "op_assign" << it.get() << "[label=<";
  init_sout_ << "<table border='1' cellborder='1'>" << endl;
  init_sout_ << "<tr>";
  init_sout_ << "<td port='1'>resource</td>";
  init_sout_ << "<td port='2'>value</td>";
  init_sout_ << "</tr>" << endl;
  init_sout_ << "<tr><td colspan=\"2\">"
             << "\"assign_" << name << "\"</td></tr>" << endl;
  init_sout_ << "</table>> shape=plaintext]" << endl;
  init_sout_ << "param" << it.get() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  init_sout_ << "const" << it.get() << "[label= \"" << name << "_const"
             << "\" shape=ellipse]" << endl;
  init_sout_ << "param" << it.get() << "->"
             << "op_assign" << it.get() << ":1" << endl;
  init_sout_ << "const" << it.get() << "->"
             << "op_assign" << it.get() << ":2" << endl;
}

void DfGraphConvertor::SetupParamInitSubGraph(const TensorOrderMap &tensors, std::vector<ge::Operator> *init_input) {
  DfGraphPtr init_graph = std::make_shared<DfGraph>("init");
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_);

  for (auto &it : nodes) {
    MS_EXCEPTION_IF_NULL(it);
    if (it->isa<ValueNode>()) {
      if (IsValueNode<SymbolicKeyInstance>(it)) {
        auto symbolic = GetValueNode<SymbolicKeyInstancePtr>(it);
        auto name = std::static_pointer_cast<Parameter>(symbolic->node())->name();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          // #ifdef DRAW_GE_GRAPH
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
          // #endif
        }
      } else if (IsValueNode<RefKey>(it)) {
        auto refkey = GetValueNode<RefKeyPtr>(it);
        MS_EXCEPTION_IF_NULL(refkey);
        auto name = refkey->tag();
        auto iter = vars_.find(name);  // get corresponding variable op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
        }
      }
    }
  }

  for (auto &it : tensors) {
    if (vars_.find(it.first) == vars_.end()) {
      MS_LOG(WARNING) << "Init parameter " << it.first << " didn't appear in graph.";
      vars_[it.first] = nullptr;
    }
  }

  // set up init sub graph
  if (init_input->size()) {
    // init sub graph needs no input
    MS_LOG(INFO) << "Build data init subgraph.";
    (void)init_graph->SetInputs(*init_input);
    this->init_graph_ = init_graph;
  } else {
    this->init_graph_ = nullptr;
  }
}

void DfGraphConvertor::MakeDatasetHandler(const std::string &name, const size_t &input_idx, const AnfNodePtr &it) {
  MS_LOG(INFO) << "The " << name << " is the " << input_idx << "(st/nd/th) input";
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    auto getnext_idx = static_cast<int64_t>(input_idx);
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    if (!param.input_indexes().empty() && input_idx <= param.input_indexes().size()) {
      getnext_idx = param.input_indexes()[input_idx] - 1;  // input_idx start from 0.
      MS_LOG(INFO) << "remap input_index:" << input_idx << " to getnext_index:" << getnext_idx << ".";
    }
    // use iterator_getnext op with output_name instead of data op in BuildGraph.
    if (dataset_iter_getnext_ != nullptr) {
      out_handle_cache_[it.get()] = OutHandler(dataset_iter_getnext_, "y" + std::to_string(getnext_idx));
    }
  }
}

void DfGraphConvertor::SetupBroadcast(const std::shared_ptr<HcomBroadcast> &broadcast,
                                      const std::vector<GeTensorDesc> &broadcast_desc,
                                      const DfGraphPtr &broadcast_graph, std::vector<ge::Operator> broadcast_input) {
  MS_LOG(INFO) << "build broadcast subgraph";
  if (broadcast_desc.size() != broadcast_input.size()) {
    MS_LOG(EXCEPTION) << "Desc number of BroadCast is not equal to number of Input";
  }
  (void)broadcast->create_dynamic_input_x(static_cast<unsigned int>(broadcast_input.size()));
  (void)broadcast->create_dynamic_output_y(static_cast<unsigned int>(broadcast_desc.size()));
  for (unsigned int i = 0; i < broadcast_input.size(); i++) {
    (void)broadcast->set_dynamic_input_x(i, broadcast_input[i]);
    (void)broadcast->update_dynamic_output_desc_y(i, broadcast_desc[i]);
  }
  (void)broadcast_graph->SetInputs(broadcast_input);
  this->broadcast_graph_ = broadcast_graph;
}

void DfGraphConvertor::InitParamWithData(const TensorOrderMap &tensors) {
  int index = 0;
  std::vector<Operator> init_input;
  for (auto it : tensors) {
    std::string name = it.first;
    auto node_itor = params_.find(name);
    // if name not in params_, create a node in graph
    if (node_itor == params_.end()) {
      MS_LOG(WARNING) << name << " is not in params, and create a new node.";
      ParameterPtr param = std::make_shared<Parameter>(nullptr);
      name = name + "_temp";
      param->set_name(name);
      (void)ConvertParameter(param);
      node_itor = params_.find(name);
    }
    auto node = node_itor->second;
    auto op_itor = op_cache_.find(node.get());
    if (op_itor == op_cache_.end()) {
      MS_LOG(EXCEPTION) << "Can not find op for node " << node->ToString() << ".";
    }
    auto adpt = FindAdapter(kNameParam, training_);
    if (adpt == nullptr) continue;
    auto param_op = adpt->generate(name + "_data");
    MS_LOG(INFO) << "Add parameter " << name << " as input, index " << index << ".";

    if (!training_) {
      auto adpt_const = FindAdapter(kNameConst, training_);
      if (adpt_const == nullptr) continue;
      auto const_op = adpt_const->generate(name + "_const");
      (void)adpt_const->setAttr(const_op, "value", it.second);

      auto const_op_desc = TransformUtil::GetGeTensorDesc(it.second->shape_c(), it.second->data_type(), kOpFormat_NCHW);
      if (const_op_desc == nullptr) {
        MS_LOG(WARNING) << "Create variable " << name << " output descriptor failed!";
        continue;
      }
      (void)std::static_pointer_cast<Constant>(const_op)->update_output_desc_y(*const_op_desc);
      const_op_to_value_[const_op] = it.second;
      vars_[name] = const_op;
      op_itor->second = const_op;
      continue;
    }

    // create tensor descriptor for output descriptor
    auto desc = TransformUtil::GetGeTensorDesc(it.second->shape_c(), it.second->data_type(), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Create variable " << name << " output descriptor failed!";
      continue;
    }

    // we need three variable ops for each graph with same name
    // build init subgraph
    if (it.second->is_init() == 0) {
      (void)std::static_pointer_cast<Data>(param_op)->set_attr_index(index++);
      auto init_var = std::make_shared<Variable>(name);
      auto assign_op = std::make_shared<Assign>("assign_" + name);
      (void)init_var->update_output_desc_y(*desc);
      (void)assign_op->set_input_ref(*init_var).set_input_value(*param_op);
      init_input.push_back(*init_var);
      init_ops_.push_back(param_op);
      init_ops_.push_back(assign_op);
      init_ops_.push_back(init_var);
    }

    auto variable = std::make_shared<Variable>(name);
    (void)variable->update_output_desc_y(*desc);
    // do not use read variable while variable sink
    MS_LOG(DEBUG) << "InitParam, op_name = " << name << ", var = " << variable->GetName() << ".";
    op_itor->second = variable;  // replace parameter with variable
    vars_[name] = variable;      // prevent the variable operator from being freed
    DrawParamInitSubGraph(name, node);
  }
  InitLoopVar(&init_input);
  SetupParamInitSubGraph(tensors, &init_input);
}

// convert all parameter need initialize to variable
DfGraphConvertor &DfGraphConvertor::InitParam(const TensorOrderMap &tensors) {
  size_t input_idx = 0;
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in InitParam.";
    return *this;
  }

  // Processing input with MakeDatasetHandler
  for (auto &it : anf_graph_->parameters()) {
    auto op_itor = op_cache_.find(it.get());  // converted node
    if (it->isa<Parameter>() && op_itor != op_cache_.end()) {
      string name = std::static_pointer_cast<Parameter>(it)->name();
      auto tensor_itor = tensors.find(name);  // in init value map
      if (tensor_itor == tensors.end()) {
        DfGraphConvertor::MakeDatasetHandler(name, input_idx, it);
        input_idx++;
      }
    }
  }
  InitParamWithData(tensors);
  init_sout_ << "}" << endl;
  return *this;
}

#if (defined ENABLE_D)
void DfGraphConvertor::BuildSaveCheckpointGraph() {
  std::vector<Operator> graph_inputs;
  ge::op::Save save_op("save_parms");
  int save_op_is_active = 0;
  size_t index = 0;
  string name;

  auto count_size = std::count_if(vars_.begin(), vars_.end(), [](const auto &it) {
    return LongToUlong(it.second == nullptr || it.first.find("/") != std::string::npos);
  });

  (void)save_op.create_dynamic_input_tensors(static_cast<uint32_t>(vars_.size() - static_cast<size_t>(count_size)));

  // for each "parameter" in anf graph excluding "input"
  for (const auto &it : vars_) {
    name = it.first;
    if (it.second == nullptr || name.find("/") != std::string::npos) continue;
    Variable variable(name);
    (void)variable.update_output_desc_y(it.second->GetOutputDesc(0));
    (void)save_op.set_dynamic_input_tensors(static_cast<uint32_t>(index++), variable);

    graph_inputs.push_back(variable);

    if (save_op_is_active == 0) {
      checkpoint_sout_ << "op_save" << &save_op << "[label=<";
      checkpoint_sout_ << "<table border='1' cellborder='1'>" << endl;
      checkpoint_sout_ << "<tr><td port='1'>tensor</td></tr>" << endl;
      checkpoint_sout_ << "<tr><td colspan=\"1\">"
                       << "\"saveop"
                       << "\"</td></tr>" << endl;
      checkpoint_sout_ << "</table>> shape=plaintext]" << endl;
    }

    checkpoint_sout_ << "param" << it.second << "[shape=octagon, label=\"" << name << "\"]" << endl;

    checkpoint_sout_ << "param" << it.second << "->"
                     << "op_save" << &save_op << ":1" << endl;
    save_op_is_active = 1;
  }
  if (save_op_is_active) {
    std::vector<Operator> graph_output;
    graph_output.emplace_back(save_op);
    DfGraphPtr checkpoint_graph = std::make_shared<DfGraph>("checkpoint");
    (void)checkpoint_graph->SetInputs(graph_inputs);
    (void)checkpoint_graph->SetOutputs(graph_output);
    this->save_ckp_graph_ = checkpoint_graph;
  } else {
    this->save_ckp_graph_ = nullptr;
  }

  checkpoint_sout_ << "}" << endl;
  return;
}
#endif

DfGraphConvertor &DfGraphConvertor::GenerateBroadcastGraph(const TensorOrderMap &tensors) {
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in generate broadcast graph";
    return *this;
  }

  DfGraphPtr broadcast_graph = std::make_shared<DfGraph>("broadcast");
  // collect the operators create for broadcast sub graph, in order to avoid auto release
  std::vector<Operator> broadcast_input;
  std::vector<GeTensorDesc> broadcast_desc;
  auto broadcast = std::make_shared<HcomBroadcast>("broadcast_parameter");
  (void)broadcast->set_attr_root_rank(0);
  (void)broadcast->set_attr_group("hccl_world_group");
  broadcast_ops_.push_back(broadcast);

  // find every parameter, build broadcast subgraph (or initialize the parameter with constant)
  for (auto &it : anf_graph_->parameters()) {
    auto op_itor = op_cache_.find(it.get());  // converted node
    if (it->isa<Parameter>() && op_itor != op_cache_.end()) {
      string name = std::static_pointer_cast<Parameter>(it)->name();
      auto tensor_itor = tensors.find(name);  // in init tensor map
      if (tensor_itor != tensors.end()) {
        auto tensor = tensor_itor->second;
        auto shape_ge = tensor->shape_c();

        // create tensor descriptor for output descriptor
        auto desc = TransformUtil::GetGeTensorDesc(shape_ge, tensor->data_type(), kOpFormat_NCHW);
        if (desc == nullptr) {
          MS_LOG(ERROR) << "Create variable " << name << " output descriptor failed!";
          continue;
        }

        // build broadcast subgraph
        if (distribute_) {
          auto broadcast_var = std::make_shared<Variable>(name);
          (void)broadcast_var->update_output_desc_y(*desc);
          broadcast_input.push_back(*broadcast_var);
          broadcast_desc.push_back(*desc);
          broadcast_ops_.push_back(broadcast_var);
        }
      }
    }
  }

  // set up broadcast sub graph
  if (!broadcast_input.empty()) {
    DfGraphConvertor::SetupBroadcast(broadcast, broadcast_desc, broadcast_graph, broadcast_input);
  } else {
    this->broadcast_graph_ = nullptr;
  }
  return *this;
}

DfGraphConvertor &DfGraphConvertor::GenerateCheckpointGraph() {
  if (error_ != SUCCESS) {
    MS_LOG(ERROR) << "Generate checkpoint graph failed, found error code " << error_ << ".";
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in GenerateCheckpointGraph";
    return *this;
  }
#ifdef ENABLE_D
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    BuildSaveCheckpointGraph();
    // Restoring from checkpoint file is done by pyfront, not in graph now.
  }
#endif
  return *this;
}

DfGraphConvertor &DfGraphConvertor::ConvertAllNode() {
  if (error_ != SUCCESS) {
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    MS_LOG(ERROR) << "Invalid AnfGraph";
    error_ = FAILED;
    return *this;
  }

  compute_sout_.clear();
  compute_sout_ << "digraph {" << endl;
  init_sout_.clear();
  init_sout_ << "digraph {" << endl;
#ifdef ENABLE_D
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    checkpoint_sout_.clear();
    checkpoint_sout_ << "digraph {" << endl;
  }
#endif
  restore_checkpoint_sout_.clear();
  restore_checkpoint_sout_ << "digraph {" << endl;
  // Convert ResizeBilinear attr size to input
  ConvertResizeBilinear(anf_graph_);
  // Convert Tile input1 to int32
  ConvertTile(anf_graph_);
  // Convert all anf node to Operator
  MS_LOG(DEBUG) << "convert all node";
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_, while_cond_node_);
  for (auto &it : nodes) {
    if (IsSubGraph() && it->isa<Parameter>()) {
      continue;
    }
    if (IsSubGraph() && (IsPartialSuccNode(it) || IsPartialCNode(it))) {
      continue;
    }
    (void)Convert(it);
    if (this->error_ != SUCCESS) {
      MS_LOG(ERROR) << "failed to convert node: " << it->DebugString() << ".";
    }
  }

  // Create dataset iterator and iterator_getnext node
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    MS_LOG(INFO) << "Dataset param is " << param.ToString() << ".";
    // GetNext
    auto iter_getnext_op = make_shared<ge::op::GetNext>("get_next_tmp");
    std::vector<enum ge::DataType> getnext_types;
    const auto &origin_ge_types = param.ge_types();
    (void)std::transform(
      origin_ge_types.begin(), origin_ge_types.end(), std::back_inserter(getnext_types),
      [](int64_t t_num) -> enum ge::DataType { return static_cast<enum ge::DataType>(t_num); });
    (void)iter_getnext_op->set_attr_output_types(getnext_types);
    (void)iter_getnext_op->set_attr_output_shapes(param.shapes());
    (void)iter_getnext_op->set_attr_channel_name(param.queue_name());

    // save iter_getnext_op for later use
    dataset_iter_getnext_ = iter_getnext_op;
  }

  // return the data flow graph
  return *this;
}

void DfGraphConvertor::TraceOutputFromTupleGetItem(const AnfNodePtr &anf_out) {
  auto it = out_handle_cache_.find(anf_out.get());
  if (it != out_handle_cache_.end()) {
    OutHandler handle = it->second;
    auto op = handle.op;
    if (op != nullptr) {
      MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType() << ", out_name: " << handle.out;
      (void)graph_outputs_.emplace_back(*op, handle.out);
    } else {
      MS_LOG(EXCEPTION) << "tuple_getitem: " << anf_out->fullname_with_scope() << " is not converted";
    }
  } else {
    // invalid tuple_getitem e.g. tuple_getitem(tuple_getitem())/tuple_getitem(depend())/tuple_getitem(make_tuple())
    MS_LOG(WARNING) << "Invalid tuple_getitem: " << anf_out->fullname_with_scope();
  }
}

void DfGraphConvertor::TraceOutput(const AnfNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  AnfNodePtr anf_out = node;
  AnfNodePtr pre_node = nullptr;

  // Trace value node
  if (node->isa<ValueNode>()) {
    auto op = Convert(anf_out);
    if (op != nullptr) {
      (void)graph_outputs_.emplace_back(*op, "");
      AddGraphConstInput(op);
    }
    return;
  }

  // Trace Parameter node
  TraceOutputFromParameter(anf_out);
  // Then trace cnode
  if (!node->isa<CNode>()) {
    return;
  }

  // trace tuple_getitem
  while (anf_out->isa<CNode>() && IsPrimitiveCNode(anf_out, prim::kPrimTupleGetItem)) {
    pre_node = anf_out;
    anf_out = anf_out->cast<CNodePtr>()->input(1);
  }
  // trace every element of make_tuple
  auto c = anf_out->cast<CNodePtr>();
  std::string name = "";
  if (anf_out->isa<CNode>()) {
    name = GetCNodeTargetFuncName(c);
  }

  if (name == "MakeTuple") {
    for (unsigned int i = 1; i < c->inputs().size(); i++) {
      TraceOutput(c->input(i));
    }
  } else if (name == prim::kPrimDepend->name() || name == prim::kPrimLoad->name()) {
    if (c->inputs().size() < 3) {  // "Depend" primitive have 3 inputs
      MS_LOG(EXCEPTION) << "length of inputs is " << c->inputs().size() << ", which is less than 3";
    }
    TraceOutput(c->input(1));
  } else if (name == prim::kTupleGetItem) {
    TraceOutputFromTupleGetItem(anf_out);
  } else {
    // add outputs
    auto op = Convert(anf_out);
    std::string index;
    if (op != nullptr) {
      if ((pre_node != nullptr) && IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
        auto item = out_handle_cache_.find(pre_node.get());
        if (item != out_handle_cache_.end()) {
          index = item->second.out;
        } else {
          MS_LOG(WARNING) << "Can't get operator: " << anf_out->fullname_with_scope() << " 's output item";
        }
      }
      MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope() << ":" << index;
      (void)graph_outputs_.emplace_back(*op, index);
    }
  }
}

void DfGraphConvertor::TraceOutputFromParameter(const AnfNodePtr &anf_out) {
  MS_EXCEPTION_IF_NULL(anf_out);
  if (!anf_out->isa<Parameter>()) {
    return;
  }
  MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope();
  auto params = anf_graph_->parameters();

  if (IsAfterGraph()) {
    auto idx = std::find(params.begin(), params.end(), anf_out) - params.begin();
    auto idx_cond = prev_after_cond_map_[idx];
    OutHandler handle;
    if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
      handle = bypass_node_prev_handle_cache_[idx_cond];
      if (handle.node == nullptr) {
        MS_LOG(INFO) << "there is  a nullptr in TraceOutputFromParameter ";
      }
    } else {
      auto idx_out = prev_cond_to_while_out_index_[idx_cond];
      handle = while_output_handle_cache_[prev_while_node_]->at(idx_out);
    }
    MS_LOG(INFO) << "op name: " << handle.op->GetName() << ", op type: " << handle.op->GetOpType()
                 << ", out_name: " << handle.out;
    (void)graph_outputs_.emplace_back(*(handle.op), handle.out);
    return;
  }
  auto it = out_handle_cache_.find(anf_out.get());
  if (it != out_handle_cache_.end()) {
    // For dataset graph mode, input parameter is converted to a "iterator_get_next:yn" OutHandler.
    OutHandler handle = it->second;
    auto op = handle.op;
    MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType() << ", out_name: " << handle.out;
    (void)graph_outputs_.emplace_back(make_pair(*op, handle.out));
  } else {
    // common parameter case
    auto op = Convert(anf_out);
    if (op != nullptr) {
      MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType();
      (void)graph_outputs_.emplace_back(std::make_pair(*op, ""));
    }
  }
  return;
}

void SetupDatasetIterGetNextNode(const OperatorPtr &op) {
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    size_t output_num = param.ge_types().size();
    MS_LOG(INFO) << "Set iterator_getnext op's output num = " << output_num << ".";
    // set iterator_getnext op's output num
    shared_ptr<ge::op::GetNext> iter_getnext = std::static_pointer_cast<ge::op::GetNext>(op);
    (void)iter_getnext->create_dynamic_output_y(static_cast<unsigned int>(output_num));

    for (uint32_t i = 0; i < output_num; i++) {
      ge::TensorDesc desc(GeShape(param.shapes()[i]), ge::FORMAT_NCHW, (ge::DataType)param.ge_types()[i]);
      // we don't SetRealDimCnt here since GE do not use this output's real-dim
      (void)iter_getnext->update_dynamic_output_desc_y((i), desc);
    }
  }
  return;
}

void DfGraphConvertor::CacheWhileGraph(const CNodePtr &cnode) {
  if (while_graph_cache_.find(cnode) != while_graph_cache_.end()) {
    return;
  }
  auto partial_node = cnode->input(0);
  auto graph_node = partial_node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr cond_graph = graph_node->value()->cast<FuncGraphPtr>();
  const auto &cond_set = cond_graph->nodes();
  for (auto beg = cond_set.begin(); beg != cond_set.end(); beg++) {
    if (!((*beg)->isa<CNode>())) {
      continue;
    }
    auto c_beg = (*beg)->cast<CNodePtr>();
    if (IsPartialSuccNode(c_beg) && c_beg->inputs().size() == kSwitchInputSize &&
        IsPartialCNode(c_beg->input(kSwitchBodyIndex)) && IsPartialCNode(c_beg->input(kSwitchAfterIndex)) &&
        GetCNodeFuncName(c_beg) == prim::kPrimSwitch->name()) {
      while_graph_cache_[cnode] = {c_beg->input(1), c_beg->input(kSwitchBodyIndex), c_beg->input(kSwitchAfterIndex)};
    }
  }
}

std::vector<Operator> DfGraphConvertor::GetWhileBodyOutputs() {
  std::vector<Operator> outputs;

  const auto &node = anf_graph_->get_return()->input(1);
  AnfNodePtr real_ret = node;
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }

  // skip input of UMonad, IOMonad
  if (HasAbstractUMonad(real_ret) || HasAbstractIOMonad(real_ret)) {
    return outputs;
  }

  // skip input of the None, UpdateState
  if (IsValueNode<None>(real_ret) || IsPrimitiveCNode(real_ret, prim::kPrimUpdateState)) {
    return outputs;
  }

  if (IsPrimitiveCNode(real_ret, prim::kPrimLoad)) {
    real_ret = ParseLoadInput(real_ret->cast<CNodePtr>());
  }

  if (!real_ret->isa<CNode>()) {
    return outputs;
  }

  auto c_node = real_ret->cast<CNodePtr>();

  auto in0 = c_node->input(0)->cast<CNodePtr>();
  std::vector<AnfNodePtr> inputs(in0->inputs().begin() + kInputOffset, in0->inputs().end());
  size_t partial_input_size = inputs.size();

  std::copy(c_node->inputs().begin() + 1, c_node->inputs().end(), std::back_inserter(inputs));

  for (size_t i = 0; i < inputs.size(); i++) {
    auto j = inputs[i];
    CNodePtr cur = nullptr;
    if (i < partial_input_size) {
      cur = in0;
    } else {
      cur = c_node;
    }
    j = GetRealInputNode(cur, j);
    if (j == nullptr) {
      continue;
    }

    if (j->isa<Parameter>()) {
      size_t idx = find(inputs_.begin(), inputs_.end(), j) - inputs_.begin();
      auto idx_cond = body_cond_map_[idx];
      if (while_used_input_index_.find(idx_cond) == while_used_input_index_.end() ||
          while_const_input_index_.find(idx_cond) != while_const_input_index_.end()) {
        continue;
      }
      outputs.push_back(*(subgraph_input_cache_[idx]));
    } else {
      outputs.push_back(*Convert(j));
    }
  }
  MS_LOG(DEBUG) << "get while body outputs size: " << outputs.size();
  return outputs;
}

std::shared_ptr<std::vector<Operator>> DfGraphConvertor::GetWhileSubGraphInput() {
  std::shared_ptr<std::vector<Operator>> graph_in = std::make_shared<std::vector<Operator>>();
  subgraph_input_cache_.clear();
  size_t i = 0;
  OperatorPtr op = nullptr;
  ParamIndexMap cond_body;
  std::string name_app = "_in_cond";
  if (graph_type_ == GraphType::kBody) {
    name_app = "_in_body";
    for (auto &p : body_cond_map_) {
      cond_body[p.second] = p.first;
    }
  }
  for (auto &idx : while_used_input_index_) {
    if (while_const_input_index_.find(idx) == while_const_input_index_.end()) {
      op = std::make_shared<Data>();
      (void)std::static_pointer_cast<Data>(op)->set_attr_index(i);
      i++;
    } else {
      auto temp = while_const_input_index_[idx].op;
      auto name = temp->GetName();
      auto value = const_op_to_value_[temp];
      MS_EXCEPTION_IF_NULL(value);
      auto adpt_const = FindAdapter(kNameConst, training_);
      if (adpt_const == nullptr) continue;
      name += name_app;
      auto const_op = adpt_const->generate(name);
      (void)adpt_const->setAttr(const_op, "value", value);
      auto const_op_desc = TransformUtil::GetGeTensorDesc(value->shape_c(), value->data_type(), kOpFormat_NCHW);
      if (const_op_desc == nullptr) {
        MS_LOG(WARNING) << "Create variable " << name << " output descriptor failed!";
        continue;
      }
      (void)std::static_pointer_cast<Constant>(const_op)->update_output_desc_y(*const_op_desc);
      op = const_op;
    }
    graph_in->push_back(*op);
    if (graph_type_ == GraphType::kCond) {
      subgraph_input_cache_[idx] = op;
    } else if (graph_type_ == GraphType::kBody) {
      subgraph_input_cache_[cond_body[idx]] = op;
    }
  }
  MS_LOG(DEBUG) << "created " << subgraph_input_cache_.size() << " data node "
                << " in graph: " << anf_graph_->ToString();
  return graph_in;
}

void DfGraphConvertor::BuildWhileSubGraph() {
  // set up dependencies
  MS_LOG(DEBUG) << "begin to build graph: " << anf_graph_->ToString();
  MS_LOG(DEBUG) << "set up dependencies";

  std::vector<Operator> graph_in = *GetWhileSubGraphInput();
  auto nodes = GetOrderedCNodes(anf_graph_, while_cond_node_);

  auto iter = std::find_if(nodes.begin(), nodes.end(), [](const AnfNodePtr &it) {
    if (IsPartialSuccNode(it) && IsWhileNode(it)) {
      return true;
    }
    return false;
  });
  if (iter != nodes.end()) {
    call_node_in_while_body_ = *iter;
  }
  AnfNodePtr real_ret = anf_graph_->get_return()->input(1);
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }
  for (auto &it : nodes) {
    if (it->isa<CNode>() && IsCaseNode(it->cast<CNodePtr>())) {
      auto node = it->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      auto input = node->input(0);
      MS_EXCEPTION_IF_NULL(input);
      auto cinput = input->cast<CNodePtr>();
      GetCaseNodeInput(node, cinput);
    }
  }
  // update tuple_out_handle_cache_
  UpdateTupleOutCache();

  for (auto &it : nodes) {
    if (it == real_ret || HasAbstractMonad(it)) {
      continue;
    }
    SetNodeInput(it);
    SetSubgraph(it);
    SetOpControlInput(it);
    UpdateOpDesc(it);
  }
  MS_LOG(DEBUG) << "trace output";
  std::vector<Operator> graph_out;
  auto graph_name = TransformUtil::NormOpName(cur_while_node_->fullname_with_scope());
  if (graph_type_ == GraphType::kCond) {
    if (op_cache_.find(while_cond_node_.get()) == op_cache_.end()) {
      return;
    }
    graph_name += "_cond_graph";
    graph_out.push_back(*(op_cache_[while_cond_node_.get()]));
  } else {
    graph_name += "_body_graph";
    graph_out = GetWhileBodyOutputs();
  }
  if (error_ == 0) {
    if (df_graph_->GetName() != graph_name) {
      MS_LOG(DEBUG) << "convert anf graph name : " << df_graph_->GetName() << " to df graph name: " << graph_name;
    }
    df_graph_ = make_shared<DfGraph>(graph_name);
  } else {
    return;
  }
  MS_LOG(DEBUG) << "set while sub graph input num: " << graph_in.size();
  MS_LOG(DEBUG) << "set while sub graph output num: " << graph_out.size();

  compute_sout_ << "}" << endl;
  df_graph_->SetInputs(graph_in).SetOutputs(graph_out);
  MS_LOG(DEBUG) << "build graph: " << anf_graph_->ToString() << " end";
}

void DfGraphConvertor::BuildWhileAfterSubGraph() {
  // update tuple_out_handle_cache_
  UpdateTupleOutCache();
  size_t i = 0;
  prev_cond_to_while_out_index_.clear();
  for (auto n : prev_while_used_input_index_) {
    if (prev_while_const_input_index_.find(n) == prev_while_const_input_index_.end()) {
      prev_cond_to_while_out_index_[n] = i;
      i++;
    }
  }
  GetCallNodeInputs(cur_while_node_);
  // set up dependencies
  MS_LOG(INFO) << "set up dependencies";
  auto nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetOpControlInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }
  MS_LOG(INFO) << "trace output";
  if (graph_outputs_.empty()) {
    TraceOutput(anf_graph_->get_return()->input(1));
  }
  compute_sout_ << "}" << endl;
  return;
}

void DfGraphConvertor::ConvertWhileBody(const AnfNodePtr &node) {
  if (!node->isa<CNode>() || GetCNodeFuncName(node->cast<CNodePtr>()) != prim::kPrimPartial->name()) {
    return;
  }
  MS_LOG(DEBUG) << "begin to convert while node body graph";
  auto graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  DfGraphConvertor converter(anf_graph);
  converter.use_inputs_ = true;

  const auto &params = anf_graph->parameters();
  converter.inputs_ = params;

  converter.graph_type_ = GraphType::kBody;
  converter.cur_while_node_ = cur_while_node_;
  converter.body_cond_map_ = body_cond_map_;
  converter.while_const_input_index_ = while_const_input_index_;
  converter.while_used_input_index_ = while_used_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;
  converter.ConvertAllNode().BuildWhileSubGraph();
  while_dfgraph_cache_[cur_while_node_]->push_back(*(converter.df_graph_));
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    converter.DrawComputeGraph(name);
  }
  MS_LOG(DEBUG) << "convert while node body graph end";
  return;
}

void DfGraphConvertor::GetWhileUsedInputIndex(const std::vector<AnfNodePtr> &graphs) {
  if (!while_used_input_index_.empty()) {
    return;
  }

  auto cond_graph_node = graphs.at(0);
  auto graph = cond_graph_node->func_graph();
  const auto &cond_params = graph->parameters();
  auto nodes = GetOrderedCNodes(graph, cond_graph_node);

  std::set<size_t> used_params_index;
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto c = n->cast<CNodePtr>();
    auto inputs = c->inputs();
    for (size_t idx = 1; idx < inputs.size(); idx++) {
      auto &i = inputs[idx];
      if (!i->isa<Parameter>() || HasAbstractMonad(i) || IsDynamicShapeNode(i)) {
        continue;
      }
      auto idx_cond = std::find(cond_params.begin(), cond_params.end(), i) - cond_params.begin();
      used_params_index.insert(idx_cond);
    }
  }

  auto body_graph_node_in_cond = graphs.at(1)->cast<CNodePtr>();
  auto body_graph_node = body_graph_node_in_cond->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(body_graph_node);
  graph = body_graph_node->value()->cast<FuncGraphPtr>();
  const auto &body_params = graph->parameters();

  auto real_ret = graph->get_return()->input(1);
  while (real_ret->isa<CNode>() && GetCNodeTargetFuncName(real_ret->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    real_ret = real_ret->cast<CNodePtr>()->input(1);
  }

  nodes = GetOrderedCNodes(graph);
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto c = n->cast<CNodePtr>();
    if (c == real_ret || c == real_ret->cast<CNodePtr>()->input(0)) {
      continue;
    }
    auto inputs = c->inputs();
    for (size_t idx = 1; idx < inputs.size(); idx++) {
      auto &i = inputs[idx];
      if (!i->isa<Parameter>() || HasAbstractMonad(i) || IsDynamicShapeNode(i)) {
        continue;
      }
      auto idx_body = std::find(body_params.begin(), body_params.end(), i) - body_params.begin();
      auto p = body_graph_node_in_cond->input(idx_body + kInputOffset);
      auto idx_cond = std::find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
      used_params_index.insert(idx_cond);
    }
  }
  while_used_input_index_ = used_params_index;
}

void DfGraphConvertor::SetParamIndexMap(const std::vector<AnfNodePtr> &graphs) {
  auto cond_graph_node = graphs.at(0);
  auto cond_graph = cond_graph_node->func_graph();
  const auto &cond_params = cond_graph->parameters();

  auto body_graph_node = graphs.at(1);
  if (!body_graph_node->isa<CNode>()) {
    return;
  }
  auto body_graph_node_inputs = body_graph_node->cast<CNodePtr>()->inputs();
  std::vector<AnfNodePtr> body_params;
  for (auto it = body_graph_node_inputs.begin() + kInputOffset; it != body_graph_node_inputs.end(); it++) {
    body_params.push_back(*it);
  }

  for (size_t i = 0; i < body_params.size(); i++) {
    auto p = body_params[i];
    size_t idx = find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
    body_cond_map_[i] = idx;
    MS_LOG(DEBUG) << "body_cond_map_'s key: " << i << " value: " << idx;
  }

  auto after_graph_node = graphs.at(kSwitchBodyIndex);
  if (!after_graph_node->isa<CNode>()) {
    return;
  }
  auto after_graph_node_inputs = after_graph_node->cast<CNodePtr>()->inputs();
  std::vector<AnfNodePtr> after_params;
  for (auto it = after_graph_node_inputs.begin() + 2; it != after_graph_node_inputs.end(); it++) {
    after_params.push_back(*it);
  }

  for (size_t i = 0; i < after_params.size(); i++) {
    auto p = after_params[i];
    size_t idx = find(cond_params.begin(), cond_params.end(), p) - cond_params.begin();
    after_cond_map_[i] = idx;
    MS_LOG(DEBUG) << "after_cond_map_'s key: " << i << " value: " << idx;
  }
  return;
}

void DfGraphConvertor::ConvertWhileCond(const AnfNodePtr &node) {
  MS_LOG(DEBUG) << "begin to convert while node cond graph";
  auto func_graph = node->func_graph();

  DfGraphConvertor converter(func_graph);
  converter.use_inputs_ = true;

  converter.inputs_ = func_graph->parameters();

  converter.graph_type_ = GraphType::kCond;
  converter.cur_while_node_ = cur_while_node_;
  converter.while_cond_node_ = node;
  converter.while_const_input_index_ = while_const_input_index_;
  converter.while_used_input_index_ = while_used_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;
  converter.ConvertAllNode().BuildWhileSubGraph();
  while_dfgraph_cache_[cur_while_node_]->push_back(*(converter.df_graph_));
  std::string name = func_graph->ToString() + "_ge_graph.dot";
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    converter.DrawComputeGraph(name);
  }

  MS_LOG(DEBUG) << "convert while node cond graph end";
}

void DfGraphConvertor::SetWhileOutputHandle(const OperatorPtr &prev_while_op) {
  if (while_output_handle_cache_.find(prev_while_node_) != while_output_handle_cache_.end()) {
    return;
  }
  auto out_handler = std::make_shared<std::vector<OutHandler>>();
  string str = "output";
  for (size_t i = 0; i < prev_while_node_out_size_; i++) {
    out_handler->emplace_back(prev_while_op, str + std::to_string(i), prev_while_node_);
  }
  while_output_handle_cache_[prev_while_node_] = out_handler;
  return;
}

void DfGraphConvertor::ConvertWhileAfter(const AnfNodePtr &node) {
  if (!node->isa<CNode>() || GetCNodeFuncName(node->cast<CNodePtr>()) != prim::kPrimPartial->name()) {
    return;
  }
  MS_LOG(DEBUG) << "begin to convert while node after graph";
  auto graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  DfGraphConvertor converter(anf_graph);
  converter.use_inputs_ = true;

  const auto &params = anf_graph->parameters();
  converter.inputs_ = params;

  converter.graph_type_ = GraphType::kAfter;
  converter.prev_after_cond_map_ = after_cond_map_;
  converter.prev_while_node_ = cur_while_node_;
  converter.prev_while_node_out_size_ = cur_while_node_out_size_;
  converter.bypass_node_prev_handle_cache_ = bypass_node_handle_cache_;
  converter.prev_while_used_input_index_ = while_used_input_index_;
  converter.prev_while_const_input_index_ = while_const_input_index_;
  converter.const_op_to_value_ = const_op_to_value_;

  auto while_op = Convert(converter.prev_while_node_);
  converter.SetWhileOutputHandle(while_op);
  converter.ConvertAllNode().BuildWhileAfterSubGraph();
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    converter.DrawComputeGraph(name);
  }
  MS_LOG(DEBUG) << "add while after graph " << converter.graph_const_inputs_.size()
                << " const inputs to main graph const inputs";
  std::transform(converter.graph_const_inputs_.begin(), converter.graph_const_inputs_.end(),
                 std::back_inserter(graph_const_inputs_), [](OperatorPtr x) { return x; });

  graph_outputs_ = converter.graph_outputs_;
  MS_LOG(DEBUG) << "convert while node after graph end";
  return;
}

void DfGraphConvertor::ConvertWhileNode(const CNodePtr &node) {
  if (IsSubGraph()) {
    return;
  }

  auto while_graph = while_graph_cache_[node];
  cur_while_node_ = node;

  auto &while_inputs = *(call_input_handle_cache_[node]);
  cur_while_node_out_size_ = while_inputs.size();
  while_dfgraph_cache_[node] = std::make_shared<std::vector<DfGraph>>();
  // convert cond graph
  MS_LOG(DEBUG) << "convert while cond begin...";
  auto cond_graph_node = while_graph[0];
  ConvertWhileCond(cond_graph_node);
  MS_LOG(DEBUG) << "convert while cond end...";

  // convert body graph
  MS_LOG(DEBUG) << "convert while body begin...";
  auto body_graph_node = while_graph[1];
  ConvertWhileBody(body_graph_node);
  MS_LOG(DEBUG) << "convert while body end...";

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(DEBUG) << "Not found adapter";
    return;
  }

  OperatorPtr op = Convert(node);
  auto graphs = while_dfgraph_cache_[node];
  adpt->setSubgraph(op, graphs);

  // convert after graph
  MS_LOG(DEBUG) << "convert while after begin...";
  auto after_graph_node = while_graph[kAfterIndexInCache];
  ConvertWhileAfter(after_graph_node);
  MS_LOG(DEBUG) << "convert while after end...";
  return;
}

void DfGraphConvertor::SetSubgraph(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return;
  }
  MS_LOG(DEBUG) << "set sub graph begin";
  auto cnode = node->cast<CNodePtr>();
  if (IsWhileNode(cnode)) {
    CacheWhileGraph(cnode);
    ConvertWhileNode(cnode);
    MS_LOG(DEBUG) << "set sub graph end....";
    return;
  }

  if (!IsCaseNode(cnode)) {
    MS_LOG(DEBUG) << "set sub graph end....";
    return;
  }
  std::vector<AnfNodePtr> case_inputs;
  std::shared_ptr<std::vector<DfGraph>> df_branches = std::make_shared<std::vector<DfGraph>>();
  case_call_input_size_ = 0;
  if (IsNormalGraph()) {
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      case_inputs.emplace_back(cnode->input(i));
      case_call_input_size_++;
    }
    auto bnode = cnode->input(0)->cast<CNodePtr>()->input(2)->cast<CNodePtr>();

    for (size_t i = 1; i < bnode->inputs().size(); i++) {
      if (!bnode->input(i)->isa<CNode>()) {
        continue;
      }
      auto branch_node = bnode->input(i)->cast<CNodePtr>();
      for (size_t j = kInputOffset; j < branch_node->inputs().size(); j++) {
        if (std::find(case_inputs.begin(), case_inputs.end(), branch_node->input(j)) == case_inputs.end()) {
          case_inputs.emplace_back(branch_node->input(j));
        }
      }
    }
    for (size_t i = 1; i < bnode->inputs().size(); i++) {
      ProcessSubgraph(bnode->input(i), case_inputs);
    }
    for (size_t i = 1; i < bnode->inputs().size(); i++) {
      df_branches->emplace_back(branches_map_[bnode->input(i).get()]);
    }
  } else {
    std::vector<AnfNodePtr> inputs;
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto node = cnode->input(i);
      if (HasAbstractMonad(node)) {
        continue;
      }
      inputs.push_back(node);
      case_call_input_size_++;
    }
    auto bnode = cnode->input(0)->cast<CNodePtr>()->input(kInputOffset);
    auto cbnode = bnode->cast<CNodePtr>();

    for (size_t i = 1; i < cbnode->inputs().size(); i++) {
      auto br = cbnode->input(i);
      if (!cbnode->input(i)->isa<CNode>()) {
        ProcessSubgraph(br, inputs);
        df_branches->emplace_back(branches_map_[br.get()]);
        continue;
      }

      auto branch_node = cbnode->input(i)->cast<CNodePtr>();
      auto branch_input = inputs;
      for (size_t j = kInputOffset; j < branch_node->inputs().size(); j++) {
        branch_input.push_back(branch_node->input(j));
      }
      ProcessSubgraph(cbnode->input(i), branch_input);
      df_branches->emplace_back(branches_map_[br.get()]);
    }
  }
  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(DEBUG) << "Not found adapter";
    return;
  }

  OperatorPtr op = Convert(node);

  adpt->setSubgraph(op, 0, df_branches);
  MS_LOG(DEBUG) << "set sub graph end....";
  return;
}

void DfGraphConvertor::GetCaseNodeInput(const CNodePtr node, const CNodePtr input_node) {
  if (case_input_handle_cache_.find(node.get()) != case_input_handle_cache_.end()) {
    return;
  }

  std::vector<AnfNodePtr> case_inputs;
  const size_t case_index = 1;
  const size_t make_tuple_index = 2;

  AnfNodePtr case_index_iter = input_node->input(case_index);
  AnfNodePtr make_tuple_iter = input_node->input(make_tuple_index);
  AnfNodePtr make_tuple_node = make_tuple_iter;

  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();

  if (IsNormalGraph()) {
    make_tuple_node = make_tuple_iter->cast<CNodePtr>();
    for (size_t i = 1; i < node->inputs().size(); i++) {
      case_inputs.emplace_back(node->input(i));
    }

    auto bnode = input_node->input(2)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(bnode);
    for (size_t i = 1; i < bnode->inputs().size(); i++) {
      if (!bnode->input(i)->isa<CNode>()) {
        continue;
      }
      auto branch_node = bnode->input(i)->cast<CNodePtr>();
      for (size_t j = 2; j < branch_node->inputs().size(); j++) {
        if (std::find(case_inputs.begin(), case_inputs.end(), branch_node->input(j)) == case_inputs.end()) {
          case_inputs.emplace_back(branch_node->input(j));
        }
      }
    }
    for (size_t i = 0; i < case_inputs.size(); i++) {
      auto item = case_inputs[i];
      tuple_items->push_back(GetHandler(item));
    }
  } else if (IsSubGraph()) {
    auto &inputs = node->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input = inputs[i];
      if (HasAbstractUMonad(input) || HasAbstractIOMonad(input)) {
        continue;
      }
      if (input->isa<Parameter>()) {
        auto idx = std::find(inputs_.begin(), inputs_.end(), input) - inputs_.begin();
        tuple_items->push_back(OutHandler(subgraph_input_cache_[idx], "", input));
      } else {
        tuple_items->push_back(GetHandler(input));
      }
    }
    MS_LOG(DEBUG) << "tuple input size of case in sub graph is " << tuple_items->size();
  } else {
    MS_LOG(ERROR) << "case in after graph is not supported.";
    return;
  }

  tuple_out_handle_cache_[make_tuple_node.get()] = tuple_items;

  std::shared_ptr<std::vector<AnfNodePtr>> case_input_items = std::make_shared<std::vector<AnfNodePtr>>();
  (void)case_input_items->emplace_back(case_index_iter);
  (void)case_input_items->emplace_back(make_tuple_iter);
  case_input_handle_cache_[node.get()] = case_input_items;
}

void DfGraphConvertor::GetCallNodeInputs(const CNodePtr &node) {
  if (node == nullptr) {
    return;
  }
  if (call_input_handle_cache_.find(node) != call_input_handle_cache_.end()) {
    return;
  }
  MS_LOG(DEBUG) << "begin to get call node inputs.";

  auto call_input_items = std::make_shared<std::vector<OutHandler>>();
  auto in0 = node->input(0)->cast<CNodePtr>();
  std::vector<AnfNodePtr> inputs(in0->inputs().begin() + kInputOffset, in0->inputs().end());
  std::copy(node->inputs().begin() + 1, node->inputs().end(), std::back_inserter(inputs));

  auto &params = anf_graph_->parameters();
  auto while_op = Convert(node);

  while_const_input_index_.clear();
  std::set<size_t> while_input_node_index;
  for (auto iter = while_used_input_index_.begin(); iter != while_used_input_index_.end(); iter++) {
    auto n = inputs[*iter];
    OutHandler out_handler;
    if (IsAfterGraph() && n->isa<Parameter>()) {
      auto idx = std::find(params.begin(), params.end(), n) - params.begin();
      auto idx_cond = prev_after_cond_map_[idx];
      if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
        out_handler = bypass_node_prev_handle_cache_[idx_cond];
      } else {
        auto idx_out = prev_cond_to_while_out_index_[idx_cond];
        out_handler = while_output_handle_cache_[prev_while_node_]->at(idx_out);
      }
    } else {
      out_handler = GetHandler(inputs[*iter]);
    }
    if ((out_handler.op->GetOpType() == "Const" || out_handler.op->GetOpType() == "Constant") &&
        const_op_to_value_.find(out_handler.op) != const_op_to_value_.end()) {
      while_const_input_index_[*iter] = out_handler;
    } else {
      while_input_node_index.insert(*iter);
      call_input_items->push_back(out_handler);
    }
  }
  cur_while_node_out_size_ = call_input_items->size();
  bypass_node_handle_cache_.clear();

  for (size_t i = 0; i < inputs.size(); i++) {
    if (while_input_node_index.find(i) == while_input_node_index.end()) {
      auto n = inputs[i];
      if (HasAbstractMonad(n)) {
        continue;
      }
      if (IsAfterGraph() && n->isa<Parameter>()) {
        auto idx = std::find(params.begin(), params.end(), n) - params.begin();
        auto idx_cond = prev_after_cond_map_[idx];
        if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
          bypass_node_handle_cache_[i] = bypass_node_prev_handle_cache_[idx_cond];
        } else {
          auto idx_out = prev_cond_to_while_out_index_[idx_cond];
          bypass_node_handle_cache_[i] = while_output_handle_cache_[prev_while_node_]->at(idx_out);
        }
      } else {
        bypass_node_handle_cache_[i] = GetHandler(n);
      }
    }
  }

  MS_LOG(DEBUG) << "while node out size: " << cur_while_node_out_size_
                << ", while const input size: " << while_const_input_index_.size()
                << ", bypass node size: " << bypass_node_handle_cache_.size();
  auto op = Convert(node);
  auto adpt = FindAdapter(node, training_);
  adpt->setDynamicOutputNum(op, cur_while_node_out_size_);

  call_input_handle_cache_[node] = call_input_items;
  MS_LOG(DEBUG) << "get call node inputs end.";
  return;
}

void DfGraphConvertor::UpdateTupleOutCache() {
  for (auto &it : tuple_out_handle_cache_) {
    std::size_t len = it.second->size();
    for (std::size_t i = 0; i < len; i++) {
      OutHandler handle = (*it.second)[i];
      if (handle.op == nullptr) {
        continue;
      }
      string name = handle.op->GetName();
      if (vars_.count(name) && (vars_[name] != nullptr)) {
        (*it.second)[i] = OutHandler(vars_[name], handle.out, handle.node);
        MS_LOG(INFO) << "update tuple_out_handle_cache_ " << name;
      }
    }
  }
}

DfGraphConvertor &DfGraphConvertor::BuildGraph() {
  SetupDatasetIterGetNextNode(dataset_iter_getnext_);

  if (error_ != SUCCESS) {
    return *this;
  }

  GetCallNodeInputs(cur_while_node_);
  // Case node set input.
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    if (it->isa<CNode>() && IsCaseNode(it->cast<CNodePtr>())) {
      auto node = it->cast<CNodePtr>();
      auto input_node = node->input(0)->cast<CNodePtr>();
      GetCaseNodeInput(node, input_node);
    }
  }

  // update tuple_out_handle_cache_
  UpdateTupleOutCache();

  // set up dependencies
  MS_LOG(DEBUG) << "set up dependencies";
  nodes = GetOrderedCNodes(anf_graph_);
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetOpControlInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }

  if (error_ == SUCCESS) {
    df_graph_ = make_shared<DfGraph>(anf_graph_->ToString());
  } else {
    return *this;
  }

  // set graph input according to the order from anf graph
  std::vector<Operator> inputs;
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    inputs.push_back(*dataset_iter_getnext_);
  } else {
    auto params = anf_graph_->parameters();
    if (use_inputs_) {
      params = inputs_;
      auto anf_params = anf_graph_->parameters();
      for (size_t i = 0; i < params.size(); i++) {
        for (size_t j = 0; j < anf_params.size(); j++) {
          if (TransformUtil::NormOpName(params[i]->ToString()) ==
              TransformUtil::NormOpName(anf_params[j]->ToString())) {
            params[i] = anf_params[j];
          }
        }
      }
    }

    int index = 0;
    for (auto &it : params) {
      auto name = std::static_pointer_cast<Parameter>(it)->name();
      //  the parameters which has not been converted to var
      if (vars_.find(name) == vars_.end()) {
        if (HasAbstractMonad(it)) {
          MS_LOG(INFO) << it->DebugString() << " is a monad parameter, skip.";
          continue;
        }
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        MS_LOG(INFO) << "add not var input " << it->ToString() << ", index " << index;
        if (op == nullptr) {
          MS_LOG(ERROR) << "Convert graph failed!";
          return *this;
        }
        UpdateDataOpDesc(it, op);

        MS_LOG(INFO) << "add input " << it->ToString() << ", index " << index;
        (void)std::static_pointer_cast<Data>(op)->set_attr_index(index++);
        inputs.push_back(*op);
      } else if (vars_[name] != nullptr) {
        MS_LOG(INFO) << "add var input " << it->ToString();
        auto op = Convert(it);
        MS_EXCEPTION_IF_NULL(op);
        UpdateConstOpDesc(it, vars_[name]);
        inputs.push_back(*op);
      }
    }
  }

  MS_LOG(DEBUG) << "trace output";
  if (cur_while_node_ == nullptr) {
    graph_outputs_.clear();
    TraceOutput(anf_graph_->get_return()->input(1));
  }

  // Add const nodes as graph input for some operator work with constant
  MS_LOG(INFO) << "graph const input size: " << graph_const_inputs_.size();
  (void)std::transform(graph_const_inputs_.begin(), graph_const_inputs_.end(), std::back_inserter(inputs),
                       [](const OperatorPtr &x) { return *x; });

  MS_LOG(INFO) << "set graph input num: " << inputs.size();
  (void)df_graph_->SetInputs(inputs);

  // set graph output
  // set the value of finale return apply node as the output of dataflow graph
  MS_LOG(DEBUG) << "set output";
  MS_LOG(INFO) << "set graph output num: " << graph_outputs_.size();
  (void)df_graph_->SetOutputs(graph_outputs_);

  compute_sout_ << "}" << endl;
  // For the graph(e.g. eval_subgraph) whose IterNum is 1, donot set NeedIteration flag.
  if (ConfigManager::GetInstance().iter_num() > 1) {
    df_graph_->SetNeedIteration(true);
  }
  return *this;
}

void DfGraphConvertor::UpdateConstOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  if (!it->isa<Parameter>()) {
    MS_LOG(DEBUG) << "It is not parameter, name: " << it->DebugString();
    return;
  }
  auto para = it->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para);
  std::string format = kOpFormat_NCHW;
  std::string param_debug_info = para->DebugString();
  auto param_format = param_format_.find(param_debug_info);
  if (param_format != param_format_.end()) {
    format = param_format->second;
    MS_LOG(DEBUG) << "Parameter debug info: " << param_debug_info << ", format is " << format;
  }
  if (format == kOpFormat_NCHW) {
    MS_LOG(DEBUG) << "Format is not changed, no need to update op desc, name: " << param_debug_info;
    return;
  }
  if (!para->has_default()) {
    MS_LOG(DEBUG) << "Parameter has no default, no need to update op desc, name: " << param_debug_info;
    return;
  }
  auto value = para->default_param();
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto const_op_desc = TransformUtil::GetGeTensorDesc(tensor->shape_c(), tensor->data_type(), format);
  if (const_op_desc == nullptr) {
    MS_LOG(WARNING) << "Create parameter " << para->name() << " output descriptor failed!";
    return;
  }
  (void)std::static_pointer_cast<Constant>(op)->update_output_desc_y(*const_op_desc);
}

void DfGraphConvertor::UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  auto node = std::static_pointer_cast<AnfNode>(it);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! Invalid node.";
    return;
  }
  std::vector<int64_t> shape;
  if (auto normal_shape_ptr = dyn_cast<abstract::Shape>(node->Shape()); normal_shape_ptr != nullptr) {
    shape = normal_shape_ptr->shape();
  } else if (auto no_shape_ptr = dyn_cast<abstract::NoShape>(node->Shape()); no_shape_ptr != nullptr) {
    shape = {};
  } else {
    MS_LOG(INFO) << "Invalid shape to update data op descriptor.";
    return;
  }
  if (node->Type() == nullptr) {
    MS_LOG(INFO) << "Invalid type to update data op descriptor.";
    return;
  }
  TypeId me_type = node->Type()->type_id();
  if (kObjectTypeTensorType == me_type) {
    me_type = dyn_cast<TensorType>(node->Type())->element()->type_id();
  }
  std::ostringstream buf;
  buf << "[" << shape << "]";
  MS_LOG(INFO) << "input shape is " << buf.str() << ", type is " << me_type;
  std::string format = "NCHW";
  if (it->isa<Parameter>()) {
    auto param = it->cast<ParameterPtr>();
    std::string param_name = param->DebugString();
    auto param_format = param_format_.find(param_name);
    if (param_format != param_format_.end()) {
      format = param_format->second;
      MS_LOG(DEBUG) << "parameter: " << param_name << ", format is " << format;
    }
  }
  auto desc = TransformUtil::GetGeTensorDesc(shape, me_type, format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! TensorDesc is null.";
  } else {
    (void)std::static_pointer_cast<Data>(op)->update_input_desc_x(*desc);
    (void)std::static_pointer_cast<Data>(op)->update_output_desc_y(*desc);
  }
}

DfGraphPtr DfGraphConvertor::GetComputeGraph() { return df_graph_; }

DfGraphPtr DfGraphConvertor::GetInitGraph() { return init_graph_; }

DfGraphPtr DfGraphConvertor::GetSaveCheckpointGraph() { return save_ckp_graph_; }

DfGraphPtr DfGraphConvertor::GetBroadcastGraph() { return broadcast_graph_; }

bool DfGraphConvertor::IsSourceEdgeNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!IsCustomCNode(cnode)) {
    std::string name = GetCNodeTargetFuncName(cnode);
    if (name.empty()) {
      return false;
    }

    // Ignore apply node Depend, UpdateState, make_tuple. make_tuple in ge pipeline.
    if ((name == prim::kPrimDepend->name()) || (name == prim::kPrimUpdateState->name()) ||
        (name == prim::kPrimReturn->name()) || (name == prim::kPrimMakeTuple->name())) {
      return false;
    }
  }
  // Load and other normal primitives which contain monad node.
  auto has_monad = std::any_of(cnode->inputs().begin(), cnode->inputs().end(),
                               [](const AnfNodePtr &node) -> bool { return HasAbstractMonad(node); });
  if (has_monad) {
    return true;
  }

  // primitive with make_tuple as input
  for (auto &input : cnode->inputs()) {
    if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      auto tuple = input->cast<CNodePtr>();
      auto ret = std::any_of(tuple->inputs().begin(), tuple->inputs().end(),
                             [](const AnfNodePtr &node) -> bool { return HasAbstractMonad(node); });
      if (ret) {
        return true;
      }
    }
  }

  return false;
}

bool DfGraphConvertor::IsControlEdgeNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!IsCustomCNode(cnode)) {
    std::string name = GetCNodeTargetFuncName(cnode);
    if (name.empty()) {
      return false;
    }

    // Ignore apply node of Load, Depend, UpdateState, make_tuple, return
    if ((name == prim::kPrimLoad->name()) || (name == prim::kPrimDepend->name()) ||
        (name == prim::kPrimUpdateState->name()) || (name == prim::kPrimMakeTuple->name()) ||
        (name == prim::kPrimReturn->name())) {
      return false;
    }
  }
  return true;
}

OperatorPtr DfGraphConvertor::ToOperatorPtr(const AnfNodePtr &node) {
  auto op = Convert(GetRealOpNode(node));
  if (op == nullptr) {
    MS_LOG(ERROR) << "Convert real op node to operator failed, " << node->ToString();
    error_ = FAILED;
    return nullptr;
  }
  return op;
}

void DfGraphConvertor::AddEdgeToCache(const AnfNodePtr &src, const AnfNodePtr &dest) {
  auto item = monad_control_edge_cache_.find(src);
  if (item == monad_control_edge_cache_.end()) {
    monad_control_edge_cache_[src] = std::vector<AnfNodePtr>{dest};
  } else {
    (void)item->second.emplace_back(dest);
  }
}

void DfGraphConvertor::AddEdgeForLoad(const AnfNodePtr &node) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(node) == manager->node_users().end()) {
    MS_LOG(EXCEPTION) << "Can't find node in nodes_users.";
  }
  auto &users = manager->node_users()[node];
  std::shared_ptr<std::vector<AnfNodePtr>> src_node_list = std::make_shared<std::vector<AnfNodePtr>>();
  std::shared_ptr<std::vector<AnfNodePtr>> dst_node_list = std::make_shared<std::vector<AnfNodePtr>>();
  for (const auto &iter : users) {
    auto user_node = iter.first;
    auto name = GetCNodeTargetFuncName(user_node->cast<CNodePtr>());
    if (name == prim::kPrimUpdateState->name()) {
      FindDestOps(user_node, dst_node_list, false);
      continue;
    }
    if (IsControlEdgeNode(user_node)) {
      src_node_list->push_back(user_node);
      continue;
    }
    FindDestOps(user_node, src_node_list, false);
  }

  // add to cache
  for (auto &dest : *dst_node_list) {
    for (auto &src : *src_node_list) {
      AddEdgeToCache(src, dest);
    }
  }
}

void DfGraphConvertor::FindDestOps(const AnfNodePtr &node, const std::shared_ptr<std::vector<AnfNodePtr>> &node_list,
                                   bool top) {
  MS_EXCEPTION_IF_NULL(node);
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto users = manager->node_users()[node];
  for (const auto &iter : users) {
    auto user_node = iter.first;
    if (IsSubGraph() && user_node == call_node_in_while_body_) {
      continue;
    }
    if (IsControlEdgeNode(user_node)) {
      if (!top) {
        node_list->push_back(user_node);
      }
    } else {
      FindDestOps(user_node, node_list, false);
    }
  }
}

void DfGraphConvertor::AutoMonadCollectInput(const AnfNodePtr &node) {
  if (!IsSourceEdgeNode(node)) {
    return;
  }

  // Add control edge if contain monad input.
  std::string name = GetCNodeTargetFuncName(node->cast<CNodePtr>());
  if (name == prim::kPrimLoad->name()) {
    AddEdgeForLoad(node);
  } else {
    auto src_ops = ToOperatorPtr(node);
    if (src_ops != nullptr) {
      // Find dest ops list
      std::shared_ptr<std::vector<AnfNodePtr>> dst_node_list = std::make_shared<std::vector<AnfNodePtr>>();
      FindDestOps(node, dst_node_list, true);
      for (auto &dest : *dst_node_list) {
        AddEdgeToCache(node, dest);
      }
    }
  }
}

void DfGraphConvertor::AutoMonadSetInput(const AnfNodePtr &node) {
  if (monad_control_edge_cache_.find(node) == monad_control_edge_cache_.end()) {
    return;
  }

  auto src_ops = ToOperatorPtr(node);
  if (src_ops != nullptr) {
    for (auto &dest : monad_control_edge_cache_[node]) {
      auto dest_ops = ToOperatorPtr(dest);
      if (dest_ops == nullptr) {
        continue;
      }
      (void)dest_ops->AddControlInput(*src_ops);
#ifdef DRAW_GE_GRAPH
      compute_sout_ << op_draw_name_[node.get()] << " -> " << op_draw_name_[dest.get()] << "[style=\"dotted\"]" << endl;
#endif
    }
  }
}

void DfGraphConvertor::AutoMonadSetControlInput(const AnfNodePtr &node) {
  AutoMonadCollectInput(node);
  AutoMonadSetInput(node);
}

void DfGraphConvertor::SetOpControlInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  AutoMonadSetControlInput(node);
  if (control_edge_cache_.find(node.get()) == control_edge_cache_.end()) {
    return;
  }

  std::vector<ControlEdge> control_edges = control_edge_cache_[node.get()];
  if ((control_edges.empty())) {
    MS_LOG(ERROR) << "Get control edge node's src or dest operator failed";
    return;
  }

  for (auto &item : control_edges) {
    (void)item.dest_op->AddControlInput(*item.src_op);
  }
}

const std::vector<std::string> trans_var_list = {string(kNameAssign), string(kNameAssignAdd), string(kNameAssignSub)};

AnfNodePtr DfGraphConvertor::ParseLoadInput(const CNodePtr &cnode) {
  if (cnode->inputs().size() < 3) {
    MS_LOG(EXCEPTION) << "input size error, " << cnode->ToString();
  }
  const size_t para_index = 1;
  return cnode->input(para_index);
}

void DfGraphConvertor::SetTupleOpInput(const OpAdapterPtr &adpt, const CNodePtr &node, const AnfNodePtr &pred,
                                       const OperatorPtr &src, int index) {
  std::shared_ptr<std::vector<OutHandler>> handler_vec = tuple_out_handle_cache_[pred.get()];
  std::shared_ptr<std::vector<OutHandler>> handler_vec_without_monad = std::make_shared<std::vector<OutHandler>>();
  bool with_monad = false;
  for (auto &handler : *handler_vec) {
    // when tuple with monad type element, the handler operator is nullptr, should be ignored.
    if (handler.op == nullptr) {
      if ((handler.node != nullptr) && !HasAbstractMonad(handler.node)) {
        MS_LOG(WARNING) << "Unsupported node in tuple : " << node->ToString();
      }
      continue;
    }
    with_monad = true;
    handler_vec_without_monad->push_back(handler);
  }
  int ret = adpt->setInput(src, index, handler_vec_without_monad);
  if ((ret == 0) && pred->isa<CNode>() && (pred->cast<CNodePtr>()->inputs().size() == handler_vec->size() + 1)) {
    for (unsigned int j = 0; j < handler_vec_without_monad->size(); j++) {
      AnfNodePtr input_node = pred->cast<CNodePtr>()->input(j + 1);
      if (with_monad) {
        input_node = handler_vec_without_monad->at(j).node;
      }
      compute_sout_ << op_draw_name_[input_node.get()] << " -> " << op_draw_name_[node.get()] << ":" << index << endl;
      AddGraphConstInput(handler_vec_without_monad->at(j).op);
    }
    return;
  }
  MS_LOG(WARNING) << "This anf node is not supported as a tuple item : " << node->ToString();
}

AnfNodePtr DfGraphConvertor::TransformConstOp(const CNodePtr &node, AnfNodePtr pred) {
  // transform "Const" op to "Variable" op when the next node is "Assign" op.
  std::string c_name = GetCNodeTargetFuncName(node);
  auto pos = std::find(trans_var_list.begin(), trans_var_list.end(), c_name);
  if (!training_ && !IsSubGraph() && pos != trans_var_list.end() && pred->isa<Parameter>()) {
    std::string name = std::static_pointer_cast<Parameter>(pred)->name();
    auto op_itor = op_cache_.find(pred.get());
    if (op_itor == op_cache_.end()) {
      MS_LOG(EXCEPTION) << "Can not find op for node " << pred->ToString() << ".";
      return nullptr;
    }
    if (op_itor->second != nullptr &&
        (op_itor->second->GetOpType() == "Constant" || op_itor->second->GetOpType() == "Const") &&
        vars_.find(name) != vars_.end()) {
      auto variable = std::make_shared<Variable>(name);
      auto desc = vars_[name]->GetOutputDesc("y");
      (void)variable->update_output_desc_y(desc);
      MS_LOG(DEBUG) << "Trans to variable, var = " << variable->GetName() << ".";
      op_itor->second = variable;  // replace parameter with variable
      vars_[name] = variable;
    }
  }
  return pred;
}

AnfNodePtr DfGraphConvertor::GetRealInputNode(const CNodePtr &node, const AnfNodePtr &input) {
  if (input == nullptr || node == nullptr) {
    return nullptr;
  }
  AnfNodePtr pred = input;
  while (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == prim::kPrimDepend->name()) {
    pred = pred->cast<CNodePtr>()->input(1);
  }

  // skip input of UMonad, IOMonad
  if (IsValueNode<UMonad>(pred) || IsValueNode<IOMonad>(pred)) {
    return nullptr;
  }
  if (HasAbstractMonad(pred)) {
    return nullptr;
  }

  // skip input of the None, UpdateState
  if (IsValueNode<None>(pred) || IsPrimitiveCNode(pred, prim::kPrimUpdateState)) {
    return nullptr;
  }

  if (IsPrimitiveCNode(pred, prim::kPrimLoad)) {
    pred = ParseLoadInput(pred->cast<CNodePtr>());
    // for scenario like: Depend->Load->TensorMove
    if (IsPrimitiveCNode(pred, prim::kPrimDepend)) {
      return GetRealInputNode(node, pred);
    }
  }

  return TransformConstOp(node, pred);
}

OutHandler DfGraphConvertor::GetNormalOpInput(const AnfNodePtr &pred) {
  OutHandler out_handler;
  if (IsSubGraph() && pred->isa<Parameter>()) {
    auto idx = std::find(inputs_.begin(), inputs_.end(), pred) - inputs_.begin();
    OperatorPtr op = subgraph_input_cache_[idx];
    out_handler.op = op;
  } else if (IsAfterGraph() && pred->isa<Parameter>()) {
    auto idx = std::find(inputs_.begin(), inputs_.end(), pred) - inputs_.begin();
    auto idx_cond = prev_after_cond_map_[idx];
    if (bypass_node_prev_handle_cache_.find(idx_cond) != bypass_node_prev_handle_cache_.end()) {
      out_handler = bypass_node_prev_handle_cache_[idx_cond];
    } else {
      auto idx_out = prev_cond_to_while_out_index_[idx_cond];
      out_handler = while_output_handle_cache_[prev_while_node_]->at(idx_out);
    }
  } else {
    OperatorPtr op = Convert(pred);
    out_handler.op = op;
  }
  return out_handler;
}

void DfGraphConvertor::DrawOpInput(const AnfNodePtr &node, const AnfNodePtr &pred, size_t i) {
  if (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == prim::kTupleGetItem) {
    compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(1).get()] << " -> " << op_draw_name_[node.get()] << ":"
                  << i << endl;
  } else if (pred->isa<Parameter>()) {
    compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
  } else {
    // don't draw anything.
    MS_LOG(INFO) << "DRAW_GE_GRAPH: Shouldn't have this case.";
  }
  return;
}

void DfGraphConvertor::SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node) {
  OperatorPtr src = Convert(node);
  int case_flag = 0;
  auto &inputs = node->inputs();
  size_t input_size = inputs.size();
  if (case_input_handle_cache_.find(node.get()) != case_input_handle_cache_.end()) {
    case_flag = 1;
    input_size = case_input_handle_cache_[node.get()]->size() + 1;
  } else if (!IsSubGraph() && call_input_handle_cache_.find(node) != call_input_handle_cache_.end()) {
    auto &handles = call_input_handle_cache_[node];
    MS_LOG(DEBUG) << "call node input size: " << handles->size();
    adpt->setInput(src, 1, handles);
    return;
  }
  MS_LOG(DEBUG) << "op:  " << src->GetName() << "'s input size is " << input_size - 1;

  for (size_t i = 1; i < input_size; i++) {
    AnfNodePtr pred = (case_flag != 0) ? case_input_handle_cache_[node.get()]->at(i - 1) : inputs[i];
    pred = GetRealInputNode(node, pred);
    if (pred == nullptr || HasAbstractMonad(pred)) {
      continue;
    }

    int index = SizeToInt(i);
    // find in out_handle_cache_ first
    auto it = out_handle_cache_.find(pred.get());
    if (it != out_handle_cache_.end()) {
      int ret = adpt->setInput(src, index, it->second);
      if (ret == 0) {
        DrawOpInput(node, pred, i);
        AddGraphConstInput(it->second.op);
      }
    } else if (tuple_out_handle_cache_.find(pred.get()) != tuple_out_handle_cache_.end()) {
      SetTupleOpInput(adpt, node, pred, src, index);
    } else {
      OutHandler handler = GetNormalOpInput(pred);
      if (handler.op != nullptr) {
        bool is_pred_handler = IsAfterGraph() && pred->isa<Parameter>();
        int ret = is_pred_handler ? adpt->setInput(src, index, handler) : adpt->setInput(src, index, handler.op);
        if (ret == 0 && !is_pred_handler) {
          compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << index << endl;
          AddGraphConstInput(handler.op);
        }
      } else if (tuple_out_handle_cache_.find(pred.get()) != tuple_out_handle_cache_.end()) {
        SetTupleOpInput(adpt, node, pred, src, index);
      }
    }
  }
  return;
}

void DfGraphConvertor::AddGraphConstInput(const OperatorPtr &op) {
  if (IsSubGraph()) {
    return;
  }

  if (op->GetOpType() == "Constant" || op->GetOpType() == "Const") {
    graph_const_inputs_.push_back(op);
  }
}

void DfGraphConvertor::SetNodeInput(const AnfNodePtr node) {
  if (!node->isa<CNode>()) {
    return;
  }
  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  OpAdapterPtr adpt = FindAdapter(cnode, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_, use adapter to set Inputs
  DfGraphConvertor::SetOpInput(adpt, cnode);
}

void DfGraphConvertor::ProcessSubgraph(const AnfNodePtr &node, const std::vector<AnfNodePtr> &inputs) {
  ValueNodePtr graph_node = nullptr;
  if (node->isa<CNode>()) {
    graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  } else if (node->isa<ValueNode>()) {
    graph_node = node->cast<ValueNodePtr>();
  } else {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph_node);
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  DfGraphConvertor converter(anf_graph);
  converter.use_inputs_ = true;
  converter.inputs_ = inputs;
  if (IsSubGraph()) {
    converter.case_call_input_size_ = case_call_input_size_;
    auto &params = anf_graph->parameters();
    for (size_t i = case_call_input_size_; i < inputs.size(); i++) {
      auto p = inputs[i];
      if (HasAbstractMonad(p)) {
        continue;
      }
      if (p->isa<Parameter>()) {
        auto idx = std::find(inputs_.begin(), inputs_.end(), p) - inputs_.begin();
        auto idx_cond = body_cond_map_[idx];
        if (while_const_input_index_.find(idx_cond) != while_const_input_index_.end()) {
          auto temp = while_const_input_index_[idx_cond].op;
          auto name = temp->GetName();
          auto value = const_op_to_value_[temp];
          MS_EXCEPTION_IF_NULL(value);
          auto adpt_const = FindAdapter(kNameConst, training_);
          if (adpt_const == nullptr) continue;
          name += "_case";
          auto const_op = adpt_const->generate(name);
          (void)adpt_const->setAttr(const_op, "value", value);
          auto const_op_desc = TransformUtil::GetGeTensorDesc(value->shape_c(), value->data_type(), kOpFormat_NCHW);
          if (const_op_desc == nullptr) {
            MS_LOG(WARNING) << "Create variable " << name << " output descriptor failed!";
            continue;
          }
          (void)std::static_pointer_cast<Constant>(const_op)->update_output_desc_y(*const_op_desc);
          auto n = params.at(i - case_call_input_size_);
          converter.op_cache_[n.get()] = const_op;
          MS_LOG(DEBUG) << "node :" << n->ToString() << " and op: " << const_op->GetName()
                        << " has cached in graph: " << anf_graph->ToString();
        }
      }
    }
  }

  (void)converter.ConvertAllNode().BuildGraph();
#ifdef ENABLE_DUMP_IR
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    converter.DrawComputeGraph(name);
  }
#endif
  branches_map_[node.get()] = *(converter.df_graph_);
  return;
}

// Update GE op's shape and type info
void DfGraphConvertor::UpdateOpDesc(const AnfNodePtr node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return;
  }

  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return;
  }

  // get Operator from op_cache_
  OperatorPtr op = Convert(node);
  adpt->updateOutputDesc(op, node->Shape(), node->Type(), node);

  std::string name = op->GetOpType();
  if (name == prim::kPrimNonZeroWithValueShape->name()) {
    MS_EXCEPTION_IF_NULL(op);
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    if (op_desc == nullptr) {
      return;
    }
    const auto output_desc0 = op_desc->MutableOutputDesc("out_value");
    ge::TensorUtils::SetReuseInput(*output_desc0, true);
    ge::TensorUtils::SetReuseInputIndex(*output_desc0, 0);

    const auto output_desc1 = op_desc->MutableOutputDesc("out_index");
    ge::TensorUtils::SetReuseInput(*output_desc1, true);
    ge::TensorUtils::SetReuseInputIndex(*output_desc1, 1);
  }
}

OperatorPtr DfGraphConvertor::Convert(const AnfNodePtr node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    error_ = NOT_FOUND;
    return nullptr;
  }
  // find in cache
  if (op_cache_.count(node.get())) {
    return op_cache_[node.get()];
  }
  if (IsSubGraph()) {
    node->set_user_data<bool>("subgraph_node", make_shared<bool>(true));
  }

  // do not convert primitive node, Load, UpdateState
  if (IsValueNode<Primitive>(node) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return nullptr;
  }
  // convert a new one
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (IsSubGraph() && IsWhileNode(cnode)) {
      return nullptr;
    }
    if (!IsSubGraph() && IsWhileNode(cnode)) {
      CacheWhileGraph(cnode);
      auto &graphs = while_graph_cache_[cnode];
      GetWhileUsedInputIndex(graphs);
      SetParamIndexMap(graphs);
      cur_while_node_ = cnode;
    }
    return ConvertCNode(cnode);
  }

  if (node->isa<Parameter>() && IsSubGraph()) {
    return nullptr;
  }

  if (node->isa<Parameter>()) {
    return ConvertParameter(node);
  }
  if (node->isa<ValueNode>()) {
    if (IsValueNode<Monad>(node)) {
      return nullptr;
    }
    return ConvertValueNode(node->cast<ValueNodePtr>());
  }

  MS_LOG(ERROR) << "Invalid AnfNode";
  error_ = INVALID_ARGUMENT;
  return nullptr;
}

void DfGraphConvertor::ConvertMakeTuple(const CNodePtr node) {
  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  // convert each tuple item to a OutHandler
  for (size_t i = 1; i < node->inputs().size(); i++) {
    AnfNodePtr item = node->input(i);
    if (IsPrimitiveCNode(item, prim::kPrimLoad)) {
      item = ParseLoadInput(item->cast<CNodePtr>());
    }
    OperatorPtr op = Convert(item);
    if (op != nullptr) {
      (void)tuple_items->emplace_back(OutHandler(op, "", item));
    } else if (out_handle_cache_.find(item.get()) != out_handle_cache_.end()) {
      tuple_items->push_back(out_handle_cache_[item.get()]);
    } else {
      tuple_items->emplace_back(OutHandler(nullptr, "", item));
    }
  }

  MS_LOG(DEBUG) << "ConvertMakeTuple: " << node.get() << " " << tuple_items->size();
  tuple_out_handle_cache_[node.get()] = tuple_items;
  return;
}

void DfGraphConvertor::ConvertTopK(const CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Convert TopK second input's type from int64 to int32.";
  auto value_ptr = node->input(2)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_ptr);
  std::ostringstream ss;
  ss << "op" << value_ptr.get();
  op_draw_name_[value_ptr.get()] = ss.str();
  compute_sout_ << ss.str() << "[label= \"" << value_ptr->value()->ToString() << "\" shape=ellipse]" << endl;
  auto input_value = value_ptr->value();
  auto int64_value = GetValue<int64_t>(input_value);
  OpAdapterPtr adpt = FindAdapter(value_ptr, training_);
  auto op = adpt->generate(value_ptr);
  (void)adpt->setAttr(op, "value", static_cast<int32_t>(int64_value));
  op_cache_[value_ptr.get()] = op;
}

void DfGraphConvertor::ConvertResizeBilinear(const FuncGraphPtr anf_graph) {
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph);
  for (auto &it : nodes) {
    if (it->isa<CNode>()) {
      auto node = it->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      std::string name = GetCNodeTargetFuncName(node);
      if (name == prim::kPrimResizeBilinear->name()) {
        AnfNodePtr op = node->input(0);
        if (IsValueNode<Primitive>(op)) {
          auto prim = GetValueNode<PrimitivePtr>(op);
          ValuePtr size_value = prim->GetAttr("size");
          auto int64_value = GetValue<std::vector<int64_t>>(size_value);
          std::vector<int32_t> int32_value;
          (void)std::transform(int64_value.begin(), int64_value.end(), std::back_inserter(int32_value), LongToInt);
          auto valuend = NewValueNode(int32_value);
          valuend->set_abstract(size_value->ToAbstract());
          node->add_input(valuend);
        }
      }
    }
  }
}

AnfNodePtr DfGraphConvertor::CreateCast(const AnfNodePtr &input, const TypePtr &dst_type) const {
  auto func_graph = input->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimCast), input, NewValueNode(dst_type)};
  auto cnode = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dst_type, input->Shape());
  cnode->set_abstract(abs_tensor);
  return cnode;
}

void DfGraphConvertor::ConvertTile(const FuncGraphPtr anf_graph) {
  std::vector<AnfNodePtr> nodes = GetOrderedCNodes(anf_graph);
  for (auto &it : nodes) {
    if (it->isa<CNode>()) {
      auto node = it->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(node);
      std::string name = GetCNodeTargetFuncName(node);
      if (name == prim::kPrimTile->name()) {
        auto type_ptr = node->input(1)->Type();
        MS_EXCEPTION_IF_NULL(type_ptr);
        auto tensor_type = type_ptr->cast<TensorTypePtr>();
        MS_EXCEPTION_IF_NULL(tensor_type);
        if (tensor_type->element()->number_type() == kNumberTypeInt64) {
          auto new_cast = CreateCast(node->input(1), kInt32);
          node->set_input(1, new_cast);
        }
      }
    }
  }
}

std::vector<int64_t> DfGraphConvertor::CastToInt(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(WARNING) << "Value ptr is nullptr.";
    return {};
  }
  std::vector<int64_t> cur_value = {};
  if (utils::isa<ValueSequencePtr>(value)) {
    auto val_seq_ptr = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(val_seq_ptr);
    if (!val_seq_ptr->value().empty()) {
      auto first_val = val_seq_ptr->value().front();
      MS_EXCEPTION_IF_NULL(first_val);
      MS_EXCEPTION_IF_NULL(first_val->type());
      if (first_val->type()->number_type() == kNumberTypeInt64) {
        cur_value = GetValue<std::vector<int64_t>>(value);
      } else {
        auto origin_value = GetValue<std::vector<int>>(value);
        (void)std::transform(origin_value.begin(), origin_value.end(), std::back_inserter(cur_value),
                             [](int index) { return static_cast<int64_t>(index); });
      }
    }
  } else {
    MS_EXCEPTION_IF_NULL(value->type());
    if (value->type()->number_type() == kNumberTypeInt64) {
      cur_value.push_back(GetValue<int64_t>(value));
    } else {
      cur_value.push_back(static_cast<int64_t>(GetValue<int>(value)));
    }
  }
  return cur_value;
}

void DfGraphConvertor::ConvertReshape(const CNodePtr node) {
  MS_LOG(INFO) << "Convert the second input of reshape to op attr.";
  const auto kInputNum = 3;
  if (node->size() < kInputNum) {
    MS_LOG(WARNING) << "Reshape must have two inputs.";
    return;
  }
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  // get shape form attr
  auto value_node = node->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  auto primitive = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(primitive);
  auto value = primitive->GetAttr("shape");
  std::vector<int64_t> list;
  list = CastToInt(value);

  (void)op->SetAttr("shape", list);
  op_cache_[node.get()] = op;
}

void DfGraphConvertor::ConvertConv2D(const CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    return;
  }
  auto op = adpt->generate(node);
  MS_EXCEPTION_IF_NULL(op);
  auto value_node = node->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  auto primitive = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(primitive);
  auto value = primitive->GetAttr("padding");
  if (value != nullptr) {
    std::string pad_mode = GetValue<std::string>(value);
    (void)op->SetAttr("padding", pad_mode);
  }
  op_cache_[node.get()] = op;
}

AnfNodePtr DfGraphConvertor::TraceTupleGetItem(const CNodePtr &node, uint64_t *index) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (node->inputs().size() < 3) {  // "tuple_getitem" primitive must have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of TupleGetItem is less than 3";
  }
  auto index_node = node->inputs()[TUPLE_GET_ITEM_INDEX];
  if (!index_node->isa<ValueNode>()) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(EXCEPTION) << "can't convert get item with non-constant index";
  }
  *index = LongToUlong(GetValue<int64_t>(GetValueNode(index_node)));
  return node->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceDepend(const CNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode->inputs().size() < 3) {  // "Depend" primitive have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of depend is less than 3";
  }
  return cnode->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceMakeTuple(const CNodePtr &node, uint64_t index) {
  if (index + 1 >= node->inputs().size()) {
    MS_LOG(EXCEPTION) << "length of make_tuple is less than index: " << index;
  }
  return node->inputs()[index + 1];
}

OutHandler DfGraphConvertor::GetHandler(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Get nullptr while getting handler from node";
    return OutHandler(nullptr, "");
  }
  auto op = Convert(node);
  if (op != nullptr) {
    auto name = op->GetName();
    if (vars_.count(name) && vars_[name] != nullptr) {
      op = vars_[name];
      MS_LOG(DEBUG) << "update tuple_out_handle_cache_ " << name;
    }
    return OutHandler(op, "", node);
  } else if (out_handle_cache_.find(node.get()) != out_handle_cache_.end()) {
    return out_handle_cache_[node.get()];
  } else {
    MS_LOG(DEBUG) << "Add an empty out handler: " << node->ToString();
    return OutHandler();
  }
}

OutHandler DfGraphConvertor::GetHandler(const AnfNodePtr &node, const std::stack<uint64_t> &index_stack,
                                        AnfNode *const draw_index) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Get nullptr while trace real op";
    return OutHandler(nullptr, "");
  }
  std::ostringstream ss;
  ss << "op" << node.get();
  if (index_stack.empty()) {
    op_draw_name_[draw_index] = ss.str();
    return OutHandler(Convert(node), "");
  } else {
    OpAdapterPtr adpt = FindAdapter(node, training_);
    if (adpt == nullptr) {
      MS_LOG(ERROR) << "Can not get node output as adpt is nullptr!";
      error_ = NOT_FOUND;
      return OutHandler(nullptr, "");
    }
    OperatorPtr op = Convert(node);
    if (op == nullptr) {
      error_ = NOT_FOUND;
      MS_LOG(ERROR) << "Can not convert node for trace real op";
      return OutHandler(nullptr, "");
    }
    op_draw_name_[draw_index] = ss.str();
    return adpt->getOutput(Convert(node), static_cast<int32_t>(index_stack.top()));
  }
}

// get the real operator through maketuple tuple_getitem depend
OutHandler DfGraphConvertor::TraceRealOp(AnfNodePtr node) {
  bool flag = IsPrimitiveCNode(node, prim::kPrimTupleGetItem) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
              IsPrimitiveCNode(node, prim::kPrimDepend);
  std::stack<uint64_t> index_stack;
  auto draw_index = node.get();
  while (flag) {
    flag = false;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      uint64_t index;
      node = TraceTupleGetItem(node->cast<CNodePtr>(), &index);
      index_stack.push(index);
      flag = true;
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      if (index_stack.empty()) {
        MS_LOG(ERROR) << "TraceRealOp find a make_tuple node";
        return OutHandler(nullptr, "");
      } else {
        node = TraceMakeTuple(node->cast<CNodePtr>(), index_stack.top());
        index_stack.pop();
        flag = true;
      }
    } else if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      node = TraceDepend(node->cast<CNodePtr>());
      flag = true;
    }
  }
  return GetHandler(node, index_stack, draw_index);
}

void DfGraphConvertor::ConvertTupleGetItem(const CNodePtr node) {
  auto handle = TraceRealOp(node);
  if (handle.op == nullptr) {
    MS_LOG(ERROR) << "Failed to trace tuple get item";
    return;
  }
  out_handle_cache_[node.get()] = handle;
}

AnfNodePtr DfGraphConvertor::GetRealOpForMakeTuple(const AnfNodePtr &node, const AnfNodePtr &make_tuple,
                                                   int64_t index) {
  MS_EXCEPTION_IF_NULL(make_tuple);
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  auto tuple_inputs = make_tuple_cnode->inputs();
  if (tuple_inputs.size() < LongToSize(index + 1L)) {
    MS_LOG(ERROR) << "Make tuple input items node not correct! size:" << tuple_inputs.size()
                  << ", item index:" << index;
    error_ = FAILED;
    return node;
  }
  return GetRealOpNode(tuple_inputs[LongToSize(index + 1L)]);
}

// Get the real op for tuple_getitem through make tuple, or depend
AnfNodePtr DfGraphConvertor::GetRealOpNode(AnfNodePtr node) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    if (node_inputs.size() != 3) {  // "tuple_getitem" primitive must have 3 inputs
      MS_LOG(ERROR) << "Tuple get item node not correct!";
      error_ = FAILED;
      return node;
    }
    MS_EXCEPTION_IF_NULL(node_inputs[TUPLE_GET_ITEM_INDEX]);
    if (!node_inputs[TUPLE_GET_ITEM_INDEX]->isa<ValueNode>()) {
      error_ = INVALID_ARGUMENT;
      MS_LOG(EXCEPTION) << "Can't convert get item with non-constant index";
    }
    auto value_ptr = GetValueNode(node_inputs[TUPLE_GET_ITEM_INDEX])->cast<Int64ImmPtr>();
    if (value_ptr == nullptr) {
      MS_LOG(ERROR) << "Can not convert get item as value is nullptr!";
      error_ = FAILED;
      return node;
    }
    int64_t index = value_ptr->value();
    // Handle scenario like: MakeTuple->Depend->TupleGetItem
    if (IsPrimitiveCNode(node_inputs[1], prim::kPrimDepend)) {
      auto depend_node = node_inputs[1]->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(depend_node);
      auto depend_inputs = depend_node->inputs();
      if (depend_inputs.size() != 3) {  // "Depend" primitive have 3 inputs
        MS_LOG(ERROR) << "Depend input items not correct";
        error_ = FAILED;
        return node;
      }
      if (IsPrimitiveCNode(depend_inputs[1], prim::kPrimMakeTuple)) {
        return GetRealOpForMakeTuple(depend_node, depend_inputs[1], index);
      }
    }
    // Make_tuple apply inputs:make_tuple, [tuple_items,]
    if (IsPrimitiveCNode(node_inputs[1], prim::kPrimMakeTuple)) {
      return GetRealOpForMakeTuple(node, node_inputs[1], index);
    }
    return GetRealOpNode(node_inputs[1]);
  }

  // Depend apply inputs: depend,output,depended_node
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto depend_inputs = node->cast<CNodePtr>()->inputs();
    if (depend_inputs.size() != 3) {  // "Depend" primitive have 3 inputs
      MS_LOG(ERROR) << "Depend input items not correct";
      error_ = FAILED;
      return node;
    }
    return GetRealOpNode(depend_inputs[1]);
  }
  return node;
}

// convert the anf node to corresponding operator list
std::vector<OperatorPtr> DfGraphConvertor::ConvertDependNode(const AnfNodePtr node) {
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    std::vector<OperatorPtr> op_lists;
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    for (size_t index = 1; index < node_inputs.size(); index++) {
      auto op = Convert(GetRealOpNode(node_inputs[index]));
      if (op == nullptr) {
        MS_LOG(ERROR) << "Convert real op node to operator failed";
        error_ = FAILED;
        return std::vector<OperatorPtr>({});
      }
      op_lists.push_back(op);
    }
    return op_lists;
  }

  auto op = Convert(GetRealOpNode(node));
  if (op == nullptr) {
    MS_LOG(ERROR) << "Convert real op node to operator failed";
    error_ = FAILED;
    return std::vector<OperatorPtr>({});
  }
  return std::vector<OperatorPtr>({op});
}

bool DfGraphConvertor::CheckCNode(const std::string &name, const CNodePtr node) {
  // ignore apply node of return
  if (name == "" || name == prim::kPrimReturn->name() || name == prim::kPrimDepend->name() ||
      name == prim::kPrimSwitchLayer->name() || name == prim::kPrimPartial->name() ||
      name == prim::kPrimSwitch->name()) {
    return false;
  }

  // Convert TopK second input from int64 to int32.
  if (name == prim::kPrimTopK->name()) {
    ConvertTopK(node);
    return true;
  }

  // Convert Reshape add const input to attr(shape)
  if (name == prim::kPrimReshape->name()) {
    ConvertReshape(node);
    return true;
  }

  // Add attr pad mode to Conv2D
  if (name == prim::kPrimConv2D->name() || name == prim::kPrimDepthwiseConv2dNative->name() ||
      name == kNameConv2DBackpropInputV2) {
    ConvertConv2D(node);
    return true;
  }

  // make_tuple is used for a dynamic_input, convert it to a vector of OutHandlers
  if (name == prim::kPrimMakeTuple->name()) {
    ConvertMakeTuple(node);
    return false;
  }

  // As for nodes with multi outputs, convert tuple_getitem to OutHandle
  if (name == prim::kPrimTupleGetItem->name()) {
    ConvertTupleGetItem(node);
    return false;
  }

  return true;
}

OperatorPtr DfGraphConvertor::ConvertCNode(const CNodePtr node) {
  SaveParamFormat(node);
  std::string name = GetCNodeTargetFuncName(node);
  if (!CheckCNode(name, node)) {
    return nullptr;
  }

  // get corresponding OpAdapter
  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return nullptr;
  }

  // get operator
  OperatorPtr op = nullptr;
  auto it_op = op_cache_.find(node.get());
  if (it_op != op_cache_.end()) {
    op = it_op->second;
  } else {
    if (cur_while_node_ == node) {
      op = adpt->generateDynOutputOp(node);
    } else {
      op = adpt->generate(node);
    }
  }

  // set attribute for primitive
  (void)adpt->setAttr(op, node);

  // add into cache
  (void)op_cache_.emplace(node.get(), op);

  DrawCNode(node, adpt);

  return op_cache_[node.get()];
}

OperatorPtr DfGraphConvertor::ConvertParameter(const AnfNodePtr node) {
  // convert Parameter in ANF to variable in DataFlow
  auto adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    MS_LOG(EXCEPTION) << "Can not find adapter for Parameter";
  }
  auto op = adpt->generate(node);
  op_cache_[node.get()] = op;

  // build index for parameter using name
  std::string name = std::static_pointer_cast<Parameter>(node)->name();
  params_[name] = node;
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[shape=octagon, label=\"" << name << "\"]" << endl;
  return op_cache_[node.get()];
}

void DfGraphConvertor::SaveParamFormat(const CNodePtr node) {
  AnfNodePtr op = node->input(0);
  if (IsValueNode<Primitive>(op)) {
    auto prim = GetValueNode<PrimitivePtr>(op);
    for (auto attr : prim->attrs()) {
      if (attr.first == "format") {
        std::string format;
        if (attr.second->isa<Int64Imm>()) {
          bool converted = CheckAndConvertUtils::ConvertAttrValueToString(prim->name(), "format", &attr.second);
          if (converted) {
            format = attr.second->ToString();
          } else {
            CheckAndConvertUtils::GetFormatStringVal(prim, &format);
          }
        } else if (attr.second->isa<StringImm>()) {
          format = attr.second->ToString();
        }
        if (format != "NCDHW" && format != "NHWC") {
          break;
        }
        for (size_t i = 1; i < node->size(); i++) {
          auto input = node->input(i);
          if (input->isa<Parameter>()) {
            param_format_[input->DebugString()] = format;
            MS_LOG(DEBUG) << "Save Param " << input->DebugString() << " format: " << format;
          }
        }
      }
    }
  }
}

Status DfGraphConvertor::TryConvertValueNodeToMultiConst(const ValueNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value = node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueList>() && !value->isa<ValueTuple>()) {
    return FAILED;
  }

  auto vec = value->isa<ValueTuple>() ? value->cast<ValueTuplePtr>()->value() : value->cast<ValueListPtr>()->value();
  if (vec.empty()) {
    return FAILED;
  }

  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  for (size_t i = 0; i < vec.size(); i++) {
    MS_EXCEPTION_IF_NULL(vec[i]);
    if (vec[i]->isa<MeTensor>()) {
      GeTensorPtr ge_tensor = transform::TransformUtil::ConvertTensor(vec[i]->cast<MeTensorPtr>(), kOpFormat_NCHW);
      auto const_op = std::make_shared<Constant>(node->fullname_with_scope() + "/const/inputs/" + std::to_string(i));
      AddGraphConstInput(const_op);
      (void)const_op->set_attr_value(*ge_tensor);
      (void)const_op->update_output_desc_y(ge_tensor->GetTensorDesc());
      (void)tuple_items->emplace_back(OutHandler(const_op, ""));
    } else {
      return FAILED;
    }
  }
  if (tuple_items->empty()) {
    return FAILED;
  }

  tuple_out_handle_cache_[node.get()] = tuple_items;
  return SUCCESS;
}

OperatorPtr DfGraphConvertor::ConvertValueNode(const ValueNodePtr node) {
  // convert valuenode in ANF to Const in DataFlow
  // find paramerte referenced by SymbolicKeyInstance of valuenode
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();
  compute_sout_ << ss.str() << "[label= \"" << node->value()->ToString() << "\" shape=ellipse]" << endl;

  if (TryConvertValueNodeToMultiConst(node) == SUCCESS) {
    MS_LOG(INFO) << "Convert value node to multi Constant OP success";
    return nullptr;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (adpt == nullptr) {
    error_ = NOT_FOUND;
    return nullptr;
  }
  auto op = adpt->generate(node);
  // set const's attrs
  if (adpt->setAttr(op, "value", node->value()) != 0) {
    MS_LOG(WARNING) << "set attr value for const failed";
  }

  auto const_op = std::static_pointer_cast<Constant>(op);
  if (const_op == nullptr) {
    MS_LOG(ERROR) << "Get Constant operator failed";
    return nullptr;
  }
  auto ge_tensor = const_op->get_attr_value();
  auto ge_desc = ge_tensor.GetTensorDesc();
  (void)const_op->update_output_desc_y(ge_desc);

  op_cache_[node.get()] = op;
  return op_cache_[node.get()];
}

void DfGraphConvertor::DrawCNode(const CNodePtr node, const OpAdapterPtr adpt) {
  if (adpt == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Failed to draw apply node as adpt or node is nullptr!";
    return;
  }
  std::ostringstream ss;
  ss << "op" << node.get();
  op_draw_name_[node.get()] = ss.str();

  compute_sout_ << ss.str() << "[label=<";
  compute_sout_ << "<table border='1' cellborder='1'>" << endl;

  auto input_map = adpt->getInputMap();
  auto dyn_input_map = adpt->getDynInputMap();
  if (input_map.size() + dyn_input_map.size() > 0) {
    compute_sout_ << "<tr>";
    for (auto &it : input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    for (auto &it : dyn_input_map) {
      compute_sout_ << "<td port='" << it.first << "'>" << it.second.name << "</td>";
    }
    compute_sout_ << "</tr>" << endl;
  }

  compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << node->ToString()
                << ":" << GetCNodeTargetFuncName(node) << "\"</td></tr>" << endl;

  // print attrs' values
  auto atts = adpt->GetAttrsFromDrawGraph();
  for (auto &it : atts) {
    compute_sout_ << "<tr><td colspan=\"" << (input_map.size() + dyn_input_map.size()) << "\">\"" << it
                  << "\"</td></tr>";
  }

  adpt->clearAttrVect();

  compute_sout_ << "</table>> shape=plaintext]" << endl;
}
void DfGraphConvertor::RegisterAdapter(const std::string &name, OpAdapterPtr adpt) {
  OpAdapterMap::get()[name] = std::make_shared<OpAdapterDesc>(adpt);
}
void DfGraphConvertor::RegisterAdapter(const std::string &name, OpAdapterPtr train_adpt, OpAdapterPtr infer_adpt) {
  OpAdapterMap::get()[name] = std::make_shared<OpAdapterDesc>(train_adpt, infer_adpt);
}
}  // namespace transform
}  // namespace mindspore
