/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/convert.h"

#include <inttypes.h>
#include <algorithm>
#include <stack>
#include "utils/utils.h"

#include "frontend/operator/ops.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "utils/symbolic.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "./common.h"
#include "utils/ms_context.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "ops/state_ops.h"
#include "ops/array_ops.h"
#include "ops/elewise_calculation_ops.h"
#include "ops/math_ops.h"

namespace mindspore {
namespace transform {
using std::endl;

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

// ---------------implement of DfGraphConvertor-------------
PrimType GetCNodeFuncType(const CNodePtr cnode) {
  if (cnode->inputs().empty()) {
    return kPrimTypeUnknown;
  }

  AnfNodePtr valuenode = cnode->input(0);
  if (IsValueNode<Primitive>(valuenode)) {
    // check whether the valuenode is primitive
    return GetValueNode<PrimitivePtr>(valuenode)->prim_type();
  }
  return kPrimTypeUnknown;
}

bool IsCaseNode(const CNodePtr node) {
  if (!node->inputs().empty() && node->input(0)->isa<CNode>() &&
      GetCNodeFuncName(node->input(0)->cast<CNodePtr>()) == "switch_layer") {
    return true;
  }
  return false;
}

std::string GetCNodeTargetFuncName(const CNodePtr cnode) {
  if (IsCaseNode(cnode)) {
    return string(kNameCase);
  }
  auto name = GetCNodeFuncName(cnode);
  if (name == "switch_layer") {
    name = "";
  }
  return name;
}

OpAdapterPtr DfGraphConvertor::FindAdapter(const AnfNodePtr node, bool train) {
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
      value = 1;
      ConfigManager::GetInstance().set_iter_num(value);
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
  std::vector<AnfNodePtr> nodes = TopoSort(anf_graph_->get_return());

  for (auto &it : nodes) {
    if (it->isa<ValueNode>()) {
      if (IsValueNode<SymbolicKeyInstance>(it)) {
        auto symbolic = GetValueNode<SymbolicKeyInstancePtr>(it);
        auto name = std::static_pointer_cast<Parameter>(symbolic->node())->name();
        auto iter = vars_.find(name);  // get correspoding varaible op
        if (iter != vars_.end()) {
          op_cache_[it.get()] = iter->second;
          // #ifdef DRAW_GE_GRAPH
          compute_sout_ << op_draw_name_[params_[name].get()] << " -> " << op_draw_name_[it.get()]
                        << "[style=\"dotted\"]" << endl;
          // #endif
        }
      } else if (IsValueNode<RefKey>(it)) {
        auto refkey = GetValueNode<RefKeyPtr>(it);
        auto name = refkey->tag();
        auto iter = vars_.find(name);  // get correspoding varaible op
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
    out_handle_cache_[it.get()] = OutHandler(dataset_iter_getnext_, "y" + std::to_string(getnext_idx));
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
        MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
        continue;
      }
      (void)std::static_pointer_cast<Constant>(const_op)->update_output_desc_y(*const_op_desc);

      vars_[name] = const_op;
      op_itor->second = const_op;
      continue;
    }

    // create tensor descriptor for output descriptor
    auto desc = TransformUtil::GetGeTensorDesc(it.second->shape_c(), it.second->data_type(), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
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
  if (error_ != 0) {
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

#if (defined ENABLE_GE)
void DfGraphConvertor::BuildSaveCheckpointGraph() {
  std::vector<Operator> graph_inputs;
  ge::op::Save save_op("save_parms");
  int save_op_is_active = 0;
  size_t index = 0;
  string name;

  int32_t count_size = std::count_if(vars_.begin(), vars_.end(), [](const std::pair<std::string, OperatorPtr> &it) {
    return (it.second == nullptr || it.first.find("/") != std::string::npos);
  });

  (void)save_op.create_dynamic_input_tensors(vars_.size() - static_cast<size_t>(count_size));

  // for each "parameter" in anf graph excluding "input"
  for (const auto &it : vars_) {
    name = it.first;
    if (it.second == nullptr || name.find("/") != std::string::npos) continue;
    Variable variable(name);
    (void)variable.update_output_desc_y(it.second->GetOutputDesc(0));
    (void)save_op.set_dynamic_input_tensors(index++, variable);

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
  if (error_ != 0) {
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
          MS_LOG(ERROR) << "Create variable " << name << " ouptut descriptor failed!";
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
  if (error_ != 0) {
    MS_LOG(ERROR) << "Generate checkpoint graph failed, found error code " << error_ << ".";
    return *this;
  }
  if (anf_graph_ == nullptr || anf_graph_->output() == nullptr) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(ERROR) << "Invalid AnfGraph in GenerateCheckpointGraph";
    return *this;
  }
#if (defined ENABLE_GE)
  BuildSaveCheckpointGraph();
  // Restoring from checkpoint file is done by pyfront, not in graph now.
#endif
  return *this;
}

DfGraphConvertor &DfGraphConvertor::ConvertAllNode() {
  if (error_ != 0) {
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
  checkpoint_sout_.clear();
  checkpoint_sout_ << "digraph {" << endl;
  restore_checkpoint_sout_.clear();
  restore_checkpoint_sout_ << "digraph {" << endl;

  // Convert all anf node to Operator
  MS_LOG(DEBUG) << "convert all node";
  std::vector<AnfNodePtr> nodes = TopoSort(anf_graph_->get_return());
  for (auto &it : nodes) {
    (void)Convert(it);
    if (this->error_ != 0) {
      MS_LOG(ERROR) << "failed to convert node: " << it->DebugString() << ".";
    }
  }

  // Create dataset iterator and iterator_getnext node
  if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE) {
    DatasetGraphParam param = ConfigManager::GetInstance().dataset_param();
    MS_LOG(INFO) << "Dataset param is " << param.ToString() << ".";
    // GetNext
    auto iter_getnext_op = make_shared<ge::op::GetNext>("get_next_tmp");
    (void)iter_getnext_op->set_attr_output_types(param.ge_types());
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
      graph_outputs_.emplace_back(std::make_pair(*op, handle.out));
    } else {
      MS_LOG(EXCEPTION) << "tuple_getitem: " << anf_out->fullname_with_scope() << " is not converted";
    }
  } else {
    // invalid tuple_getitem e.g. tuple_getitem(tuple_getitem())/tuple_getitem(depend())/tuple_getitem(make_tuple())
    MS_LOG(WARNING) << "Invalid tuple_getitem: " << anf_out->fullname_with_scope();
  }
}

void DfGraphConvertor::TraceOutput(const AnfNodePtr node) {
  AnfNodePtr anf_out = node;
  AnfNodePtr pre_node = nullptr;

  // trace Parameter node
  TraceOutputFromParameter(anf_out);
  // then trace cnode
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

  if (name == "make_tuple") {
    for (unsigned int i = 1; i < c->inputs().size(); i++) {
      TraceOutput(c->input(i));
    }
  } else if (name == "Depend") {
    if (c->inputs().size() < 3) {  // "Depend" primitive have 3 inputs
      MS_LOG(EXCEPTION) << "length of inputs is " << c->inputs().size() << ", which is less than 3";
    }
    TraceOutput(c->input(1));
  } else if (name == "tuple_getitem") {
    TraceOutputFromTupleGetItem(anf_out);
  } else {
    // add outputs;
    auto op = Convert(anf_out);
    std::string index;
    if (op != nullptr) {
      if ((pre_node != nullptr) && IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
        auto item = out_handle_cache_.find(pre_node.get());
        if (item != out_handle_cache_.end()) {
          index = item->second.out;
        } else {
          MS_LOG(WARNING) << "Can't get operater: " << anf_out->fullname_with_scope() << " 's output item";
        }
      }
      MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope() << ":" << index;
      graph_outputs_.emplace_back(make_pair(*op, index));
    }
  }
}

void DfGraphConvertor::TraceOutputFromParameter(const AnfNodePtr &anf_out) {
  if (anf_out->isa<Parameter>()) {
    MS_LOG(INFO) << "Add graph output: " << anf_out->fullname_with_scope();
    auto it = out_handle_cache_.find(anf_out.get());
    if (it != out_handle_cache_.end()) {
      // For dataset graph mode, input parameter is converted to a "iterator_get_next:yn" OutHandler.
      OutHandler handle = it->second;
      auto op = handle.op;
      MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType() << ", out_name: " << handle.out;
      graph_outputs_.emplace_back(make_pair(*op, handle.out));
    } else {
      // common parameter case
      auto op = Convert(anf_out);
      if (op != nullptr) {
        MS_LOG(INFO) << "op name: " << op->GetName() << ", op type: " << op->GetOpType();
        graph_outputs_.emplace_back(std::make_pair(*op, ""));
      }
    }
  }
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

void DfGraphConvertor::SetSubgraph(AnfNodePtr node) {
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!IsCaseNode(cnode)) {
    return;
  }
  std::vector<AnfNodePtr> case_inputs;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    case_inputs.emplace_back(cnode->input(i));
  }
  std::shared_ptr<std::vector<DfGraph>> branches = std::make_shared<std::vector<DfGraph>>();
  auto bnode = cnode->input(0)->cast<CNodePtr>()->input(2)->cast<CNodePtr>();

  for (size_t i = 1; i < bnode->inputs().size(); i++) {
    auto branch_node = bnode->input(i)->cast<CNodePtr>();
    for (size_t j = 2; j < branch_node->inputs().size(); j++) {
      if (std::find(case_inputs.begin(), case_inputs.end(), branch_node->input(j)) == case_inputs.end()) {
        case_inputs.emplace_back(branch_node->input(j));
      }
    }
  }

  for (size_t i = 1; i < bnode->inputs().size(); i++) {
    ProcessSubgraph(bnode->input(i), case_inputs);
  }

  for (size_t i = 1; i < bnode->inputs().size(); i++) {
    branches->emplace_back(branches_map_[bnode->input(i).get()]);
  }

  if (op_cache_.find(node.get()) == op_cache_.end()) {
    return;
  }

  OpAdapterPtr adpt = FindAdapter(node, training_);
  if (nullptr == adpt) {
    MS_LOG(DEBUG) << "Not found adapter";
    return;
  }

  OperatorPtr op = Convert(node);
  adpt->setSubgraph(op, 0, branches);
  return;
}

void DfGraphConvertor::GetCaseNodeInput(const CNodePtr node, const CNodePtr input_node) {
  std::vector<AnfNodePtr> case_inputs;
  for (size_t i = 1; i < node->inputs().size(); i++) {
    case_inputs.emplace_back(node->input(i));
  }
  std::shared_ptr<std::vector<DfGraph>> branches = std::make_shared<std::vector<DfGraph>>();
  auto bnode = input_node->input(2)->cast<CNodePtr>();

  for (size_t i = 1; i < bnode->inputs().size(); i++) {
    auto branch_node = bnode->input(i)->cast<CNodePtr>();
    for (size_t j = 2; j < branch_node->inputs().size(); j++) {
      if (std::find(case_inputs.begin(), case_inputs.end(), branch_node->input(j)) == case_inputs.end()) {
        case_inputs.emplace_back(branch_node->input(j));
      }
    }
  }

  const size_t case_index = 1;
  const size_t make_tuple_index = 2;

  AnfNodePtr case_index_iter = input_node->input(case_index);
  AnfNodePtr make_tuple_iter = input_node->input(make_tuple_index);
  auto make_tuple_node = make_tuple_iter->cast<CNodePtr>();
  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();

  for (size_t i = 0; i < case_inputs.size(); i++) {
    auto item = case_inputs[i];
    auto op = Convert(item);
    if (op != nullptr) {
      tuple_items->emplace_back(OutHandler(op, ""));
    } else if (out_handle_cache_.find(item.get()) != out_handle_cache_.end()) {
      tuple_items->push_back(out_handle_cache_[item.get()]);
    } else {
      MS_LOG(WARNING) << "This anf node is not supported as a case input: " << item->ToString();
      continue;
    }
  }

  tuple_out_handle_cache_[make_tuple_node.get()] = tuple_items;

  std::shared_ptr<std::vector<AnfNodePtr>> case_input_items = std::make_shared<std::vector<AnfNodePtr>>();
  case_input_items->emplace_back(case_index_iter);
  case_input_items->emplace_back(make_tuple_iter);
  case_input_handle_cache_[node.get()] = case_input_items;
}

DfGraphConvertor &DfGraphConvertor::BuildGraph() {
  SetupDatasetIterGetNextNode(dataset_iter_getnext_);

  if (error_ != 0) {
    return *this;
  }

  // Case node set input.
  std::vector<AnfNodePtr> nodes = ::mindspore::TopoSort(anf_graph_->get_return());
  for (auto &it : nodes) {
    if (it->isa<CNode>() && IsCaseNode(it->cast<CNodePtr>())) {
      auto node = it->cast<CNodePtr>();
      auto input_node = node->input(0)->cast<CNodePtr>();
      GetCaseNodeInput(node, input_node);
    }
  }

  // update tuple_out_handle_cache_
  for (auto it : tuple_out_handle_cache_) {
    std::size_t len = it.second->size();
    for (std::size_t i = 0; i < len; i++) {
      OutHandler handle = (*it.second)[i];
      if (handle.op) {
        string name = handle.op->GetName();
        if (vars_.count(name)) {
          OperatorPtr new_op = vars_[name];
          if (new_op != nullptr) {
            MS_LOG(INFO) << "update tuple_out_handle_cache_ " << name;
            (*it.second)[i] = OutHandler(new_op, handle.out);
          }
        }
      }
    }
  }

  // set up dependices
  MS_LOG(DEBUG) << "set up dependices";
  nodes = ::mindspore::TopoSort(anf_graph_->get_return());
  for (auto &it : nodes) {
    SetNodeInput(it);
    SetOpControlInput(it);
    SetSubgraph(it);
    UpdateOpDesc(it);
  }

  if (error_ == 0) {
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
          if (params[i]->ToString() == anf_params[j]->ToString()) {
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
        inputs.push_back(*op);
      }
    }
  }

  // Add const nodes as graph input for some operator work with constant
  std::transform(graph_const_inputs_.begin(), graph_const_inputs_.end(), std::back_inserter(inputs),
                 [](OperatorPtr x) { return *x; });

  MS_LOG(INFO) << "set graph input num: " << inputs.size();
  (void)df_graph_->SetInputs(inputs);

  // set graph output
  // set the value of finale return apply node as the output of dataflow graph
  MS_LOG(DEBUG) << "set output";
  graph_outputs_.clear();
  TraceOutput(anf_graph_->get_return()->input(1));
  MS_LOG(INFO) << "set graph output num: " << graph_outputs_.size();
  (void)df_graph_->SetOutputs(graph_outputs_);

  compute_sout_ << "}" << endl;
  // For the graph(e.g. eval_subgraph) whose IterNum is 1, donot set NeedIteration flag.
  if (ConfigManager::GetInstance().iter_num() > 1) {
    df_graph_->SetNeedIteration(true);
  }
  return *this;
}

void DfGraphConvertor::UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const {
  auto node = std::static_pointer_cast<AnfNode>(it);
  if (node == nullptr) {
    MS_LOG(ERROR) << "Update data op descriptor failed! Invalid node.";
    return;
  }
  auto normal_shape_ptr = dyn_cast<abstract::Shape>(node->Shape());
  vector<int> shape;
  if (normal_shape_ptr == nullptr) {
    MS_LOG(INFO) << "Invalid shape to update data op descriptor.";
    return;
  }
  shape = normal_shape_ptr->shape();
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
  auto desc = TransformUtil::GetGeTensorDesc(shape, me_type, "NCHW");
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

void DfGraphConvertor::SetOpControlInput(const AnfNodePtr node) {
  if (control_depend_cache_.find(node.get()) == control_depend_cache_.end()) {
    return;
  }

  std::vector<ControlEdge> control_edges = control_depend_cache_[node.get()];
  if ((control_edges.empty())) {
    MS_LOG(ERROR) << "Get control depend node's src or dest operator failed";
    return;
  }

  for (auto &item : control_edges) {
    (void)item.dest_op->AddControlInput(*item.src_op);
  }
}

const std::vector<std::string> trans_var_list = {string(kNameAssign), string(kNameAssignAdd), string(kNameAssignSub)};

void DfGraphConvertor::SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node) {
  OperatorPtr src = Convert(node);
  int case_flag = 0;
  auto &inputs = node->inputs();
  size_t input_size = inputs.size();
  if (case_input_handle_cache_.find(node.get()) != case_input_handle_cache_.end()) {
    case_flag = 1;
    input_size = case_input_handle_cache_[node.get()]->size() + 1;
  }

  for (size_t i = 1; i < input_size; i++) {
    AnfNodePtr pred = nullptr;
    if (case_flag != 0) {
      pred = case_input_handle_cache_[node.get()]->at(i - 1);
    } else {
      pred = inputs[i];
    }

    while (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == "Depend") {
      pred = pred->cast<CNodePtr>()->input(1);
    }
    // skip the None input
    if (IsValueNode<None>(pred)) {
      continue;
    }
    // transform "Const" op to "Variable" op when the next node is "Assign" op.
    std::string c_name = GetCNodeTargetFuncName(node);
    auto pos = std::find(trans_var_list.begin(), trans_var_list.end(), c_name);
    if (!training_ && pos != trans_var_list.end() && pred->isa<Parameter>()) {
      std::string name = std::static_pointer_cast<Parameter>(pred)->name();
      auto op_itor = op_cache_.find(pred.get());
      if (op_itor == op_cache_.end()) {
        MS_LOG(EXCEPTION) << "Can not find op for node " << pred->ToString() << ".";
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
    // find in out_hadnle_cache_ first
    auto it = out_handle_cache_.find(pred.get());
    if (it != out_handle_cache_.end()) {
      int ret = adpt->setInput(src, SizeToInt(i), it->second);
      if (ret == 0) {
        if (pred->isa<CNode>() && GetCNodeTargetFuncName(pred->cast<CNodePtr>()) == "tuple_getitem") {
          compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(1).get()] << " -> " << op_draw_name_[node.get()]
                        << ":" << i << endl;
        } else if (pred->isa<Parameter>()) {
          compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
        } else {
          // don't draw anything.
          MS_LOG(INFO) << "DRAW_GE_GRAPH: Shouldn't have this case.";
        }
        AddGraphConstInput(it->second.op);
      }
    } else if (tuple_out_handle_cache_.find(pred.get()) != tuple_out_handle_cache_.end()) {
      std::shared_ptr<std::vector<OutHandler>> handler_vec = tuple_out_handle_cache_[pred.get()];
      int ret = adpt->setInput(src, SizeToInt(i), handler_vec);
      if ((ret == 0) && pred->isa<CNode>() && (pred->cast<CNodePtr>()->inputs().size() == handler_vec->size() + 1)) {
        for (unsigned int j = 0; j < handler_vec->size(); j++) {
          compute_sout_ << op_draw_name_[pred->cast<CNodePtr>()->input(j + 1).get()] << " -> "
                        << op_draw_name_[node.get()] << ":" << i << endl;
          AddGraphConstInput(handler_vec->at(j).op);
        }
      } else {
        MS_LOG(WARNING) << "Convert tuple node setInput failed : " << node->ToString();
      }
    } else {
      auto op = Convert(pred);
      int ret = adpt->setInput(src, SizeToInt(i), op);
      if (ret == 0) {
        compute_sout_ << op_draw_name_[pred.get()] << " -> " << op_draw_name_[node.get()] << ":" << i << endl;
        AddGraphConstInput(op);
      }
    }
  }
}

void DfGraphConvertor::AddGraphConstInput(const OperatorPtr &op) {
  if (op->GetOpType() == "Constant") {
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

void DfGraphConvertor::ProcessSubgraph(AnfNodePtr node, const std::vector<AnfNodePtr> &inputs) {
  if (!node->isa<CNode>() || GetCNodeFuncName(node->cast<CNodePtr>()) != "Partial") {
    return;
  }
  auto graph_node = node->cast<CNodePtr>()->input(1)->cast<ValueNodePtr>();
  FuncGraphPtr anf_graph = graph_node->value()->cast<FuncGraphPtr>();
  DfGraphConvertor convertor(anf_graph);
  convertor.use_inputs_ = true;
  convertor.inputs_ = inputs;
  (void)convertor.ConvertAllNode().BuildGraph();
  std::string name = graph_node->ToString() + "_ge_graph.dot";
  if (MsContext::GetInstance()->save_graphs_flag()) {
    convertor.DrawComputeGraph(name);
  }
  branches_map_[node.get()] = *(convertor.df_graph_);
}

// Update GE op's shape and type info
void DfGraphConvertor::UpdateOpDesc(const AnfNodePtr node) {
  if (nullptr == node || !node->isa<CNode>()) {
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

  // do not convert primitive node
  if (IsValueNode<Primitive>(node)) {
    return nullptr;
  }

  // convert a new one
  if (node->isa<CNode>()) {
    return ConvertCNode(node->cast<CNodePtr>());
  }
  if (node->isa<Parameter>()) {
    return ConvertParameter(node);
  }
  if (node->isa<ValueNode>()) {
    return ConvertValueNode(node->cast<ValueNodePtr>());
  }

  MS_LOG(ERROR) << "Invalide AnfNode";
  error_ = INVALID_ARGUMENT;
  return nullptr;
}

void DfGraphConvertor::ConvertMakeTuple(const CNodePtr node) {
  std::shared_ptr<std::vector<OutHandler>> tuple_items = std::make_shared<std::vector<OutHandler>>();
  // convert each tuple item to a OutHandler
  for (size_t i = 1; i < node->inputs().size(); i++) {
    AnfNodePtr item = node->input(i);
    OperatorPtr op = Convert(item);
    if (op != nullptr) {
      tuple_items->emplace_back(OutHandler(op, ""));
    } else if (out_handle_cache_.find(item.get()) != out_handle_cache_.end()) {
      tuple_items->push_back(out_handle_cache_[item.get()]);
    } else {
      MS_LOG(WARNING) << "This anf node is not supported as a tuple item : " << item->ToString();
      return;
    }
  }

  MS_LOG(WARNING) << "ConvertMakeTuple: " << node.get() << " " << tuple_items->size();
  tuple_out_handle_cache_[node.get()] = tuple_items;
}

AnfNodePtr DfGraphConvertor::TraceTupleGetItem(const CNodePtr &node, unsigned int *index) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (node->inputs().size() < 3) {  // "tuple_getitem" primitive must have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of TupleGetItem is less than 3";
  }
  auto index_node = node->inputs()[TUPLE_GET_ITEM_INDEX];
  if (!index_node->isa<ValueNode>()) {
    error_ = INVALID_ARGUMENT;
    MS_LOG(EXCEPTION) << "can't convert get item with non-constant index";
  }
  *index = IntToUint(GetValue<int>(GetValueNode(index_node)));
  return node->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceDepend(const CNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode->inputs().size() < 3) {  // "Depend" primitive have 3 inputs
    MS_LOG(EXCEPTION) << "length of inputs of depend is less than 3";
  }
  return cnode->inputs()[1];
}

AnfNodePtr DfGraphConvertor::TraceMakeTuple(const CNodePtr &node, unsigned int index) {
  if (index + 1 >= node->inputs().size()) {
    MS_LOG(EXCEPTION) << "length of make_tuple is less than index: " << index;
  }
  return node->inputs()[index + 1];
}

OutHandler DfGraphConvertor::GetHandler(const AnfNodePtr &node, const std::stack<unsigned int> &index_stack,
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
    if (nullptr == adpt) {
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
    return adpt->getOutput(Convert(node), UintToInt(index_stack.top()));
  }
}

// get the real operator through maketuple tuple_getitem depend
OutHandler DfGraphConvertor::TraceRealOp(AnfNodePtr node) {
  bool flag = IsPrimitiveCNode(node, prim::kPrimTupleGetItem) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
              IsPrimitiveCNode(node, prim::kPrimDepend);
  std::stack<unsigned int> index_stack;
  auto draw_index = node.get();
  while (flag) {
    flag = false;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      unsigned int index;
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

// Get the real op for tuple_getitem through make tuple, or depend
AnfNodePtr DfGraphConvertor::GetRealOpNode(AnfNodePtr node) {
  const int TUPLE_GET_ITEM_INDEX = 2;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    if (node_inputs.size() != 3) {  // "tuple_getitem" primitive must have 3 inputs
      MS_LOG(ERROR) << "tuple get item node not correct!";
      error_ = FAILED;
      return node;
    }
    MS_EXCEPTION_IF_NULL(node_inputs[TUPLE_GET_ITEM_INDEX]);
    if (!node_inputs[TUPLE_GET_ITEM_INDEX]->isa<ValueNode>()) {
      error_ = INVALID_ARGUMENT;
      MS_LOG(EXCEPTION) << "can't convert get item with non-constant index";
    }
    auto value_ptr = GetValueNode(node_inputs[TUPLE_GET_ITEM_INDEX])->cast<Int32ImmPtr>();
    if (value_ptr == nullptr) {
      MS_LOG(ERROR) << "Can not convert get item as value is nullptr!";
      error_ = FAILED;
      return node;
    }
    int index = value_ptr->value();

    // make_tuple apply inputs:make_tuple, [tuple_items,]
    if (IsPrimitiveCNode(node_inputs[1], prim::kPrimMakeTuple)) {
      auto tuple_inputs = node->cast<CNodePtr>()->inputs();
      if (tuple_inputs.size() < IntToSize(index + 1)) {
        MS_LOG(ERROR) << "make tuple input items node not correct! size:" << tuple_inputs.size()
                      << ", item index:" << index;
        error_ = FAILED;
        return node;
      }
      return GetRealOpNode(tuple_inputs[IntToSize(index + 1)]);
    }
    return GetRealOpNode(node_inputs[1]);
  }

  // depend apply inputs: depend,output,depended_node
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto depend_inputs = node->cast<CNodePtr>()->inputs();
    if (depend_inputs.size() != 3) {  // "Depend" primitive have 3 inputs
      MS_LOG(ERROR) << "depend input items not correct";
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
        MS_LOG(ERROR) << "Convert control depend node to operator failed";
        error_ = FAILED;
        return std::vector<OperatorPtr>({});
      }
      op_lists.push_back(op);
    }
    return op_lists;
  }

  auto op = Convert(GetRealOpNode(node));
  if (op == nullptr) {
    MS_LOG(ERROR) << "Convert control depend node to operator failed";
    error_ = FAILED;
    return std::vector<OperatorPtr>({});
  }
  return std::vector<OperatorPtr>({op});
}

// get the anf node list for depend
std::vector<AnfNodePtr> DfGraphConvertor::GetDependNodes(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> nodes;
  // for make tuple, should control depend on the tuple items
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto node_inputs = node->cast<CNodePtr>()->inputs();
    for (size_t index = 1; index < node_inputs.size(); index++) {
      nodes.push_back(GetRealOpNode(node_inputs[index]));
    }
    return nodes;
  }

  // for parameter ,find the apply that used the parameter as the control depended node
  if (node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend))) {
        nodes.push_back(GetRealOpNode(use_node));
      }
    }
    return nodes;
  }
  nodes.push_back(GetRealOpNode(node));
  return nodes;
}

void DfGraphConvertor::DrawControlDepend(const AnfNodePtr &src_node, const AnfNodePtr &dest_node) {
#ifdef DRAW_GE_GRAPH
  auto src_depend_nodes = GetDependNodes(src_node);
  auto dst_depend_nodes = GetDependNodes(dest_node);
  if (src_depend_nodes.size() == 1 && dst_depend_nodes.size() > 1) {
    for (auto &item : dst_depend_nodes) {
      compute_sout_ << op_draw_name_[src_depend_nodes[0].get()] << " -> " << op_draw_name_[item.get()]
                    << "[style=\"dotted\"]" << endl;
    }
  } else if (src_depend_nodes.size() > 1 && dst_depend_nodes.size() == 1) {
    for (auto &item : src_depend_nodes) {
      compute_sout_ << op_draw_name_[item.get()] << " -> " << op_draw_name_[dst_depend_nodes[0].get()]
                    << "[style=\"dotted\"]" << endl;
    }
  } else if (src_depend_nodes.size() == 1 && dst_depend_nodes.size() == 1) {
    compute_sout_ << op_draw_name_[src_depend_nodes[0].get()] << " -> " << op_draw_name_[dst_depend_nodes[0].get()]
                  << "[style=\"dotted\"]" << endl;
  }
#endif
}

void DfGraphConvertor::GetDependOnParameterUse(const CNodePtr &node, const AnfNodePtr &src_node,
                                               const AnfNodePtr &dest_node,
                                               const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                                               const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list) {
  if (src_node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[src_node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) &&
          (!IsPrimitiveCNode(use_node, prim::kPrimMakeTuple))) {
        auto converted_list = ConvertDependNode(use_node);
        src_ops_list->insert(src_ops_list->end(), converted_list.begin(), converted_list.end());
      }
    }
  }

  if (dest_node->isa<Parameter>()) {
    auto uses = node->func_graph()->manager()->node_users()[dest_node];
    for (auto &use : uses) {
      auto use_node = use.first;
      if ((use_node->isa<CNode>()) && (!IsPrimitiveCNode(use_node, prim::kPrimControlDepend)) &&
          (!IsPrimitiveCNode(use_node, prim::kPrimMakeTuple))) {
        auto converted_list = ConvertDependNode(use_node);
        dst_ops_list->insert(dst_ops_list->end(), converted_list.begin(), converted_list.end());
      }
    }
  }
}

bool DfGraphConvertor::GetControlDependList(const CNodePtr &node,
                                            const std::shared_ptr<std::vector<OperatorPtr>> &src_ops_list,
                                            const std::shared_ptr<std::vector<OperatorPtr>> &dst_ops_list) {
  const int CONTROL_DEPEND_INDEX = 0;
  const int SRC_NODE_INDEX = 1;
  const int DEST_NODE_INDEX = 2;
  const int DEPEND_MODE_NORMAL_USE = 0;
  const int DEPEND_MODE_ON_PARAMETER_USE = 1;

  auto node_inputs = node->inputs();
  if (node_inputs.size() <= DEST_NODE_INDEX) {
    MS_LOG(WARNING) << "Control depend node input size error";
    return false;
  }
  auto src_node = node_inputs[SRC_NODE_INDEX];
  auto dest_node = node_inputs[DEST_NODE_INDEX];
  if ((src_node == nullptr) || (dest_node == nullptr)) {
    MS_LOG(ERROR) << "Control depend node miss src or dest node";
    error_ = FAILED;
    return false;
  }
  AnfNodePtr fn = node_inputs[CONTROL_DEPEND_INDEX];
  PrimitivePtr prim_ptr = GetValueNode<PrimitivePtr>(fn);
  ValuePtr mode_ptr = prim_ptr->GetAttr("depend_mode");
  int depend_mode = DEPEND_MODE_NORMAL_USE;
  if (mode_ptr != nullptr) {
    auto mode_int = mode_ptr->cast<Int32ImmPtr>();
    MS_EXCEPTION_IF_NULL(mode_int);
    depend_mode = mode_int->value();
    MS_LOG(DEBUG) << "depend_mode = " << depend_mode;
  }
  if (depend_mode == DEPEND_MODE_ON_PARAMETER_USE) {
    GetDependOnParameterUse(node, src_node, dest_node, src_ops_list, dst_ops_list);
  }

  if (src_node->isa<CNode>()) {
    auto converted_list = ConvertDependNode(src_node);
    src_ops_list->insert(src_ops_list->end(), converted_list.begin(), converted_list.end());
  }

  if (dest_node->isa<CNode>()) {
    auto converted_list = ConvertDependNode(dest_node);
    dst_ops_list->insert(dst_ops_list->end(), converted_list.begin(), converted_list.end());
  }
  if (src_ops_list->empty() || dst_ops_list->empty()) {
    MS_LOG(DEBUG) << "Control depend node's src or dest node is not a CNode, ignore it";
    error_ = SUCCESS;
  }
  return true;
}

void DfGraphConvertor::ConvertControlDependNode(const CNodePtr node) {
  const int SRC_NODE_INDEX = 1;
  const int DEST_NODE_INDEX = 2;
  if (control_depend_cache_.find(node.get()) != control_depend_cache_.end()) {
    return;
  }
  auto node_inputs = node->inputs();
  if (node_inputs.size() <= DEST_NODE_INDEX) {
    MS_LOG(WARNING) << "Control depend node input size error";
    return;
  }
  auto src_node = node_inputs[SRC_NODE_INDEX];
  auto dest_node = node_inputs[DEST_NODE_INDEX];
  if ((src_node == nullptr) || (dest_node == nullptr)) {
    MS_LOG(ERROR) << "Control depend node miss src or dest node";
    error_ = FAILED;
    return;
  }
  std::shared_ptr<std::vector<OperatorPtr>> src_ops_list = std::make_shared<std::vector<OperatorPtr>>();
  std::shared_ptr<std::vector<OperatorPtr>> dst_ops_list = std::make_shared<std::vector<OperatorPtr>>();
  if (!GetControlDependList(node, src_ops_list, dst_ops_list)) {
    MS_LOG(ERROR) << "Get depend list failed";
    error_ = FAILED;
    return;
  }
  std::vector<ControlEdge> control_edges;
  if (src_ops_list->size() == 1 && dst_ops_list->size() > 1) {
    (void)std::transform(dst_ops_list->begin(), dst_ops_list->end(), std::back_inserter(control_edges),
                         [src_ops_list](const OperatorPtr &op) -> ControlEdge {
                           return {(*src_ops_list)[0], op};
                         });
  } else if (src_ops_list->size() > 1 && dst_ops_list->size() == 1) {
    (void)std::transform(src_ops_list->begin(), src_ops_list->end(), std::back_inserter(control_edges),
                         [dst_ops_list](const OperatorPtr &op) -> ControlEdge {
                           return {op, (*dst_ops_list)[0]};
                         });
  } else if (src_ops_list->size() == 1 && dst_ops_list->size() == 1) {
    control_edges.push_back({(*src_ops_list)[0], (*dst_ops_list)[0]});
  } else if (src_ops_list->empty() || dst_ops_list->empty()) {
    MS_LOG(DEBUG) << "Depend list of src or dst is empty, ignore it";
  } else {
    MS_LOG(ERROR) << "Convert control depend node to operator failed, depend src:" << src_ops_list->size()
                  << " -> dst:" << dst_ops_list->size();
    error_ = FAILED;
    return;
  }
  control_depend_cache_[node.get()] = control_edges;

#ifdef DRAW_GE_GRAPH
  DrawControlDepend(src_node, dest_node);
#endif
}

bool DfGraphConvertor::CheckCNode(const std::string &name, const CNodePtr node) {
  // ignore apply node of return
  if (name == "return" || name == "Depend") {
    return false;
  }

  if (name == "" && GetCNodeFuncName(node) == "switch_layer") {
    return false;
  }

  if (name == "Partial") {
    return false;
  }

  // make_tuple is used for a dynamic_input, convert it to a vector of OutHandlers
  if (name == "make_tuple") {
    ConvertMakeTuple(node);
    return false;
  }

  // As for nodes with multi outputs, convert tuple_getitem to OutHandle
  if (name == "tuple_getitem") {
    ConvertTupleGetItem(node);
    return false;
  }

  if (name == "ControlDepend") {
    ConvertControlDependNode(node);
    return false;
  }

  return true;
}

OperatorPtr DfGraphConvertor::ConvertCNode(const CNodePtr node) {
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
    op = adpt->generate(node);
  }

  // set attribute for primitive
  (void)adpt->setAttr(op, node);

  // add into cache
  (void)op_cache_.insert(std::make_pair(node.get(), op));

  DrawCNode(node, adpt);

  return op_cache_[node.get()];
}

OperatorPtr DfGraphConvertor::ConvertParameter(const AnfNodePtr node) {
  // convert Parameter in ANF to variable in DataFlow
  auto op = FindAdapter(node, training_)->generate(node);
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
      (void)const_op->set_attr_value(*ge_tensor);
      (void)const_op->update_output_desc_y(ge_tensor->GetTensorDesc());
      tuple_items->emplace_back(OutHandler(const_op, ""));
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

#if (defined ENABLE_GE)
  auto const_op = std::static_pointer_cast<Constant>(op);
  if (const_op == nullptr) {
    MS_LOG(ERROR) << "Get Constant operator failed";
    return nullptr;
  }
  auto ge_tensor = const_op->get_attr_value();
  auto ge_desc = ge_tensor.GetTensorDesc();
  (void)const_op->update_output_desc_y(ge_desc);
#endif

  op_cache_[node.get()] = op;
  return op_cache_[node.get()];
}

void DfGraphConvertor::DrawCNode(const CNodePtr node, const OpAdapterPtr adpt) {
  if (nullptr == adpt || nullptr == node) {
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
