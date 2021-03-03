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

#include <ir/func_graph_cloner.h>
#include "prim_bprop_optimizer.h"
#include "pass.h"

namespace mindspore {
namespace pipeline {

void PrimBpropOptGraphLevel2Info::TryFreeArgsValue(const ValuePtrList &op_args, const ValuePtr &out) {
  if (args_value_using_info.size() != op_args.size() + 1) {
    MS_LOG(EXCEPTION) << "param size :" << args_value_using_info.size() << " of bp_graph:"
                      << opt_func_graph_->ToString()
                      << " not match input arguments num:" << op_args.size();
  }

  ValuePtrList new_args(op_args);
  new_args.emplace_back(out);
  TryFreeOneValue(new_args, args_value_using_info);
}

void PrimBpropOptGraphLevel2Info::TryFreeOneValue(const ValuePtrList &op_args,
                                                  const std::vector<ParamUsingInfo> &param_info_vec) {
  if (param_info_vec.size() != op_args.size()) {
    MS_LOG(EXCEPTION) << "param size :" << param_info_vec.size() << " of bp_graph:"
                      << opt_func_graph_->ToString()
                      << " not match input arguments num:" << op_args.size();
  }

  for (size_t i = 0; i < op_args.size(); ++i) {
    if (!param_info_vec[i].using_flg && !param_info_vec[i].tuple_flg &&
        op_args[i]->isa<tensor::Tensor>()) {
      auto value = op_args[i]->cast<tensor::TensorPtr>();
      value->set_device_address(nullptr);
    } else if (param_info_vec[i].tuple_flg && op_args[i]->isa<ValueTuple>()) {
      auto value = op_args[i]->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value);
      TryFreeOneValue(value->value(), param_info_vec[i].sub_using_info);
    }
  }
}

void PrimBpropOptGraphLevel2Info::AnalysisArgUsingInfo(FuncGraphManagerPtr &manager) {
  if (analysis_finish_flg) {
    return;
  }
  MS_EXCEPTION_IF_NULL(opt_func_graph_);
  auto &params = opt_func_graph_->parameters();
  auto &node_users = manager->node_users();
  args_value_using_info.resize(params.size() - 1);
  // analysis value using flg except dout
  for (size_t i = 0; i < params.size() - 1; ++i) {
    auto &param = params[i];
    auto &arg_info = args_value_using_info[i];
    ArgInfoRefresh(param, arg_info);
    AnalysisNodeUsingInfo(node_users, param, arg_info);
  }
  analysis_finish_flg = true;
}

void PrimBpropOptGraphLevel2Info::AnalysisNodeUsingInfo(
  const NodeUsersMap &node_users, const std::shared_ptr<AnfNode> &param, ParamUsingInfo &arg_info) const {
  auto iter = node_users.find(param);

  if (iter == node_users.end()) {
    arg_info.using_flg = false;
    return;
  }

  // tensor return directly
  if (!arg_info.tuple_flg) {
    arg_info.using_flg = true;
    return;
  }

  // specific process for tuple parameter, may only partial items used
  auto &users_info = iter->second;
  for (auto &user_info : users_info) {
    auto user_node = user_info.first;
    arg_info.using_flg = true;
    MS_LOG(WARNING) << "param:" << param->ToString() << " used by node:" << user_node->ToString();
    if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
      MS_LOG(EXCEPTION) << "tuple param:" << param->ToString() << " of bp_graph:" << opt_func_graph_->ToString()
                        << " unexpect used by node:" << user_node->ToString();
    }
    auto cnode = user_node->cast<CNodePtr>();
    if (cnode->size() != 3) {
      MS_LOG(EXCEPTION) << "TupleGetItem Node:" << user_node->ToString() << " of bp_graph:"
                        << opt_func_graph_->ToString()
                        << "input size is:" << cnode->size();
    }
    auto idx_node = cnode->input(2);
    if (!idx_node->isa<ValueNode>()) {
      MS_LOG(EXCEPTION) << "tuple :" << param->ToString() << " of bp_graph:" << opt_func_graph_->ToString()
                        << " unexpect used by node:" << user_node->ToString() << " TupleGetItem idx node:"
                        << idx_node->ToString();
    }

    auto vnode = idx_node->cast<ValueNodePtr>();
    auto value_ptr = vnode->value();
    if (value_ptr == nullptr || !value_ptr->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "tuple :" << param->ToString() << " of bp_graph:" << opt_func_graph_->ToString()
                        << " unexpect used by node:" << user_node->ToString() << " TupleGetItem idx node:"
                        << idx_node->ToString() << " idx Value :" << value_ptr;
    }

    auto idx = value_ptr->cast<Int64ImmPtr>()->value();
    arg_info.sub_using_info[idx].using_flg = true;
    ArgInfoRefresh(cnode, arg_info.sub_using_info[idx]);

    if (arg_info.tuple_flg) {
      AnalysisNodeUsingInfo(node_users, cnode, arg_info.sub_using_info[idx]);
    }
  }
}

void PrimBpropOptGraphLevel2Info::ArgInfoRefresh(
  const std::shared_ptr<AnfNode> &param, ParamUsingInfo &arg_info) const {
  auto abs = param->abstract();
  if (abs->isa<abstract::AbstractTensor>()) {
    arg_info.tuple_flg = false;
    MS_LOG(DEBUG) << "param abstract:" << param->ToString() << " is a AbstractTensor";
  } else if (abs->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    MS_LOG(DEBUG) << "param abstract:" << param->ToString() << " is a AbstractTuple";
    arg_info.tuple_flg = true;
    arg_info.tuple_size = abs_tuple->size();
    arg_info.sub_using_info.resize(abs_tuple->size());
  } else {
    MS_LOG(EXCEPTION) << "param:" << param->ToString() << " of bp_graph:" << opt_func_graph_->ToString()
                      << " not tensor or tuple";
  }
}


PrimBpropOptimizer &PrimBpropOptimizer::GetPrimBpropOptimizerInst() {
  static PrimBpropOptimizer g_prim_bprop_opt;
  return g_prim_bprop_opt;
}

PrimBpropOptimizer::PrimBpropOptimizer() {
  prim_bprop_opt_res = std::make_shared<pipeline::Resource>();
  prim_bprop_opt_manage = prim_bprop_opt_res->manager();
}

PrimBpropOptimizer::~PrimBpropOptimizer() {}

void PrimBpropOptimizer::Clear() {
  prim_bprop_cache.clear();
  prim_bprop_opt_manage = nullptr;
  prim_bprop_opt_res = nullptr;
}

// bprop_fg has the signature:
// (sens_input1, sens_input2,...)bprop_fg(input1, input2, ..., out, d_out)
// c_node contains the prim(input 0) and the input parameters of that prim;
// op_args contains the arguments list of each input parameters, it maybe tensor or tuple
// out contains the out of c_node;
FuncGraphPtr PrimBpropOptimizer::OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node,
                                                        const ValuePtrList &op_args, const ValuePtr &out) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  MS_EXCEPTION_IF_NULL(c_node);
  MS_EXCEPTION_IF_NULL(out);
  auto &inputs = c_node->inputs();
  if (inputs.size() < 1 || inputs.size() - 1 != op_args.size()) {
    MS_LOG(EXCEPTION) << "The parameters num " << inputs.size() - 1 << " not match arguments num " << op_args.size()
                      << ", CNode:" << c_node->ToString() << " grap:" << bprop_fg->ToString();
  }

  if (!IsValueNode<Primitive>(inputs[0])) {
    MS_LOG(EXCEPTION) << "CNode:" << c_node->ToString()
                      << " not a primitive node, input_0 is:" << inputs[0]->ToString();
  }

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(inputs[0]);
  MS_LOG(DEBUG) << "Hash of prim " << prim->ToString() << " is:" << prim->hash();

  abstract::AbstractBasePtrList abs_list;
  ArgsToAbs(prim, op_args, abs_list);

  PrimBpropOptGraphLevel2InfoPtr level_2_graph_info;
  PrimBpropOptGraphInfoPtr level_1_graph_info;
  ECacheQrtRes cache_res = GetOptBpfgFromCache(prim, abs_list, level_2_graph_info, level_1_graph_info);

  MS_LOG(DEBUG) << "Cache match result " << cache_res << ", prim: " << prim->ToString();
  if (cache_res == E_LEVEL_2) {
    MS_LOG(DEBUG) << "Level 2 cache matched, prim: " << prim->ToString();
    level_2_graph_info->TryFreeArgsValue(op_args, out);
    return BasicClone(level_2_graph_info->opt_func_graph());
  }

  // do step1 opt
  if (cache_res == E_NOT_FOUND) {
    bprop_fg->debug_info()->set_name(prim->ToString());
    level_1_graph_info = PrimBpropOptStep1(bprop_fg);
    prim_bprop_cache[prim] = level_1_graph_info;
  }
  FuncGraphPtr level_1_graph = BasicClone(level_1_graph_info->opt_func_graph);

  // do step2 opt
  auto new_abs_list = AddOutToAbsList(out, abs_list);
  level_2_graph_info = PrimBpropOptStep2(level_1_graph, new_abs_list);
  level_1_graph_info->graph_level_2_cache[abs_list] = level_2_graph_info;
  level_2_graph_info->TryFreeArgsValue(op_args, out);
  return BasicClone(level_2_graph_info->opt_func_graph());
}

PrimBpropOptGraphInfoPtr PrimBpropOptimizer::PrimBpropOptStep1(const FuncGraphPtr &bprop_fg) {
  auto level_1_graph_info = std::make_shared<PrimBpropOptGraphInfo>();
  prim_bprop_opt_res->set_func_graph(bprop_fg);
  prim_bprop_opt_manage->AddFuncGraph(bprop_fg);
  auto opt_bprop_fg = PrimBpOptPassStep1(irpass, prim_bprop_opt_res);
  level_1_graph_info->opt_func_graph = opt_bprop_fg;
  return level_1_graph_info;
}

void PrimBpropOptimizer::BindAbsToParameters(const FuncGraphPtr &bprop_fg,
                                             abstract::AbstractBasePtrList &abs_list_input) {
  auto &params = bprop_fg->parameters();
  if (abs_list_input.size() != params.size()) {
    MS_LOG(EXCEPTION) << "Param num:" << params.size() << " not match inputs num " << abs_list_input.size();
  }

  for (size_t i = 0; i < abs_list_input.size(); i++) {
    params[i]->set_abstract(abs_list_input[i]);
  }
}

PrimBpropOptGraphLevel2InfoPtr PrimBpropOptimizer::PrimBpropOptStep2(const FuncGraphPtr &bprop_fg,
                                                                     abstract::AbstractBasePtrList &abs_list_input) {
  BindAbsToParameters(bprop_fg, abs_list_input);
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  auto manager = resource->manager();
  resource->set_func_graph(bprop_fg);
  manager->AddFuncGraph(bprop_fg);
  auto opt_bprop_fg = PrimBpOptPassStep2(irpass, resource);
  auto level_2_graph_info = std::make_shared<PrimBpropOptGraphLevel2Info>(opt_bprop_fg);
  level_2_graph_info->AnalysisArgUsingInfo(manager);
  return level_2_graph_info;
}

FuncGraphPtr PrimBpropOptimizer::BpropGraphFinalOpt(const ResourcePtr &res) {
  MS_EXCEPTION_IF_NULL(res);
  auto after_opt_bg = BpropGraphFinalOptPass(res);
  return after_opt_bg;
}

ECacheQrtRes PrimBpropOptimizer::GetOptBpfgFromCache(
  const PrimitivePtr &prim,
  const abstract::AbstractBasePtrList &abs_list,
  PrimBpropOptGraphLevel2InfoPtr &level_2_graph_info,
  PrimBpropOptGraphInfoPtr &level_1_graph_info) {
  auto attrs_ = prim->attrs();
  for (auto &item : attrs_) {
    MS_LOG(DEBUG) << "prim:" << prim->ToString() << " attr: " << item.first << " value:" << item.second->ToString();
  }

  auto iter = prim_bprop_cache.find(prim);
  if (iter == prim_bprop_cache.end()) {
    return E_NOT_FOUND;
  }

  level_1_graph_info = iter->second;
  auto second_iter = level_1_graph_info->graph_level_2_cache.find(abs_list);
  if (second_iter == level_1_graph_info->graph_level_2_cache.end()) {
    return E_LEVEL_1;
  }
  level_2_graph_info = second_iter->second;
  return E_LEVEL_2;
}

void PrimBpropOptimizer::ArgsToAbs(PrimitivePtr &prim, const ValuePtrList &op_args,
                                   abstract::AbstractBasePtrList &abs_list) {
  auto const_input_index = prim->get_const_input_indexes();
  bool have_const_input = !const_input_index.empty();
  bool is_const_prim = prim->is_const_prim();
  for (size_t i = 0; i < op_args.size(); ++i) {
    bool is_const_input =
      have_const_input && std::find(const_input_index.begin(), const_input_index.end(), i) != const_input_index.end();
    auto &arg_value = op_args[i];
    auto arg_abs = arg_value->ToAbstract();
    if (!is_const_prim && !is_const_input) {
      auto config = abstract::AbstractBase::kBroadenTensorOnly;
      arg_abs = arg_abs->Broaden(config);
      MS_LOG(DEBUG) << "Broaden for " << prim->ToString() << " " << config;
    }
    abs_list.emplace_back(arg_abs);
  }
}

abstract::AbstractBasePtrList PrimBpropOptimizer::AddOutToAbsList(const ValuePtr &out,
                                                                  const abstract::AbstractBasePtrList &abs_list) {
  if (!out->isa<tensor::Tensor>() && !out->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Out value not Tensor or Tuple, please check the input arguments.";
  }
  abstract::AbstractBasePtrList new_abs_list(abs_list);
  auto out_abs = out->ToAbstract();
  auto config = abstract::AbstractBase::kBroadenTensorOnly;
  out_abs = out_abs->Broaden(config);
  new_abs_list.emplace_back(out_abs);
  new_abs_list.emplace_back(out_abs);
  return new_abs_list;
}

FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                    const ValuePtr &out) {
  return PrimBpropOptimizer::GetPrimBpropOptimizerInst().OptimizeBPropFuncGraph(bprop_fg, c_node, op_args, out);
}

}  // namespace pipeline
}  // namespace mindspore