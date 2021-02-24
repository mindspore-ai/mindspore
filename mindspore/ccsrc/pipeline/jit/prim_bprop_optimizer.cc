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

PrimBpropOptimizer &PrimBpropOptimizer::GetPrimBpropOptimizerInst() {
  static PrimBpropOptimizer g_prim_bprop_opt;
  return g_prim_bprop_opt;
}

PrimBpropOptimizer::PrimBpropOptimizer() {
  prim_bprop_opt_res = std::make_shared<pipeline::Resource>();
  prim_bprop_opt_manage = prim_bprop_opt_res->manager();
}

PrimBpropOptimizer::~PrimBpropOptimizer() {
}

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

  abstract::AbstractBasePtrList abs_list;
  ArgsToAbs(op_args, abs_list);

  FuncGraphPtr ret_bprop_fg;
  PrimBpropOptGraphInfoPtr ret_bprop_info;
  ECacheQrtRes cache_res = GetOptBpfgFromCache(prim, abs_list, ret_bprop_fg, ret_bprop_info);

  if (cache_res == E_LEVEL_2) {
    FreeTensorValue(op_args, out, ret_bprop_info);
    return ret_bprop_fg;
  }

  // do step1 opt
  if (cache_res == E_NOT_FOUND) {
    ret_bprop_info = std::make_shared<PrimBpropOptGraphInfo>();
    ret_bprop_fg = PrimBpropOptStep1(bprop_fg);
    ret_bprop_info->opt_fungraph = BasicClone(ret_bprop_fg);
    prim_bprop_cache[prim] = ret_bprop_info;
    DumpIR(ret_bprop_fg->ToString(), ret_bprop_fg);
  }

  // do step2 opt
  AddOutToAbsList(out, abs_list);
  ret_bprop_fg = PrimBpropOptStep2(ret_bprop_fg, abs_list);
  ret_bprop_info->graph_level_2_cache[abs_list] = BasicClone(ret_bprop_fg);
  return ret_bprop_fg;
}

FuncGraphPtr PrimBpropOptimizer::PrimBpropOptStep1(const FuncGraphPtr &bprop_fg) {
  prim_bprop_opt_res->set_func_graph(bprop_fg);
  prim_bprop_opt_manage->AddFuncGraph(bprop_fg);
  auto opt_bprop_fg = PrimBpOptPassStep1(irpass, prim_bprop_opt_res);
  return opt_bprop_fg;
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

FuncGraphPtr PrimBpropOptimizer::PrimBpropOptStep2(const FuncGraphPtr &bprop_fg,
                                                   abstract::AbstractBasePtrList &abs_list_input) {
  BindAbsToParameters(bprop_fg, abs_list_input);
  prim_bprop_opt_res->set_func_graph(bprop_fg);
  prim_bprop_opt_manage->AddFuncGraph(bprop_fg);
  auto opt_bprop_fg = PrimBpOptPassStep2(irpass, prim_bprop_opt_res);
  return opt_bprop_fg;
}

ECacheQrtRes PrimBpropOptimizer::GetOptBpfgFromCache(const PrimitivePtr &prim,
                                                     const abstract::AbstractBasePtrList &abs_list,
                                                     FuncGraphPtr &bprop_fg, PrimBpropOptGraphInfoPtr &bprop_info) {
  auto iter = prim_bprop_cache.find(prim);
  if (iter == prim_bprop_cache.end()) {
    return E_NOT_FOUND;
  }

  bprop_info = iter->second;
  auto second_iter = bprop_info->graph_level_2_cache.find(abs_list);
  if (second_iter == bprop_info->graph_level_2_cache.end()) {
    bprop_fg = BasicClone(bprop_info->opt_fungraph);
    return E_LEVEL_1;
  }
  bprop_fg = BasicClone(second_iter->second);
  return E_LEVEL_2;
}

void PrimBpropOptimizer::ArgsToAbs(const ValuePtrList &op_args, abstract::AbstractBasePtrList &abs_list) {
  for (auto &item : op_args) {
    MS_EXCEPTION_IF_NULL(item);
    auto abs = item->ToAbstract();
    abs_list.emplace_back(abs);
  }
}

void PrimBpropOptimizer::AddOutToAbsList(const ValuePtr &out, abstract::AbstractBasePtrList &abs_list) {
  if (!out->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "Just suport tensor out now, tuple out need support later.";
  }

  auto tens = out->cast<tensor::TensorPtr>();
  if (tens->is_parameter()) {
    abs_list.emplace_back(out->ToAbstract());
    abs_list.emplace_back(out->ToAbstract());
  }

  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber but got: " << dtype->ToString() << ".";
  }
  auto tensor_shape = tens->shape();
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  std::string param_name("dout");
  auto ref_key = std::make_shared<RefKey>(param_name);
  auto abs_ref_key = ref_key->ToAbstract();
  auto ref_out = std::make_shared<abstract::AbstractRef>(abs_ref_key, abs_tensor);
  abs_list.emplace_back(ref_out);
  abs_list.emplace_back(ref_out);
}

FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                    const ValuePtr &out) {
  return PrimBpropOptimizer::GetPrimBpropOptimizerInst().OptimizeBPropFuncGraph(bprop_fg, c_node, op_args, out);
}

}  // namespace pipeline
}  // namespace mindspore