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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PRIM_BPROP_OPTIMIZER_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_PRIM_BPROP_OPTIMIZER_H

#include "frontend/optimizer/irpass.h"
#include "ir/func_graph.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace pipeline {
struct PrimBpropOptGraphInfo;
struct PrimitiveTotalEqual;

using PrimBpropOptGraphInfoPtr = std::shared_ptr<PrimBpropOptGraphInfo>;

using PrimBpropCache = std::unordered_map<PrimitivePtr, PrimBpropOptGraphInfoPtr, PrimitiveHasher, PrimitiveTotalEqual>;

using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr,
  abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;

struct PrimitiveTotalEqual {
  bool operator()(PrimitivePtr const &t1, PrimitivePtr const &t2) const {
    MS_EXCEPTION_IF_NULL(t1);
    MS_EXCEPTION_IF_NULL(t2);
    return *t1 == *t2;
  }
};

enum ECacheQrtRes {
  E_NOT_FOUND, E_LEVEL_1, E_LEVEL_2
};

struct PrimBpropOptGraphInfo {
  // the opt funcgraph without infer, level1 cache
  FuncGraphPtr opt_fungraph;
  // the opt funcgraph with infer, level2 cache
  // key: hash value of arguments
  AbstractListMap graph_level_2_cache;
  // to indicate using tencer value or not, if flg is false release value
  std::vector<bool> args_value_using_flg;
};

class PrimBpropOptimizer {
public:
  ~PrimBpropOptimizer();

  void Clear();

  static PrimBpropOptimizer &GetPrimBpropOptimizerInst();

  // bprop_fg has the signature:
  // (sens_input1, sens_input2,...)bprop_fg(input1, input2, ..., out, d_out)
  // c_node contains the prim(input 0) and the input parameters of that prim;
  // op_args contains the arguments list of each input parameters, it maybe tensor or tuple
  // out contains the out of c_node;
  FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                      const ValuePtr &out);

  // need ? how to shrink ?
  // void CacheShrink();

private:
  PrimBpropOptimizer();

  ECacheQrtRes GetOptBpfgFromCache(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list,
                                   FuncGraphPtr &bprop_fg, PrimBpropOptGraphInfoPtr &bprop_info);

  // converter tensor args to abs value;
  void ArgsToAbs(PrimitivePtr &prim, const ValuePtrList &op_args, abstract::AbstractBasePtrList &abs_list);

  // add out && dout to abs list
  void AddOutToAbsList(const ValuePtr &out, abstract::AbstractBasePtrList &abs_list);

  // TODO: how To?
  void FreeTensorValue(const ValuePtrList &op_args, const ValuePtr &out, PrimBpropOptGraphInfoPtr &bprop_info) {};

  // do opt without input info, no infer
  FuncGraphPtr PrimBpropOptStep1(const FuncGraphPtr &bprop_fg);

  // do opt with input info
  FuncGraphPtr PrimBpropOptStep2(const FuncGraphPtr &bprop_fg, abstract::AbstractBasePtrList &abs_list_input);

  void BindAbsToParameters(const FuncGraphPtr &bprop_fg, abstract::AbstractBasePtrList &abs_list_input);

private:
  FuncGraphManagerPtr prim_bprop_opt_manage;
  ResourcePtr prim_bprop_opt_res;
  // cache optimized bprop graph
  PrimBpropCache prim_bprop_cache;
  opt::irpass::OptimizeIRPassLib irpass;
};

// bprop_fg has the signature:
// (sens_input1, sens_input2,...)bprop_fg(input1, input2, ..., out, d_out)
// c_node contains the prim(input 0) and the input parameters of that prim;
// op_args contains the arguments list of each input parameters, it maybe tensor or tuple
// out contains the out of c_node;
FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                    const ValuePtr &out);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PRIM_BPROP_OPTIMIZER_H