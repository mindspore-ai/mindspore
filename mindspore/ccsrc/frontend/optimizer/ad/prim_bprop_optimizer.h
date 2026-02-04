/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PRIM_BPROP_OPTIMIZER_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PRIM_BPROP_OPTIMIZER_H

#include <vector>
#include <utility>
#include <memory>
#include <unordered_map>
#include <string>

#include "frontend/optimizer/irpass.h"
#include "ir/func_graph.h"
#include "frontend/jit/ps/resource.h"

namespace mindspore {
namespace ad {
struct PrimBpropOptGraphInfo;

class PrimBpropOptGraphLevel2Info;

struct PrimitiveTotalEqual;

struct PrimitiveTupleListHasher;

struct PrimitiveTupleListEqual;

using PrimBpropOptGraphInfoPtr = std::shared_ptr<PrimBpropOptGraphInfo>;

using PrimBpropOptGraphLevel2InfoPtr = std::shared_ptr<PrimBpropOptGraphLevel2Info>;

using PrimBpropCache = std::unordered_map<PrimitivePtr, PrimBpropOptGraphInfoPtr, PrimitiveHasher, PrimitiveTotalEqual>;

using TupleListKey = std::pair<PrimitivePtr, abstract::AbstractBasePtrList>;

using PrimBpropGragFlagCache = std::unordered_map<std::vector<bool>, PrimBpropOptGraphLevel2InfoPtr>;

using PrimBpropLevel2Cache =
  std::unordered_map<abstract::AbstractBasePtrList, PrimBpropGragFlagCache, abstract::AbstractBasePtrListHasher,
                     abstract::AbstractBasePtrListEqual>;

using PrimTupleListCache =
  std::unordered_map<TupleListKey, FuncGraphPtr, PrimitiveTupleListHasher, PrimitiveTupleListEqual>;

struct PrimitiveTupleListHasher {
  std::size_t operator()(const TupleListKey &key) const {
    abstract::AbstractBasePtrListHasher hasher;
    return hasher(key.second);
  }
};

struct PrimitiveTupleListEqual {
  bool operator()(TupleListKey const &t1, TupleListKey const &t2) const {
    MS_EXCEPTION_IF_NULL(t1.first);
    MS_EXCEPTION_IF_NULL(t2.first);

    if (!(*t1.first == *t2.first)) {
      return false;
    }
    abstract::AbstractBasePtrListEqual cmp;
    return cmp(t1.second, t2.second);
  }
};

struct PrimitiveTotalEqual {
  bool operator()(PrimitivePtr const &t1, PrimitivePtr const &t2) const {
    MS_EXCEPTION_IF_NULL(t1);
    MS_EXCEPTION_IF_NULL(t2);
    return *t1 == *t2;
  }
};

enum ECacheQrtRes { E_NOT_FOUND, E_LEVEL_1, E_LEVEL_2 };

struct PrimBpropOptGraphInfo {
  // the level1 opt func_graph without infer, no shape/type info provide
  FuncGraphPtr opt_func_graph_;
  // the opt func_graph after infer, func_graph level2 cache
  PrimBpropLevel2Cache graph_level_2_cache_;
};

struct ParamUsingInfo {
  bool using_flg_{false};
  bool tuple_flg_{false};
  size_t tuple_size_;
  std::vector<ParamUsingInfo> sub_using_info_;
};

class PrimBpropOptGraphLevel2Info {
 public:
  explicit PrimBpropOptGraphLevel2Info(const FuncGraphPtr &func_graph) : opt_func_graph_(func_graph) {}
  ~PrimBpropOptGraphLevel2Info() = default;

  const FuncGraphPtr &opt_func_graph() const { return opt_func_graph_; }

  void TryFreeArgsValue(const ValuePtrList &op_args, const ValuePtr &out);

  void AnalysisArgUsingInfo(const FuncGraphManagerPtr &manager);

 private:
  void ArgInfoRefresh(const std::shared_ptr<AnfNode> &param, ParamUsingInfo *arg_info) const;

  void AnalysisNodeUsingInfo(const NodeUsersMap &node_users, const std::shared_ptr<AnfNode> &param,
                             ParamUsingInfo *arg_info) const;

  void TryFreeOneValue(const ValuePtrList &op_args, const std::vector<ParamUsingInfo> &param_info_vec);

  void AalysisForTupleGetItem(const NodeUsersMap &node_users, const std::shared_ptr<AnfNode> &param,
                              ParamUsingInfo *arg_info, const AnfNodePtr &user_node) const;

  // the level2 opt func_graph
  FuncGraphPtr opt_func_graph_;
  // to indicate arguments value using or not, if not using should free device memory
  std::vector<ParamUsingInfo> args_value_using_info_;
  bool analysis_finish_flg_{false};
};

class PrimBpropOptimizer {
 public:
  ~PrimBpropOptimizer() = default;

  void Clear();

  static PrimBpropOptimizer &GetPrimBpropOptimizerInst();

  // bprop_fg has the signature:
  // (sens_input1, sens_input2,...)bprop_fg(input1, input2, ..., out, d_out)
  // c_node contains the prim(input 0) and the input parameters of that prim;
  // op_args contains the arguments list of each input parameters, it maybe tensor or tuple
  // out contains the out of c_node;
  FuncGraphPtr OptimizeBPropFuncGraph(const FuncGraphPtr &bprop_fg, const CNodePtr &c_node, const ValuePtrList &op_args,
                                      const ValuePtr &out);

  // do inline opt for final bprop graph
  FuncGraphPtr BpropGraphFinalOpt(const pipeline::ResourcePtr &res) const;

 private:
  PrimBpropOptimizer() = default;

  ECacheQrtRes GetOptBpfgFromCache(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list,
                                   const std::vector<bool> &need_grad_flags,
                                   PrimBpropOptGraphLevel2InfoPtr *level_2_graph_info,
                                   PrimBpropOptGraphInfoPtr *level_1_graph_info);

  // converter tensor args to abs value;
  void ArgsToAbs(const PrimitivePtr &prim, const ValuePtrList &op_args, abstract::AbstractBasePtrList *abs_list) const;

  // add out && dout to abs list
  abstract::AbstractBasePtrList AddOutToAbsList(const ValuePtr &out, const abstract::AbstractBasePtrList &abs_list,
                                                const FuncGraphPtr &bprop_fg) const;

  // do opt without input info, no infer
  PrimBpropOptGraphInfoPtr PrimBpropOptStep1(const FuncGraphPtr &bprop_fg) const;

  // do opt with input info
  PrimBpropOptGraphLevel2InfoPtr PrimBpropOptStep2(
    const FuncGraphPtr &bprop_fg, const abstract::AbstractBasePtrList &abs_list_input,
    const std::vector<bool> &need_grad_flags = std::vector<bool>()) const;

  void BindAbsToParameters(const FuncGraphPtr &bprop_fg, const abstract::AbstractBasePtrList &abs_list_input) const;

  FuncGraphPtr GetOptBpropFromCache(const FuncGraphPtr &bprop_fg, const ValuePtrList &op_args, const ValuePtr &out,
                                    const PrimitivePtr &prim, const std::vector<bool> &need_grad_flags);

  FuncGraphPtr GenSpecOptBprop(const FuncGraphPtr &bprop_fg, const ValuePtrList &op_args, const ValuePtr &out,
                               const PrimitivePtr &prim, bool hook_flg);

  // cache optimized bprop graph
  PrimBpropCache prim_bprop_cache_;
  PrimTupleListCache tuple_list_bprop_cache_;
};
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PRIM_BPROP_OPTIMIZER_H
