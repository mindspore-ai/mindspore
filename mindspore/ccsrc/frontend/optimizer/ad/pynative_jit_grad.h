/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "pynative/utils/base.h"
#include "include/utils/pynative/variable.h"
#include "frontend/jit/ps/compile_cache_manager.h"
#include "include/frontend/optimizer/ad/jit_grad_interface.h"

namespace mindspore {
namespace ad {
class BpropGenerator {
 public:
  BpropGenerator(const FuncGraphPtr &fprop_graph, const abstract::AbstractBasePtrList &input_abs,
                 const std::vector<ValuePtr> &input_value, const abstract::AbstractBasePtr &out_abs, bool need_reuse)
      : fprop_graph_(fprop_graph),
        input_abs_(input_abs),
        input_value_(input_value),
        out_abs_(out_abs),
        need_reuse_forward_node_(need_reuse) {
    Init();
  }
  ~BpropGenerator() = default;

  void Init();
  FuncGraphPtr GenerateBpropGraph();
  FuncGraphPtr GenerateForwardGraph(const FuncGraphPtr &jit_forward_graph, bool do_renormalize);
  void SetForwardOutputAbs(const abstract::AbstractBasePtr &forward_abs, const FuncGraphPtr &bprop_graph);
  void EraseUnusedReuseCNode(const FuncGraphPtr &bprop_fg);

 private:
  void ReusePrimalCNode(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg, bool top_cell_do_recompute);
  void ReuseCustomBpropForwardOutput(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg);

  FuncGraphPtr fprop_graph_;
  FuncGraphPtr forward_graph_;
  FuncGraphPtr basic_graph_;
  abstract::AbstractBasePtrList input_abs_{};
  std::vector<ValuePtr> input_value_{};
  abstract::AbstractBasePtr out_abs_{nullptr};
  bool need_reuse_forward_node_{false};
  size_t bprop_origin_param_size_{0};
  std::vector<FuncGraphPtr> fprop_sub_fgs_{};
  AnfNodePtrList fprop_modified_params_{};
  AnfNodePtrList replace_nodes_{};
  abstract::AbstractBasePtrList replace_nodes_abs_{};
};
using BpropGeneratorPtr = std::shared_ptr<BpropGenerator>;

std::pair<FuncGraphPtr, FuncGraphPtr> GetGradAndForwardGraph(const std::string &key);
void StoreOriginGradGraph(const std::string &key, const FuncGraphPtr &fg);
FuncGraphPtr GetOriginGradGraph(const std::string &key);
bool HasOriginGradGraph(const std::string &key);
size_t StoreFilteredGradGraph(const std::string &cache_key, size_t hash_key, const FuncGraphPtr &fg);
FuncGraphPtr GetFilteredGradGraph(const std::string &cache_key, size_t hash_key);
FuncGraphPtr FilterGraphOutput(const bool is_filtered, const std::pair<VectorRef, VectorRef> arg_pair,
                               const FuncGraphPtr &func_graph, const std::string &cache_key,
                               std::vector<pynative::autograd::Edge> *next_edges);
VectorRef FilterGraphInputOutput(bool is_filtered, const std::pair<VectorRef, VectorRef> arg_pair,
                                 const FuncGraphPtr &func_graph, const std::string &cache_key,
                                 std::vector<pynative::autograd::Edge> *next_edges);
bool FilterGradOutput(const std::vector<bool> &need_grad, const FuncGraphPtr &func_graph, const VectorRef &args,
                      std::vector<pynative::autograd::Edge> *next_edges);
void FilterGradInput(const std::vector<bool> &need_filter, const FuncGraphPtr &func_graph, size_t add_args_size,
                     size_t skip_filter_size);
VectorRef RefreshAddedArgs(const VectorRef &added_args, const std::vector<bool> &need_filter, size_t add_args_size);
void FilterForwardOutput(const std::vector<bool> &need_filter, const std::string &cache_key, size_t add_args_size);
void UpdateNextEdge(std::vector<pynative::autograd::Edge> *next_edges, const FuncGraphPtr &func_graph,
                    const VectorRef &args);
std::pair<std::vector<bool>, int> CollectFilterMsg(const VectorRef &added_args, const FuncGraphPtr &func_graph);
std::vector<bool> GetNeedGradIndexes(const VectorRef &args);
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_PYNATIVE_JIT_GRAD_H
