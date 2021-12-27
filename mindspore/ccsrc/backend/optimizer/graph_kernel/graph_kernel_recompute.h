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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_RECOMPUTE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_RECOMPUTE_H_

#include <map>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "utils/context/graph_kernel_flags.h"
#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
/*
 * Recompute some operator to reduce temporary memory peak.
 *
 *   (a)  (b)                  (a)   (b)
 *     \  /                     |     |
 *      Gs                     Gs1    |
 *  (c)/ |                   (c)|     |
 *    /  |                      Go    |
 *  Go   |(d)   =========>      │└─depend
 *    \  |                      │     │
 *  (e)\ |                   (e)│    Gs2
 *      \|                      │     │
 *      Gt                      ├────(d)
 *                              Gt
 *
 * Where, split Gs to Gs1 and Gs2, and (x) means the temporary tensor.
 * For left graph, the memory is (a+b) -> (c+d) -> (d+e)
 * As for right graph, memory is (a+b) -> (b+c) -> (b+e) -> (d+e)
 * If the (c+d) reach the threshold memory, and (b+c) or (b+e) is less than it,
 * it may ease the memory burden.
 */
enum class EdgeLifeTimeType : char { ShortTerm, LongTerm };
inline std::ostream &operator<<(std::ostream &os, EdgeLifeTimeType type) {
  std::map<EdgeLifeTimeType, std::string> out_str = {{EdgeLifeTimeType::ShortTerm, "[ShortTerm]"},
                                                     {EdgeLifeTimeType::LongTerm, "[LongTerm]"}};
  return os << out_str[type];
}
using OutPosLinkList = std::vector<std::tuple<AnfNodePtr, std::vector<int>, EdgeLifeTimeType>>;
using OutPosLinkMap = std::map<AnfNodePtr, std::vector<int>>;
using MemorySize = int64_t;
struct Candidate {
  AnfNodePtr source_graph;
  AnfNodePtr target_graph;
  EdgeLifeTimeType type;
  AnfNodePtrList recompute_edges;  // getitem list for recompute edges.
};

class AutoRecompute {
 public:
  std::vector<Candidate> Run(const FuncGraphPtr &func_graph) {
    lifetime_threshold_ = GraphKernelFlags::GetInstance().recompute_increment_threshold;
    local_peak_threshold_ = GraphKernelFlags::GetInstance().recompute_peak_threshold;
    FindCandidates(func_graph);
    return candidates_;
  }

 private:
  OutPosLinkList JudegeTargetAndCaptureSource(const AnfNodePtr &node, const FuncGraphManagerPtr &mng);
  AnfNodePtrList Filter(const AnfNodePtr &source_node, const AnfNodePtr &end_node, int edge_pos,
                        const FuncGraphManagerPtr &mng);
  void FindCandidates(const FuncGraphPtr &func_graph);
  int GetSourceLinkOutPos(const AnfNodePtr &target, int pos);
  std::tuple<OrderedSet<AnfNodePtr>, OutPosLinkMap, MemorySize> GetValidUsers(const AnfNodePtr &node,
                                                                              const FuncGraphManagerPtr &mng);
  MemorySize SelectThreshold(EdgeLifeTimeType type);

  std::map<AnfNodePtr, MemorySize> topo_indice_;
  std::vector<Candidate> candidates_;
  MemorySize lifetime_threshold_{0};
  MemorySize local_peak_threshold_{0};

  void RecomputeCandidatesLog(const std::vector<Candidate> &candidates) const;
};

class GraphKernelRecompute : public opt::Pass {
 public:
  GraphKernelRecompute() : Pass("graph_kernel_recompute") {}
  ~GraphKernelRecompute() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void Process(const Candidate &candidate);
  std::pair<FuncGraphPtr, AnfNodePtrList> CloneGraph(const CNodePtr &source_graph,
                                                     const AnfNodePtrList &recompute_edge);
  void LinkIntoTargetFuncGraph(const Candidate &candidate, const FuncGraphPtr &cloned_func,
                               const AnfNodePtrList &cloned_inputs);

  std::vector<Candidate> candidates_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_RECOMPUTE_H_
