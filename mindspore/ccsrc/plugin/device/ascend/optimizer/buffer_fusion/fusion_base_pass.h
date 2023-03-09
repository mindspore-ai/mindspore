/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_FUSION_BASE_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_FUSION_BASE_PASS_H_
#include <vector>
#include <string>
#include <utility>
#include <unordered_set>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"
#include "plugin/device/ascend/optimizer/fusion_id_allocator.h"
#include "plugin/device/ascend/optimizer/ascend_pass_control.h"
#include "include/backend/kernel_info.h"
#include "kernel/kernel.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace opt {
constexpr size_t MAX_ELTWISE_NUM = 3;
constexpr size_t MIN_ELTWISE_SIZE = 2;
constexpr size_t ELTWISE_INPUT_SIZE = 2;
constexpr size_t ELTWISE_DOUBLE_IN_INPUT_SIZE = 3;
constexpr size_t ELTWISE_SINGLE_OUTPUT_SIZE = 1;
constexpr size_t ELTWISE_DOUBLE_OUTPUT_SIZE = 2;
constexpr size_t CONV_DOUBLE_IN_INPUT_SIZE = 3;
constexpr size_t CONV_QUART_IN_INPUT_SIZE = 5;
constexpr size_t ELTWISE_USE = 1;
constexpr size_t ELTWISE_MULTI_USE = 2;
constexpr size_t MAX_ELTWISE_SIZE = 6;
constexpr size_t MULTI_ELTWISE_SIZE = 4;

constexpr int64_t kBNTrainingUpdateOutputUsedTotalNum = 5;
constexpr int64_t kConvOutputUsedTotalNum = 4;
using FusedNodeRecord = std::vector<mindspore::HashSet<AnfNodePtr>>;

struct BufferFusionInfo_t {
  std::string full_name;
  std::string core_type;
  std::vector<AnfNodePtr> anf_nodes;
  std::vector<AnfNodePtr> inputs_list;
  std::vector<AnfNodePtr> outputs_list;
  // node_id of anf_node corresponding to each input
  std::vector<size_t> nodes_id;
  kernel::KernelBuildInfoPtr kernel_build_info;
  bool all_inputs_to_first_node = true;
  bool all_outputs_from_last_node = true;
};

class FusionBasePass : public PassWithSwitch {
 public:
  FusionBasePass(const std::string &name, FusionIdAllocatorPtr idAllocator)
      : PassWithSwitch(name), fusion_id_allocator(std::move(idAllocator)) {}
  ~FusionBasePass() override = default;
  bool MatchUBFusionPattern(const session::KernelGraph &kernel_graph);
  virtual void MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                        FusedNodeRecord *candidate_fusion) = 0;

 protected:
  bool RunPass(const FuncGraphPtr &graph) override;
  void SetRecordFusionId(const mindspore::HashSet<AnfNodePtr> &record);
  bool CheckEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                        const std::unordered_set<std::string> &fusion_types, size_t input_size,
                        size_t not_updatestate_size);

  bool CheckSingleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) {
    return CheckSingleInEltWiseNode(kernel_graph, node, {kernel::kPatternElemWise});
  }
  bool CheckSingleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                const std::unordered_set<std::string> &fusion_types);

  bool CheckDoubleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) {
    return CheckDoubleInEltWiseNode(kernel_graph, node, {kernel::kPatternElemWise});
  }
  bool CheckDoubleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                const std::unordered_set<std::string> &fusion_types);

  bool CheckMultiOutputEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) {
    return CheckMultiOutputEltWiseNode(kernel_graph, node, {kernel::kPatternElemWise});
  }
  bool CheckMultiOutputEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                   const std::unordered_set<std::string> &fusion_types);

  size_t GetNotUpdateStateUserNums(const session::KernelGraph &kernel_graph, const AnfNodePtr &node) const;
  FusionIdAllocatorPtr fusion_id_allocator;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_FUSION_BASE_PASS_H_
