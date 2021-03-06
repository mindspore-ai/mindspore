/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "ir/anf.h"
#include "backend/optimizer/common/pass.h"
#include "backend/optimizer/common/fusion_id_allocator.h"
#include "runtime/device/kernel_info.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
const int8_t MAX_ELTWISE_NUM = 3;
const int8_t MIN_ELTWISE_SIZE = 2;
const int8_t ELTWISE_INPUT_SIZE = 2;
const int8_t ELTWISE_DOUBLE_IN_INPUT_SIZE = 3;
const int8_t ELTWISE_DOUBLE_OUTPUT_SIZE = 2;
const int8_t CONV_DOUBLE_IN_INPUT_SIZE = 3;
const int8_t CONV_QUART_IN_INPUT_SIZE = 5;
const int8_t ELTWISE_USE = 1;
const int8_t ELTWISE_MULTI_USE = 2;
const int8_t MAX_ELTWISE_SIZE = 6;
const int8_t MULTI_ELTWISE_SIZE = 4;
using FusedNodeRecord = std::vector<std::unordered_set<AnfNodePtr>>;

struct BufferFusionInfo_t {
  std::string full_name;
  std::vector<AnfNodePtr> anf_nodes;
  std::vector<AnfNodePtr> inputs_list;
  std::vector<AnfNodePtr> outputs_list;
  kernel::KernelBuildInfoPtr kernel_build_info;
};

class FusionBasePass : public Pass {
 public:
  FusionBasePass(const std::string &name, FusionIdAllocatorPtr idAllocator)
      : Pass(name), fusion_id_allocator(idAllocator) {}
  ~FusionBasePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
  bool MatchUBFusionPattern(const session::KernelGraph &kernel_graph);

 protected:
  virtual void MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                        FusedNodeRecord *candidate_fusion) = 0;
  void SetRecordFusionId(const std::unordered_set<AnfNodePtr> &record);
  bool CheckEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node);
  bool CheckDoubleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node);
  bool CheckMultiOutputEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node);
  FusionIdAllocatorPtr fusion_id_allocator;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_FUSION_BASE_PASS_H_
