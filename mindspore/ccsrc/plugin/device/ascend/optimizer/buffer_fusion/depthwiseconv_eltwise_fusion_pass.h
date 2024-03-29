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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_DEPTHWISECONV_ELTWISE_FUSION_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_PASS_DEPTHWISECONV_ELTWISE_FUSION_PASS_H_

#include <utility>
#include "utils/hash_set.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/fusion_base_pass.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"
#include "plugin/device/ascend/optimizer/fusion_id_allocator.h"
#include "include/backend/kernel_info.h"
#include "kernel/kernel.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace opt {
class DepthwiseConvEltwiseFusionPass : public FusionBasePass {
 public:
  explicit DepthwiseConvEltwiseFusionPass(FusionIdAllocatorPtr idAllocator)
      : FusionBasePass("DepthwiseConvEltwiseFusionPass", std::move(idAllocator)) {}
  ~DepthwiseConvEltwiseFusionPass() override = default;
  void MatchSingleFusionPattern(const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) override;

 private:
  void MatchDepthwiseConvRelu(const CNodePtr &cnode, FusedNodeRecord *candidate_fusion, bool is_order);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_DEPTHWISECONV_ELTWISE_FUSION_PASS_H_
