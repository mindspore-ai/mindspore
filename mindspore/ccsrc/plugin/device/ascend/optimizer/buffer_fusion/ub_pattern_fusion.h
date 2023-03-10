/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
#include <string>
#include <map>
#include "utils/hash_map.h"
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
class UbPatternFusion : public PassWithSwitch {
 public:
  UbPatternFusion() : PassWithSwitch("TbeBufferFusion") {}
  ~UbPatternFusion() override = default;

 protected:
  bool RunPass(const FuncGraphPtr &graph) override;

 private:
  void GetBufferFusionInfo(session::KernelGraph *kernel_graph,
                           mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos) const;
  bool ReplaceFusionOp(mindspore::HashMap<int64_t, BufferFusionInfo_t> *buffer_fusion_infos, int64_t fusion_id,
                       session::KernelGraph *kernel_graph) const;
  bool FuseBufferFusionPattern(session::KernelGraph *kernel_graph) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_BUFFER_FUSION_UB_PATTERN_FUSION_H_
