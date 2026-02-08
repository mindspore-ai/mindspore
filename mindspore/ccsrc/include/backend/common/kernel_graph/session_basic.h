/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_SESSION_BASIC_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_SESSION_BASIC_H

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include <set>

#include "include/backend/common/kernel_graph/kernel_graph_mgr.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/tensor_future.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "include/utils/contract.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "utils/ms_context.h"
#include "pynative/utils/base.h"

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
class GraphCompiler;
}  // namespace runtime
}  // namespace mindspore

namespace mindspore {
const char kSessionBasic[] = "SessionBasic";

namespace session {
struct BackendOpRunInfo {
  ~BackendOpRunInfo() = default;
  BackendOpRunInfo(pynative::BaseOpRunInfo base_op_run_info, PrimitivePtr prim, bool is_infer, bool is_gradient_out)
      : base_op_run_info(std::move(base_op_run_info)),
        op_prim(std::move(prim)),
        is_infer(is_infer),
        is_gradient_out(is_gradient_out) {}

  pynative::BaseOpRunInfo base_op_run_info;
  PrimitivePtr op_prim;
  bool is_infer = false;
  bool is_gradient_out = false;
};
using BackendOpRunInfoPtr = std::shared_ptr<BackendOpRunInfo>;

class BACKEND_COMMON_EXPORT SessionBasic : public KernelGraphMgr, public std::enable_shared_from_this<SessionBasic> {
 public:
  using KernelGraphMgr::ConstructKernelGraph;
  SessionBasic() : device_id_(0) {}
  void RegisterSummaryCallBackFunc();
  // create a single run op graph
  std::shared_ptr<KernelGraph> ConstructSingleOpGraph(const BackendOpRunInfoPtr &op_run_info,
                                                      const std::vector<ValuePtr> &input_values,
                                                      const std::vector<InputType> &input_type);
  void DumpGraphs(const std::vector<KernelGraphPtr> &graphs) const;
  void RecurseSetSummaryNodesForAllGraphs(KernelGraph *graph);
  void Summary(KernelGraph *graph);

 protected:
  friend class mindspore::runtime::GraphCompiler;
  // create graph output for RunOp
  void CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) const;

  uint32_t device_id_;
  // rank id of physical device
  uint32_t rank_id_{0};
};

using SessionPtr = std::shared_ptr<session::SessionBasic>;
}  // namespace session
BACKEND_COMMON_EXPORT void DumpGraphExeOrder(const std::string &file_name, const std::string &target_dir,
                                             const std::vector<CNodePtr> &execution_order);
uint32_t GetRankId();
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_SESSION_BASIC_H
