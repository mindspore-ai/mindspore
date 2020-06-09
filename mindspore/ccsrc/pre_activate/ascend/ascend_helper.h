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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ASCEND_HELPER_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ASCEND_HELPER_H_

#include <memory>
#include <string>
#include <vector>
#include "device/ascend/kernel_select_ascend.h"
#include "kernel/kernel_query.h"
#include "kernel/tbe/tbe_kernel_select.h"

namespace mindspore {
namespace opt {
class KernelSelect {
 public:
  KernelSelect() = default;
  virtual ~KernelSelect() = default;
  virtual void SelectKernel(const CNodePtr &cnode) { device::ascend::SelectKernelInfo(cnode); }
};
using KernelSelectPtr = std::shared_ptr<KernelSelect>;

class SupportedChecker {
 public:
  SupportedChecker() = default;
  virtual ~SupportedChecker() = default;
  virtual bool CheckAICoreSupported(const AnfNodePtr &anf_node,
                                    const kernel::KernelBuildInfoPtr &select_kernel_build_info) {
    return kernel::IsSupportedByAICore(anf_node, select_kernel_build_info);
  }
  virtual bool CheckAICPUSupported(const AnfNodePtr &anf_node,
                                   const kernel::KernelBuildInfoPtr &select_kernel_build_info) {
    return kernel::IsSupportedByAICPU(anf_node, select_kernel_build_info);
  }
};
using SupportedCheckerPtr = std::shared_ptr<SupportedChecker>;

class KernelQuery {
 public:
  KernelQuery() = default;
  virtual ~KernelQuery() = default;
  virtual void Query(const CNodePtr &kernel_node,
                     std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) {
    kernel::KernelQuery(kernel_node, kernel_info_list);
  }
};
using KernelQueryPtr = std::shared_ptr<KernelQuery>;
void RefreshKernelBuildInfo(const std::string &input_format, const std::string &output_format, const TypeId device_type,
                            const AnfNodePtr &trans_data, const std::vector<kernel::Axis> &reshape_type = {});

CNodePtr NewTransOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const KernelSelectPtr &kernel_select,
                        const bool need_padding, const std::string &op_name);

AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const std::vector<size_t> &origin_shape, const TypeId &origin_type);

AnfNodePtr InsertTransOpForInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select);

AnfNodePtr InsertTransOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                  const KernelSelectPtr &kernel_select);

CNodePtr InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

AnfNodePtr CreateMemcpyAsyncOp(const FuncGraphPtr &graph, const AnfNodePtr &node);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ASCEND_HELPER_H_
