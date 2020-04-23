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

namespace mindspore {
namespace opt {
class KernelSelect {
 public:
  KernelSelect() = default;
  virtual ~KernelSelect() = default;
  virtual void SelectKernel(const CNodePtr &cnode) { device::ascend::SelectKernelInfo(cnode); }
  virtual bool CheckKernelAccuracySupported(const CNodePtr &kernel_node,
                                            const kernel::KernelBuildInfoPtr &new_kernel_build_info) {
    return device::ascend::CheckKernelAccuracySupported(kernel_node, new_kernel_build_info);
  }
};
using KernelSelectPtr = std::shared_ptr<KernelSelect>;

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

AnfNodePtr AddTransOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select, size_t insert_index,
                                 const std::string &origin_format, const std::string &dest_format,
                                 const std::string &op_name, bool is_insert_input);

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
