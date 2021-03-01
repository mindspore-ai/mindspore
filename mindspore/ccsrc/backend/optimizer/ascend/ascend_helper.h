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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_HELPER_H_

#include <memory>
#include <string>
#include <set>
#include <vector>
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "backend/kernel_compiler/kernel_query.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"

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
  virtual bool IsTbeRef(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      return false;
    }
    auto op_info = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(AnfAlgo::GetCNodeName(node), node);
    if (op_info != nullptr) {
      return op_info->is_ref();
    }
    return false;
  }
};
using KernelQueryPtr = std::shared_ptr<KernelQuery>;

class OpFinder {
 public:
  OpFinder() = default;
  virtual ~OpFinder() = default;
  virtual int GetOpRegisteredOutputNum(const std::string &op_name, const CNodePtr &cnode) {
    auto op_info = kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, cnode);
    if (op_info == nullptr) {
      return -1;
    }
    return op_info->outputs_ptr().size();
  }
};
using OpFinderPtr = std::shared_ptr<OpFinder>;

void RefreshKernelBuildInfo(const std::string &input_format, const std::string &output_format,
                            const AnfNodePtr &trans_data, const std::string &reshape_type = {""},
                            const TypeId &type_id = kTypeUnknown);

CNodePtr NewTransOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const KernelSelectPtr &kernel_select,
                        const bool need_padding, const std::string &op_name,
                        const std::vector<int64_t> &perm = std::vector<int64_t>{});

CNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                              const TypeId &input_type, const TypeId &output_type,
                              const std::vector<size_t> &origin_shape, const TypeId &origin_type,
                              const std::string &reshape_type = std::string{});

AnfNodePtr InsertTransOpForInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select);

AnfNodePtr InsertTransOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                  const KernelSelectPtr &kernel_select);

CNodePtr InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

AnfNodePtr CreateMemcpyAsyncOp(const FuncGraphPtr &graph, const AnfNodePtr &node);

AnfNodePtr AddTransOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select, size_t insert_index, bool is_insert_input);

const std::set<std::string> kCommonFormatSet = {kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NCDHW};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_HELPER_H_
