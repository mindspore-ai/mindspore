/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/kernel_query.h"
#include "kernel/oplib/oplib.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_kernel_select.h"

namespace mindspore {
const std::set<TypeId> kFloatDataTypeSet = {kNumberTypeFloat16, kNumberTypeFloat32};
namespace opt {
class KernelSelect {
 public:
  KernelSelect() = default;
  virtual ~KernelSelect() = default;
  virtual void SelectKernel(const CNodePtr &cnode) { device::ascend::SelectKernelInfo(cnode); }
};
using KernelSelectPtr = std::shared_ptr<KernelSelect>;

bool CheckAICoreSupported(const AnfNodePtr &anf_node);
bool CheckAICoreSupportedAny(const AnfNodePtr &anf_node);
bool CheckAICoreSupportedSpec(const AnfNodePtr &anf_node, const kernel::KernelBuildInfoPtr &select_kernel_build_info);
bool CheckAICPUSupportedSpec(const AnfNodePtr &anf_node, const kernel::KernelBuildInfoPtr &select_kernel_build_info);
std::string GetInputName(const CNodePtr &origin_op, size_t input_index);
ValuePtr UpdateValueByAttrDataType(const ValuePtr &value, const std::string &attr_data_type);

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
    auto op_info = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(common::AnfAlgo::GetCNodeName(node), node);
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

void RefreshKernelBuildInfo(const KernelSelectPtr &kernel_select, const std::string &input_format,
                            const std::string &output_format, const AnfNodePtr &trans_node,
                            const std::string &reshape_type = {""}, const TypeId &type_id = kTypeUnknown);

CNodePtr NewTransOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const AnfNodePtr &orig_node,
                        const KernelSelectPtr &kernel_select, const bool need_padding, const std::string &op_name,
                        const std::vector<int64_t> &perm = std::vector<int64_t>{});

ValueNodePtr CreatePermValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &perm);

CNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const AnfNodePtr &orig_node,
                              const std::string &format, const TypeId &input_type, const TypeId &output_type,
                              const abstract::BaseShapePtr &origin_shape, const TypeId &origin_type,
                              const std::string &reshape_type = std::string{});

AnfNodePtr InsertTransOpForInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select);

AnfNodePtr InsertTransOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &orig_node, const AnfNodePtr &node,
                                  const KernelSelectPtr &kernel_select);

CNodePtr InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

AnfNodePtr AddTransOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select, size_t insert_index, bool is_insert_input);

AnfNodePtr AddTransOpNodeToGraphWithFormat(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                           const AnfNodePtr &node, const KernelSelectPtr &kernel_select,
                                           const std::string &input_format, const std::string &dst_format,
                                           const std::string &reshape_type, const TypeId &type_id = kTypeUnknown,
                                           int64_t groups = 1);

void SetInputOutputNames(const std::vector<std::string> &input_names, const std::vector<std::string> &output_names,
                         const AnfNodePtr &node);

void SelectCallInlineKernelInfo(const CNodePtr &node);

const std::set<std::string> kCommonFormatSet = {kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NCDHW};

inline bool NeedInsertTransData(const ShapeVector &origin_shape, const std::string &format) {
  bool shape_check =
    origin_shape.size() > 1 || (origin_shape.size() == 1 && origin_shape[0] % SizeToLong(kCubeSize) != 0);
  return kCommonFormatSet.find(format) == kCommonFormatSet.end() && (shape_check || format == kOpFormat_ND_RNN_BIAS);
}

void NormalizeReduceAttrAxis(const CNodePtr &cnode);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_ASCEND_HELPER_H_
