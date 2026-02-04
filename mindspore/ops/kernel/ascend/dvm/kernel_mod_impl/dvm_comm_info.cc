/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/dvm/kernel_mod_impl/dvm_comm_info.h"

#include <string>
#include <unordered_set>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "plugin/ascend/res_manager/collective/multi_ascend_collective_comm_lib.h"

namespace mindspore {
namespace graphkernel {
bool EnableDvmComm() {
  // There are 2 requirements for enabling collective communication in DVM
  // 1. Jit level is set to O1
  // 2. At least one of the collective communication primitives is included in `enable_cluster_ops` or
  // `enable_cluster_ops_only` of graph kernel flags
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b") {
    return false;
  }

  const auto &jit_level = MsContext::GetInstance()->GetJitLevel();
  if (jit_level != "O1") {
    return false;
  }
  const auto &gk_flags = graphkernel::GraphKernelFlags::GetInstance();
  const auto &enable_cluster_ops = gk_flags.enable_cluster_ops;
  const auto &enable_cluster_ops_only = gk_flags.enable_cluster_ops_only;
  auto check_func = [](const std::string &op) {
    return op == "AllReduce" || op == "AllGather" || op == "ReduceScatter";
  };
  if (std::any_of(enable_cluster_ops.begin(), enable_cluster_ops.end(), check_func)) {
    return true;
  }
  if (std::any_of(enable_cluster_ops_only.begin(), enable_cluster_ops_only.end(), check_func)) {
    return true;
  }
  return false;
}

DvmCommInfo::DvmCommInfo() = default;

DvmCommInfo::~DvmCommInfo() = default;

bool DvmCommInfo::EnableComm() {
  static bool enable_comm = EnableDvmComm();
  return enable_comm;
}

bool DvmCommInfo::IsTargetCommOp(const AnfNodePtr node) {
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr || !prim->HasAttr(kAttrGroup)) {
    return false;
  }
  const std::string &group = GetValue<std::string>(prim->GetAttr(kAttrGroup));
  const std::unordered_set<std::string> &dvm_comm_enabled_groups =
    device::ascend::MultiAscendCollectiveCommLib::GetInstance().GetDvmCommEnabledGroups();
  if (EnableComm() && dvm_comm_enabled_groups.count(group) != 0) {
    return true;
  }
  return false;
}
}  // namespace graphkernel
}  // namespace mindspore
