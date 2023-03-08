/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/hccl/hcom_all_to_all.h"
#include <algorithm>
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"

namespace mindspore::kernel {
HcomAllToAllKernel::HcomAllToAllKernel() {}

HcomAllToAllKernel::~HcomAllToAllKernel() {}

bool HcomAllToAllKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcclAllToAll launch";
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid AllToAll input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllToAll(inputs[0]->addr, outputs[0]->addr, params_,
                                                                   data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllToAll failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool HcomAllToAllKernel::Init(const AnfNodePtr &anf_node) {
  bool ret = HcclKernel::Init(anf_node);
  if (!ret) {
    return ret;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrNeedDropInput, cnode)) {
    need_drop_input_ = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrNeedDropInput);
  }

  if (hccl_data_type_list_.empty()) {
    auto recv_type = common::AnfAlgo::GetNodeAttr<TypePtr>(anf_node, kAttrRecvType);
    MS_EXCEPTION_IF_NULL(recv_type);
    data_type_ = HcomUtil::ConvertHcclType(recv_type->type_id());
  } else {
    data_type_ = hccl_data_type_list_[0];
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK)) {
    mutable_workspace_size_list_ = {
      LongToSize(hccl::HcclAdapter::GetInstance().CalcWorkspaceSize(anf_node, data_type_))};
  }
  uint32_t rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(group_, &rank_size)) {
    MS_LOG(EXCEPTION) << "Get hccl rank size for group " << group_ << " failed.";
  }
  hccl::AllToAllvCalcParam calc(cnode, rank_size);
  calc.CalcOpParam();
  std::transform(calc.GetSendCounts().begin(), calc.GetSendCounts().end(), std::back_inserter(params_.sendcounts),
                 [](int64_t elem) { return static_cast<uint64_t>(elem); });
  std::transform(calc.GetSendDispls().begin(), calc.GetSendDispls().end(), std::back_inserter(params_.sdispls),
                 [](int64_t elem) { return static_cast<uint64_t>(elem); });
  std::transform(calc.GetRecvCounts().begin(), calc.GetRecvCounts().end(), std::back_inserter(params_.recvcounts),
                 [](int64_t elem) { return static_cast<uint64_t>(elem); });
  std::transform(calc.GetRecvDispls().begin(), calc.GetRecvDispls().end(), std::back_inserter(params_.rdispls),
                 [](int64_t elem) { return static_cast<uint64_t>(elem); });
  return true;
}

const std::vector<size_t> &HcomAllToAllKernel::GetOutputSizeList() const {
  if (!mutable_output_size_list_.empty()) {
    return mutable_output_size_list_;
  }
  size_t size = 0;
  for (size_t i = 0; i < hccl_kernel_output_shape_list_.size(); ++i) {
    if (!HcomUtil::GetHcclOpSize(data_type_, hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(EXCEPTION) << "AllToAllv get output size failed.";
    }
    mutable_output_size_list_.push_back(size);
  }
  return mutable_output_size_list_;
}

void HcomAllToAllKernel::UpdateOutputSizeList() {
  auto anf_node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(anf_node);
  size_t size = 0;
  hccl_kernel_output_shape_list_.clear();
  mutable_output_size_list_.clear();
  if (!HcomUtil::GetKernelOutputShape(anf_node, &hccl_kernel_output_shape_list_)) {
    MS_LOG(EXCEPTION) << "GetKernelOutputShape fail!";
  }

  for (size_t i = 0; i < hccl_kernel_output_shape_list_.size(); ++i) {
    if (!HcomUtil::GetHcclOpSize(data_type_, hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(EXCEPTION) << "AllToAllv get output size failed in Update stage";
    }
    mutable_output_size_list_.push_back(size);
  }
}

std::vector<TaskInfoPtr> HcomAllToAllKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  auto anf_node = anf_node_.lock();
  if (!anf_node) {
    MS_LOG(EXCEPTION) << "anf_node pointer is expired.";
  }

  stream_id_ = stream_id;
  void *input_data_addr = inputs.empty() ? nullptr : inputs.at(0)->addr;
  void *output_data_addr = outputs.empty() ? nullptr : outputs.at(0)->addr;

  // if send empty, remove the input that added for depend
  if (need_drop_input_) {
    input_data_addr = nullptr;
  }

  std::vector<uint8_t> private_def;
  std::vector<hccl::HcclTaskInfo> task_info;
  bool ret = hccl::HcclAdapter::GetInstance().GenTask(anf_node, data_type_, &task_info);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Gen Task for " << anf_node->DebugString() << " failed.";
  }

  std::vector<TaskInfoPtr> results;
  for (auto &task : task_info) {
    MS_LOG(INFO) << "AlltoAll Task : stream_id=" << stream_id << ", count=" << hccl_count_ << ", root_id=" << root_id_
                 << ", op_type=" << static_cast<int>(op_type_) << ", data_type=" << static_cast<int>(data_type_)
                 << ", workspace_size=" << task.workspace_size << ", stream_num=" << task.stream_num
                 << ", private_def_size=" << task.private_def.size();

    private_def.resize(task.private_def.size());
    auto sec_ret = memcpy_s(private_def.data(), private_def.size(), task.private_def.data(), task.private_def.size());
    if (sec_ret != EOK) {
      MS_LOG(EXCEPTION) << "Set data memcpy_s failed, ret = " << sec_ret;
    }

    void *workspace_addr = nullptr;
    if (task.workspace_size != 0) {
      if (workspace.empty()) {
        MS_LOG(EXCEPTION) << "Workspace size list of " << anf_node->DebugString() << " is empty";
      }
      MS_EXCEPTION_IF_NULL(workspace.at(0));
      workspace_addr = workspace.at(0)->addr;
    }

    results.emplace_back(std::make_shared<ge::model_runner::HcclTaskInfo>(
      unique_name_, stream_id, hccl::HcclAdapter::GetHcclType(anf_node), input_data_addr, output_data_addr,
      workspace_addr, task.workspace_size, task.stream_num, private_def,
      hccl::HcclAdapter::GetInstance().GetHcclOpsKernelInfoStore(), hccl_count_, root_id_, op_type_, data_type_, group_,
      NeedDump()));
  }

  return results;
}
MS_HCCL_REG_KERNEL(AllToAllv, HcomAllToAllKernel);
}  // namespace mindspore::kernel
