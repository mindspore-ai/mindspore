/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/hccl/hcom_all_to_all.h"
#include "runtime/hccl_adapter/hccl_adapter.h"
#include "runtime/device/ascend/ge_runtime/task_info.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore::kernel {
HcomAllToAllKernel::HcomAllToAllKernel() {}

HcomAllToAllKernel::~HcomAllToAllKernel() {}

bool HcomAllToAllKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &, void *) {
  return true;
}

bool HcomAllToAllKernel::Init(const AnfNodePtr &anf_node) {
  bool ret = HcclKernel::Init(anf_node);
  if (!ret) {
    return ret;
  }

  if (hccl_data_type_list_.empty()) {
    auto recv_type = AnfAlgo::GetNodeAttr<TypePtr>(anf_node, kAttrRecvType);
    MS_EXCEPTION_IF_NULL(recv_type);
    data_type_ = HcomUtil::ConvertHcclType(recv_type->type_id());
  } else {
    data_type_ = hccl_data_type_list_[0];
  }

  workspace_size_list_ = {LongToSize(hccl::HcclAdapter::GetInstance().CalcWorkspaceSize(anf_node, data_type_))};
  return true;
}

const std::vector<size_t> &HcomAllToAllKernel::GetOutputSizeList() const {
  if (!output_size_list_.empty()) {
    return output_size_list_;
  }
  for (size_t i = 0; i < hccl_kernel_output_shape_list_.size(); ++i) {
    size_t size = 0;
    if (!HcomUtil::GetHcclOpSize(data_type_, hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(EXCEPTION) << "AllToAllv get output size failed.";
    }
    output_size_list_.push_back(size);
  }
  return output_size_list_;
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
    if (sec_ret != 0) {
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
