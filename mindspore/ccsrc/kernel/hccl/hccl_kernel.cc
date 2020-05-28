/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/hccl/hccl_kernel.h"
#include "device/ascend/tasksink/runtime_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"

using HcclTaskInfoPtr = std::shared_ptr<ge::model_runner::HcclTaskInfo>;
using ge::model_runner::HcclTaskInfo;
using mindspore::device::ascend::tasksink::RuntimeUtils;

namespace mindspore {
namespace kernel {
void HcclKernelFactory::Registe(const std::string &name, HcclKernelCreater &&fun) {
  hcclKernelMap_.emplace(name, std::move(fun));
}

std::shared_ptr<HcclKernel> HcclKernelFactory::Get(const std::string &name) {
  const auto &map = Get().hcclKernelMap_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

HcclKernelFactory &HcclKernelFactory::Get() {
  static HcclKernelFactory _this;
  return _this;
}

HcclKernel::HcclKernel() : hccl_count_(0), op_type_(HCCL_REP_OP_SUM), root_id_(0), anf_node_(nullptr) {}

HcclKernel::~HcclKernel() {
  hccl_kernel_input_shape_list_.clear();
  hccl_kernel_output_shape_list_.clear();
  hccl_data_type_list_.clear();
  hccl_count_ = 0;
  op_type_ = HCCL_REP_OP_SUM;
  root_id_ = 0;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  anf_node_ = nullptr;
}

bool HcclKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  op_name_ = AnfAlgo::GetCNodeName(anf_node);

  if (!HcomUtil::GetKernelInputShape(anf_node, &hccl_kernel_input_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelInputShape fail!";
    return false;
  }
  if (!HcomUtil::GetKernelOutputShape(anf_node, &hccl_kernel_output_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelOutputShape fail!";
    return false;
  }
  if (!HcomUtil::GetHcomDataType(anf_node, &hccl_data_type_list_)) {
    MS_LOG(ERROR) << "GetHcomDataType fail!";
    return false;
  }
  if (!HcomUtil::GetHcomCount(anf_node, hccl_data_type_list_, hccl_kernel_input_shape_list_, &hccl_count_)) {
    MS_LOG(ERROR) << "GetHcomCount fail!";
    return false;
  }
  if (op_name_ == kAllReduce || op_name_ == kReduceScatter) {
    if (!HcomUtil::GetHcomOperationType(anf_node, &op_type_)) {
      MS_LOG(ERROR) << "GetHcomOperationType fail!";
      return false;
    }
  }
  if (op_name_ == kBroadcast) {
    if (!HcomUtil::GetHcomRootId(anf_node, &root_id_)) {
      MS_LOG(ERROR) << "GetHcomRootId fail!";
      return false;
    }
  }
  HcomUtil::GetHcomGroup(NOT_NULL(anf_node), NOT_NULL(&group_));
  anf_node_ = anf_node;
  return true;
}

const std::vector<size_t> &HcclKernel::GetInputSizeList() const {
  size_t size = 0;
  if (!input_size_list_.empty()) {
    return input_size_list_;
  }
  for (ulong i = 0; i < hccl_data_type_list_.size(); ++i) {
    if (!HcomUtil::GetHcclOpSize(hccl_data_type_list_[i], hccl_kernel_input_shape_list_[i], &size)) {
      MS_LOG(ERROR) << "GetHcclOpInputSize failed";
    }
    input_size_list_.push_back(size);
  }
  return input_size_list_;
}

const std::vector<size_t> &HcclKernel::GetOutputSizeList() const {
  size_t size = 0;
  if (!output_size_list_.empty()) {
    return output_size_list_;
  }
  for (ulong i = 0; i < hccl_data_type_list_.size(); ++i) {
    if (!HcomUtil::GetHcclOpSize(hccl_data_type_list_[i], hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(ERROR) << "GetHcclOpOutputSize failed";
    }
    output_size_list_.push_back(size);
  }
  return output_size_list_;
}

const std::vector<size_t> &HcclKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

std::vector<TaskInfoPtr> HcclKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "inputs or outputs is empty";
  }
  stream_id_ = stream_id;
  std::string hccl_type = AnfAlgo::GetCNodeName(anf_node_);
  MS_EXCEPTION_IF_NULL(inputs.at(0));
  auto input_data_addr = inputs.at(0)->addr;
  MS_EXCEPTION_IF_NULL(outputs.at(0));
  auto output_data_addr = outputs.at(0)->addr;
  void *workspace_address = nullptr;
  const int64_t workspace_num = 0;
  std::vector<uint8_t> private_def;
  hcclDataType_t data_type = hccl_data_type_list_[0];

  MS_LOG(INFO) << "HCCL Task : stream_id=" << stream_id << ", ws_num=" << workspace_num << ", count=" << hccl_count_
               << ", root_id=" << root_id_ << ", op_type=" << static_cast<int>(op_type_)
               << ", data_type=" << static_cast<int>(data_type);

  HcclTaskInfoPtr task_info_ptr = std::make_shared<HcclTaskInfo>(
    stream_id, hccl_type, input_data_addr, output_data_addr, workspace_address, workspace_num, 0, private_def, nullptr,
    hccl_count_, root_id_, op_type_, data_type, group_, RuntimeUtils::HcomBindModel, RuntimeUtils::HcomUnbindModel,
    RuntimeUtils::HcomDistribute);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  return {task_info_ptr};
}
}  // namespace kernel
}  // namespace mindspore
