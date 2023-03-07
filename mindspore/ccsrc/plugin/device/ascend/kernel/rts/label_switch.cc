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

#include "plugin/device/ascend/kernel/rts/label_switch.h"
#include <string>
#include "runtime/stream.h"
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

using mindspore::ge::model_runner::LabelSwitchTaskInfo;
using LabelSwitchTaskInfoPtr = std::shared_ptr<LabelSwitchTaskInfo>;

namespace mindspore {
namespace kernel {
LabelSwitchKernel::LabelSwitchKernel() { label_list_ = {}; }

LabelSwitchKernel::~LabelSwitchKernel() { cond_ = nullptr; }

bool LabelSwitchKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_LOG(INFO) << "LabelSwitchKernel init";
  auto cnode = anf_node->cast<CNodePtr>();
  if (!common::AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cnode)) {
    MS_LOG(EXCEPTION) << "LabelSwitchKernel has no attr label_switch_list";
  }
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  label_list_ = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrLabelSwitchList));
  label_size_ = SizeToUint(label_list_.size());
  MS_LOG(INFO) << "LabelSwitchKernel get attr label size:" << label_size_;
  for (auto label : label_list_) {
    MS_LOG(INFO) << "label: " << label;
  }
  return true;
}

bool LabelSwitchKernel::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &, void *) {
  MS_LOG(INFO) << "LabelSwitchKernel launch";
  return true;
}

std::vector<TaskInfoPtr> LabelSwitchKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                    uint32_t stream_id) {
  MS_LOG(INFO) << "LabelSwitchKernel GenTask label size:" << label_size_ << ", stream id:" << stream_id;
  std::vector<TaskInfoPtr> task_info_list;
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "LabelSwitchKernel is empty";
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  cond_ = inputs[0]->addr;
  auto task_info_ptr = std::make_shared<LabelSwitchTaskInfo>(unique_name_, stream_id, label_size_, label_list_, cond_);
  MS_EXCEPTION_IF_NULL(task_info_ptr);
  (void)task_info_list.emplace_back(task_info_ptr);
  return task_info_list;
}

std::vector<std::shared_ptr<kernel::KernelBuildInfo>> LabelSwitchDesc::GetKernelInfo(const CNodePtr &) {
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> label_switch_build_info{};
  std::vector<string> input_format{kOpFormat_DEFAULT};
  std::vector<TypeId> input_type{kNumberTypeUInt64};
  if (input_format.size() != input_type.size()) {
    MS_LOG(EXCEPTION) << "Invalid param num, input_format size " << input_format.size() << " input_type size "
                      << input_type.size();
  }
  for (size_t i = 0; i < input_format.size(); ++i) {
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    builder.SetInputsFormat({input_format[i]});
    builder.SetInputsDeviceType({input_type[i]});
    builder.SetProcessor(AICORE);
    builder.SetKernelType(RT_KERNEL);
    builder.SetFusionType(kPatternOpaque);
    // LabelSwitch always return UMonad.
    builder.SetOutputsFormat({kOpFormat_DEFAULT});
    builder.SetOutputsDeviceType({TypeId::kObjectTypeUMonad});
    builder.SetInputsKernelObjectType({KernelObjectType::TENSOR});
    builder.SetOutputsKernelObjectType({KernelObjectType::TENSOR});
    label_switch_build_info.emplace_back(builder.Build());
  }
  return label_switch_build_info;
}
}  // namespace kernel
}  // namespace mindspore
