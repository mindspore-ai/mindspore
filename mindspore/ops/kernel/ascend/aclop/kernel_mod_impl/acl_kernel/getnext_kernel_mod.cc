/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel/getnext_kernel_mod.h"
#include <memory>
#include <vector>
#include <string>
#include "plugin/ascend/res_manager/data_queue/ascend_data_queue.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "primitive/structure_op_name.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "tools/profiler/mstx/mstx_impl.h"

namespace mindspore {
namespace kernel {
bool GetNextAclKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ == kDynamicGetNextV2OpName) {
    kernel_name_ = kDynamicGetNextAscendOpName;
  }
  converter_ = std::make_shared<device::ascend::AclConverter>();
  converter_->ConvertToAclOpType(kernel_name_);
  converter_->ProcessRunnerSpecialInfo(kernel_name_, output_params_, is_dynamic_);
  return true;
}

int GetNextAclKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  GilReleaseWithCheck gil_release;
  RefreshAclConverter(inputs);
  GetInputInfo(inputs);

  std::vector<std::vector<int64_t>> new_output_shapes;
  if (is_dynamic_) {
    auto wingman_queue = GetTdtWingManQueue(primitive_);
    MS_EXCEPTION_IF_NULL(wingman_queue);
    std::vector<device::DataQueueItem> data;
    RetryPeakItemFromDataQueue(nullptr, wingman_queue, &data);
    MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == data.size(), "Size of output is not equal to size of data");

    output_size_list_.clear();
    output_size_list_.resize(outputs.size(), 0);
    for (size_t i = 0; i < outputs.size(); i++) {
      auto cur_shape = data[i].shapes;
      if (cur_shape.empty()) {
        cur_shape.push_back(1);
      }
      PackageOutput(i, cur_shape);
      if (output_size_list_[i] != data[i].data_len) {
        MS_LOG(EXCEPTION) << "GetNext calc error data_len:" << output_size_list_[i]
                          << ", right data_len is:" << data[i].data_len << ", input index is:" << i;
      }
      // update output info for data_source_actor
      outputs[i]->SetShapeVector(data[i].shapes);
      outputs[i]->set_size(data[i].data_len);
      (void)new_output_shapes.emplace_back(cur_shape);
    }

    if (primitive_->GetAttr("shapes") == nullptr) {
      MS_LOG(EXCEPTION) << "output shapes attr must be in getnext, please check!";
    }
  } else {
    for (size_t i = 0; i < outputs.size(); i++) {
      auto cur_shape = outputs[i]->GetShapeVector();
      if (cur_shape.empty()) {
        cur_shape.push_back(1);
      }
      PackageOutput(i, cur_shape);
      (void)new_output_shapes.emplace_back(cur_shape);
    }
  }
  primitive_->set_attr("shapes", MakeValue(new_output_shapes));

  if (device::ascend::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, nullptr);
  }
  return 0;
}

bool GetNextAclKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto wingman_queue = GetTdtWingManQueue(primitive_);
  MS_EXCEPTION_IF_NULL(wingman_queue);
  if (wingman_queue->Size() > 0) {
    (void)wingman_queue->Pop();
  }

  uint64_t range_id = 0;
  MSTX_START_WITHOUT_DOMAIN(range_id, mindspore::profiler::MSTX_GETNEXT, stream_ptr);
  auto ret = AclKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  MSTX_END_WITHOUT_DOMAIN(range_id);
  return ret;
}
MS_KERNEL_FACTORY_REG(AclKernelMod, GetNext, GetNextAclKernelMod);

bool IsGetNextOp(const std::string &op_name) { return op_name == kGetNextOpName || op_name == kDynamicGetNextV2OpName; }

std::shared_ptr<device::BlockingQueue> GetTdtWingManQueue(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!IsGetNextOp(prim->name())) {
    return nullptr;
  }
  auto queue_name = GetValue<std::string>(prim->GetAttr("shared_name"));
  if (!device::DataQueueMgr::GetInstance().IsCreated(queue_name)) {
    return nullptr;
  }
  return device::DataQueueMgr::GetInstance().GetDataQueue(queue_name);
}

std::shared_ptr<device::BlockingQueue> GetTdtWingManQueue(const std::shared_ptr<AnfNode> &node) {
  if (!common::AnfAlgo::IsGetNextNode(node)) {
    return nullptr;
  }
  return GetTdtWingManQueue(common::AnfAlgo::GetCNodePrimitive(node));
}

void CloseTdtWingManQueue(const PrimitivePtr &prim) {
  if (!IsGetNextOp(prim->name())) {
    return;
  }
  auto wingman = GetTdtWingManQueue(prim);
  if (wingman && wingman->IsOpen()) {
    wingman->Close();
  }
}

void CloseTdtWingManQueue(const std::shared_ptr<AnfNode> &node) {
  if (!common::AnfAlgo::IsGetNextNode(node)) {
    return;
  }
  CloseTdtWingManQueue(common::AnfAlgo::GetCNodePrimitive(node));
}
}  // namespace kernel
}  // namespace mindspore
