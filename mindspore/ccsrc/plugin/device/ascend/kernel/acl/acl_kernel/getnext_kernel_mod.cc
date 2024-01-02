/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/acl/acl_kernel/getnext_kernel_mod.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "transform/acl_ir/acl_helper.h"
#include "ops/structure_op_name.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace kernel {
bool GetNextAclKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ == kDynamicGetNextV2OpName) {
    kernel_name_ = kDynamicGetNextAscendOpName;
  }
  converter_ = std::make_shared<transform::AclConverter>();
  converter_->ConvertToAclOpType(kernel_name_);
  converter_->ResizeAclOpInputs(primitive_);
  converter_->ConvertAttrToAclInput(primitive_->attrs(), kernel_name_);
  converter_->ConvertInputToAclAttr(inputs, kernel_name_);
  if (transform::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, nullptr);
  }
  converter_->ProcessRunnerSpecialInfo(kernel_name_, output_params_);
  return true;
}

int GetNextAclKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  GilReleaseWithCheck gil_release;
  GetInputInfo(inputs);

  auto wingman_queue = device::GetTdtWingManQueue(primitive_);
  std::vector<device::DataQueueItem> data;
  RetryPeakItemFromDataQueue(nullptr, wingman_queue, &data);
  (void)wingman_queue->Pop();
  MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == data.size(), "Size of output is not equal to size of data");

  output_size_list_.clear();
  output_size_list_.resize(outputs.size(), 0);
  std::vector<std::vector<int64_t>> new_output_shapes;
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
  primitive_->set_attr("shapes", MakeValue(new_output_shapes));

  CreateAclConverter();
  if (transform::AclHelper::IsPrintDebugString()) {
    ms_attr_str_.clear();
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, &ms_attr_str_);
  } else {
    converter_->ConvertToAclAttr(primitive_->attrs(), kernel_name_, nullptr);
  }
  return 0;
}
}  // namespace kernel
}  // namespace mindspore
