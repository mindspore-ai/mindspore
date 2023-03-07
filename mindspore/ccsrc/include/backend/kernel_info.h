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

#ifndef MINDSPORE_DEVICE_KERNEL_INFO_H_
#define MINDSPORE_DEVICE_KERNEL_INFO_H_

#include <vector>
#include <memory>
#include <utility>
#include "ir/kernel_info_dev.h"
#include "kernel/kernel_build_info.h"
#include "kernel/kernel.h"
#include "include/backend/device_address.h"
#include "include/backend/visible.h"

namespace mindspore {
const uint32_t kInvalidGraphId = UINT32_MAX;
const uint32_t kInvalidDistincLabel = UINT32_MAX;
namespace device {
class BACKEND_EXPORT KernelInfo : public KernelInfoDevice {
 public:
  KernelInfo() {
    kernel_mod_ = nullptr;
    is_feature_map_ = false;
    select_kernel_build_info_ = nullptr;
    output_address_list_ = {};
    workspace_address_list_ = {};
    stream_id_ = UINT32_MAX;
    stream_distinction_label_ = kInvalidDistincLabel;
    graph_id_ = kInvalidGraphId;
  }
  virtual ~KernelInfo() = default;

  bool has_build_info() const override { return select_kernel_build_info_ != nullptr; }
  const kernel::KernelBuildInfo *select_kernel_build_info() const;
  kernel::KernelBuildInfoPtr GetMutableSelectKernelBuildInfo() const;
  void set_select_kernel_build_info(const kernel::KernelBuildInfoPtr &select_kernel_build_info) {
    select_kernel_build_info_ = select_kernel_build_info;
  }
  void set_feature_map_flag(bool flag) { is_feature_map_ = flag; }
  const DeviceAddress *GetOutputAddr(size_t index) const;
  DeviceAddressPtr GetMutableOutputAddr(size_t index) const;
  bool OutputAddrExist(size_t index) const;
  bool SetOutputAddr(const DeviceAddressPtr &output_address, size_t index);
  DeviceAddress *GetWorkspaceAddr(size_t index) const;
  DeviceAddressPtr GetMutableWorkspaceAddr(size_t index) const;
  bool WorkspaceAddrExist(size_t index) const;
  bool SetWorkspaceAddr(const DeviceAddressPtr &output_address, size_t index);
  // The number of workspace may change after kernel Resize.
  void set_workspace_address_list(const std::vector<DeviceAddressPtr> &workspace_address_list) {
    workspace_address_list_ = workspace_address_list;
  }
  void set_kernel_mod(const kernel::KernelModPtr &kernel_mod);
  kernel::KernelMod *MutableKernelMod() const;
  const kernel::KernelMod *kernel_mod() const;
  uint32_t stream_id() const { return stream_id_; }
  void set_stream_id(uint32_t stream_id) { stream_id_ = stream_id; }
  uint32_t stream_distinction_label() const { return stream_distinction_label_; }
  void set_stream_distinction_label(uint32_t stream_distinction_label) {
    stream_distinction_label_ = stream_distinction_label;
  }
  void set_graph_id(uint32_t graph_id) { graph_id_ = graph_id; }
  uint32_t graph_id() const { return graph_id_; }
  bool operator==(const KernelInfo &other) const;
  bool is_feature_map() const { return is_feature_map_; }

  const std::vector<std::shared_ptr<DeviceAddress>> &output_address_list() const { return output_address_list_; }
  const std::vector<std::shared_ptr<DeviceAddress>> &workspace_address_list() const { return workspace_address_list_; }

  // Set output and input reference. If all_ref is set true, each output is a reference to the input with the same
  // index.
  void set_ref_map(const bool &all_ref, const OutputInputRefMap &ref_map);
  const OutputInputRefMap &out_in_ref_map() const { return out_in_ref_map_; }

  // The interface of somas.
  bool SetSomasResult(std::vector<std::pair<size_t, size_t>> &&output_somas_result,
                      std::vector<std::pair<size_t, size_t>> &&workspace_somas_result);
  size_t GetTensorSomasOffset(const std::vector<std::pair<size_t, size_t>> &somas_result, size_t tensor_index) const;
  size_t GetTensorSomasAlignedSize(const std::vector<std::pair<size_t, size_t>> &somas_result,
                                   size_t tensor_index) const;
  bool IsTensorEnableSomas(const std::vector<std::pair<size_t, size_t>> &somas_result, size_t tensor_index) const;
  const std::vector<std::pair<size_t, size_t>> &somas_output_result() const { return somas_output_result_; }
  const std::vector<std::pair<size_t, size_t>> &somas_workspace_result() const { return somas_workspace_result_; }

 private:
  bool is_feature_map_;
  kernel::KernelBuildInfoPtr select_kernel_build_info_;
  std::vector<std::shared_ptr<DeviceAddress>> output_address_list_;
  std::vector<std::shared_ptr<DeviceAddress>> workspace_address_list_;
  // pair<size_t, size_t> : (offset, aligned_size)
  // aligned_size of 0 means no memory allocation
  std::vector<std::pair<size_t, size_t>> somas_output_result_;
  // pair<size_t, size_t> : (offset, aligned_size)
  // aligned_size of 0 means no memory allocation
  std::vector<std::pair<size_t, size_t>> somas_workspace_result_;
  kernel::KernelModPtr kernel_mod_;
  // stream_id_ is the index of stream object vector
  uint32_t stream_id_;
  // stream_distinction_label_ is used mark different op in different stream
  uint32_t stream_distinction_label_;
  // record which graph the node belong to
  uint32_t graph_id_;
  // The map between kernel's output and input ref relationship.
  OutputInputRefMap out_in_ref_map_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_KERNEL_INFO_H_
