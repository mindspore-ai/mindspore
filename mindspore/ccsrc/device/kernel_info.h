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
#include "kernel/kernel_build_info.h"
#include "device/ascend/ascend_device_address.h"
#include "kernel/kernel.h"

namespace mindspore {
const uint32_t kInvalidGraphId = UINT32_MAX;
const uint32_t kInvalidDistincLabel = UINT32_MAX;
namespace device {
class KernelInfo {
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

  const kernel::KernelBuildInfo *select_kernel_build_info() const;
  kernel::KernelBuildInfoPtr GetMutableSelectKernelBuildInfo() const;
  void set_select_kernel_build_info(const kernel::KernelBuildInfoPtr &select_kernel_build_info) {
    select_kernel_build_info_ = select_kernel_build_info;
  }
  void SetFeatureMapFlag(bool flag) { is_feature_map_ = flag; }
  const DeviceAddress *GetOutputAddr(size_t index) const;
  DeviceAddressPtr GetMutableOutputAddr(size_t index) const;
  bool OutputAddrExist(size_t index) const;
  bool SetOutputAddr(const DeviceAddressPtr &output_address, size_t index);
  DeviceAddress *GetWorkspaceAddr(size_t index) const;
  bool SetWorkspaceAddr(const DeviceAddressPtr &output_address, size_t index);
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

 private:
  bool is_feature_map_;
  kernel::KernelBuildInfoPtr select_kernel_build_info_;
  std::vector<std::shared_ptr<DeviceAddress>> output_address_list_;
  std::vector<std::shared_ptr<DeviceAddress>> workspace_address_list_;
  kernel::KernelModPtr kernel_mod_;
  // stream_id_ is the index of stream object vector
  uint32_t stream_id_;
  // stream_distinction_label_ is used mark different op in different stream
  uint32_t stream_distinction_label_;
  // record which graph the node belong to
  uint32_t graph_id_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_KERNEL_INFO_H_
