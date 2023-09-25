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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_HAL_DEVICE_LAUNCH_TRANSDATA_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_HAL_DEVICE_LAUNCH_TRANSDATA_H_
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include "kernel/kernel.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::device::ascend {
class LaunchTransData {
 public:
  LaunchTransData(void *stream, TypeId dtype, size_t total_size, std::string src_format, std::string dst_format,
                  ShapeVector host_shape, int64_t groups)
      : stream_(stream),
        dtype_(dtype),
        total_size_(total_size),
        src_format_(std::move(src_format)),
        dst_format_(std::move(dst_format)),
        shape_(std::move(host_shape)),
        groups_(groups) {}

  ~LaunchTransData() = default;
  void LaunchOpKernel();
  std::vector<uint8_t *> GetKernelOutputAddr();
  void SetInputAddr(void *input_addr);
  void FreeDeviceMem();

 private:
  void AclKernelBuild();
  void ConstructKernelGraph();
  void SetKernelBuildInfo();
  uint8_t *AllocDeviceMem(size_t size);
  std::vector<kernel::AddressPtr> CreateOutputAddr(const std::vector<size_t> &outputs_list);
  void *stream_;
  TypeId dtype_;
  size_t total_size_;
  std::string src_format_;
  std::string dst_format_;
  ShapeVector shape_;
  int64_t groups_;
  kernel::KernelModPtr kernel_mod_{nullptr};
  std::vector<uint8_t *> outputs_addr_;
  void *input_addr_{nullptr};
  KernelGraphPtr kernel_graph_;
};

}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_HAL_DEVICE_LAUNCH_TRANSDATA_H_
