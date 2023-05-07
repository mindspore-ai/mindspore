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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeDeviceResManager : public DeviceResManager {
 public:
  GeDeviceResManager() : mem_manager_(nullptr) {}
  ~GeDeviceResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list) const override;

  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const ShapeVector &shape, const UserDataPtr &user_data = nullptr) const override;

  static void CreateSessionAndGraphRunner(bool is_training);

  bool LoadCollectiveCommLib() override {
    collective_comm_lib_ = &AscendCollectiveCommLib::GetInstance();
    return true;
  }

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size) const override;
  void FreeMemory(void *ptr) const override;

 private:
  static void GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr, transform::SessionOptions *options);
  std::shared_ptr<MemoryManager> mem_manager_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_DEVICE_RES_MANAGER_H_
