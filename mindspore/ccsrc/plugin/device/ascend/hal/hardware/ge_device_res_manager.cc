/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
void GeDeviceResManager::Initialize() {
  if (mem_manager_ == nullptr) {
    mem_manager_ = std::make_shared<cpu::CPUMemoryManager>();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
}

void GeDeviceResManager::Destroy() {
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

void *GeDeviceResManager::AllocateMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false);
}

void GeDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

std::vector<void *> GeDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list) const {
  return mem_manager_->MallocContinuousMemFromMemPool(size_list);
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(void *const device_ptr, size_t device_size,
                                                         const string &format, TypeId type_id, const ShapeVector &shape,
                                                         const UserDataPtr &user_data) const {
  auto device_address = std::make_shared<cpu::CPUDeviceAddress>(device_ptr, device_size, format, type_id,
                                                                device_context_->device_context_key_.device_name_,
                                                                device_context_->device_context_key_.device_id_);
  device_address->set_host_shape(shape);
  return device_address;
}

void GeDeviceResManager::GeSetContextOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                             transform::SessionOptions *options) {
  MS_EXCEPTION_IF_NULL(options);
  if (ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.graphMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*options)["ge.variableMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }
}

void GeDeviceResManager::CreateSessionAndGraphRunner(bool is_training) {
  std::shared_ptr<::ge::Session> sess = transform::GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    if (is_training) {
      options["ge.trainFlag"] = "1";
      options["ge.streamNum"] = "100";
      options["ge.enabledLocalFmkop"] = "1";
      options["ge.hcomParallel"] = "1";
    } else {
      options["ge.trainFlag"] = "0";
    }

    options["ge.enablePrintOpPass"] = "0";
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    GeSetContextOptions(ms_context, &options);

    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = transform::NewGraphRunner(options);
  transform::SetGraphRunner(graph_runner);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
