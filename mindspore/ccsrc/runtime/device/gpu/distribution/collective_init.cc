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

#include "runtime/device/gpu/distribution/collective_init.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
CollectiveInitializer &CollectiveInitializer::instance() {
  static CollectiveInitializer instance = {};
  return instance;
}

bool CollectiveInitializer::collective_inited() const { return collective_inited_; }

const void *CollectiveInitializer::collective_handle() const { return collective_handle_; }

void CollectiveInitializer::InitCollective() {
  void *handle = dlopen("libgpu_collective.so", RTLD_LAZY);
  if (handle == nullptr) {
    MS_LOG(EXCEPTION)
      << "Loading libgpu_collective.so failed. Many reasons could cause this:\n"
         "1.libgpu_collective.so is not found, please check this MindSpore package is GPU version and built "
         "with distributed feature.\n"
         "2.NCCL is not found or the user-installed NCCL version installed is incompatible: MindSpore "
         "requires NCCL-2.7.6.\n"
         "3.OpenMPI is not found or the user-installed OpenMPI version is incompatible: MindSpore "
         "requires OpenMPI-4.0.3.\n";
  }
  auto mpi_init_funcptr = reinterpret_cast<InitMPI>(dlsym(handle, "InitMPI"));
  MS_EXCEPTION_IF_NULL(mpi_init_funcptr);
  (*mpi_init_funcptr)();

  CollectiveInitializer::instance().collective_inited_ = true;
  CollectiveInitializer::instance().collective_handle_ = handle;
}

void CollectiveInitializer::FinalizeCollective() {
  if (CollectiveInitializer::instance().collective_handle_ != nullptr) {
    if (dlclose(CollectiveInitializer::instance().collective_handle_) != 0) {
      MS_LOG(EXCEPTION) << "Closing libgpu_collective.so handle failed.";
    }
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
