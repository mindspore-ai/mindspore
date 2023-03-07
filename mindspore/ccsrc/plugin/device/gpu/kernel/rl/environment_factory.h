/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENV_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENV_FACTORY_H_

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
// Interface of MindSpore built-in environment.
// The derive class should override all virtual interface.
class Environment {
 public:
  // Initialize Environment.
  virtual bool Init(const CNodePtr &cnode, void *stream_ptr) = 0;
  // Reset Environment.
  virtual bool Reset(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                     const std::vector<AddressPtr> &outputs, void *stream_ptr) = 0;
  // Run one timestep.
  virtual bool Step(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr) = 0;

  // Return Environment specification and the framework will malloc device memory for it.
  virtual size_t ActionSizeInBytes() = 0;
  virtual size_t StateSizeInBytes() = 0;
  virtual size_t RewardSizeInBytes() = 0;
  virtual size_t DoneSizeInBytes() = 0;
  virtual size_t WorkspaceSizeInBytes() { return 0; }
};

constexpr int64_t kInvalidHandle = -1;
using EnvCreator = std::function<Environment *()>;

// Class for environment registration, environment instances managerment with factory design pattern.
class EnvironmentFactory {
 public:
  // Create a factory instance with lazy mode.
  static EnvironmentFactory &GetInstance();
  // Create an environment instance with unique handle and instance returned.
  std::tuple<int, std::shared_ptr<Environment>> Create(const std::string &name);
  // Delete the environment instance.
  void Delete(int64_t handle);
  // Get environment instance by handle.
  std::shared_ptr<Environment> GetByHandle(int64_t handle);
  // Register environment creator.
  void Register(const std::string &name, EnvCreator &&creator);

 private:
  EnvironmentFactory() = default;
  ~EnvironmentFactory() = default;
  DISABLE_COPY_AND_ASSIGN(EnvironmentFactory)

  int64_t handle_ = kInvalidHandle;
  std::map<int64_t, std::shared_ptr<Environment>> map_env_handle_to_instances_;
  std::map<std::string, EnvCreator> map_env_name_to_creators_;
};

// Class for environment registration.
// The registration depend on global variable initialization(constructor) which means that
// registration executed is before than `main()` and execution order is not guaranteed to be same.
class EnvironmentRegister {
 public:
  EnvironmentRegister(const std::string &name, EnvCreator &&creator) {
    EnvironmentFactory::GetInstance().Register(name, std::move(creator));
  }
};

// Helper macro for environment registration.
#define MS_REG_GPU_ENV(NAME, ENVCLASS)                                                          \
  static_assert(std::is_base_of<Environment, ENVCLASS>::value, " must be base of Environment"); \
  static const EnvironmentRegister g_##NAME##_gpu_env_reg(#NAME, []() { return new ENVCLASS(); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ENV_FACTORY_H_
