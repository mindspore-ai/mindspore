/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_ENVIRON_AICPU_ENVIRON_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_ENVIRON_AICPU_ENVIRON_MANAGER_H_

#include <utility>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include "environ/aicpu_environ.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"
#include "common/distinct_uniform_int_distribution.h"
#include "common/tensor.h"

namespace aicpu {
class EnvironMgr {
 public:
  static EnvironMgr &GetInstance() noexcept {
    static EnvironMgr instance = EnvironMgr();
    return instance;
  }

  EnvironMgr(const EnvironMgr &) = delete;
  EnvironMgr(EnvironMgr &&) = delete;
  EnvironMgr &operator=(const EnvironMgr &) = delete;
  EnvironMgr &operator=(EnvironMgr &&) = delete;

  // Create the env object and return the unique env handle.
  int64_t Create();

  EnvironPtr Get(int64_t handle);

  void Clear();

  // Check whether the inputs of EnvironGet kernel or EnvironSet kernel are valid.
  bool CheckEnvInput(const aicpuops::NodeDef &node_def) const;
  // Check whether is scalar tensor. Environ handle and env key only support scalar tensor currently.
  bool IsScalarTensor(const aicpuops::Tensor &tensor) const;

 private:
  EnvironMgr() = default;
  ~EnvironMgr() = default;

  // Store the envs in map, as <handle, env>.
  std::map<int64_t, EnvironPtr> envs_;

  int64_t env_handles_count_{0};

  std::mutex mutex;
};
}  // namespace aicpu

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_OPS_ENVIRON_AICPU_ENVIRON_MANAGER_H_
