/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_ENV_CHECKER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_ENV_CHECKER_H_

#include <set>
#include <string>

namespace mindspore {
namespace device {
namespace gpu {
class CudaEnvChecker {
 public:
  bool CheckNvccInPath();

  static CudaEnvChecker &GetInstance() {
    static CudaEnvChecker instance;
    return instance;
  }

 private:
  CudaEnvChecker() = default;
  ~CudaEnvChecker() = default;
  CudaEnvChecker(const CudaEnvChecker &);
  CudaEnvChecker &operator=(const CudaEnvChecker &);

  void GetRealPaths(std::set<std::string> *paths) const;

  bool already_check_nvcc_ = false;
  bool find_nvcc_ = false;
  static constexpr auto kPathEnv = "PATH";
  static constexpr auto kNvcc = "nvcc";
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_CUDA_ENV_CHECKER_H_
