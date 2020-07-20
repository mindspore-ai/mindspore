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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_

#include <dlfcn.h>
#include <vector>
#include <string>

namespace mindspore {
namespace device {
namespace gpu {
using InitMPI = void (*)();
using InitNCCLComm = void (*)();
using GetLocalRankId = int (*)();
using CreateCommGroupFunc = bool (*)(const std::string &, const std::vector<unsigned int> &);
using GetRankIDByGroupFunc = int (*)(const std::string &);
using GetGroupSizeFunc = int (*)(const std::string &);
using DestroyGroupFunc = bool (*)(const std::string &);

class CollectiveInitializer {
 public:
  CollectiveInitializer(CollectiveInitializer const &) = delete;
  CollectiveInitializer &operator=(const CollectiveInitializer &) = delete;
  static CollectiveInitializer &instance();
  bool collective_inited() const;
  const void *collective_handle() const;
  static void InitCollective();
  static void FinalizeCollective();

 private:
  CollectiveInitializer() : collective_inited_(false) {}
  ~CollectiveInitializer() = default;

  bool collective_inited_;
  void *collective_handle_{nullptr};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_INIT_H_
