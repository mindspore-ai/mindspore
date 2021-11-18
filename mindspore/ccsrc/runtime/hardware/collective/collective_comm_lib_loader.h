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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_

#ifndef _WIN32
#include "utils/dlopen_macro.h"
#else
#include <windows.h>
#undef ERROR
#undef SM_DEBUG
#undef Yield
#include "utils/log_adapter.h"
#endif
#include <string>
#include <memory>
#include <vector>

namespace mindspore {
namespace device {
class CollectiveCommLibLoader {
 public:
  explicit CollectiveCommLibLoader(const std::string &comm_lib_name)
      : collective_comm_lib_ptr_(nullptr), comm_lib_name_(comm_lib_name) {}
  ~CollectiveCommLibLoader() = default;

  // Dynamically load the communication library.
  bool Initialize();

  // Finalize the communication library.
  bool Finalize();

  // Return the handle for this collective communication library. Caller should use this handle to call all methods of
  // the collective communication.
  void *collective_comm_lib_ptr() const { return collective_comm_lib_ptr_; }

 private:
  // The library handle 'dlopen' returns.
  void *collective_comm_lib_ptr_;

  // Name of the communication library.
  std::string comm_lib_name_;
};
using CollectiveCommLibLoaderPtr = std::shared_ptr<CollectiveCommLibLoader>;
}  // namespace device
}  // namespace mindspore

#ifndef _WIN32
// The exported symbols of collective communication shared library is registered here.
ORIGIN_METHOD(InitializeCollectiveLib, bool, uint32_t, uint32_t)
ORIGIN_METHOD(FinalizeCollectiveLib, bool)
ORIGIN_METHOD(CreateCommunicationGroup, bool, const std::string &, const std::vector<uint32_t> &)
ORIGIN_METHOD(DestroyCommunicationGroup, bool, const std::string &)
ORIGIN_METHOD(GetRankId, uint32_t, const std::string &)
ORIGIN_METHOD(GetCommunicationGroupSize, uint32_t, const std::string &)
ORIGIN_METHOD(AssignLocalRank, bool)
ORIGIN_METHOD(global_rank_id, uint32_t)
ORIGIN_METHOD(local_rank_id, uint32_t)
ORIGIN_METHOD(global_rank_size, uint32_t)
#endif
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_
