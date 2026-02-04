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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_

#include <string>
#include <memory>
#include "utils/dlopen_macro.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace device {
class RUNTIME_HARDWARE_EXPORT CollectiveCommLibLoader {
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

// The exported symbols of collective communication shared library is registered here.
ORIGIN_METHOD(communication_lib_instance, mindspore::device::CollectiveCommunicationLib *)
ORIGIN_METHOD(ascend_communication_lib_instance, mindspore::device::CollectiveCommunicationLib *)
ORIGIN_METHOD(lowlatency_communication_lib_instance, mindspore::device::CollectiveCommunicationLib *)
ORIGIN_METHOD(multi_ascend_communication_lib_instance, mindspore::device::CollectiveCommunicationLib *)
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_LIB_LOADER_H_
