/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pynative {

namespace {
bool ms_dispatch_enable_{true};
}

bool ms_dispatch_enabled() { return ms_dispatch_enable_; }

void SetMsDispatchEnabled(bool enabled) { ms_dispatch_enable_ = enabled; }

// ============================================================================
// Handler functions
// ============================================================================

namespace detail {
py::object GetMsDispatchFromArgs(const py::args &packed_args) {
  // Iterate through packed_args to find a tensor with __ms_dispatch__
  for (size_t i = 0; i < packed_args.size(); ++i) {
    py::object item = packed_args[i];
    // Check if the item is a Tensor with __ms_dispatch__ attribute
    if (py::hasattr(item, GetMsDispatchStr().c_str())) {
      py::object handler = py::getattr(item, GetMsDispatchStr().c_str());
      if (!handler.is_none()) {
        MS_LOG(DEBUG) << "Found __ms_dispatch__ handler at arg index " << i;
        return handler;
      }
    }
  }
  MS_LOG(EXCEPTION) << "No __ms_dispatch__ handler found in packed_args";
  return py::none();
}
}  // namespace detail

}  // namespace pynative
}  // namespace mindspore
