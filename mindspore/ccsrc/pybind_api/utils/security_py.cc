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

#include "include/common/utils/utils.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// Get whether security is enable
bool EnableSecurity() {
#ifdef ENABLE_SECURITY
  return true;
#else
  return false;
#endif
}

// Define python wrapper to judge security enable.
void RegSecurity(py::module *m) {
  auto m_sub = m->def_submodule("security", "submodule for security");
  (void)m_sub.def("enable_security", &EnableSecurity, "enable security");
}
}  // namespace mindspore
