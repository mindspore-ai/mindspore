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
#include <string>
#include "ir/anf.h"

#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// Define python 'CNode' class.
void RegCNode(const py::module *m) {
  (void)py::class_<CNode, CNodePtr>(*m, "CNode")
    .def("expanded_str", (std::string(CNode::*)(int32_t) const) & CNode::DebugString,
         "Get CNode string representation with specified expansion level.");
}
}  // namespace mindspore
