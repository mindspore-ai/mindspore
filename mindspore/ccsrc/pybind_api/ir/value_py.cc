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

#include "ir/value.h"

#include "include/common/pybind_api/api_register.h"
#include "abstract/abstract_value.h"

namespace mindspore {
// Define python class for values.
void RegValues(const py::module *m) {
  (void)py::class_<UMonad, std::shared_ptr<UMonad>>(*m, "UMonad").def(py::init());
  (void)py::class_<IOMonad, std::shared_ptr<IOMonad>>(*m, "IOMonad").def(py::init());
}
}  // namespace mindspore
