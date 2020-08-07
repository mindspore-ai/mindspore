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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/engine/cache/cache_client.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(CacheClient, 0, ([](const py::module *m) {
                  (void)py::class_<CacheClient, std::shared_ptr<CacheClient>>(*m, "CacheClient")
                    .def(py::init<uint32_t, uint64_t, bool>());
                }));

}  // namespace dataset
}  // namespace mindspore
