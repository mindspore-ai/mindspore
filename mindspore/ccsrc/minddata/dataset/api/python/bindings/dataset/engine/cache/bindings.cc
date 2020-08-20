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

#include <optional>
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/engine/cache/cache_client.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(CacheClient, 0, ([](const py::module *m) {
                  (void)py::class_<CacheClient, std::shared_ptr<CacheClient>>(*m, "CacheClient")
                    .def(py::init([](session_id_type id, uint64_t mem_sz, bool spill,
                                     std::optional<std::string> hostname, std::optional<int32_t> port,
                                     std::optional<int32_t> num_connections, std::optional<int32_t> prefetch_sz) {
                      std::shared_ptr<CacheClient> cc;
                      CacheClient::Builder builder;
                      builder.SetSessionId(id).SetCacheMemSz(mem_sz).SetSpill(spill);
                      if (hostname) builder.SetHostname(hostname.value());
                      if (port) builder.SetPort(port.value());
                      if (num_connections) builder.SetNumConnections(num_connections.value());
                      if (prefetch_sz) builder.SetPrefetchSize(prefetch_sz.value());
                      THROW_IF_ERROR(builder.Build(&cc));
                      return cc;
                    }))
                    .def("GetStat", [](CacheClient &cc) {
                      CacheServiceStat stat{};
                      THROW_IF_ERROR(cc.GetStat(&stat));
                      return stat;
                    });
                  (void)py::class_<CacheServiceStat>(*m, "CacheServiceStat")
                    .def(py::init<>())
                    .def_readwrite("avg_cache_sz", &CacheServiceStat::avg_cache_sz)
                    .def_readwrite("num_mem_cached", &CacheServiceStat::num_mem_cached)
                    .def_readwrite("num_disk_cached", &CacheServiceStat::num_disk_cached);
                }));

}  // namespace dataset
}  // namespace mindspore
