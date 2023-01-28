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
#include "ir/param_info.h"
#include "pybind11/pybind11.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
namespace py = pybind11;

void RegParamInfo(py::module *m) {
  (void)py::class_<ParamInfo, ParamInfoPtr>(*m, "ParamInfo", py::dynamic_attr())
    .def(py::init())
    .def(py::init([](const ParamInfo &param_info) { return std::make_shared<ParamInfo>(param_info); }),
         py::arg("input"))
    .def("clone", &ParamInfo::Clone)
    .def_property("name", &ParamInfo::name, &ParamInfo::set_name)
    .def_property("key", &ParamInfo::key, &ParamInfo::set_key)
    .def_property("requires_grad", &ParamInfo::requires_grad, &ParamInfo::set_requires_grad)
    .def_property("init_in_server", &ParamInfo::init_in_server, &ParamInfo::set_init_in_server)
    .def_property("layerwise_parallel", &ParamInfo::layerwise_parallel, &ParamInfo::set_layerwise_parallel)
    .def_property("parallel_optimizer", &ParamInfo::parallel_optimizer, &ParamInfo::set_parallel_optimizer)
    .def_property("comm_fusion", &ParamInfo::comm_fusion, &ParamInfo::set_comm_fusion)
    .def_property("parallel_optimizer_comm_recompute", &ParamInfo::parallel_optimizer_comm_recompute,
                  &ParamInfo::set_parallel_optimizer_comm_recompute)
    .def_property("parameter_shape", &ParamInfo::parameter_shape, &ParamInfo::set_parameter_shape)
    .def_property("origin_shape", &ParamInfo::origin_shape, &ParamInfo::set_origin_shape)
    .def_property("use_persistent_storage", &ParamInfo::use_persistent_storage, &ParamInfo::set_use_persistent_storage)
    .def_property("cache_enable", &ParamInfo::cache_enable, &ParamInfo::set_cache_enable)
    .def_property("cache_shape", &ParamInfo::cache_shape, &ParamInfo::set_cache_shape)
    .def_property("requires_aggr", &ParamInfo::requires_aggr, &ParamInfo::set_requires_aggr)
    .def_property("param_strategy", &ParamInfo::param_strategy, &ParamInfo::set_param_strategy)
    .def(py::pickle(
      [](const ParamInfo &p) {  // __getstate__
        return py::make_tuple(p.name(), p.requires_grad(), p.layerwise_parallel());
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 6) {
          std::runtime_error("Invalid state for ParamInfo!");
        }
        ParamInfoPtr p = std::make_shared<ParamInfo>();
        p->set_name(t[1].cast<std::string>());
        p->set_requires_grad(t[2].cast<bool>());
        p->set_layerwise_parallel(t[3].cast<bool>());
        return p;
      }));
}
}  // namespace mindspore
