/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/pynative/parallel/reducer.h"
#include "pybind_api/pynative/pynative_api.h"
#include "pybind_api/pynative/tensor/tensor_py.h"
#include "include/pynative/backward/function.h"

namespace mindspore {
namespace pynative {
namespace distributed {
// Define python class for reducer.
void RegReducer(py::module *m) {
  py::class_<Reducer>(*m, "Reducer")
    .def(py::init<tensor::TensorPtrList, std::string, size_t, bool, bool, bool, bool>())
    .def("prepare_for_backward", &Reducer::prepare_for_backward)
    .def("prepare_for_forward", &Reducer::prepare_for_forward)
    .def("get_bucket_for_debug", &Reducer::get_bucket_for_debug)
    .def("rebuild_buckets", &Reducer::rebuild_buckets)
    .def("find_unused_parameters", &Reducer::find_unused_parameters)
    .def("zero_grad", &Reducer::zero_grad)
    .def_readonly("bucket_indices", &Reducer::bucket_indices);
  m->def("_find_unused_parameters", &_find_unused_parameters, "find unused parameters");
}
}  // namespace distributed
}  // namespace pynative
}  // namespace mindspore
