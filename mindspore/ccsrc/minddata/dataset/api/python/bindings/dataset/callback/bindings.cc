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
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/callback/py_ds_callback.h"
#include "minddata/dataset/callback/ds_callback.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(PyDSCallback, 0, ([](const py::module *m) {
                  (void)py::class_<PyDSCallback, std::shared_ptr<PyDSCallback>>(*m, "PyDSCallback")
                    .def(py::init<int32_t>())
                    .def("set_begin", &PyDSCallback::SetBegin)
                    .def("set_end", &PyDSCallback::SetEnd)
                    .def("set_epoch_begin", &PyDSCallback::SetEpochBegin)
                    .def("set_epoch_end", &PyDSCallback::SetEpochEnd)
                    .def("set_step_begin", &PyDSCallback::SetStepBegin)
                    .def("set_step_end", &PyDSCallback::SetStepEnd);
                }));

PYBIND_REGISTER(CallbackParam, 0, ([](const py::module *m) {
                  (void)py::class_<CallbackParam, std::shared_ptr<CallbackParam>>(*m, "CallbackParam")
                    .def(py::init<int64_t, int64_t, int64_t>())
                    .def_readonly("cur_epoch_num", &CallbackParam::cur_epoch_num_)
                    .def_readonly("cur_step_num_in_epoch", &CallbackParam::cur_epoch_step_num_)
                    .def_readonly("cur_step_num", &CallbackParam::cur_step_num_);
                }));
}  // namespace dataset
}  // namespace mindspore
