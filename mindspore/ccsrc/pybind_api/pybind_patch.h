/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef PYBIND_API_PYBIND_PATCH_H_
#define PYBIND_API_PYBIND_PATCH_H_

namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(attribute_error, PyExc_AttributeError)
PYBIND11_RUNTIME_EXCEPTION(name_error, PyExc_NameError)
PYBIND11_RUNTIME_EXCEPTION(assertion_error, PyExc_AssertionError)
PYBIND11_RUNTIME_EXCEPTION(base_exception, PyExc_BaseException)
PYBIND11_RUNTIME_EXCEPTION(keyboard_interrupt, PyExc_KeyboardInterrupt)
PYBIND11_RUNTIME_EXCEPTION(overflow_error, PyExc_OverflowError)
PYBIND11_RUNTIME_EXCEPTION(zero_division_error, PyExc_ZeroDivisionError)
PYBIND11_RUNTIME_EXCEPTION(environment_error, PyExc_EnvironmentError)
PYBIND11_RUNTIME_EXCEPTION(io_error, PyExc_IOError)
PYBIND11_RUNTIME_EXCEPTION(os_error, PyExc_OSError)
PYBIND11_RUNTIME_EXCEPTION(memory_error, PyExc_MemoryError)
PYBIND11_RUNTIME_EXCEPTION(unbound_local_error, PyExc_UnboundLocalError)
PYBIND11_RUNTIME_EXCEPTION(not_implemented_error, PyExc_NotImplementedError)
PYBIND11_RUNTIME_EXCEPTION(indentation_error, PyExc_IndentationError)
PYBIND11_RUNTIME_EXCEPTION(runtime_warning, PyExc_RuntimeWarning)
}  // namespace pybind11

#endif  // PYBIND_API_PYBIND_PATCH_H_
