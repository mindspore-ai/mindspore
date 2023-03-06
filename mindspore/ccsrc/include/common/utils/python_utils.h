/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_

#include <functional>
#include <string>
#include "utils/info.h"

namespace mindspore {
template <typename ThrowExceptionType, typename CatchedExceptionType>
static void ThrowException(const std::function<void(void)> &other_error_handler, const DebugInfoPtr &debug_info,
                           const CatchedExceptionType &ex) {
  if (other_error_handler) {
    other_error_handler();
  }
  if (debug_info == nullptr) {
    throw ThrowExceptionType(ex.what());
  } else {
    std::stringstream ss;
    ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
    throw ThrowExceptionType(ss.str());
  }
}

inline void HandleExceptionRethrow(const std::function<void(void)> &main_func,
                                   const std::function<void(void)> &already_set_error_handler,
                                   const std::function<void(void)> &other_error_handler,
                                   const std::function<void(void)> &default_error_handler,
                                   const DebugInfoPtr &debug_info = nullptr) {
  try {
    if (!main_func) {
      MS_LOG(ERROR) << "The 'main_func' should not be empty.";
      return;
    }
    main_func();
  } catch (const py::error_already_set &ex) {
    if (already_set_error_handler) {
      already_set_error_handler();
    }
    // Re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    ThrowException<py::type_error, py::type_error>(other_error_handler, debug_info, ex);
  } catch (const py::value_error &ex) {
    ThrowException<py::value_error, py::value_error>(other_error_handler, debug_info, ex);
  } catch (const py::index_error &ex) {
    ThrowException<py::index_error, py::index_error>(other_error_handler, debug_info, ex);
  } catch (const py::key_error &ex) {
    ThrowException<py::key_error, py::key_error>(other_error_handler, debug_info, ex);
  } catch (const py::attribute_error &ex) {
    ThrowException<py::attribute_error, py::attribute_error>(other_error_handler, debug_info, ex);
  } catch (const py::name_error &ex) {
    ThrowException<py::name_error, py::name_error>(other_error_handler, debug_info, ex);
  } catch (const py::assertion_error &ex) {
    ThrowException<py::assertion_error, py::assertion_error>(other_error_handler, debug_info, ex);
  } catch (const py::base_exception &ex) {
    ThrowException<py::base_exception, py::base_exception>(other_error_handler, debug_info, ex);
  } catch (const py::keyboard_interrupt &ex) {
    ThrowException<py::keyboard_interrupt, py::keyboard_interrupt>(other_error_handler, debug_info, ex);
  } catch (const py::stop_iteration &ex) {
    ThrowException<py::stop_iteration, py::stop_iteration>(other_error_handler, debug_info, ex);
  } catch (const py::overflow_error &ex) {
    ThrowException<py::overflow_error, py::overflow_error>(other_error_handler, debug_info, ex);
  } catch (const py::zero_division_error &ex) {
    ThrowException<py::zero_division_error, py::zero_division_error>(other_error_handler, debug_info, ex);
  } catch (const py::environment_error &ex) {
    ThrowException<py::environment_error, py::environment_error>(other_error_handler, debug_info, ex);
  } catch (const py::io_error &ex) {
    ThrowException<py::io_error, py::io_error>(other_error_handler, debug_info, ex);
  } catch (const py::os_error &ex) {
    ThrowException<py::os_error, py::os_error>(other_error_handler, debug_info, ex);
  } catch (const py::memory_error &ex) {
    ThrowException<py::memory_error, py::memory_error>(other_error_handler, debug_info, ex);
  } catch (const py::unbound_local_error &ex) {
    ThrowException<py::unbound_local_error, py::unbound_local_error>(other_error_handler, debug_info, ex);
  } catch (const py::not_implemented_error &ex) {
    ThrowException<py::not_implemented_error, py::not_implemented_error>(other_error_handler, debug_info, ex);
  } catch (const py::indentation_error &ex) {
    ThrowException<py::indentation_error, py::indentation_error>(other_error_handler, debug_info, ex);
  } catch (const py::runtime_warning &ex) {
    ThrowException<py::runtime_warning, py::runtime_warning>(other_error_handler, debug_info, ex);
  } catch (const std::runtime_error &ex) {
    ThrowException<std::runtime_error, std::runtime_error>(other_error_handler, debug_info, ex);
  } catch (const std::exception &ex) {
    ThrowException<std::runtime_error, std::exception>(other_error_handler, debug_info, ex);
  } catch (...) {
    if (default_error_handler) {
      default_error_handler();
    }

#ifndef _MSC_VER
    auto exception_type = abi::__cxa_current_exception_type();
    MS_EXCEPTION_IF_NULL(exception_type);
    std::string ex_name(exception_type->name());
    MS_LOG(EXCEPTION) << "Error occurred. Exception name: " << ex_name;
#else
    MS_LOG(EXCEPTION) << "Error occurred.";
#endif
  }
}
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PYTHON_UTILS_H_
