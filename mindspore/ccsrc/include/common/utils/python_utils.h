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
inline void HandleExceptionRethrow(const std::function<void(void)> &main_func,
                                   const std::function<void(void)> &already_set_error_handler,
                                   const std::function<void(void)> &other_error_handler,
                                   const std::function<void(void)> &default_error_handler,
                                   const DebugInfoPtr &debug_info = nullptr, bool force_rethrow = false) {
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
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    // Re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::type_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::type_error(ss.str());
    }
  } catch (const py::value_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::value_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::value_error(ss.str());
    }
  } catch (const py::index_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::index_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::index_error(ss.str());
    }
  } catch (const py::key_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::key_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::key_error(ss.str());
    }
  } catch (const py::attribute_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::attribute_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::attribute_error(ss.str());
    }
  } catch (const py::name_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::name_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::name_error(ss.str());
    }
  } catch (const py::assertion_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::assertion_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::assertion_error(ss.str());
    }
  } catch (const py::base_exception &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::base_exception(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::base_exception(ss.str());
    }
  } catch (const py::keyboard_interrupt &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::keyboard_interrupt(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::keyboard_interrupt(ss.str());
    }
  } catch (const py::stop_iteration &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::stop_iteration(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::stop_iteration(ss.str());
    }
  } catch (const py::overflow_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::overflow_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::overflow_error(ss.str());
    }
  } catch (const py::zero_division_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::zero_division_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::zero_division_error(ss.str());
    }
  } catch (const py::environment_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::environment_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::environment_error(ss.str());
    }
  } catch (const py::io_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::io_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::io_error(ss.str());
    }
  } catch (const py::os_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::os_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::os_error(ss.str());
    }
  } catch (const py::memory_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::memory_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::memory_error(ss.str());
    }
  } catch (const py::unbound_local_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::unbound_local_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::unbound_local_error(ss.str());
    }
  } catch (const py::not_implemented_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::not_implemented_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::not_implemented_error(ss.str());
    }
  } catch (const py::indentation_error &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::indentation_error(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::indentation_error(ss.str());
    }
  } catch (const py::runtime_warning &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    if (debug_info == nullptr) {
      throw py::runtime_warning(ex);
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw py::runtime_warning(ss.str());
    }
  } catch (const std::exception &ex) {
    if (other_error_handler) {
      other_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    // Re-throw this exception to Python interpreter to handle it.
    if (debug_info == nullptr) {
      throw std::runtime_error(ex.what());
    } else {
      std::stringstream ss;
      ss << ex.what() << ".\n\n" << trace::GetDebugInfo(debug_info);
      throw std::runtime_error(ss.str());
    }
  } catch (...) {
    if (default_error_handler) {
      default_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
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
