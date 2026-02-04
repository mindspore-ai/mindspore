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

#include "pybind_api/utils/log_adapter_py.h"
#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"
#include "utils/log_adapter.h"
#include "include/utils/python_utils.h"
#include "utils/trace_base.h"
#include "pybind_api/utils/utils_api.h"

namespace py = pybind11;
namespace mindspore {
PyExceptionInitializer::PyExceptionInitializer() {
  MS_LOG(INFO) << "Set exception handler";
  mindspore::LogWriter::SetExceptionHandler(HandleExceptionPy);
  mindspore::ExceptionHandler::Get().SetHandler(
    [](const std::function<void(void)> &main_func, const std::function<void(void)> &already_set_error_handler,
       const std::function<void(void)> &other_error_handler, const std::function<void(void)> &default_error_handler,
       const DebugInfoPtr &debug_info, bool force_rethrow) {
      HandleExceptionRethrow(main_func, already_set_error_handler, other_error_handler, default_error_handler,
                             debug_info, force_rethrow);
    });
}

void PyExceptionInitializer::HandleExceptionPy(ExceptionType exception_type, const std::string &str) {
  if (exception_type == IndexError) {
    throw py::index_error(str);
  }
  if (exception_type == ValueError) {
    throw py::value_error(str);
  }
  if (exception_type == TypeError) {
    throw py::type_error(str);
  }
  if (exception_type == KeyError) {
    throw py::key_error(str);
  }
  if (exception_type == AttributeError) {
    throw py::attribute_error(str);
  }
  if (exception_type == NameError) {
    throw py::name_error(str);
  }
  if (exception_type == AssertionError) {
    throw py::assertion_error(str);
  }
  if (exception_type == BaseException) {
    throw py::base_exception(str);
  }
  if (exception_type == KeyboardInterrupt) {
    throw py::keyboard_interrupt(str);
  }
  if (exception_type == StopIteration) {
    throw py::stop_iteration(str);
  }
  if (exception_type == OverflowError) {
    throw py::overflow_error(str);
  }
  if (exception_type == ZeroDivisionError) {
    throw py::zero_division_error(str);
  }
  if (exception_type == EnvironmentError) {
    throw py::environment_error(str);
  }
  if (exception_type == IOError) {
    throw py::io_error(str);
  }
  if (exception_type == OSError) {
    throw py::os_error(str);
  }
  if (exception_type == MemoryError) {
    throw py::memory_error(str);
  }
  if (exception_type == UnboundLocalError) {
    throw py::unbound_local_error(str);
  }
  if (exception_type == NotImplementedError) {
    throw py::not_implemented_error(str);
  }
  if (exception_type == IndentationError) {
    throw py::indentation_error(str);
  }
  if (exception_type == RuntimeWarning) {
    throw py::runtime_warning(str);
  }
  throw std::runtime_error(str);
}

void PyExceptionInitializer::HandleExceptionRethrow(const std::function<void(void)> &main_func,
                                                    const std::function<void(void)> &already_set_error_handler,
                                                    const std::function<void(void)> &other_error_handler,
                                                    const std::function<void(void)> &default_error_handler,
                                                    const DebugInfoPtr &debug_info, bool force_rethrow) {
  try {
    if (!main_func) {
      MS_LOG(ERROR) << "The 'main_func' should not be empty.";
      return;
    }
    main_func();
  } catch (const py::error_already_set &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
    if (already_set_error_handler) {
      already_set_error_handler();
    }
    if (force_rethrow) {
      std::rethrow_exception(std::current_exception());
    }

    // Re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::type_error(ss.str());
    }
  } catch (const py::value_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::value_error(ss.str());
    }
  } catch (const py::index_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::index_error(ss.str());
    }
  } catch (const py::key_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::key_error(ss.str());
    }
  } catch (const py::attribute_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::attribute_error(ss.str());
    }
  } catch (const py::name_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::name_error(ss.str());
    }
  } catch (const py::assertion_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::assertion_error(ss.str());
    }
  } catch (const py::base_exception &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::base_exception(ss.str());
    }
  } catch (const py::keyboard_interrupt &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::keyboard_interrupt(ss.str());
    }
  } catch (const py::stop_iteration &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::stop_iteration(ss.str());
    }
  } catch (const py::overflow_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::overflow_error(ss.str());
    }
  } catch (const py::zero_division_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::zero_division_error(ss.str());
    }
  } catch (const py::environment_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::environment_error(ss.str());
    }
  } catch (const py::io_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::io_error(ss.str());
    }
  } catch (const py::os_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::os_error(ss.str());
    }
  } catch (const py::memory_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::memory_error(ss.str());
    }
  } catch (const py::unbound_local_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::unbound_local_error(ss.str());
    }
  } catch (const py::not_implemented_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::not_implemented_error(ss.str());
    }
  } catch (const py::indentation_error &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::indentation_error(ss.str());
    }
  } catch (const py::runtime_warning &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
      throw py::runtime_warning(ss.str());
    }
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Caught exception: " << ex.what();
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
      ss << ex.what() << ".\n\n" << trace::GetDebugInfoStr(debug_info);
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
static PyExceptionInitializer py_exception_initializer;
}  // namespace mindspore
