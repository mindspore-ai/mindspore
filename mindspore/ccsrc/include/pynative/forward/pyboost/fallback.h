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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_

#include <string>
#include "pybind11/pybind11.h"
#include "include/utils/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {
/// \brief Check whether PyNative fallback mechanism is enabled.
/// When enabled, certain PyNative execution paths will call back into
/// Python via the registered __fallback__ handlers instead of running
/// the default C++ implementation.
/// \return true if fallback is enabled; false otherwise.
PYNATIVE_EXPORT bool fallback_enabled();

/// \brief Get the attribute name used to look up the Python fallback handler.
/// The returned string is typically "__fallback__", and is used as the
/// attribute name on Tensor-like Python objects to retrieve the
/// corresponding Python fallback function.
/// \return A const reference to the fallback attribute name string.
PYNATIVE_EXPORT const std::string &GetFallbackStr();

/// \brief Handle PyNative fallback with explicit self / args / kwargs.
/// This overload searches for a Python __fallback__ function from the
/// given \p self / \p py_args / \p py_kwargs (typically by inspecting
/// Tensor-like inputs). Once the __fallback__ handler is found, it is
/// invoked from C++ with the provided \p callable object and the
/// merged arguments (self + args + kwargs), so that the actual
/// computation is delegated back to Python.
/// \param self       The Python "self" object (must not be nullptr).
/// \param py_args    Positional arguments (may be nullptr, a tuple, or a single object).
/// \param py_kwargs  Keyword arguments (may be nullptr).
/// \param callable   The Python callable object passed as an argument to the __fallback__ handler.
/// \return           The result of calling the Python __fallback__ handler (new reference).
/// \throws           C++ exception with MS_LOG(EXCEPTION) if the __fallback__ handler
///                   cannot be found or the Python call fails.
PYNATIVE_EXPORT PyObject *HandleFallback(PyObject *self, PyObject *py_args, PyObject *py_kwargs,
                                         const py::object &callable);

/// \brief Handle PyNative fallback using only a PyObject* args container.
/// This overload searches for a Python __fallback__ function from
/// \p py_args (e.g. a Tensor, a tuple/list of Tensors, or nested containers).
/// After locating the __fallback__ handler, it is called from C++ with
/// the given \p callable object and \p py_args as its arguments, so that
/// the actual execution is redirected back to Python.
/// \param py_args    Positional arguments from which the __fallback__ handler is discovered.
/// \param callable   The Python callable object passed as an argument to the __fallback__ handler.
/// \return           The result of calling the Python __fallback__ handler (new reference).
/// \throws           C++ exception with MS_LOG(EXCEPTION) if the __fallback__ handler
///                   cannot be found or the Python call fails.
PYNATIVE_EXPORT PyObject *HandleFallback(PyObject *py_args, const py::object &callable);

/// \brief Handle PyNative fallback using pybind11-style args/kwargs.
/// This overload is intended for C++ functions exposed via pybind11.
/// It inspects \p args and \p kwargs to locate a Python __fallback__
/// function (typically attached to Tensor-like inputs). Once the
/// __fallback__ handler is found, it is invoked from C++ with the
/// provided \p callable object and the original \p args / \p kwargs,
/// delegating the operation back to Python.
/// \param args       Positional arguments from Python.
/// \param kwargs     Keyword arguments from Python.
/// \param callable   The Python callable object passed as an argument to the __fallback__ handler.
/// \return           A py::object wrapping the result of the Python __fallback__ handler.
/// \throws           C++ exception with MS_LOG(EXCEPTION) if the __fallback__ handler
///                   cannot be found or the Python call fails.
PYNATIVE_EXPORT py::object HandleFallback(const py::args &args, const py::kwargs &kwargs, const py::object &callable);

/// \brief Register PyNative fallback-related helpers into the given Python module.
/// This function exposes the RAII-style guard class `NoFallbackGuard`
/// to Python as a context manager. The guard is typically used in
/// code like:
/// \code
///   with NoFallbackGuard():
///       # PyNative fallback is temporarily disabled in this scope
///       ...
/// \endcode
/// The `__enter__` method uses `reference_internal` return policy so
/// that the lifetime of the returned object is tied to the owning
/// `NoFallbackGuard` instance managed by pybind11.
/// \param m  The pybind11 module to which fallback helpers will be registered.
PYNATIVE_EXPORT void RegFallback(py::module *m);
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_
