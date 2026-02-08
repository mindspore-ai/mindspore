/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_

#include <utility>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <limits>
#include <Python.h>
#include <mutex>
#include <atomic>
#include "pybind11/pybind11.h"
#include "tools/profiler/python_obj_pointer.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace profiler {
namespace py = pybind11;

constexpr size_t max_py_threads = std::numeric_limits<uint8_t>::max() + 1;

enum class PROFILER_EXPORT Command { kStartOne = 0, kStartAll, kStop, kClear };

enum class PROFILER_EXPORT TraceTag { kPy_Call = 0, kPy_Return, kC_Call, kC_Return };

class PROFILER_EXPORT PythonCApi {
 public:
  static PyFrameObject *PyEval_GetFrame_MS() {
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);
    return frame;
  }
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 11)
  static PythonCodeObjPtr PyFrame_GetCode_MS(PyFrameObject *frame) { return PythonCodeObjPtr(frame->f_code); }
  static PythonObjPtr PyFrame_GetLocals_MS(PyFrameObject *frame) { return PythonObjPtr(frame->f_locals); }
  static PyFrameObject *PyFrame_GetBack_MS(PyFrameObject *frame) { return frame->f_back; }
#else
  static PythonCodeObjPtr PyFrame_GetCode_MS(PyFrameObject *frame) { return PythonCodeObjPtr(PyFrame_GetCode(frame)); }
  static PythonObjPtr PyFrame_GetLocals_MS(PyFrameObject *frame) { return PythonObjPtr(PyFrame_GetLocals(frame)); }
  static PyFrameObject *PyFrame_GetBack_MS(PyFrameObject *frame) { return PyFrame_GetBack(frame); }
#endif
};

struct PROFILER_EXPORT TraceContext {
  PyObject_HEAD PyThreadState *thread_state_;
};

struct PROFILER_EXPORT PythonFuncCallData {
  uint64_t start_time_{0};
  uint64_t end_time_{0};
  uint32_t map_index_{0};
  PythonFuncCallData(uint64_t start_time, uint64_t end_time, uint32_t map_index)
      : start_time_{start_time}, end_time_{end_time}, map_index_{map_index} {}
};

class PROFILER_EXPORT PythonTracer final {
 public:
  static void call(Command c, uint32_t rank_id);
  static int pyProfileFn(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg);

 private:
  PythonTracer();
  static PythonTracer &singleton();

  void start(size_t max_threads = max_py_threads, uint32_t rank_id = 0);
  void stop();
  void clear();
  void recordPyCall(TraceContext *ctx, PyFrameObject *frame);
  void recordCCall(TraceContext *ctx, PyFrameObject *frame, PyObject *arg);
  void recordReturn(TraceContext *ctx, PyFrameObject *frame, PyObject *arg, TraceTag tag);
  void Flush();

  bool active_{false};
  PyObject *module_call_code_{nullptr};
  std::vector<TraceContext *> trace_contexts_;
  std::mutex flush_mutex_;
  std::stack<uint64_t> call_syscnt_;
  uint64_t tid_{0};
  uint32_t rank_id_{0};
  uint64_t stack_cnt{0};
  std::deque<std::string> sys_path_list;
  std::deque<std::unique_ptr<PythonFuncCallData>> data_chunk_buf_;
  std::unordered_map<std::string, uint32_t> op_map_;
  std::atomic<uint32_t> op_index_{0};
  uint32_t max_call_data_count_{20 * 1000 * 1000};
};
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_PROFILING_PYTHON_H_
