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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_UTILS_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_UTILS_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <tuple>
#include <Python.h>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/graph_guard/trace.h"

namespace mindspore {
namespace jit {
namespace graph {

class GuardItem {
 public:
  explicit GuardItem(TracePtr var);
  virtual ~GuardItem() = default;
  virtual bool Check(const PyFrameObject *frame, std::map<std::string, PyObject *> *cache = nullptr) = 0;
  virtual bool Check(PyObject *obj) = 0;
  virtual std::string ToString() = 0;
  virtual void Replace(TracePtr dst, TracePtr src);
  virtual TracePtr GetTrace();

 protected:
  TracePtr var_;
};
using GuardItemPtr = std::shared_ptr<GuardItem>;

/// \brief check whether elements are equal
/// \param[in] obj
/// \param[in] needSpecialize to check the content of buffer
/// \param[in] recurseDepth to check the hierarchy element access like a.b.c by depth
GuardItemPtr GuardEqual(TracePtr obj, bool needSpecialize = true, int recurseDepth = INT_MAX);
GuardItemPtr GuardType(TracePtr obj);
GuardItemPtr GuardId(TracePtr obj);
GuardItemPtr GuardAttr(TracePtr obj);
GuardItemPtr GuardRepr(TracePtr obj);
bool IsPyObjectEqual(PyObject *src, PyObject *dst);
PyObject *GetMsModule();
PyObject *GetMsType();
PyObject *GetMsTensorType();

}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GUARD_UTILS_H
