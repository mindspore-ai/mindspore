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
#ifndef MINDSPORE_PI_JIT_INFER_H
#define MINDSPORE_PI_JIT_INFER_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"

namespace mindspore {
namespace pijit {

class InferEngine : public std::enable_shared_from_this<InferEngine> {
 public:
  static std::shared_ptr<InferEngine> GetInstance();
  PyObject *InferPrimitive(PyObject *primitive, const std::vector<PyObject *> &args, bool *is_abstract);
  PyObject *InferSpecialPrimitive(PyObject *primitive, const std::vector<PyObject *> &arglist,
                                  const PrimitivePyPtr &prim);
  bool SupportInfer(PyObject *primitive);
  bool Init();
  bool Deinit();

 protected:
  InferEngine();
  bool bInit_ = false;
};

using InferEnginePtr = std::shared_ptr<InferEngine>;

namespace py = pybind11;

template <class T>
PyTypeObject *GetPybindType() {
  py::handle mapped_type = py::detail::get_type_handle(typeid(T), false);
  return reinterpret_cast<PyTypeObject *>(mapped_type.ptr());
}

template <class T, bool sub = true>
bool IsPybindType(PyTypeObject *tp) {
  PyTypeObject *tar = GetPybindType<T>();
  if (tar == nullptr || tp == nullptr) {
    return false;
  }
  return tp == tar || (sub ? PyType_IsSubtype(tp, tar) : false);
}

template <bool sub>
bool IsGradOperationType(PyTypeObject *tp);
template <bool sub>
bool IsVmapOperationType(PyTypeObject *tp);
template <bool sub>
bool IsShardType(PyTypeObject *tp);
template <bool sub>
bool IsStubTensorType(PyTypeObject *tp);
template <bool sub>
bool IsTensorType(PyTypeObject *tp);
template <bool sub>
bool IsCellListType(PyTypeObject *tp);
template <bool sub>
bool IsCellType(PyTypeObject *tp);
template <bool sub>
bool IsPrimitiveType(PyTypeObject *tp);
template <bool sub>
bool IsMetaFuncGraphType(PyTypeObject *tp);
template <bool sub>
bool IsMSDTypeType(PyTypeObject *tp);

bool FindTensorName(const std::string &name);

bool CheckTensorDataInitialized(const py::object &tensor);

using SpecialPrimitiveInferFuncMap =
  std::unordered_map<std::string, PyObject *(*)(PyObject *, const std::vector<PyObject *> &)>;
const SpecialPrimitiveInferFuncMap &GetSpecialPrimitiveInferFunc();

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_INFER_H
