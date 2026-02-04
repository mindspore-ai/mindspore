/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <unordered_set>
#include <shared_mutex>

#include "pybind11/pybind11.h"
#include "utils/any.h"
#include "base/base_ref.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/utils/visible.h"

namespace py = pybind11;

namespace mindspore {
// Detect recursion for python object sych as a = [..., 1].
class PyRecursionScope {
 public:
  explicit PyRecursionScope(const py::object &obj) {
    address_ = obj.ptr();
    {
      std::unique_lock lock(recursion_mutex_);
      if (!recursion_set_.insert(address_).second) {
        MS_LOG(EXCEPTION) << "Detect recursion when converting python object.";
      }
    }
  }

  ~PyRecursionScope() {
    std::unique_lock lock(recursion_mutex_);
    recursion_set_.erase(address_);
  }

 private:
  PyObject *address_;
  static inline std::unordered_set<PyObject *> recursion_set_;
  static inline std::shared_mutex recursion_mutex_;
};

py::object AnyToPyData(const Any &value);
COMMON_EXPORT py::object BaseRefToPyData(const BaseRef &value, const AbstractBasePtr &abs = nullptr);
COMMON_EXPORT py::object ValueToPyData(const ValuePtr &value, const AbstractBasePtr &abs = nullptr);
COMMON_EXPORT py::object CValueToPybindObj(const ValuePtr &val);
COMMON_EXPORT ValuePtr PyStubNodeCast(const py::handle &obj);
COMMON_EXPORT ValuePtr ConvertTensorNode(const py::object &obj);
COMMON_EXPORT bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                                     const std::shared_ptr<py::object> &ret_val);
COMMON_EXPORT ValuePtr ShallowCopyTensorValue(const ValuePtr &value);
COMMON_EXPORT ValuePtr ConvertPyObjectToCTensor(const py::object &input_object);
COMMON_EXPORT void ConvertPyObjectToCTensor(const py::object &input_object, std::vector<ValuePtr> *tensors,
                                            bool is_base_tensor = false);
COMMON_EXPORT void ConvertPybindTupleGradToCValue(const py::tuple &input_tuple, std::vector<ValuePtr> *gradient_values);
COMMON_EXPORT py::object ConvertCTensorToPyTensor(const py::object &input_arg);
COMMON_EXPORT std::string ConvertPyObjToString(const py::object &obj);
COMMON_EXPORT tensor::TensorPtr ConvertPyObjToTensor(const py::object &obj);
COMMON_EXPORT py::tuple CheckBpropOut(const py::object &grads_obj, const py::tuple &py_args,
                                      const std::string &bprop_cls_name);
py::object ScalarPtrToPyData(const ScalarPtr &value);
py::object CheckAndConvertToScalar(const tensor::TensorPtr &tensor, const AbstractBasePtr &abs);
COMMON_EXPORT tensor::TensorPtr ConvertTensorAndSyncCompiling(const py::handle &obj);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_PY_H_
