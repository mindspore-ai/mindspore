/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "pybind_api/pynative/pynative_api.h"

#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/tensor_py.h"
#include "pybind_api/pynative/tensor/tensor_py.h"

namespace mindspore {
namespace tensor {

static void SwapTensorImpl(py::object t1, py::object t2) {
  PyObject *t1_ptr = t1.ptr();
  PyObject *t2_ptr = t2.ptr();

  if (!PyObject_TypeCheck(t1_ptr, GetTensorPyType()) || !PyObject_TypeCheck(t2_ptr, GetTensorPyType())) {
    MS_LOG(EXCEPTION) << "_swap_tensor_impl: both arguments must be Tensor";
  }

  runtime::Pipeline::Get().WaitAll();

  auto *p1 = reinterpret_cast<PyType<TensorPy> *>(t1_ptr);
  auto *p2 = reinterpret_cast<PyType<TensorPy> *>(t2_ptr);

  auto tensor1 = p1->value.GetTensor();
  auto tensor2 = p2->value.GetTensor();
  p1->value.SetTensor(tensor2);
  p2->value.SetTensor(tensor1);

  p1->value.SetStorage(py::none());
  p2->value.SetStorage(py::none());
}

void RegTensorUtilsApi(py::module *m) {
  (void)m->def("_swap_tensor_impl", &SwapTensorImpl,
               "Swap the underlying TensorImpl pointers of two Tensors. Internal use only.");
}

}  // namespace tensor
}  // namespace mindspore
