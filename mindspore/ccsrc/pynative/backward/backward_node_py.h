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

#ifndef MINDSPORE_CCSRC_PYBIND_API_BACKWARD_NODE_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_BACKWARD_NODE_PY_H

#include "pybind11/pybind11.h"
#include "include/utils/pynative/variable.h"

namespace mindspore::pynative::autograd {
struct BackwardNodePy {
  PyObject_HEAD BackwardNodePtr cdata;
};

PYNATIVE_EXPORT PyObject *Wrap(const BackwardNodePtr &backward_node);
PyObject *BackwardNode_get_next_edges(const BackwardNodePtr &backward_node);
PyObject *BackwardNode_register_pre_hook(const BackwardNodePtr &backward_node, PyObject *hook_fn);
PyObject *BackwardNode_register_post_hook(const BackwardNodePtr &backward_node, PyObject *hook_fn);
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PYBIND_API_BACKWARD_NODE_PY_H
