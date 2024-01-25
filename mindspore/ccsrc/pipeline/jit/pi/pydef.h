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
#ifndef MINDSPORE_PI_JIT_PYDEF_H
#define MINDSPORE_PI_JIT_PYDEF_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include <opcode.h>

#define NO_IMPL_OPCODE 257

#if (PY_MAJOR_VERSION == 3)
#if (PY_MINOR_VERSION == 7)

inline _PyFrameEvalFunction _PyInterpreterState_GetEvalFrameFunc(PyInterpreterState *interp) {
  return interp->eval_frame;
}

inline void _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState *interp, _PyFrameEvalFunction eval_frame) {
  interp->eval_frame = eval_frame;
}

inline PyObject *PyObject_Vectorcall(PyObject *func, PyObject *const *stack, Py_ssize_t nargs, PyObject *kwnames) {
  return _PyObject_FastCallKeywords(func, stack, nargs, kwnames);
}

inline PyObject *PyObject_CallOneArg(PyObject *func, PyObject *arg) {
  return _PyObject_FastCallKeywords(func, &arg, 1, nullptr);
}

#define PY_VECTORCALL_ARGUMENTS_OFFSET 0
#define Py_IS_TYPE(ob, type) Py_TYPE(ob) == type

#define _PyEval_EvalFrameDefault(state, f, exc) _PyEval_EvalFrameDefault(f, exc)
#define _PyTuple_CAST(op) (MS_ASSERT(PyTuple_Check(op)), reinterpret_cast<PyTupleObject *>(op))

#define ROT_FOUR (NO_IMPL_OPCODE + 1)
#define LIST_TO_TUPLE (NO_IMPL_OPCODE + 2)
#define LIST_EXTEND (NO_IMPL_OPCODE + 3)
#define DICT_MERGE (NO_IMPL_OPCODE + 4)
#define DICT_UPDATE (NO_IMPL_OPCODE + 5)
#define SET_UPDATE (NO_IMPL_OPCODE + 6)
#define IS_OP (NO_IMPL_OPCODE + 7)
#define CONTAINS_OP (NO_IMPL_OPCODE + 8)
#define LOAD_ASSERTION_ERROR (NO_IMPL_OPCODE + 9)
#define WITH_EXCEPT_START (NO_IMPL_OPCODE + 10)
#define END_ASYNC_FOR (NO_IMPL_OPCODE + 11)
#define RERAISE (NO_IMPL_OPCODE + 12)
#define JUMP_IF_NOT_EXC_MATCH (NO_IMPL_OPCODE + 13)

#endif
#endif

#endif
