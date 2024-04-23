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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
#define Py_BUILD_CORE
// <stdatomic.h> is unsupported by g++
#include <internal/pycore_pystate.h>
#undef Py_BUILD_CORE
#endif

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)

_PyFrameEvalFunction _PyInterpreterState_GetEvalFrameFunc(PyInterpreterState *state) { return state->eval_frame; }
void _PyInterpreterState_SetEvalFrameFunc(PyInterpreterState *state, _PyFrameEvalFunction eval_frame_function) {
  state->eval_frame = eval_frame_function;
}

#endif
