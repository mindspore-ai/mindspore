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
#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_GRAPH_JIT_MS_ADAPTER_INFER_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_GRAPH_JIT_MS_ADAPTER_INFER_H

#include "pybind11/pybind11.h"

namespace mindspore {
namespace jit {
namespace graph {

namespace py = pybind11;
bool IsMSAdapterModuleForwardCall(PyFrameObject *f);
bool IsMSAdapterModuleType(PyTypeObject *tp, bool sub_type);
void SpecializeForMSAdapterModule(PyFrameObject *f);

}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif
