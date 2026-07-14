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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H

// Re-export Stream class from pynative runtime for backward compatibility
#include "include/pynative/utils/runtime/stream.h"

namespace mindspore {
namespace hal {
// Backward compatibility aliases for Python binding
using StreamPy = Stream;
using StreamPyPtr = StreamPtr;

// Use the functions from stream.h, with alias for backward compatibility
inline void Synchronize() { SynchronizeAllStreams(); }
}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_STREAM_PY_H
