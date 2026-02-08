/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_COMPILER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_COMPILER_H_

#include <functional>
#include <string>
#include <utility>
#include "include/utils/python_adapter.h"
#include "frontend/jit/pi/runtime.h"
#include "frontend/jit/pi/python_adapter/pydef.h"

namespace mindspore {
namespace pijit {
using CallableGraph = std::function<PyObject *(PyObject *, PyObject *)>;

class GraphCompiler {
 public:
  struct CompileInfo {
    std::string co_filename_;
    std::string co_name_;
    int co_firstlineno_;
    int co_argcount_;
    int co_kwonlyargcount_;
    int co_flags_;
    size_t origin_top_input_num_;
  };
  static CallableGraph Compile(const FuncGraphPtr &func_graph, const py::tuple &args, const py::dict &kwargs,
                               const std::string &phase, const CompileInfo &compile_info);
  static std::pair<std::string, CallableGraph> Compile(const FuncGraphPtr &func_graph, const CompileInfo &compile_info);

 private:
  GraphCompiler() = default;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_COMPILER_H_
