/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_COMPILER_H_
#define MINDSPORE_PI_JIT_COMPILER_H_

#include <functional>
#include <string>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/common.h"

namespace mindspore {
namespace pijit {
using CallableGraph = std::function<PyObject *(PyObject *, PyObject *)>;
// Compiler to parse python byte code
class Compiler {
 public:
  static CallableGraph Compile(const PyFunctionObject &func, const PyFrameObject &frame, const std::string &phase);

 private:
  Compiler() = default;
};

class MindCompiler {
 public:
  struct CompileInfo {
    std::string co_name_;
    int co_argcount_;
    int co_kwonlyargcount_;
    int co_flags_;
  };
  static CallableGraph Compile(const FuncGraphPtr &func_graph, const py::tuple &args, const py::dict &kwargs,
                               const std::string &phase, const CompileInfo &compile_info);

 private:
  MindCompiler() = default;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_COMPILER_H_
