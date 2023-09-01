/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_BASE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_BASE_H_

#include <mutex>
#include <memory>
#include <string>
#include <sstream>

#include "ir/anf.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace pipeline {
struct ExecutorInfo {
  FuncGraphPtr jit_primal_func_graph{nullptr};
  FuncGraphPtr func_graph{nullptr};
  // The grad graph of func_graph, it will create in PyNative mode when @jit is used.
  FuncGraphPtr jit_grad_graph{nullptr};
  ResourcePtr resource{nullptr};
  // The num of input data.
  std::size_t arg_list_size;
  // The all args of graph,including input data and weight.
  VectorRef arg_list;
};
using ExecutorInfoPtr = std::shared_ptr<ExecutorInfo>;

inline std::string GetPhasePrefix(const std::string &phase) {
  auto pos = phase.find('.');
  if (pos == std::string::npos) {
    MS_LOG(INTERNAL_EXCEPTION) << "Phase has no . for prefix" << phase;
  }
  return phase.substr(0, pos);
}
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_BASE_H_
