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
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace pipeline {
struct ExecutorInfo {
  FuncGraphPtr primal_func_graph{nullptr};
  FuncGraphPtr func_graph;
  // The grad graph of func_graph, it will create in PyNative mode when @jit is used.
  FuncGraphPtr grad_graph;
  ResourcePtr resource;
  // The num of input data.
  std::size_t arg_list_size;
  // The all args of graph,including input data and weight.
  VectorRef arg_list;
};
using ExecutorInfoPtr = std::shared_ptr<ExecutorInfo>;

inline std::string GetPhasePrefix(const std::string &phase) {
  auto pos = phase.find('.');
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Phase has no . for prefix" << phase;
  }
  return phase.substr(0, pos);
}
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_BASE_H_
