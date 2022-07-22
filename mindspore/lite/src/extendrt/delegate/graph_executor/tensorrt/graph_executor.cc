/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "extendrt/delegate/graph_executor/tensorrt/graph_executor.h"
#include "extendrt/delegate/graph_executor/factory.h"

namespace mindspore {
const char tensorrt_provider[] = "tensorrt";
bool TensorRTGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  return true;
}

bool TensorRTGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                                     std::vector<tensor::Tensor> *outputs,
                                     const std::map<string, string> &compile_options) {
  return true;
}

static std::shared_ptr<device::GraphExecutor> TensorRTGraphExecutorCreator() {
  return std::make_shared<TensorRTGraphExecutor>();
}

REG_GRAPH_EXECUTOR(kGPU, tensorrt_provider, TensorRTGraphExecutorCreator);
}  // namespace mindspore
