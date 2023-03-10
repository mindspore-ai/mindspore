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

#include "src/extendrt/graph_compiler/single_graph_compiler.h"
#include <memory>
#include <iostream>
#include "src/extendrt/execution_flow.h"
#include "src/extendrt/graph_compiler/compile_result_builder.h"

namespace mindspore {
namespace infer {
CompileResultPtr SingleGraphCompiler::Build(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                                            const AnfNodePtrList &outputs) {
  // assume graph_format is NHWC
  auto graph_format = NHWC;
  auto builder = std::make_shared<CompileResultBuilder>(graph_format);
  return builder->Build(segment, inputs, outputs);
}

abstract::ExecutionFlowPtr SingleGraphCompiler::Schedule(const CompileResultPtr &node_list) {
  // infershape
  // choose lite-kernel else choose kernel-mod
  // tensor data type/format transform
  if (MS_UNLIKELY(scheduler_ == nullptr)) {
    scheduler_ = std::make_shared<SingleGraphScheduler>(this->context_.get());
  }
  return scheduler_->Schedule(node_list);
}

abstract::ExecutionFlowPtr SingleGraphCompiler::Compile(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                                                        const AnfNodePtrList &outputs) {
  auto node_list = this->Build(segment, inputs, outputs);
  std::cout << node_list->Dump() << std::endl;
  return this->Schedule(node_list);
}
}  // namespace infer
}  // namespace mindspore
