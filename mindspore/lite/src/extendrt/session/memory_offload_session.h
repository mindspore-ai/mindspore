/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_MEMORY_LOAD_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_MEMORY_LOAD_SESSION_H_

#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include "src/extendrt/kernel/default/kernel_mod_kernel.h"
#include "src/extendrt/session/single_op_session.h"
#include "src/extendrt/graph_compiler/compile_result_builder.h"
#include "src/extendrt/memory_offload/infer_strategy_builder.h"
namespace mindspore::lite {
/// \brief memory offload implementation.
class MemoryOffloadInferSession : public SingleOpInferSession {
 public:
  MemoryOffloadInferSession() = default;
  ~MemoryOffloadInferSession() override = default;

  Status Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info = {}) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0,
                      uint32_t *graph_id = nullptr) override;

 private:
  Status BuildCustomAscendKernel(const CNodePtr &cnode, lite::CompileNodePtr compile_node);
  kernel::KernelModKernel *BuildCustomAscendKernelImpl(const CNodePtr &cnode, const lite::CompileNodePtr &compile_node);

  lite::CompileResultPtr compile_result_;
  std::vector<kernel::KernelModKernel *> kernels_;
  std::shared_ptr<device::SwapContext> swap_context_;
  std::shared_ptr<device::SwapStrategy> strategy_;
  std::shared_ptr<Context> context_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_EXTENDRT_MEMORY_LOAD_SESSION_H_
