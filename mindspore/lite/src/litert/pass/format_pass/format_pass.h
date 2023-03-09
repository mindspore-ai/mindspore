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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_FORMAT_PASS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_FORMAT_PASS_H_

#include <vector>
#include <memory>
#include "src/litert/kernel_exec.h"
#include "src/litert/sub_graph_kernel.h"

namespace mindspore::lite::pass {
class FormatPass {
 public:
  explicit FormatPass(mindspore::Format format) : format_(format) {}
  virtual ~FormatPass() = default;
  virtual int RunPass(kernel::SubGraphKernel *graph, std::vector<lite::Tensor *> *tensors) = 0;

 protected:
  Format format_ = DEFAULT_FORMAT;
};
using FormatPassPtr = std::shared_ptr<FormatPass>;

class FormatOptimize {
 public:
  int AddPass(FormatPassPtr pass);
  int RunPass(kernel::SubGraphKernel *graph, std::vector<Tensor *> *tensors);

 private:
  std::vector<FormatPassPtr> pass_list_;
};
using FormatOptimizePtr = std::shared_ptr<FormatOptimize>;

int RuntimeFormatPass(std::vector<mindspore::kernel::KernelExec *> *subgraph_list,
                      std::vector<mindspore::lite::Tensor *> *tensors,
                      mindspore::Format format = mindspore::Format::NHWC);
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_FORMAT_PASS_H_
