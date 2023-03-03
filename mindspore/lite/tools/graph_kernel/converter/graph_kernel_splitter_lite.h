/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_SPLITTER_LITE_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_SPLITTER_LITE_H_
#include <string>
#include "backend/common/graph_kernel/core/split_schemer.h"
#include "backend/common/graph_kernel/core/graph_kernel_splitter.h"

namespace mindspore::graphkernel {
class GraphKernelSplitterWithTuning : public GraphKernelSplitter {
 public:
  GraphKernelSplitterWithTuning() = default;
  ~GraphKernelSplitterWithTuning() = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  SplitSchemerPtr GetSplitSchema(const std::string &processor) override;

 protected:
  bool StartTuning(const std::string &dir_path) const;

  std::string tuning_path_;
  bool tuning_flag_{true};
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_SPLITTER_LITE_H_
