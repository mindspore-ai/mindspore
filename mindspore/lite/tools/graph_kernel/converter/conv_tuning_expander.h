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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CONV_TUNING_EXPANDER_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CONV_TUNING_EXPANDER_H_
#include <vector>
#include "ir/func_graph.h"
#include "tools/graph_kernel/converter/graph_kernel_expander_lite.h"

namespace mindspore::graphkernel {
class ConvTuningExpander : public GraphKernelExpanderLite {
 public:
  ConvTuningExpander() : GraphKernelExpanderLite("conv_tuning_expander") {}
  ~ConvTuningExpander() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  std::vector<PrimitivePtr> InitOpList() override;
};

bool InvalidConvAttr(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &pad,
                     const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation);
}  // namespace mindspore::graphkernel

#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_CONV_TUNING_EXPANDER_H_
