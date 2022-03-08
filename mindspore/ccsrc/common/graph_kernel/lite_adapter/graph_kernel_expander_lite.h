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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_GRAPH_KERNEL_EXPANDER_LITE_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_GRAPH_KERNEL_EXPANDER_LITE_H_
#include <memory>
#include <vector>

#include "common/graph_kernel/core/graph_kernel_expander.h"
#include "ir/func_graph.h"
#include "utils/hash_set.h"

namespace mindspore::graphkernel {
class LiteExpander : public DefaultExpander {
 public:
  explicit LiteExpander(HashSet<size_t> input_idx) : input_idx_(input_idx) {}
  ~LiteExpander() = default;
  AnfNodePtr Run(const AnfNodePtr &node) override;

 private:
  HashSet<size_t> input_idx_;
};

class GraphKernelExpanderLite : public GraphKernelExpander {
 public:
  GraphKernelExpanderLite() : GraphKernelExpander() {}
  ~GraphKernelExpanderLite() override = default;

 protected:
  std::vector<PrimitivePtr> InitOpList() override;
  ExpanderPtr GetExpander(const AnfNodePtr &node) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_GRAPH_KERNEL_EXPANDER_LITE_H_
