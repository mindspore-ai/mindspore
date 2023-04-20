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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_EXPANDER_LITE_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_EXPANDER_LITE_H_
#include <memory>
#include <string>
#include <vector>

#include "backend/common/graph_kernel/core/graph_kernel_expander.h"
#include "ir/func_graph.h"
#include "utils/hash_set.h"

namespace mindspore::graphkernel {
class FixFormatDeco : public ExpanderDecorator {
 public:
  explicit FixFormatDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~FixFormatDeco() = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<FixFormatDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class InferValueDeco : public ExpanderDecorator {
 public:
  explicit InferValueDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~InferValueDeco() = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<InferValueDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class PoolLayoutDeco : public ExpanderDecorator {
 public:
  explicit PoolLayoutDeco(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~PoolLayoutDeco() = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<PoolLayoutDeco>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;
};

class GraphKernelExpanderLite : public GraphKernelExpander {
 public:
  GraphKernelExpanderLite() : GraphKernelExpander() {}
  explicit GraphKernelExpanderLite(const std::string &name) : GraphKernelExpander(name) {}
  ~GraphKernelExpanderLite() override = default;

 protected:
  bool DisableConvTuning();
  std::vector<PrimitivePtr> ConvTuningExpanderOps();
  std::vector<PrimitivePtr> InitOpList() override;
  ExpanderPtr InitExpander(const AnfNodePtr &node) override;
  bool CanExpand(const CNodePtr &node) const override;
  void PreProcessAllNode(const CNodePtr &node) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_EXPANDER_LITE_H_
