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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PREPROCESS_WEIGHT_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PREPROCESS_WEIGHT_H_
#include <memory>

#include "backend/common/graph_kernel/core/graph_kernel_expander.h"

namespace mindspore::graphkernel {
class SubstituteConv2D : public ExpanderDecorator {
 public:
  using ExpanderDecorator::ExpanderDecorator;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<SubstituteConv2D>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  AnfNodePtr InferWeightValue(const AnfNodePtr &node);
};

class MatmulPackB : public ExpanderDecorator {
 public:
  using ExpanderDecorator::ExpanderDecorator;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<MatmulPackB>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override;

 protected:
  AnfNodePtr InferValue(const AnfNodePtr &node);
  tensor::TensorPtr PackB(const tensor::TensorPtr &tensor, const ShapeVector &shape, bool transpose);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_PREPROCESS_WEIGHT_H_
