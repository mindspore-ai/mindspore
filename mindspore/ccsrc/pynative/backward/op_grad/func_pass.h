/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_PASS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_PASS_H_

#include <string>
#include <utility>
#include <memory>
#include "ir/anf.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"

namespace mindspore {
namespace pynative {
namespace autograd {
class FuncBuilder;
}

namespace bprop_pass {
using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;

struct FuncPassForward {
  explicit FuncPassForward(autograd::FuncBuilder *func_builder, device::DeviceType device_target)
      : func_builder_(func_builder), device_target_(device_target) {}

  // Pass for expander outputs
  NodePtrList PassForOpInput(const PrimitivePtr &prim, const NodePtrList &inputs);
  NodePtr BatchNormGradToBNInferGrad(const NodePtrList &inputs, bool is_scale_or_bias_grad);
  NodePtr GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(const NodePtrList &inputs, const expander::DAttr &attrs,
                                                             const NodePtr &out, const NodePtr &dout,
                                                             bool is_graph_mode);

 private:
  // Plant op input which is tuple, and set kAttrDynInputSizes attr
  NodePtrList ConvertMakeTupleInputToDynamicInput(const PrimitivePtr &prim, const NodePtrList &inputs);

  autograd::FuncBuilder *func_builder_{nullptr};
  device::DeviceType device_target_;
};
using FuncPassForwardPtr = std::shared_ptr<FuncPassForward>;
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_PASS_H_
