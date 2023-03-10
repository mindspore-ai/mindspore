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
#include "plugin/device/ascend/optimizer/ir_fusion/conv2d_backprop_input_dilation_fusion.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"
#include "plugin/device/ascend/hal/device/lic_manager.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kConv2DbpInput1 = 1;
constexpr auto kAttrInputSizes = "input_sizes";
constexpr auto kNCHWDimNum = 4;
constexpr auto kAttrStride = "stride";
constexpr auto kAttrHAxisSize = 2;
constexpr auto kAttrWAxisSize = 1;

bool CheckSupported(const AnfNodePtr &input) {
  if (input == nullptr) {
    return false;
  }
  if (!common::AnfAlgo::HasNodeAttr(kAttrStride, input->cast<CNodePtr>())) {
    MS_LOG(INFO) << "The node " << input->DebugString() << " has no " << kAttrStride << " attr.";
    return false;
  }

  auto stride_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(input, kAttrStride);
  if (stride_size.size() != kNCHWDimNum) {
    return false;
  }

  auto filter_size = common::AnfAlgo::GetPrevNodeOutputInferShape(input, kConv2DbpInput1);
  if (filter_size.size() != kNCHWDimNum) {
    return false;
  }

  auto grad = common::AnfAlgo::GetInputNode(input->cast<CNodePtr>(), 0);
  MS_EXCEPTION_IF_NULL(grad);
  auto shapes = common::AnfAlgo::GetOutputInferShape(grad, 0);
  if (shapes.size() != kNCHWDimNum) {
    return false;
  }

  // currently mindspore only support kOpFormat_NCHW
  if (stride_size[kDim2] == kAttrHAxisSize && filter_size[kDim2] == kAttrWAxisSize) {
    return true;
  }
  return false;
}
}  // namespace

const BaseRef Conv2dBackpropInputDilationFusion::DefinePattern() const {
  VectorRef pattern({conv2d_bp_input_var_, x0_, x1_});
  return pattern;
}

AnfNodePtr Conv2dBackpropInputDilationFusion::CreateConv2DbpInput(const FuncGraphPtr &func_graph,
                                                                  const AnfNodePtr &node, const AnfNodePtr &grad,
                                                                  const EquivPtr &equiv) const {
  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimConv2DBackpropInputD->name())));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x0_));
  (void)inputs.emplace_back(GetAnfNodeByVar(equiv, x1_));
  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttrs(node, new_node);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = common::AnfAlgo::GetOutputInferShape(grad, 0);
  auto filter_size = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kConv2DbpInput1);
  shapes[kDim1] = filter_size[kDim1];
  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shapes}, new_node.get());
  common::AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(std::vector<int64_t>{1, 1, 1, 1}), new_node);
  std::vector<int64_t> input_sizes;
  input_sizes.push_back(shapes[kDim0]);
  input_sizes.push_back(shapes[kDim1]);
  input_sizes.push_back(shapes[kDim2]);
  input_sizes.push_back(shapes[kDim3]);
  common::AnfAlgo::SetNodeAttr(kAttrInputSizes, MakeValue(std::vector<int64_t>{input_sizes}), new_node);
  return new_node;
}

AnfNodePtr Conv2dBackpropInputDilationFusion::CreateDilation(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                             const AnfNodePtr &in) const {
  std::vector<AnfNodePtr> dilation_inputs = {NewValueNode(std::make_shared<Primitive>("Dilation")), in};
  auto dilation = NewCNode(dilation_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(dilation);
  dilation->set_scope(node->scope());
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  auto shapes = {output_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(node, 0)}, {shapes},
                                              dilation.get());
  std::vector<int64_t> dilations = {1, 1, kAttrHAxisSize, kAttrHAxisSize, 1};
  common::AnfAlgo::SetNodeAttr("dilations", MakeValue(dilations), dilation);
  common::AnfAlgo::SetNodeAttr("padding_value", MakeValue<float>(0.0), dilation);
  common::AnfAlgo::SetNodeAttr("pads", MakeValue(std::vector<int64_t>{0, 1, 0, 1}), dilation);
  return dilation;
}

const AnfNodePtr Conv2dBackpropInputDilationFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                            const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_CHECK_CUBE_VECTOR_NOT_SPLIT();
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (GetBoolAttr(cnode, kAttrVisited)) {
    return nullptr;
  }
  if (!CheckSupported(node)) {
    MS_LOG(INFO) << "Op stride not support Conv2dBackpropInput and Dilation split.";
    return nullptr;
  }

  auto input = common::AnfAlgo::GetInputNode(cnode, 0);
  AnfNodePtr conv = CreateConv2DbpInput(graph, node, input, equiv);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), conv);
  AnfNodePtr dilation = CreateDilation(graph, node, conv);
  return dilation;
}
}  // namespace opt
}  // namespace mindspore
