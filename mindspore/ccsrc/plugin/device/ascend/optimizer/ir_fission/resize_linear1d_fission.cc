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
#include "plugin/device/ascend/optimizer/ir_fission/resize_linear1d_fission.h"
#include <memory>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kResizeLinear1DInputNum = 2;
constexpr int64_t kExpandDim = -1;
constexpr int64_t kSqueezeDim = 2;

AnfNodePtr AddExpandDimsNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                             const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);

  // Add ExpandDims Node
  std::vector<AnfNodePtr> expand_dims_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimExpandDims->name())), input_node};
  auto expand_dims = pass.NewCNode(expand_dims_inputs, func_graph);

  // Set ExpandDims OutShape and Type
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto expand_shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  (void)expand_shape.insert(expand_shape.end() + kExpandDim, 1);
  (void)common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {expand_shape}, expand_dims.get());

  // Set ExpandDims Attr
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(kExpandDim), expand_dims);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), expand_dims);

  return expand_dims;
}
}  // namespace

const BaseRef ResizeLinear1DFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto resize_linear1d_prim = std::make_shared<Primitive>(prim::kPrimResizeLinear1D->name());
  return VectorRef({resize_linear1d_prim, Xs});
}

const AnfNodePtr ResizeLinear1DFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto resize_linear1d = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(resize_linear1d);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }

  if (resize_linear1d->size() != kResizeLinear1DInputNum + 1) {
    MS_LOG(INFO) << "The node " << resize_linear1d->DebugString() << " is not equal to " << kResizeLinear1DInputNum
                 << "inputs";
    return nullptr;
  }

  if (!common::AnfAlgo::HasNodeAttr("coordinate_transformation_mode", resize_linear1d)) {
    MS_LOG(EXCEPTION) << "ResizeLinear1D need to set coordinate_transformation_mode attribute.";
  }

  const auto ori_inputs = resize_linear1d->inputs();

  // Add ExpandDims Node
  auto expand_dims = AddExpandDimsNode(func_graph, ori_inputs[kDim1], *this);

  // Get ResizeD Node
  std::vector<AnfNodePtr> resize_d_inputs = {NewValueNode(std::make_shared<Primitive>(kResizeDOpName)), expand_dims};
  auto resize_d = func_graph->NewCNode(resize_d_inputs);
  MS_EXCEPTION_IF_NULL(resize_d);

  // Set ResizeD OutShape and Type
  auto out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto out_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  (void)out_shape.insert(out_shape.end() + kExpandDim, 1);
  (void)common::AnfAlgo::SetOutputInferTypeAndShape({out_type}, {out_shape}, resize_d.get());

  // Set ResizeD Attr
  std::vector<int64_t> size_value = {out_shape[kIndex3]};
  auto x_shape = common::AnfAlgo::GetOutputInferShape(ori_inputs[kDim1], 0);
  float scale = static_cast<float>(size_value[kDim0]) / static_cast<float>(x_shape[kDim2]);
  std::vector<float> scales = {scale};
  common::AnfAlgo::SetNodeAttr("sizes", MakeValue(size_value), resize_d);
  common::AnfAlgo::SetNodeAttr("scales", MakeValue(scales), resize_d);
  common::AnfAlgo::SetNodeAttr("mode", MakeValue("linear"), resize_d);
  common::AnfAlgo::CopyNodeAttr("coordinate_transformation_mode", resize_linear1d, resize_d);

  // Get Squeeze Node
  std::vector<AnfNodePtr> squeeze_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSqueeze->name())),
                                            resize_d};
  auto squeeze = func_graph->NewCNode(squeeze_inputs);
  MS_EXCEPTION_IF_NULL(squeeze);

  // Set Squeeze Attr
  std::vector<int64_t> axis = {kSqueezeDim};
  common::AnfAlgo::SetNodeAttr("axis", MakeValue(axis), squeeze);

  // Set abstract and scope
  squeeze->set_abstract(resize_linear1d->abstract());
  squeeze->set_scope(resize_linear1d->scope());

  return squeeze;
}

const BaseRef ResizeLinear1DGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto resize_linear1d_grad_prim = std::make_shared<Primitive>(prim::kPrimResizeLinear1DGrad->name());
  return VectorRef({resize_linear1d_grad_prim, Xs});
}

const AnfNodePtr ResizeLinear1DGradFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto resize_linear1d_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(resize_linear1d_grad);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  if (resize_linear1d_grad->size() != kResizeLinear1DInputNum + 1) {
    MS_LOG(INFO) << "The node " << resize_linear1d_grad->DebugString() << " is not equal to " << kResizeLinear1DInputNum
                 << "inputs";
    return nullptr;
  }

  if (!common::AnfAlgo::HasNodeAttr("coordinate_transformation_mode", resize_linear1d_grad)) {
    MS_LOG(EXCEPTION) << "ResizeLinear1DGrad need to set coordinate_transformation_mode attribute.";
  }

  const auto ori_inputs = resize_linear1d_grad->inputs();

  // Add ExpandDims Node
  auto expand_dims = AddExpandDimsNode(func_graph, ori_inputs[kDim1], *this);

  // Get ResizeGradD Node
  std::vector<AnfNodePtr> resize_grad_d_inputs = {NewValueNode(std::make_shared<Primitive>(kResizeGradDOpName)),
                                                  expand_dims};
  auto resize_grad_d = func_graph->NewCNode(resize_grad_d_inputs);
  MS_EXCEPTION_IF_NULL(resize_grad_d);

  // Set ResizeGradD OutShape and Type
  auto out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto out_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  (void)out_shape.insert(out_shape.end() + kExpandDim, 1);
  (void)common::AnfAlgo::SetOutputInferTypeAndShape({out_type}, {out_shape}, resize_grad_d.get());

  // Set ResizeGradD Attr
  auto origin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex1);
  auto x_shape = common::AnfAlgo::GetOutputInferShape(ori_inputs[kDim1], 0);
  float scale = static_cast<float>(x_shape[kDim2]) / static_cast<float>(origin_shape[kDim2]);
  std::vector<float> scales = {scale};
  common::AnfAlgo::SetNodeAttr("original_size", MakeValue(origin_shape), resize_grad_d);
  common::AnfAlgo::SetNodeAttr("scales", MakeValue(scales), resize_grad_d);
  common::AnfAlgo::SetNodeAttr("mode", MakeValue("linear"), resize_grad_d);
  common::AnfAlgo::CopyNodeAttr("coordinate_transformation_mode", resize_linear1d_grad, resize_grad_d);

  // Get Squeeze Node
  std::vector<AnfNodePtr> squeeze_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSqueeze->name())),
                                            resize_grad_d};
  auto squeeze = func_graph->NewCNode(squeeze_inputs);
  MS_EXCEPTION_IF_NULL(squeeze);

  // Set Squeeze Attr
  std::vector<int64_t> axis = {kSqueezeDim};
  common::AnfAlgo::SetNodeAttr("axis", MakeValue(axis), squeeze);

  // Set abstract and scope
  squeeze->set_abstract(resize_linear1d_grad->abstract());
  squeeze->set_scope(resize_linear1d_grad->scope());

  return squeeze;
}
}  // namespace opt
}  // namespace mindspore
