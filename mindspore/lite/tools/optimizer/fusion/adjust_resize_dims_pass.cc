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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/adjust_resize_dims_pass.h"
#include <memory>
#include <vector>
#include "ops/resize.h"
#include "ops/op_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/squeeze.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/unsqueeze.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int32_t kShapeMinus_1 = -1;
constexpr int32_t kShape_0 = 0;
constexpr size_t kShape_1 = 1;
constexpr size_t kShape_2 = 2;
constexpr size_t kInputIndex_0 = 0;
constexpr size_t kInputIndex_1 = 1;
constexpr size_t kInputIndex_2 = 2;
constexpr size_t kIndices_1 = 1;
constexpr size_t kIndices_2 = 2;
constexpr size_t kIndices_3 = 3;
constexpr size_t kIndices_4 = 4;
constexpr size_t kDimIndex_0 = 0;
constexpr size_t kAxis_0 = 0;
constexpr size_t kResizeInputDim_3 = 3;
constexpr size_t kResizeInputDim_5 = 5;
}  // namespace

CNodePtr CreateBeforeReshapeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_LOG(INFO) << "create reshape node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto reshape_op = std::make_unique<ops::Reshape>();
  MS_CHECK_TRUE_RET(reshape_op != nullptr, nullptr);
  auto reshape_prim_c = reshape_op->GetPrim();
  MS_CHECK_TRUE_RET(reshape_prim_c != nullptr, nullptr);
  std::vector<int32_t> shape = {static_cast<int32_t>(kShape_1), kShapeMinus_1, kShape_0, kShape_0, kShape_0};
  AnfNodePtr shape_node = BuildIntVecParameterNode(func_graph, shape, cnode->fullname_with_scope() + "_shape");
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "shape_node is nullptr!";
    return nullptr;
  }
  if (cnode->inputs().size() < kInputIndex_1 + 1) {
    MS_LOG(INFO) << "The inputs num of " << cnode->fullname_with_scope() << " is smaller than " << (kInputIndex_1 + 1)
                 << ", please check it!";
    return nullptr;
  }
  auto inputs = {cnode->input(kInputIndex_1), shape_node};
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, inputs);
  MS_CHECK_TRUE_RET(reshape_node != nullptr, nullptr);
  reshape_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_reshape");
  if (cnode->abstract() != nullptr) {
    reshape_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create reshape node end.";
  return reshape_node;
}

CNodePtr CreateSqueezeCnode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_LOG(INFO) << "create squeeze node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto squeeze_op = std::make_unique<ops::Squeeze>();
  MS_CHECK_TRUE_RET(squeeze_op != nullptr, nullptr);
  squeeze_op->set_axis({kDimIndex_0});
  auto squeeze_prim_c = squeeze_op->GetPrim();
  if (squeeze_prim_c == nullptr) {
    MS_LOG(ERROR) << "squeeze_prim_c is nullptr!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {cnode};
  auto squeeze_node = func_graph->NewCNode(squeeze_prim_c, inputs);
  MS_CHECK_TRUE_RET(squeeze_node != nullptr, nullptr);
  squeeze_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_squeeze");
  if (cnode->abstract() != nullptr) {
    squeeze_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create squeeze node end.";
  return squeeze_node;
}

CNodePtr CreateResizeCNodeFor5D(const FuncGraphPtr &func_graph, const CNodePtr &resize_cnode,
                                const CNodePtr &squeeze_cnode) {
  MS_LOG(INFO) << "create resize node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(resize_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(squeeze_cnode != nullptr, nullptr);
  auto resize_op = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(resize_op != nullptr, nullptr);
  auto resize_prim_c = resize_op->GetPrim();
  MS_CHECK_TRUE_RET(resize_prim_c != nullptr, nullptr);
  if (resize_cnode->inputs().size() < kInputIndex_0 + 1) {
    MS_LOG(INFO) << "The inputs num of " << resize_cnode->fullname_with_scope() << " is smaller than "
                 << (kInputIndex_0 + 1) << ", please check it!";
    return nullptr;
  }
  if (!utils::isa<ValueNodePtr>(resize_cnode->input(kInputIndex_0))) {
    MS_LOG(INFO) << "The first input of resize_cnode is not ValueNode!";
    return nullptr;
  }
  ValueNodePtr value_node = resize_cnode->input(kInputIndex_0)->cast<ValueNodePtr>();
  PrimitivePtr src_prim = GetValueNode<PrimitivePtr>(value_node);
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "src_prim is nullptr!";
    return nullptr;
  }
  resize_prim_c->SetAttrs(src_prim->attrs());
  (void)resize_prim_c->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  std::vector<float> shape = {kShape_2, kShape_2};
  AnfNodePtr shape_node =
    BuildFloatVecParameterNode(func_graph, shape, squeeze_cnode->fullname_with_scope() + "_shape");
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "shape_node is nullptr!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {squeeze_cnode, shape_node};
  auto new_resize_cnode = func_graph->NewCNode(resize_prim_c, inputs);
  MS_CHECK_TRUE_RET(new_resize_cnode != nullptr, nullptr);
  new_resize_cnode->set_fullname_with_scope(squeeze_cnode->fullname_with_scope() + "_resize");
  if (squeeze_cnode->abstract() != nullptr) {
    new_resize_cnode->set_abstract(squeeze_cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create resize node end.";
  return new_resize_cnode;
}

CNodePtr CreateShapeCNode(const FuncGraphPtr &func_graph, const CNodePtr &resize_cnode) {
  MS_LOG(INFO) << "create shape node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(resize_cnode != nullptr, nullptr);
  auto reshape_op = std::make_unique<ops::Shape>();
  MS_CHECK_TRUE_RET(reshape_op != nullptr, nullptr);
  auto shape_prim_c = reshape_op->GetPrim();
  MS_CHECK_TRUE_RET(shape_prim_c != nullptr, nullptr);
  if (resize_cnode->inputs().size() < kInputIndex_1 + 1) {
    MS_LOG(INFO) << "The inputs num of " << resize_cnode->fullname_with_scope() << " is smaller than "
                 << (kInputIndex_1 + 1) << ", please check it!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {resize_cnode->input(kInputIndex_1)};
  auto shape_cnode = func_graph->NewCNode(shape_prim_c, inputs);
  if (shape_cnode == nullptr) {
    MS_LOG(ERROR) << "shape_cnode is nullptr!";
    return nullptr;
  }
  shape_cnode->set_fullname_with_scope(shape_cnode->fullname_with_scope() + "_shape");
  if (resize_cnode->abstract() != nullptr) {
    shape_cnode->set_abstract(resize_cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create shape node end.";
  return shape_cnode;
}

CNodePtr CreateGatherCNode(const FuncGraphPtr &func_graph, const CNodePtr &shape_node) {
  MS_LOG(INFO) << "create gather node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(shape_node != nullptr, nullptr);
  auto gather_op = std::make_unique<ops::Gather>();
  MS_CHECK_TRUE_RET(gather_op != nullptr, nullptr);
  auto gather_prim_c = gather_op->GetPrim();
  MS_CHECK_TRUE_RET(gather_prim_c != nullptr, nullptr);
  auto indices_node =
    BuildIntVecParameterNode(func_graph, {kIndices_3, kIndices_4}, shape_node->fullname_with_scope() + "_indices");
  if (indices_node == nullptr) {
    MS_LOG(ERROR) << "shape_node is nullptr!";
    return nullptr;
  }
  auto axis_node = BuildIntVecParameterNode(func_graph, {kAxis_0}, shape_node->fullname_with_scope() + "_axis");
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "axis_node is nullptr!";
    return nullptr;
  }
  auto cnode = func_graph->NewCNode(gather_prim_c, {shape_node, indices_node, axis_node});
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope(shape_node->fullname_with_scope() + "_gather");
  if (shape_node->abstract() != nullptr) {
    cnode->set_abstract(shape_node->abstract()->Clone());
  }
  MS_LOG(INFO) << "create gather node end.";
  return cnode;
}

CNodePtr CreateAfterReshapeNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const CNodePtr &shape_node) {
  MS_LOG(INFO) << "create reshape node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(shape_node != nullptr, nullptr);
  auto reshape_op = std::make_unique<ops::Reshape>();
  MS_CHECK_TRUE_RET(reshape_op != nullptr, nullptr);
  auto reshape_prim_c = reshape_op->GetPrim();
  MS_CHECK_TRUE_RET(reshape_prim_c != nullptr, nullptr);
  std::vector<AnfNodePtr> inputs = {cnode, shape_node};
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, inputs);
  if (reshape_node == nullptr) {
    MS_LOG(ERROR) << "reshape_node is nullptr!";
    return nullptr;
  }
  reshape_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_reshape_after");
  if (cnode->abstract() != nullptr) {
    reshape_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create reshape node end.";
  return reshape_node;
}

CNodePtr CreateMulNode(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode) {
  MS_LOG(INFO) << "create mul_fusion node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_cnode != nullptr, nullptr);
  auto mul_fusion_op = std::make_unique<ops::MulFusion>();
  MS_CHECK_TRUE_RET(mul_fusion_op != nullptr, nullptr);
  auto mul_fusion_prim_c = mul_fusion_op->GetPrim();
  MS_CHECK_TRUE_RET(mul_fusion_prim_c != nullptr, nullptr);
  auto indices_node = BuildIntVecParameterNode(func_graph, {kIndices_1, kIndices_1, kIndices_1, kIndices_2, kIndices_2},
                                               input_cnode->fullname_with_scope() + "_indices");
  if (indices_node == nullptr) {
    MS_LOG(ERROR) << "indices_node is nullptr!";
    return nullptr;
  }
  auto cnode = func_graph->NewCNode(mul_fusion_prim_c, {input_cnode, indices_node});
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_mul_fusion");
  if (input_cnode->abstract() != nullptr) {
    cnode->set_abstract(input_cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create mul_fusion node end.";
  return cnode;
}

CNodePtr CreateUnsqueezeCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_LOG(INFO) << "create unsqueeze node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(cnode != nullptr, nullptr);
  auto unsqueeze_op = std::make_unique<ops::Unsqueeze>();
  MS_CHECK_TRUE_RET(unsqueeze_op != nullptr, nullptr);
  unsqueeze_op->set_axis({0});
  auto unsqueeze_prim_c = unsqueeze_op->GetPrim();
  if (unsqueeze_prim_c == nullptr) {
    MS_LOG(ERROR) << "unsqueeze_prim_c is nullptr!";
    return nullptr;
  }
  if (cnode->inputs().size() < kInputIndex_1 + 1) {
    MS_LOG(INFO) << "The inputs num of " << cnode->fullname_with_scope() << " is smaller than " << (kInputIndex_1 + 1)
                 << ", please check it!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {cnode->input(kInputIndex_1)};
  auto unsqueeze_node = func_graph->NewCNode(unsqueeze_prim_c, inputs);
  if (unsqueeze_node == nullptr) {
    MS_LOG(ERROR) << "unsqueeze_node is nullptr!";
    return nullptr;
  }
  unsqueeze_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_unsqueeze");
  if (cnode->abstract() != nullptr) {
    unsqueeze_node->set_abstract(cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create unsqueeze node end.";
  return unsqueeze_node;
}

CNodePtr CreateResizeCNodeFor3D(const FuncGraphPtr &func_graph, const CNodePtr &resize_cnode,
                                const CNodePtr &unsqueeze_cnode) {
  MS_LOG(INFO) << "create resize node start.";
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(resize_cnode != nullptr, nullptr);
  MS_CHECK_TRUE_RET(unsqueeze_cnode != nullptr, nullptr);
  if (resize_cnode->inputs().size() < kInputIndex_2 + 1) {
    MS_LOG(INFO) << "The inputs num of " << resize_cnode->fullname_with_scope() << " is smaller than "
                 << (kInputIndex_2 + 1) << ", please check it!";
    return nullptr;
  }
  if (!utils::isa<ParameterPtr>(resize_cnode->input(kInputIndex_2))) {
    MS_LOG(INFO) << "The third input of resize_cnode is not Parameter!";
    return nullptr;
  }
  auto resize_shape_node = resize_cnode->input(kInputIndex_2)->cast<ParameterPtr>();
  if (resize_shape_node->default_param() == nullptr) {
    MS_LOG(INFO) << "resize_shape_node->default_param() is nullptr!";
    return nullptr;
  }
  auto shape_tensor = std::dynamic_pointer_cast<tensor::Tensor>(resize_shape_node->default_param());
  if (shape_tensor == nullptr) {
    MS_LOG(INFO) << "Get resize shape_tensor failed!";
    return nullptr;
  }
  if (shape_tensor->data_type() == kNumberTypeFloat32) {
    auto tensor_data = static_cast<float *>(shape_tensor->data_c());
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr!";
      return nullptr;
    }
  } else {
    MS_LOG(INFO) << "not support data type, " << shape_tensor->data_type();
    return nullptr;
  }
  float *tensor_data = reinterpret_cast<float *>(shape_tensor->data_c());
  if (tensor_data == nullptr) {
    MS_LOG(INFO) << "tensor_data is nullptr!";
    return nullptr;
  }
  if (shape_tensor->ElementsNum() < static_cast<int>(kShape_2)) {
    MS_LOG(INFO) << "resize shape tensor should contain at least 2 elements, but got "
                 << " shape_tensor->ElementsNum()!";
    return nullptr;
  }
  std::vector<float> shape = {tensor_data[shape_tensor->ElementsNum() - 2],
                              tensor_data[shape_tensor->ElementsNum() - 1]};
  auto resize_op = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(resize_op != nullptr, nullptr);
  auto resize_prim_c = resize_op->GetPrim();
  if (resize_prim_c == nullptr) {
    MS_LOG(ERROR) << "resize_prim_c is nullptr!";
    return nullptr;
  }
  if (!utils::isa<ValueNodePtr>(resize_cnode->input(kInputIndex_0))) {
    MS_LOG(INFO) << "The first input of resize_cnode is not ValueNode!";
    return nullptr;
  }
  ValueNodePtr value_node = resize_cnode->input(kInputIndex_0)->cast<ValueNodePtr>();
  PrimitivePtr src_prim = GetValueNode<PrimitivePtr>(value_node);
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "src_prim is nullptr!";
    return nullptr;
  }
  resize_prim_c->SetAttrs(src_prim->attrs());
  (void)resize_prim_c->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));

  AnfNodePtr shape_node =
    BuildFloatVecParameterNode(func_graph, shape, unsqueeze_cnode->fullname_with_scope() + "_shape");
  if (shape_node == nullptr) {
    MS_LOG(ERROR) << "shape_node is nullptr!";
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs = {unsqueeze_cnode, shape_node};
  auto new_resize_cnode = func_graph->NewCNode(resize_prim_c, inputs);
  if (new_resize_cnode == nullptr) {
    MS_LOG(ERROR) << "new_resize_cnode is nullptr!";
    return nullptr;
  }
  new_resize_cnode->set_fullname_with_scope(unsqueeze_cnode->fullname_with_scope() + "_resize");
  if (unsqueeze_cnode->abstract() != nullptr) {
    new_resize_cnode->set_abstract(unsqueeze_cnode->abstract()->Clone());
  }
  MS_LOG(INFO) << "create resize node end.";
  return new_resize_cnode;
}

int GetResizeInputDims(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, 0);
  if (cnode->inputs().size() < kInputIndex_2 + 1) {
    MS_LOG(INFO) << "The inputs num of " << cnode->fullname_with_scope() << " is smaller than " << (kInputIndex_2 + 1)
                 << ", please check it!";
    return 0;
  }
  auto size_node = cnode->input(kInputIndex_2);
  if (size_node == nullptr) {
    MS_LOG(ERROR) << "size_node is nullptr!";
    return 0;
  }
  if (!utils::isa<ParameterPtr>(size_node)) {
    MS_LOG(INFO) << "size_node is not Parameter!";
    return 0;
  }
  auto resize_shape_node = size_node->cast<ParameterPtr>();

  if (resize_shape_node == nullptr) {
    MS_LOG(ERROR) << "resize_shape_node is nullptr!";
    return 0;
  }
  if (resize_shape_node->default_param() == nullptr) {
    MS_LOG(INFO) << "resize_shape_node->default_param() is nullptr!";
    return 0;
  }
  auto shape_tensor = std::dynamic_pointer_cast<tensor::Tensor>(resize_shape_node->default_param());
  if (shape_tensor == nullptr) {
    MS_LOG(INFO) << "shape_tensor is nullptr!";
    return 0;
  }
  if (shape_tensor->shape().size() < kDimIndex_0 + 1) {
    MS_LOG(INFO) << "The dimension size of resize's shape tensor is smaller than " << (kDimIndex_0 + 1)
                 << ", please check it!";
    return 0;
  }
  int input_dim = shape_tensor->shape().at(kDimIndex_0);
  MS_LOG(INFO) << "shape_tensor dimension: " << input_dim;
  return input_dim;
}

bool AdjustResize5DToResize4D(const FuncGraphPtr &func_graph, const CNodePtr &resize_cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto input_shape_cnode = CreateShapeCNode(func_graph, resize_cnode);
  if (input_shape_cnode == nullptr) {
    MS_LOG(INFO) << "input_shape_cnode is nullptr!";
    return false;
  }
  auto before_reshape_cnode = CreateBeforeReshapeNode(func_graph, resize_cnode);
  if (before_reshape_cnode == nullptr) {
    MS_LOG(INFO) << "before_reshape_cnode is nullptr!";
    return false;
  }
  auto squeeze_cnode = CreateSqueezeCnode(func_graph, before_reshape_cnode);
  if (squeeze_cnode == nullptr) {
    MS_LOG(INFO) << "squeeze_cnode is nullptr!";
    return false;
  }
  auto new_resize_cnode = CreateResizeCNodeFor5D(func_graph, resize_cnode, squeeze_cnode);
  if (new_resize_cnode == nullptr) {
    MS_LOG(INFO) << "new_resize_cnode is nullptr!";
    return false;
  }
  auto mul_node = CreateMulNode(func_graph, input_shape_cnode);
  if (mul_node == nullptr) {
    MS_LOG(INFO) << "mul_node is nullptr!";
    return false;
  }
  auto reshape_node = CreateAfterReshapeNode(func_graph, new_resize_cnode, mul_node);
  if (reshape_node == nullptr) {
    MS_LOG(INFO) << "reshape_node is nullptr!";
    return false;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return false;
  }
  return manager->Replace(resize_cnode, reshape_node);
}

bool AdjustResize3DToResize4D(const FuncGraphPtr &func_graph, const CNodePtr &resize_cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto unsqueze_cnode = CreateUnsqueezeCNode(func_graph, resize_cnode);
  if (unsqueze_cnode == nullptr) {
    MS_LOG(INFO) << "unsqueze_cnode is nullptr!";
    return false;
  }
  auto new_resize_cnode = CreateResizeCNodeFor3D(func_graph, resize_cnode, unsqueze_cnode);
  if (new_resize_cnode == nullptr) {
    MS_LOG(INFO) << "new_resize_cnode is nullptr!";
    return false;
  }
  auto squeeze_cnode = CreateSqueezeCnode(func_graph, new_resize_cnode);
  if (squeeze_cnode == nullptr) {
    MS_LOG(INFO) << "squeeze_cnode is nullptr!";
    return false;
  }
  auto manager = Manage(func_graph);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return false;
  }
  return manager->Replace(resize_cnode, squeeze_cnode);
}

bool AdjustResizeDimsPass::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  MS_LOG(INFO) << "AdjustResizeDimsPass start.";
  auto node_list = TopoSort(func_graph->get_return());
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return false;
  }
  bool adjust_result = true;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!opt::CheckPrimitiveType(node, prim::kPrimResize)) {
      continue;
    }
    auto resize_cnode = node->cast<CNodePtr>();
    MS_LOG(INFO) << resize_cnode->fullname_with_scope();
    int resize_input_dims = GetResizeInputDims(resize_cnode);
    if (resize_input_dims == kResizeInputDim_3) {
      MS_LOG(INFO) << "Find a 3D resize cnode.";
      adjust_result = AdjustResize3DToResize4D(func_graph, resize_cnode);
    } else if (resize_input_dims == kResizeInputDim_5) {
      MS_LOG(INFO) << "Find a 5D resize cnode.";
      adjust_result = AdjustResize5DToResize4D(func_graph, resize_cnode);
    } else {
      continue;
    }
    if (!adjust_result) {
      MS_LOG(INFO) << "Adjust resize node failed!";
      return adjust_result;
    }
  }
  MS_LOG(INFO) << "AdjustResizeDimsPass end.";
  return true;
}
}  // namespace opt
}  // namespace mindspore
