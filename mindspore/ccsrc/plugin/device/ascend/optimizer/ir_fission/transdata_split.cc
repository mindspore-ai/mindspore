/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/transdata_split.h"
#include <set>
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kNchwDimNum = 4;
constexpr size_t kDimC = 1;

const std::set<std::pair<string, string>> invalid_formats_pair = {
  {kOpFormat_C1HWNCoC0, kOpFormat_NCHW},          {kOpFormat_NCHW, kOpFormat_C1HWNCoC0},
  {kOpFormat_C1HWNCoC0, kOpFormat_DEFAULT},       {kOpFormat_DEFAULT, kOpFormat_FRACTAL_ZN_LSTM},
  {kOpFormat_FRACTAL_ZN_LSTM, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT, kOpFormat_C1HWNCoC0}};

bool IsDepthwiseCase(const CNodePtr &node, const std::string &input_format, const std::string &output_format) {
  MS_EXCEPTION_IF_NULL(node);
  abstract::BaseShapePtr base_shape;
  if (input_format == kOpFormat_FRAC_Z && output_format == kOpFormat_DEFAULT) {
    base_shape = AnfAlgo::GetPrevNodeOutputDetailShape(node, 0);
  } else if (input_format == kOpFormat_DEFAULT && output_format == kOpFormat_FRAC_Z) {
    base_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  } else {
    return false;
  }
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    auto shape_vec = shape_ptr->shape();
    return shape_vec.size() == kNchwDimNum && shape_vec[kDimC] == 1;
  }
  return false;
}
}  // namespace

const AnfNodePtr TransDataSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (node != nullptr && node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kTransDataOpName) {
    CheckCNodeInputSize(node->cast<CNodePtr>(), kTransOpInputTensorNum);
    if (IsFormatInvaild(node)) {
      TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));
      return DoSplit(func_graph, node);
    }
  }
  return nullptr;
}

bool TransDataSplit::IsFormatInvaild(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  auto format_pair = std::make_pair(input_format, output_format);

  return invalid_formats_pair.find(format_pair) != invalid_formats_pair.end() ||
         IsDepthwiseCase(cnode, input_format, output_format);
}

const BaseRef TransDataSplit::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimTransData, X});
}

// transdata cannot support frac_z to nchw need split transdata(frac_z-HWCN) and transpose(HWCN-NCHW)
CNodePtr TransDataSplit::DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_node);

  auto input_format = AnfAlgo::GetInputFormat(node, 0);
  auto output_format = AnfAlgo::GetOutputFormat(node, 0);
  CNodePtr new_transdata_node = nullptr;
  CNodePtr new_transpose_node = nullptr;
  CNodePtr new_replace_node = nullptr;
  auto padding_axis = AnfAlgo::GetOutputReshapeType(node, 0);
  // if output_format=default transdata need split transdata->transpose else transpose->transdata
  if (output_format == kOpFormat_DEFAULT || output_format == kOpFormat_NCHW) {
    // trans input_format to hwcn
    new_transdata_node = NewTransOpNode(func_graph, common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), 0), node,
                                        kernel_select_, false, prim::kPrimTransData->name());
    RefreshKernelBuildInfo(kernel_select_, input_format, kOpFormat_HWCN, new_transdata_node, padding_axis);
    // trans hwcn to default_format
    new_transpose_node = NewTransOpNode(func_graph, new_transdata_node, node, kernel_select_, false,
                                        prim::kPrimTranspose->name(), std::vector<int64_t>{3, 2, 0, 1});
    RefreshKernelBuildInfo(kernel_select_, kOpFormat_HWCN, output_format, new_transpose_node);
    new_replace_node = new_transpose_node;
  } else {
    // trans default to hwcn
    new_transpose_node =
      NewTransOpNode(func_graph, common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), 0), node, kernel_select_, false,
                     prim::kPrimTranspose->name(), std::vector<int64_t>{2, 3, 1, 0});
    if (output_format == kOpFormat_FRACTAL_ZN_LSTM) {
      common::AnfAlgo::SetNodeAttr(kAttrNopOp, MakeValue(true), new_transpose_node);
    }
    RefreshKernelBuildInfo(kernel_select_, input_format, kOpFormat_HWCN, new_transpose_node);

    // trans hwcn to output_format
    new_transdata_node =
      NewTransOpNode(func_graph, new_transpose_node, node, kernel_select_, false, prim::kPrimTransData->name());
    RefreshKernelBuildInfo(kernel_select_, kOpFormat_HWCN, output_format, new_transdata_node, padding_axis);
    new_transdata_node->set_abstract(node->abstract());
    new_replace_node = new_transdata_node;
  }
  // replace ref pair since transdata will be inserted in DealRefOutput
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->ReplaceRefPair({node, 0}, {new_replace_node, 0});
  MS_LOG(INFO) << "Transdata node:" << cnode->DebugString() << "split success.";
  return new_replace_node;
}
}  // namespace opt
}  // namespace mindspore
