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

#include "pre_activate/ascend/ascend_helper.h"
#include <set>
#include "common/trans.h"
#include "common/utils.h"
#include "device/kernel_info.h"
#include "kernel/oplib/oplib.h"
#include "operator/ops.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"
#include "utils/context/ms_context.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
namespace {
AnfNodePtr CreatTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx) {
  auto idx = NewValueNode(SizeToInt(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int32Imm>(SizeToInt(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  AnfNodePtr tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(node, output_idx);
  TypeId origin_type = AnfAlgo::GetOutputInferDataType(node, output_idx);
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, tuple_getitem.get());
  return tuple_getitem;
}

kernel::KernelBuildInfoPtr CreateKernelBuildInfo(const std::string &input_format, const std::string &output_format,
                                                 const AnfNodePtr &node, const kernel::KernelBuildInfo ori_build_info) {
  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsDeviceType({ori_build_info.GetInputDeviceType(0)});
  builder.SetOutputsDeviceType({ori_build_info.GetOutputDeviceType(0)});
  builder.SetKernelType(ori_build_info.kernel_type());
  builder.SetFusionType(ori_build_info.fusion_type());
  builder.SetProcessor(ori_build_info.processor());
  return builder.Build();
}

CNodePtr NewTransOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const KernelSelectPtr &kernel_select,
                        const bool need_padding, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  std::vector<AnfNodePtr> trans_inputs;
  auto prim = std::make_shared<Primitive>(op_name);
  trans_inputs.push_back(NewValueNode(prim));
  trans_inputs.push_back(input);
  CNodePtr trans_node = func_graph->NewCNode(trans_inputs);
  MS_EXCEPTION_IF_NULL(trans_node);
  if (need_padding) {
    AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input, 0)},
                                        {trans::TransShapeTo4d(AnfAlgo::GetOutputInferShape(input, 0))},
                                        trans_node.get());
  } else {
    AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input, 0)},
                                        {AnfAlgo::GetOutputInferShape(input, 0)}, trans_node.get());
  }
  // special handle for ut
  if (trans_node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    trans_node->set_kernel_info(kernel_info);
  }
  MS_EXCEPTION_IF_NULL(kernel_select);
  kernel_select->SelectKernel(trans_node);
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), trans_node);
  MS_EXCEPTION_IF_NULL(trans_node);
  trans_node->set_scope(input->scope());
  return trans_node;
}

AnfNodePtr CreateReshapeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                             const KernelSelectPtr &kernel_select, const std::vector<size_t> &dst_shape) {
  std::vector<AnfNodePtr> trans_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimReshape->name());
  trans_inputs.emplace_back(NewValueNode(prim));
  trans_inputs.emplace_back(input_node);
  auto reshape = func_graph->NewCNode(trans_inputs);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input_node, 0)}, {dst_shape}, reshape.get());
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), reshape);
  AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(dst_shape), reshape);
  reshape->set_scope(input_node->scope());
  kernel_select->SelectKernel(reshape);
  return reshape;
}

AnfNodePtr GetTransInputNodePtr(const FuncGraphPtr &func_graph, const CNodePtr &node, size_t index,
                                const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  bool padding_flag = false;
  auto input_node = AnfAlgo::GetInputNode(node, index);
  if (!AnfAlgo::IsFeatureMapInput(node, index)) {
    input_node = InsertTransOpForOutput(func_graph, input_node, kernel_select);
    MS_EXCEPTION_IF_NULL(input_node);
    AnfAlgo::SetNodeInput(node, input_node, index);
  }
  if (AnfAlgo::GetInputFormat(node, index) == kOpFormat_NC1KHKWHWC0) {
    MS_LOG(EXCEPTION) << "got the format " << AnfAlgo::GetInputFormat(node, index)
                      << "when inserting the transdata node " << node->DebugString();
  }
  std::vector<size_t> origin_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, index);
  std::string origin_format = kOpFormat_DEFAULT;
  std::string dest_format = AnfAlgo::GetInputFormat(node, index);
  if (dest_format == kOpFormat_C1HWNCoC0) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, node, kernel_select, index, padding_flag,
                                                     origin_format, dest_format, kTransDataOpName, true);
    MS_EXCEPTION_IF_NULL(replace_input);
    return replace_input;
  }
  if (dest_format == kOpFormat_NC1HWC0 && origin_shape.size() > 1) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, node, kernel_select, index, padding_flag,
                                                     origin_format, dest_format, kTransDataOpName, true);
    MS_EXCEPTION_IF_NULL(replace_input);
    MS_LOG(DEBUG) << "Inserted Translate45, index: " << index;
    return replace_input;
  } else if (dest_format == kOpFormat_FRAC_NZ) {
    AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, node, kernel_select, index, padding_flag,
                                                     origin_format, dest_format, kTransDataOpName, true);
    MS_EXCEPTION_IF_NULL(replace_input);
    MS_LOG(DEBUG) << "inserted translate " << AnfAlgo::GetInputFormat(node, index) << " To default, index: " << index;
    return replace_input;
  } else if (dest_format == kOpFormat_FRAC_Z && !origin_shape.empty()) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, node, kernel_select, index, padding_flag,
                                                     origin_format, dest_format, kTransDataOpName, true);
    MS_EXCEPTION_IF_NULL(replace_input);
    MS_LOG(DEBUG) << "Inserted Translate45, index: " << index;
    return replace_input;
  }
  return input_node;
}

AnfNodePtr InsertTransOpForSingleOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  bool padding_flag = false;
  std::string output_format;
  std::vector<size_t> origin_shape;
  if (!AnfAlgo::IsRealKernel(node)) {
    output_format = AnfAlgo::GetPrevNodeOutputFormat(node, 0);
    origin_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  } else {
    output_format = AnfAlgo::GetOutputFormat(node, 0);
    origin_shape = AnfAlgo::GetOutputInferShape(node, 0);
  }
  if (output_format == kOpFormat_NC1KHKWHWC0) {
    MS_LOG(EXCEPTION) << "got the hw format " << output_format << "when insert the transdata node "
                      << node->DebugString();
  }
  std::string origin_format = output_format;
  std::string dest_format = kOpFormat_DEFAULT;
  if (output_format == kOpFormat_C1HWNCoC0) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, node, kernel_select, 0, padding_flag, origin_format,
                                                     dest_format, kTransDataOpName, false);
    MS_EXCEPTION_IF_NULL(replace_input);
    return replace_input;
  }
  if (output_format == kOpFormat_NC1HWC0 && origin_shape.size() > 1) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, node, kernel_select, 0, padding_flag, origin_format,
                                                      dest_format, kTransDataOpName, false);
    MS_EXCEPTION_IF_NULL(replace_output);
    MS_LOG(DEBUG) << "Inserted Trans54";
    return replace_output;
  } else if (output_format == kOpFormat_FRAC_NZ) {
    AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, node, kernel_select, 0, padding_flag, origin_format,
                                                      dest_format, kTransDataOpName, false);
    MS_EXCEPTION_IF_NULL(replace_output);
    MS_LOG(DEBUG) << "Inserted Translate " << output_format << " To default, index: 0";
    return replace_output;
  } else if (output_format == kOpFormat_FRAC_Z && !origin_shape.empty()) {
    padding_flag = (origin_shape.size() != kShape4dDims);
    AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, node, kernel_select, 0, padding_flag, origin_format,
                                                      dest_format, kTransDataOpName, false);
    MS_EXCEPTION_IF_NULL(replace_output);
    MS_LOG(DEBUG) << "Inserted Trans54";
    return replace_output;
  }
  return node;
}

void GetTransDataInputFormat(const AnfNodePtr &node, std::string *input_format) {
  MS_EXCEPTION_IF_NULL(input_format);
  if (AnfAlgo::IsRealKernel(node)) {
    *input_format = AnfAlgo::GetOutputFormat(node, 0);
  } else {
    *input_format = AnfAlgo::GetPrevNodeOutputFormat(node, 0);
  }
}

AnfNodePtr InsertTransOpForMultipleOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t output_idx = 0; output_idx < AnfAlgo::GetOutputTensorNum(node); ++output_idx) {
    bool padding_flag = false;

    std::string output_format;
    GetTransDataInputFormat(node, &output_format);
    if (output_format == kOpFormat_NC1KHKWHWC0) {
      MS_LOG(EXCEPTION) << "got the hw format" << output_format << " when insert the transdata node "
                        << node->DebugString();
    }
    auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, output_idx);
    std::vector<size_t> origin_shape = AnfAlgo::GetOutputInferShape(node, output_idx);
    std::string origin_format = output_format;
    std::string dest_format = kOpFormat_DEFAULT;
    if (output_format == kOpFormat_C1HWNCoC0) {
      padding_flag = (origin_shape.size() != kShape4dDims);
      AnfNodePtr replace_input = AddTransOpNodeToGraph(func_graph, tuple_getitem, kernel_select, 0, padding_flag,
                                                       origin_format, dest_format, kTransDataOpName, false);
      MS_EXCEPTION_IF_NULL(replace_input);
      return replace_input;
    }
    if (output_format == kOpFormat_NC1HWC0 && origin_shape.size() > 1) {
      padding_flag = (origin_shape.size() != kShape4dDims);
      // Insert a 5to4 trans op.
      AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, tuple_getitem, kernel_select, 0, padding_flag,
                                                        origin_format, dest_format, kTransDataOpName, false);
      MS_EXCEPTION_IF_NULL(replace_output);
      MS_LOG(DEBUG) << "Inserted Translate54";
      make_tuple_inputs.push_back(replace_output);
    } else if (output_format == kOpFormat_FRAC_NZ) {
      AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, tuple_getitem, kernel_select, 0, padding_flag,
                                                        origin_format, dest_format, kTransDataOpName, false);
      MS_EXCEPTION_IF_NULL(replace_output);
      MS_LOG(DEBUG) << "Inserted Translate " << output_format << " To default, index: " << output_idx;
      make_tuple_inputs.push_back(replace_output);
    } else if (output_format == kOpFormat_FRAC_Z && !origin_shape.empty()) {
      padding_flag = (origin_shape.size() != kShape4dDims);
      AnfNodePtr replace_output = AddTransOpNodeToGraph(func_graph, tuple_getitem, kernel_select, 0, padding_flag,
                                                        origin_format, dest_format, kTransDataOpName, false);
      MS_EXCEPTION_IF_NULL(replace_output);
      MS_LOG(DEBUG) << "Inserted Translate54";
      make_tuple_inputs.push_back(replace_output);
    } else {
      // No need insert trans op.
      make_tuple_inputs.push_back(tuple_getitem);
    }
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace
AnfNodePtr AddTransOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select, size_t insert_index, const bool padding_flag,
                                 const std::string &origin_format, const std::string &dest_format,
                                 const std::string &op_name, bool is_insert_input) {
  AnfNodePtr trans_node = nullptr;
  AnfNodePtr input_node = nullptr;
  AnfNodePtr trans_data = nullptr;
  MS_EXCEPTION_IF_NULL(node);
  if (origin_format.empty() || dest_format.empty()) {
    MS_LOG(EXCEPTION) << "trans op format is error, origin = " << origin_format << ", dest " << origin_format;
  }
  if (is_insert_input) {
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "cannot insert a transdata node to a node's input which the node is not a cnode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    input_node = AnfAlgo::GetInputNode(cnode, insert_index);
    if (padding_flag) {
      auto padd_shape = trans::TransShapeTo4d(AnfAlgo::GetOutputInferShape(input_node, 0));
      auto reshape_node = CreateReshapeNode(func_graph, input_node, kernel_select, padd_shape);
      trans_data = NewTransOpNode(func_graph, reshape_node, kernel_select, padding_flag, op_name);
    } else {
      trans_data = NewTransOpNode(func_graph, input_node, kernel_select, padding_flag, op_name);
    }
    trans_node = trans_data;
  } else {
    input_node = node;
    trans_data = NewTransOpNode(func_graph, input_node, kernel_select, padding_flag, op_name);
    if (padding_flag) {
      auto reshape_node =
        CreateReshapeNode(func_graph, trans_data, kernel_select, AnfAlgo::GetOutputInferShape(input_node, 0));
      trans_node = reshape_node;
    } else {
      trans_node = trans_data;
    }
  }
  MS_EXCEPTION_IF_NULL(trans_data);
  MS_EXCEPTION_IF_NULL(trans_data->kernel_info());
  auto trans_ori_build_info = trans_data->kernel_info()->select_kernel_build_info();
  auto kernel_build_info = CreateKernelBuildInfo(origin_format, dest_format, input_node, *trans_ori_build_info);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, trans_data.get());
  return trans_node;
}

AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const std::vector<size_t> &origin_shape, const TypeId &origin_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::string input_format = format;
  std::string output_format = format;
  std::vector<AnfNodePtr> new_cast_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  new_cast_inputs.push_back(NewValueNode(prim));
  new_cast_inputs.push_back(input);
  CNodePtr cast = func_graph->NewCNode(new_cast_inputs);
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  builder.SetFusionType(kernel::FusionType::OPAQUE);
  builder.SetProcessor(kernel::Processor::AICORE);
  if (kernel::OpLib::FindOp(prim::kPrimCast->name(), kernel::kTBE) != nullptr) {
    builder.SetKernelType(KernelType::TBE_KERNEL);
  } else {
    builder.SetKernelType(KernelType::AUTO_DIFF_KERNEL);
  }
  // if kernel info is null , it remarks this function is running ut
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, cast.get());
  return cast;
}

AnfNodePtr InsertTransOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                  const KernelSelectPtr &kernel_select) {
  size_t outputs_num = AnfAlgo::GetOutputTensorNum(node);
  if (outputs_num == 0) {
    return node;
  }
  // Single output
  if (outputs_num == 1 && (!AnfAlgo::IsTupleOutput(node))) {
    return InsertTransOpForSingleOutput(func_graph, node, kernel_select);
  }
  // Multiple output
  return InsertTransOpForMultipleOutput(func_graph, node, kernel_select);
}

AnfNodePtr InsertTransOpForInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(cnode); ++input_index) {
    AnfNodePtr input_node = GetTransInputNodePtr(func_graph, cnode, input_index, kernel_select);
    MS_EXCEPTION_IF_NULL(input_node);
    new_inputs.push_back(input_node);
  }
  CNodePtr new_cnode = nullptr;
  // cnode changed so make a new cnode to differ from original one.
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph == nullptr) {
    new_cnode = std::make_shared<CNode>(*cnode);
  } else {
    new_cnode = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_inputs(new_inputs);
  return new_cnode;
}

CNodePtr InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t input_index = 0; input_index < AnfAlgo::GetInputTensorNum(cnode); ++input_index) {
    TypeId origin_type;
    auto cur_input = AnfAlgo::GetInputNode(cnode, input_index);
    if (!AnfAlgo::IsFeatureMapInput(cnode, input_index)) {
      // weight
      origin_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(cnode, input_index);
    } else {
      // feature map
      origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index);
    }
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const std::vector<size_t> origin_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, input_index);
    const TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index);
    if (origin_type != device_type) {
      auto cast =
        AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, origin_type, device_type, origin_shape, origin_type);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), cast);
      new_inputs.push_back(cast);
    } else {
      new_inputs.push_back(cur_input);
    }
  }
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
