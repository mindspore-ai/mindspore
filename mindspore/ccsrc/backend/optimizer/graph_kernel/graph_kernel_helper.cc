/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_decoder.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/const_input_to_attr_registry.h"
#include "ir/func_graph_cloner.h"
#include "ir/func_graph.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/action.h"
#include "vm/segment_runner.h"
#include "utils/ms_context.h"
#if ENABLE_D
#include "runtime/device/ascend/kernel_select_ascend.h"
#elif ENABLE_GPU
#include "runtime/device/gpu/kernel_info_setter.h"
#endif

namespace mindspore {
namespace opt {
namespace {
bool IsMakeTupleOut(const AnfNodePtr &out, AnfNodePtrList *real_outs) {
  MS_EXCEPTION_IF_NULL(real_outs);
  if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
    auto &inputs = out->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      real_outs->push_back(inputs[i]);
    }
    return true;
  }

  if (auto fg = AnfAlgo::GetCNodeFuncGraphPtr(out); fg != nullptr) {
    auto fg_out = fg->output();
    if (IsPrimitiveCNode(fg_out, prim::kPrimMakeTuple)) {
      auto inputs = fg_out->cast<CNodePtr>()->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        real_outs->push_back(inputs[i]);
      }
      return true;
    }
  }
  return false;
}

AbstractBasePtr GetOutputAbstract(const AnfNodePtr &node, size_t output_idx) {
  auto out_spec = node->abstract();
  if (out_spec->isa<abstract::AbstractTuple>()) {
    return out_spec->cast<abstract::AbstractTuplePtr>()->elements()[output_idx];
  }
  return out_spec;
}

AnfNodePtrList EliminateMakeTuple(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
  AnfNodePtrList outs;
  auto out_node = fg->output();
  if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
    std::vector<AnfNodePtr> output_args;
    auto out_cnode = out_node->cast<CNodePtr>();
    for (auto out : out_cnode->inputs()) {
      if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
        auto inputs = out->cast<CNodePtr>()->inputs();
        for (size_t i = 1; i < inputs.size(); ++i) {
          output_args.push_back(inputs[i]);
        }
      } else {
        output_args.push_back(out);
      }
    }
    if (output_args.size() != out_cnode->inputs().size()) {
      auto new_out = fg->NewCNode(output_args);
      mng->Replace(out_node, new_out);
    }

    for (size_t i = 1; i < output_args.size(); ++i) {
      outs.push_back(output_args[i]);
    }
    return outs;
  }

  outs.push_back(out_node);
  return outs;
}

bool GenJson(const AnfNodePtrList &op_nodes, const AnfNodePtrList &inputs, const AnfNodePtrList &outputs,
             const DumpOption &dump_option, nlohmann::json *op_desc,
             std::map<std::string, AnfNodePtr> *address_node_map = nullptr) {
  kernel::AkgKernelJsonGenerator akg_kernel_json_generator(dump_option);
  if (!akg_kernel_json_generator.CollectFusedJson(op_nodes, inputs, outputs)) {
    MS_LOG(ERROR) << "Collect json desc failed.";
    return false;
  }

  *op_desc = akg_kernel_json_generator.kernel_json();
  if (address_node_map != nullptr) {
    *address_node_map = akg_kernel_json_generator.address_node_map();
  }
  std::string fused_name;
  std::for_each(op_nodes.begin(), op_nodes.end(), [&fused_name](const AnfNodePtr &node) {
    (void)fused_name.append(AnfAlgo::GetCNodeName(node)).append("_");
  });
  MS_LOG(INFO) << "Collect fusion json: " << fused_name;
  return true;
}

bool TensorElementAllTheSame(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->DataSize() == 1) {
    return true;
  }

  auto data = static_cast<char *>(tensor->data_c());
  auto itemsize = static_cast<size_t>(tensor->data().itemsize());
  auto total_cnt = static_cast<size_t>(tensor->DataSize());
  for (size_t i = 1; i < total_cnt; ++i) {
    for (size_t ei = 0; ei < itemsize; ++ei) {
      if (data[ei] != data[i * itemsize + ei]) {
        return false;
      }
    }
  }
  return true;
}

AnfNodePtr ConvertToScalarTensor(const AnfNodePtr &value_node) {
  auto tensor = GetValueNode<tensor::TensorPtr>(value_node);
  MS_EXCEPTION_IF_NULL(tensor);
  auto type_id = tensor->data_type();
  ShapeVector new_shape;
  auto origin_ndim = static_cast<size_t>(tensor->DataDim());
  for (size_t i = 0; i < origin_ndim; ++i) {
    new_shape.push_back(1);
  }
  tensor::TensorPtr scalar_tensor = std::make_shared<tensor::Tensor>(type_id, new_shape);
  scalar_tensor->set_device_info(tensor->device_info());
  auto data_ptr = scalar_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto itemsize = static_cast<size_t>(tensor->data().itemsize());
  if (memcpy_s(data_ptr, static_cast<size_t>(itemsize), tensor->data_c(), itemsize) != 0) {
    MS_LOG(EXCEPTION) << "Failed to copy data from tensor into scalar.";
  }

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(scalar_tensor);
  new_value_node->set_abstract(scalar_tensor->ToAbstract());
  new_value_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{GetFormat(value_node)});
  kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{type_id});
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());

  return new_value_node;
}

void ReplaceTensorWithScalar(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &scalar_tensors) {
  MS_EXCEPTION_IF_NULL(fg);
  if (scalar_tensors.empty()) {
    return;
  }

  auto sub_mng = fg->manager();
  if (sub_mng == nullptr) {
    sub_mng = Manage(fg, true);
    fg->set_manager(sub_mng);
  }

  std::map<AnfNodePtr, AnfNodePtr> to_be_replaced;
  for (auto scalar_tensor_node : scalar_tensors) {
    auto scalar = ConvertToScalarTensor(scalar_tensor_node);
    auto format = GetFormat(scalar_tensor_node);
    auto dst_shape_vec = GetShape(scalar_tensor_node);
    AnfNodePtrList new_broadcast_inputs = {NewValueNode(prim::kPrimBroadcastTo), scalar};
    auto broadcast_node = CreateCNode(new_broadcast_inputs, fg,
                                      {.format = format, .shape = dst_shape_vec, .type = GetType(scalar_tensor_node)});
    auto device_shape = GetDeviceShape(scalar_tensor_node);
    SetNodeAttrSafely("shape", MakeValue(device_shape), broadcast_node);
    to_be_replaced[scalar_tensor_node] = broadcast_node;
  }

  for (auto [old_value_node, new_node] : to_be_replaced) {
    sub_mng->Replace(old_value_node, new_node);
  }
}
}  // namespace

bool ConvertNonscalarTensorToParameter(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr) {
  MS_EXCEPTION_IF_NULL(inputs_ptr);
  auto nodes = TopoSort(fg->get_return());

  OrderedMap<ValuePtr, AnfNodePtrList> vmap;
  std::vector<AnfNodePtr> scalar_tensors;
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto &inputs = node->cast<CNodePtr>()->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      const auto &tnode = inputs[i];
      auto tensor = GetValueNode<tensor::TensorPtr>(tnode);
      if (tensor == nullptr || tensor->DataSize() == 1) {
        continue;
      }
      if (TensorElementAllTheSame(tensor)) {
        scalar_tensors.emplace_back(tnode);
      } else {
        vmap[GetValueNode(tnode)].push_back(tnode);
      }
    }
  }

  ReplaceTensorWithScalar(fg, scalar_tensors);

  if (vmap.empty()) {
    return false;
  }

  auto mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }

  auto &inputs = *inputs_ptr;
  for (auto iter : vmap) {
    auto value_nodes = iter.second;
    if (value_nodes.empty()) {
      MS_LOG(EXCEPTION) << "Invalid value in map!";
    }

    auto vnode = value_nodes[0];
    auto parameter = fg->add_parameter();
    parameter->set_abstract(vnode->abstract());
    parameter->set_kernel_info(vnode->kernel_info_ptr());
    for (const auto &value_node : value_nodes) {
      mng->Replace(value_node, parameter);
    }

    inputs.push_back(vnode);
  }

  return true;
}

// Transform nodes(including basic and composite node) to a new graph, and collect their inputs and outputs.
std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> MixedNodesTransToGraph(const AnfNodePtrList &fuse_nodes,
                                                                                AnfNodePtrList *src_outputs) {
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  AnfNodePtrList *soutputs = (src_outputs != nullptr) ? src_outputs : &outputs;
  std::tie(fg, inputs, *soutputs) = compile::TransformSegmentToAnfGraph(fuse_nodes);

  FuncGraphManagerPtr mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }

  // Inline origin graphkernel
  auto cnodes = fg->GetOrderedCnodes();
  for (const auto &n : cnodes) {
    if (!AnfAlgo::IsGraphKernel(n)) {
      continue;
    }
    auto graph_kernel_g = GetValueNode<FuncGraphPtr>(n->input(0));
    AnfNodePtrList ins;
    ins.insert(ins.end(), n->inputs().begin() + 1, n->inputs().end());
    auto out = InlineClone(graph_kernel_g, fg, ins, n->input(0)->scope());
    mng->Replace(n, out);
  }

  EliminateMakeTuple(fg, mng);
  ConvertNonscalarTensorToParameter(fg, &inputs);

  outputs.clear();
  kernel::GetFuncGraphOutputNodes(fg, &outputs);
  return std::make_tuple(fg, inputs, outputs);
}

void SetNewKernelInfo(const AnfNodePtr &new_node, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                      const AnfNodePtrList &outputs, kernel::Processor processor) {
  std::vector<std::string> graph_input_format;
  std::vector<TypeId> graph_input_type;
  std::vector<std::string> graph_output_format;
  std::vector<TypeId> graph_output_type;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
    auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    graph_input_format.push_back(input_format);
    auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
    graph_input_type.push_back(input_type);
    auto input_abs = GetOutputAbstract(kernel_with_index.first, kernel_with_index.second);
    fg->parameters()[i]->set_abstract(input_abs);
  }
  auto new_outputs = outputs;
  if (outputs.size() == 1 && AnfAlgo::IsGraphKernel(outputs[0])) {
    std::vector<AnfNodePtr> real_outs;
    if (IsMakeTupleOut(outputs[0], &real_outs)) {
      new_outputs = real_outs;
    }
  }
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    auto kernel_with_index = AnfAlgo::VisitKernel(new_outputs[i], 0);
    auto output_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
    graph_output_format.push_back(output_format);
    graph_output_type.push_back(output_type);
  }
  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(graph_input_format);
  graph_info_builder.SetInputsDeviceType(graph_input_type);
  graph_info_builder.SetOutputsFormat(graph_output_format);
  graph_info_builder.SetOutputsDeviceType(graph_output_type);
  graph_info_builder.SetProcessor(processor);
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto graph_selected_info = graph_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, new_node.get());
}

AnfNodePtr CreateNewFuseCNode(const FuncGraphPtr &func_graph, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                              const AnfNodePtrList &outputs) {
  auto func_node = NewValueNode(fg);
  std::vector<AnfNodePtr> fn_inputs;
  fn_inputs.push_back(func_node);
  fn_inputs.insert(fn_inputs.end(), inputs.begin(), inputs.end());
  auto fuse_cnode = func_graph->NewCNode(fn_inputs);
  // Set output abstract
  if (outputs.size() > 1) {
    std::vector<AbstractBasePtr> out_specs;
    for (size_t i = 0; i < outputs.size(); ++i) {
      out_specs.push_back(outputs[i]->abstract());
    }
    auto out_spec = std::make_shared<abstract::AbstractTuple>(out_specs);
    fuse_cnode->set_abstract(out_spec);
  } else {
    fuse_cnode->set_abstract(outputs[0]->abstract());
  }
  // Set parameter abstract.
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
    auto input_abs = GetOutputAbstract(kernel_with_index.first, kernel_with_index.second);
    fg->parameters()[i]->set_abstract(input_abs);
  }
  return fuse_cnode;
}

void ReplaceNewFuseCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // single out
  if (outputs.size() == 1) {
    mng->Replace(outputs[0], new_fuse_cnode);
    return;
  }

  std::vector<AnfNodePtr> fn_inputs;
  size_t offset = 0;
  for (size_t out_idx = 0; out_idx < outputs.size(); out_idx++) {
    AnfNodePtrList real_outs;
    // not make tuple out, replace
    if (!IsMakeTupleOut(outputs[out_idx], &real_outs)) {
      fn_inputs.clear();
      fn_inputs.push_back(NewValueNode(prim::kPrimTupleGetItem));
      fn_inputs.push_back(new_fuse_cnode);
      fn_inputs.push_back(NewValueNode(MakeValue(SizeToLong(out_idx + offset))));
      auto new_out = func_graph->NewCNode(fn_inputs);
      new_out->set_abstract(outputs[out_idx]->abstract());
      mng->Replace(outputs[out_idx], new_out);
      continue;
    }

    // the out is make tuple , modify the get_item node's value
    auto users = mng->node_users()[outputs[out_idx]];
    for (auto &user : users) {
      auto use_node = user.first;
      if (!use_node->isa<CNode>() || !IsPrimitiveCNode(use_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto get_item_cnode = use_node->cast<CNodePtr>();
      auto value_input = get_item_cnode->input(kInputNodeOutputIndexInTupleGetItem);
      MS_EXCEPTION_IF_NULL(value_input);
      auto value_node = value_input->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto item_idx = GetValue<int64_t>(value_node->value());
      int64_t new_item_idx = SizeToLong(out_idx) + offset + item_idx;
      fn_inputs.clear();
      fn_inputs.push_back(NewValueNode(prim::kPrimTupleGetItem));
      fn_inputs.push_back(new_fuse_cnode);
      fn_inputs.push_back(NewValueNode(new_item_idx));
      auto new_out = func_graph->NewCNode(fn_inputs);
      new_out->set_abstract(get_item_cnode->abstract());
      mng->Replace(get_item_cnode, new_out);
    }

    offset += real_outs.size() - 1;
  }
}

std::tuple<AnfNodePtr, AnfNodePtrList> FuseNodesToSubGraph(const std::vector<AnfNodePtr> &fuse_nodes,
                                                           const FuncGraphPtr &kernel_graph,
                                                           const std::string &postfix) {
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList src_outputs;
  AnfNodePtrList outputs;

  std::tie(fg, inputs, outputs) = MixedNodesTransToGraph(fuse_nodes, &src_outputs);
  auto fuse_new_node = CreateNewFuseCNode(kernel_graph, fg, inputs, outputs);
  SetNewKernelInfo(fuse_new_node, fg, inputs, outputs, AnfAlgo::GetProcessor(fuse_nodes[0]));
  // Handle get-item probleam.
  ReplaceNewFuseCNode(kernel_graph, fuse_new_node, src_outputs);

  // set graphKernel attr
  std::string fuse_op_name = "";
  for (auto &fuse_node : fuse_nodes) {
    if (IsPrimitiveCNode(fuse_node)) {
      fuse_op_name += AnfAlgo::GetCNodePrimitive(fuse_node)->name() + "_";
    } else if (AnfAlgo::IsGraphKernel(fuse_node)) {
      auto fuse_cnode = fuse_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(fuse_cnode);
      auto graph_kernel_fg = GetValueNode<FuncGraphPtr>(fuse_cnode->input(kAnfPrimitiveIndex));
      auto fg_flag_val = graph_kernel_fg->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
      auto fuse_fg_name = GetValue<std::string>(fg_flag_val);
      fuse_op_name += fuse_fg_name + "_";
    }
  }
  fuse_op_name += postfix;
  fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(fuse_op_name));

  return std::make_tuple(fuse_new_node, src_outputs);
}

bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc,
                   std::map<std::string, AnfNodePtr> *address_node_map) {
  MS_EXCEPTION_IF_NULL(op_desc);
  if (nodes.empty()) {
    MS_LOG(ERROR) << "Input nodes is empty.";
    return false;
  }
  bool has_graph_kernel = std::any_of(nodes.begin(), nodes.end(), AnfAlgo::IsGraphKernel);
  bool is_single_graph_kernel = has_graph_kernel && nodes.size() == 1;

  FuncGraphPtr fg;
  AnfNodePtrList op_nodes, inputs, outputs;
  if (is_single_graph_kernel) {
    fg = AnfAlgo::GetCNodeFuncGraphPtr(nodes[0]);
    kernel::GetValidKernelNodes(fg, &op_nodes, &inputs, &outputs);
  } else if (!has_graph_kernel) {
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(nodes);
    op_nodes = nodes;
  } else {
    // When there are basic and composite ops, the composite ops should be inline to the basic ones' graph,
    // so a new graph generation should be done (because they may in the main graph!).
    // If address_node_map is wanted, we should map the new node in new graph to the old nodes. But... not support now.
    MS_LOG(EXCEPTION) << "No support mixed with basic and composite ops now!";
  }

  return GenJson(op_nodes, inputs, outputs, dump_option, op_desc, address_node_map);
}

bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc) {
  MS_EXCEPTION_IF_NULL(op_desc);
  if (nodes.empty()) {
    MS_LOG(ERROR) << "Input nodes is empty.";
    return false;
  }

  FuncGraphPtr fg;
  AnfNodePtrList op_nodes, inputs, outputs;
  if (nodes.size() == 1 && AnfAlgo::IsGraphKernel(nodes[0])) {
    fg = AnfAlgo::GetCNodeFuncGraphPtr(nodes[0]);
  } else {
    std::tie(fg, inputs, outputs) = MixedNodesTransToGraph(nodes);
    inputs.clear();
    outputs.clear();
  }

  kernel::GetValidKernelNodes(fg, &op_nodes, &inputs, &outputs);

  auto mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }

  return GenJson(op_nodes, inputs, outputs, dump_option, op_desc);
}

bool AnfToJsonDesc(const std::vector<AnfNodePtrList> &graphs, const DumpOption &dump_option, nlohmann::json *op_desc) {
  MS_EXCEPTION_IF_NULL(op_desc);
  std::vector<nlohmann::json> graphs_desc;
  for (auto const &graph_nodes : graphs) {
    nlohmann::json desc;
    if (!AnfToJsonDesc(graph_nodes, dump_option, &desc)) {
      MS_LOG(ERROR) << "Collect json desc failed.";
      return false;
    }
    graphs_desc.push_back(desc);
  }
  if (graphs_desc.empty()) {
    MS_LOG(ERROR) << "Collect zero json desc.";
    return false;
  }

  if (graphs_desc.size() > 1) {
    nlohmann::json op_json_desc;
    op_json_desc[kJsonKeyMultiGraph] = true;
    op_json_desc[kJsonKeyGraphDesc] = graphs_desc;
    *op_desc = op_json_desc;
    return true;
  }

  *op_desc = graphs_desc[0];
  return true;
}

FuncGraphPtr JsonDescToAnf(const std::string &json_desc, const std::vector<AnfNodePtr> &inputs) {
  kernel::AkgKernelJsonDecoder akg_kernel_json_decoder;
  auto fg = akg_kernel_json_decoder.DecodeFusedNodes(json_desc);
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Akg decode json to graph failed.";
    return nullptr;
  }
  return fg;
}

std::string ExtractGraphKernelName(const AnfNodePtrList &cnodes, const string &prefix, const string &postfix) {
  std::stringstream name;
  if (prefix != "") {
    name << prefix << "_";
  }
  for (const auto &node : cnodes) {
    if (node->isa<CNode>() && AnfAlgo::IsRealKernel(node)) {
      name << AnfAlgo::GetCNodeName(node) << "_";
    }
  }
  if (postfix != "") {
    name << postfix;
  }
  return name.str();
}

std::vector<PrimitivePtr> GetFusibleOpList() {
#if ENABLE_D
  std::vector<PrimitivePtr> fusible_basic_ops = {
    prim::kPrimAbs,        prim::kPrimRound,  prim::kPrimNeg,      prim::kPrimExp,       prim::kPrimAdd,
    prim::kPrimExpandDims, prim::kPrimMul,    prim::kPrimMinimum,  prim::kPrimMaximum,   prim::kPrimLog,
    prim::kPrimPow,        prim::kPrimSub,    prim::kPrimRsqrt,    prim::kPrimSqrt,      prim::kPrimEqual,
    prim::kPrimReciprocal, prim::kPrimTanh,   prim::kPrimReshape,  prim::kPrimTranspose, prim::kPrimCast,
    prim::kPrimRealDiv,    prim::kPrimAssign, prim::kPrimReduceSum};
#elif ENABLE_GPU
  std::vector<PrimitivePtr> fusible_basic_ops = {
    prim::kPrimAbs,     prim::kPrimRound,      prim::kPrimNeg,       prim::kPrimExp,     prim::kPrimAdd,
    prim::kPrimRealDiv, prim::kPrimMul,        prim::kPrimMinimum,   prim::kPrimMaximum, prim::kPrimLog,
    prim::kPrimPow,     prim::kPrimSub,        prim::kPrimRsqrt,     prim::kPrimSqrt,    prim::kPrimAddN,
    prim::kPrimEqual,   prim::kPrimReciprocal, prim::KPrimTransData, prim::kPrimSelect,  prim::kPrimGreater,
    prim::kPrimCast,    prim::kPrimReduceSum,  prim::kPrimTanh,      prim::kPrimReshape, prim::kPrimTranspose,
    prim::kPrimAssign,  prim::kPrimExpandDims};
#else
  std::vector<PrimitivePtr> fusible_basic_ops;
#endif
  return fusible_basic_ops;
}

bool CheckProcessor(const AnfNodePtr &node, kernel::Processor processor = kernel::Processor::AICORE) {
  MS_EXCEPTION_IF_NULL(node);

  auto node_kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  if (node_kernel_info == nullptr) {
    return false;
  }

  auto node_build_info = node_kernel_info->GetMutableSelectKernelBuildInfo();
  if (node_build_info == nullptr) {
    return false;
  }

  return node_build_info->processor() == processor;
}

bool IsBasicFuseOp(const AnfNodePtr &node) {
  std::vector<PrimitivePtr> basic_ops = GetFusibleOpList();
#if ENABLE_D
  if (!CheckProcessor(node)) {
    return false;
  }
#endif
  return std::any_of(basic_ops.begin(), basic_ops.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

void ResetKernelInfo(const AnfNodePtr &node, KernelType kernel_type) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
#if ENABLE_D
  device::ascend::SetKernelInfo(cnode, kernel_type);
#elif ENABLE_GPU
  device::gpu::SetKernelInfo(cnode, kernel_type);
#endif
}

void InitDependPrior(const std::vector<AnfNodePtr> &todos,
                     std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> *depend_prior) {
  for (auto iter = todos.cbegin(); iter != todos.cend(); ++iter) {
    auto cnode = (*iter)->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (!AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimControlDepend)) {
      continue;
    }

    auto prior_node = cnode->input(kControlDependPriorIndex);
    auto depend_node = cnode->input(kControlDependBehindIndex);
    MS_EXCEPTION_IF_NULL(prior_node);
    MS_EXCEPTION_IF_NULL(depend_node);
    std::vector<AnfNodePtr> prior_nodes = {prior_node};
    std::vector<AnfNodePtr> depend_nodes = {depend_node};

    int depend_mode = 0;
    if (AnfAlgo::HasNodeAttr(kControlDependMode, cnode)) {
      depend_mode = AnfAlgo::GetNodeAttr<int64_t>(cnode, kControlDependMode);
    }

    auto GetOutputNodes = [cnode](const AnfNodePtr &param) -> std::vector<AnfNodePtr> {
      std::vector<AnfNodePtr> out_nodes;
      auto user_set = param->func_graph()->manager()->node_users()[param];
      for (auto iter = user_set.cbegin(); iter != user_set.cend(); ++iter) {
        if (iter->first != cnode) {
          out_nodes.push_back(iter->first);
        }
      }
      return out_nodes;
    };

    if (prior_node->isa<Parameter>() && depend_mode == 1) {
      prior_nodes = GetOutputNodes(prior_node);
    }
    if (depend_node->isa<Parameter>()) {
      depend_nodes = depend_mode == 1 ? GetOutputNodes(depend_node) : std::vector<AnfNodePtr>{};
    }

    std::vector<AnfNodePtr> real_prior_nodes;
    std::set<AnfNodePtr> prior_visited;
    for (const auto &tmp : prior_nodes) {
      AnfAlgo::GetAllFatherRealNode(tmp, &real_prior_nodes, &prior_visited);
    }
    prior_visited.clear();
    std::vector<AnfNodePtr> real_depend_nodes;
    std::set<AnfNodePtr> depend_visited;
    for (const auto &tmp : depend_nodes) {
      AnfAlgo::GetAllFatherRealNode(tmp, &real_depend_nodes, &depend_visited);
    }
    depend_visited.clear();

    for (auto &prior : real_prior_nodes) {
      if (AnfAlgo::CheckPrimitiveType(prior, prim::kPrimControlDepend)) {
        continue;
      }
      for (auto &depend : real_depend_nodes) {
        if (AnfAlgo::CheckPrimitiveType(depend, prim::kPrimControlDepend)) {
          continue;
        }
        depend_prior->insert({depend, std::make_pair(prior, cnode)});
      }
    }
    real_prior_nodes.clear();
    real_depend_nodes.clear();
  }
}

void ReplaceNewFuseCNodeForDependPrior(std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> *depend_prior,
                                       const AnfNodePtr &new_fuse_cnode, const AnfNodePtrList &outputs) {
  std::multimap<AnfNodePtr, std::pair<AnfNodePtr, AnfNodePtr>> new_fuse_cnode_dep_pri;

  for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
    if (IsPrimitiveCNode(outputs[out_idx], prim::kPrimMakeTuple)) {
      MS_LOG(ERROR) << "Need real outputs of makeTuple";
    }
    if (IsPrimitiveCNode(outputs[out_idx], prim::kPrimTupleGetItem)) {
      continue;
    }
    for (auto iter = (*depend_prior).begin(); iter != (*depend_prior).end();) {
      if (iter->first == outputs[out_idx]) {
        new_fuse_cnode_dep_pri.insert({new_fuse_cnode, iter->second});
        iter = depend_prior->erase(iter);
        continue;
      }
      if (iter->second.first == outputs[out_idx]) {
        new_fuse_cnode_dep_pri.insert({iter->first, std::make_pair(new_fuse_cnode, iter->second.second)});
        iter = depend_prior->erase(iter);
        continue;
      }
      ++iter;
    }
  }

  for (auto item : new_fuse_cnode_dep_pri) {
    depend_prior->insert(item);
  }
}

std::string GetFormat(const AnfNodePtr &node) {
  auto kernel_info = static_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_build_info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_info);
  return kernel_build_info->GetOutputFormat(0);
}

TypePtr GetType(const AnfNodePtr &node) {
  const auto &abstract = node->abstract();
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  return type;
}

ShapeVector GetShape(const AnfNodePtr &node) {
  auto abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape = abstract->GetShapeTrack();
  if (shape == nullptr || !shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "Cannot get shape from " << node->fullname_with_scope();
  }
  auto shape_vec = shape->cast<abstract::ShapePtr>()->shape();
  if (shape_vec.empty()) {
    shape_vec.push_back(1);
  }
  return shape_vec;
}

ShapeVector GetDeviceShape(const AnfNodePtr &node) {
  ShapeVector res_device_shape;
  auto device_shape = AnfAlgo::GetOutputDeviceShape(node, 0);
  if (device_shape.empty()) {
    res_device_shape.push_back(1);
  } else {
    std::transform(device_shape.begin(), device_shape.end(), std::back_inserter(res_device_shape), SizeToLong);
  }
  return res_device_shape;
}

std::vector<int64_t> GetReduceAxis(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  const auto &attrs = prim->attrs();
  auto iter = attrs.find("axis");
  if (iter == attrs.end()) {
    MS_LOG(EXCEPTION) << "Origin node have no attributes!";
  }

  std::vector<int64_t> axis;

  auto &v = iter->second;
  if (v->isa<ValueList>() || v->isa<ValueTuple>()) {
    auto vec = v->isa<ValueList>() ? v->cast<ValueListPtr>()->value() : v->cast<ValueTuplePtr>()->value();
    for (auto value : vec) {
      if (value->isa<Int64Imm>()) {
        axis.push_back(GetValue<int64_t>(value));
      } else {
        MS_LOG(EXCEPTION) << "Reduce axis type should be int64!";
      }
    }
  } else if (v->isa<Int64Imm>()) {
    axis.push_back(GetValue<int64_t>(v));
  } else {
    MS_LOG(EXCEPTION) << "Reduce axis should be a list or tuple!";
  }

  return axis;
}

kernel::Processor GetProcessorFromContext() {
  kernel::Processor processor = kernel::Processor::UNKNOWN;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_info = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_info == kGPUDevice) {
    processor = kernel::Processor::CUDA;
  } else if (device_info == kAscendDevice) {
    processor = kernel::Processor::AICORE;
  }
  return processor;
}

CNodePtr CreateCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph, const DataInfo &out_info) {
  // Limitation: 1. Node's attributes should be set out of this function; 2. only one output.
  MS_EXCEPTION_IF_NULL(out_info.type);
  auto out_type = out_info.type;
  if (auto otype = out_info.type->cast<TensorTypePtr>(); otype != nullptr) {
    out_type = otype->element();
  }

  // Create CNode.
  auto cnode = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);

  // Setup abstract.
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(out_type, out_info.shape);
  cnode->set_abstract(abs_tensor);

  // Setup kernel info.
  auto kernel_info = std::make_shared<device::KernelInfo>();
  cnode->set_kernel_info(kernel_info);
  std::vector<size_t> feature_map_input_indexs;
  kernel_info->set_feature_map_flag(false);
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (AnfAlgo::IsFeatureMapOutput(inputs[i])) {
      kernel_info->set_feature_map_flag(true);
      feature_map_input_indexs.push_back(i);
    }
  }
  if (inputs.size() == 1) {
    kernel_info->set_feature_map_flag(true);
  }
  if (AnfAlgo::IsRealKernel(cnode)) {
    // if the node only has the primitive(such as getNext) or the node's input has a feature map input
    // then the node's output is a feature map output
    SetNodeAttrSafely(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), cnode);
    SetNodeAttrSafely(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), cnode);
  }

  // Setup kernel build info.
  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto kernel_with_index = AnfAlgo::VisitKernel(inputs[i], 0);
    auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    input_formats.push_back(input_format);
    auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
    input_types.push_back(input_type);
  }

  std::vector<std::string> output_formats = {out_info.format};
  std::vector<TypeId> output_types = {out_type->type_id()};

  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat(input_formats);
  info_builder.SetInputsDeviceType(input_types);
  info_builder.SetOutputsFormat(output_formats);
  info_builder.SetOutputsDeviceType(output_types);
  info_builder.SetProcessor(GetProcessorFromContext());
  info_builder.SetKernelType(KernelType::AKG_KERNEL);
  info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto selected_info = info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(selected_info, cnode.get());

  func_graph->AddNode(cnode);
  return cnode;
}

void SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  // Make CNode safe to set attr firstly.
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  AnfNodePtrList new_inputs = {NewValueNode(AnfAlgo::GetCNodePrimitive(cnode)->Clone())};
  auto inputs = cnode->inputs();
  new_inputs.insert(new_inputs.end(), inputs.begin() + 1, inputs.end());
  cnode->set_inputs(new_inputs);

  // Set attr secondly.
  AnfAlgo::SetNodeAttr(key, value, node);
}

bool IsKeepBasicNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  static std::vector<std::string> contagious_attrs = {"inplace_group", "inplace_algo", "inplace_output_index",
                                                      "aggregate", "aggregate_input_indexx"};
  static std::vector<std::function<bool(const AnfNodePtr &node)>> attrs_with_value = {
    [](const AnfNodePtr &n) -> bool { return AnfAlgo::GetBooleanAttr(n, "skip"); }};
  // If node contain attribute in contagious_attrs, it have to keep basic no matter what the value is.
  // If node contain attribute in attrs_with_value, it only have to keep basic when the check result is true.
  if (std::any_of(contagious_attrs.cbegin(), contagious_attrs.cend(),
                  [&cnode](const std::string &attr_name) -> bool { return AnfAlgo::HasNodeAttr(attr_name, cnode); }) ||
      std::any_of(attrs_with_value.cbegin(), attrs_with_value.cend(),
                  [&cnode](std::function<bool(const AnfNodePtr &node)> func) -> bool { return func(cnode); })) {
    return true;
  }

  return false;
}
}  // namespace opt
}  // namespace mindspore
