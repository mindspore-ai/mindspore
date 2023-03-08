/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/graph_kernel_helper.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include "utils/hash_set.h"
#include "kernel/common_utils.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "kernel/akg/akg_kernel_json_decoder.h"
#include "kernel/kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/adapter/fake_abstract_shape.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "ir/func_graph_cloner.h"
#include "ir/func_graph.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/action.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"

namespace mindspore::graphkernel {
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

  if (auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(out); fg != nullptr) {
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

bool GenJson(const AnfNodePtrList &op_nodes, const std::pair<AnfNodePtrList, AnfNodePtrList> &in_and_out,
             const DumpOption &dump_option, nlohmann::json *op_desc,
             std::map<std::string, AnfNodePtr> *address_node_map = nullptr) {
  AkgKernelJsonGenerator akg_kernel_json_generator(dump_option);
  if (!akg_kernel_json_generator.CollectFusedJson(op_nodes, in_and_out.first, in_and_out.second)) {
    MS_LOG(ERROR) << "Collect json desc failed.";
    return false;
  }

  *op_desc = akg_kernel_json_generator.kernel_json();
  if (address_node_map != nullptr) {
    *address_node_map = akg_kernel_json_generator.address_node_map();
  }
  std::string fused_name;
  (void)std::for_each(op_nodes.begin(), op_nodes.end(), [&fused_name](const AnfNodePtr &node) {
    (void)fused_name.append(common::AnfAlgo::GetCNodeName(node)).append("_");
  });
  MS_LOG(DEBUG) << "Collect fusion json: " << fused_name;
  return true;
}
}  // namespace

AbstractBasePtr GetOutputAbstract(const AnfNodePtr &node, size_t output_idx) {
  auto out_spec = node->abstract();
  if (out_spec->isa<abstract::AbstractTuple>()) {
    return out_spec->cast<abstract::AbstractTuplePtr>()->elements()[output_idx];
  }
  return out_spec;
}

// Build for new node, processor comes from context
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types) {
  return BuildSelectKernelBuildInfo(inputs_format, inputs_type, output_formats, output_types,
                                    kernel::GetProcessorFromContext());
}

// Build for new node with given processor
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types,
                                                      const kernel::Processor &processor) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(inputs_format);
  graph_info_builder.SetInputsDeviceType(inputs_type);
  graph_info_builder.SetOutputsFormat(output_formats);
  graph_info_builder.SetOutputsDeviceType(output_types);
  graph_info_builder.SetProcessor(processor);
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::kPatternOpaque);
  return graph_info_builder.Build();
}

// Deprecated. use Callback->SetGraphKernelNodeKernelInfo.
void SetNewKernelInfo(const AnfNodePtr &new_node, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                      const AnfNodePtrList &outputs) {
  std::vector<std::string> graph_input_format;
  std::vector<TypeId> graph_input_type;
  std::vector<std::string> graph_output_format;
  std::vector<TypeId> graph_output_type;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto kernel_with_index = common::AnfAlgo::VisitKernel(inputs[i], 0);
    if (kernel_with_index.first->isa<ValueNode>()) {
      auto tensor = GetValueNode<tensor::TensorPtr>(kernel_with_index.first);
      MS_EXCEPTION_IF_NULL(tensor);
      (void)graph_input_type.emplace_back(tensor->data_type());
      (void)graph_input_format.emplace_back(kOpFormat_DEFAULT);
    } else {
      auto input_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
      (void)graph_input_type.emplace_back(input_type);
      auto input_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
      (void)graph_input_format.emplace_back(std::move(input_format));
    }
    auto input_abs = GetOutputAbstract(kernel_with_index.first, kernel_with_index.second);
    fg->parameters()[i]->set_abstract(input_abs);
    fg->parameters()[i]->set_kernel_info(std::make_shared<device::KernelInfo>());
    kernel::KernelBuildInfo::KernelBuildInfoBuilder para_info_builder;
    para_info_builder.SetOutputsFormat({graph_input_format.back()});
    para_info_builder.SetOutputsDeviceType({graph_input_type.back()});
    para_info_builder.SetKernelType(KernelType::AKG_KERNEL);
    para_info_builder.SetProcessor(kernel::GetProcessorFromContext());
    AnfAlgo::SetSelectKernelBuildInfo(para_info_builder.Build(), fg->parameters()[i].get());
  }
  auto new_outputs = outputs;
  if (outputs.size() == 1 && common::AnfAlgo::IsGraphKernel(outputs[0])) {
    std::vector<AnfNodePtr> real_outs;
    if (IsMakeTupleOut(outputs[0], &real_outs)) {
      new_outputs = real_outs;
    }
  }
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    auto kernel_with_index = common::AnfAlgo::VisitKernel(new_outputs[i], 0);
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
  graph_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::kPatternOpaque);
  auto graph_selected_info = graph_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(graph_selected_info, new_node.get());
}

bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc,
                   std::map<std::string, AnfNodePtr> *address_node_map) {
  MS_EXCEPTION_IF_NULL(op_desc);
  if (nodes.empty()) {
    MS_LOG(ERROR) << "Input nodes is empty.";
    return false;
  }
  bool has_graph_kernel = std::any_of(nodes.begin(), nodes.end(), common::AnfAlgo::IsGraphKernel);
  bool is_single_graph_kernel = has_graph_kernel && nodes.size() == 1;

  FuncGraphPtr fg;
  AnfNodePtrList op_nodes, inputs, outputs;
  if (is_single_graph_kernel) {
    fg = common::AnfAlgo::GetCNodeFuncGraphPtr(nodes[0]);
    kernel::GetValidKernelNodes(fg, &op_nodes, &inputs, &outputs);
  } else if (!has_graph_kernel) {
    std::tie(fg, inputs, outputs) = BuildGraphFromNodes(nodes);
    op_nodes = nodes;
  } else {
    // When there are basic and composite ops, the composite ops should be inline to the basic ones' graph,
    // so a new graph generation should be done (because they may in the main graph!).
    // If address_node_map is wanted, we should map the new node in new graph to the old nodes. But... not support now.
    MS_LOG(EXCEPTION) << "No support mixed with basic and composite ops now!";
  }
  std::pair<AnfNodePtrList, AnfNodePtrList> in_and_out = std::make_pair(inputs, outputs);
  return GenJson(op_nodes, in_and_out, dump_option, op_desc, address_node_map);
}

bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc) {
  MS_EXCEPTION_IF_NULL(op_desc);
  if (nodes.empty()) {
    MS_LOG(ERROR) << "Input nodes is empty.";
    return false;
  }

  FuncGraphPtr fg;

  if (nodes.size() == 1 && common::AnfAlgo::IsGraphKernel(nodes[0])) {
    fg = common::AnfAlgo::GetCNodeFuncGraphPtr(nodes[0]);
  } else {
    std::tie(fg, std::ignore, std::ignore) = BuildSingleGraphFromNodes(nodes);
  }

  AnfNodePtrList op_nodes, inputs, outputs;
  kernel::GetValidKernelNodes(fg, &op_nodes, &inputs, &outputs);

  auto mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }
  std::pair<AnfNodePtrList, AnfNodePtrList> in_and_out = std::make_pair(inputs, outputs);
  return GenJson(op_nodes, in_and_out, dump_option, op_desc);
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

FuncGraphPtr JsonDescToAnf(const std::string &json_desc) {
  kernel::AkgKernelJsonDecoder akg_kernel_json_decoder;
  auto fg = akg_kernel_json_decoder.DecodeFusedNodes(json_desc);
  if (fg == nullptr) {
    MS_LOG(ERROR) << "Akg decode json to graph failed. json is: " << json_desc;
    return nullptr;
  }
  return fg;
}

std::string GetFormat(const AnfNodePtr &node) { return AnfAlgo::GetOutputFormat(node, 0); }

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
  if (shape == nullptr) {
    MS_LOG(EXCEPTION) << "The shape of node " << node->fullname_with_scope() << " is nullptr";
  } else if (!shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "The shape of node " << node->fullname_with_scope() << " should be of type Shape, but got "
                      << shape->ToString();
  }
  auto shape_vec = shape->cast<abstract::ShapePtr>()->shape();
  if (shape_vec.empty()) {
    shape_vec.push_back(1);
  }
  return shape_vec;
}

ShapeVector GetDeviceShape(const AnfNodePtr &node) {
  ShapeVector res_device_shape = AnfAlgo::GetOutputDeviceShape(node, 0);
  return res_device_shape.empty() ? ShapeVector({1}) : res_device_shape;
}

std::vector<int64_t> GetReduceAxis(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  const auto &attrs = prim->attrs();
  auto iter = attrs.find(kAttrAxis);
  if (iter == attrs.end()) {
    MS_LOG(EXCEPTION) << "Can not find attribute '" << kAttrAxis << "' in node " << node->fullname_with_scope();
  }

  std::vector<int64_t> axis;

  auto &v = iter->second;
  if (v->isa<ValueList>() || v->isa<ValueTuple>()) {
    auto vec = v->isa<ValueList>() ? v->cast<ValueListPtr>()->value() : v->cast<ValueTuplePtr>()->value();
    for (auto value : vec) {
      if (value->isa<Int64Imm>()) {
        axis.push_back(GetValue<int64_t>(value));
      } else {
        MS_LOG(EXCEPTION) << "Element in attribute 'axis' should be of type int64 in node "
                          << node->fullname_with_scope();
      }
    }
  } else if (v->isa<Int64Imm>()) {
    axis.push_back(GetValue<int64_t>(v));
  } else {
    MS_LOG(EXCEPTION) << "Attribute 'axis' should be a list or tuple in node " << node->fullname_with_scope();
  }

  return axis;
}

// Deprecated. use GkUtils::NewRealCNode
CNodePtr CreateCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph, const DataInfo &out_info,
                     bool use_fake_abstract) {
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
  if (use_fake_abstract) {
    auto abs_shape = GetFakeAbstractShape(out_info.shape, out_info.format);
    auto abs_tensor = std::make_shared<abstract::AbstractTensor>(out_type, abs_shape);
    cnode->set_abstract(abs_tensor);
  } else {
    auto abs_tensor = std::make_shared<abstract::AbstractTensor>(out_type, out_info.shape);
    cnode->set_abstract(abs_tensor);
  }

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
  if (AnfUtils::IsRealKernel(cnode)) {
    // if the node only has the primitive(such as getNext) or the node's input has a feature map input
    // then the node's output is a feature map output
    SetNodeAttrSafely(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), cnode);
    SetNodeAttrSafely(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), cnode);
  }

  // Setup kernel build info.
  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto kernel_with_index = common::AnfAlgo::VisitKernel(inputs[i], 0);
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
  info_builder.SetProcessor(kernel::GetProcessorFromContext());
  info_builder.SetKernelType(KernelType::AKG_KERNEL);
  info_builder.SetFusionType(kernel::kPatternOpaque);
  auto selected_info = info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(selected_info, cnode.get());

  func_graph->AddNode(cnode);
  return cnode;
}

void SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  common::AnfAlgo::SetNodeAttrSafely(key, value, node);
}

bool IsBufferStitchNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input = cnode->input(kAnfPrimitiveIndex);
  if (!IsValueNode<FuncGraph>(input)) {
    return common::AnfAlgo::HasNodeAttr(kAttrStitch, cnode);
  }

  auto func_graph = GetValueNode<FuncGraphPtr>(input);
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList sub_nodes;
  kernel::GetValidKernelNodes(func_graph, &sub_nodes);
  for (auto sub_node : sub_nodes) {
    auto sub_cnode = sub_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(sub_cnode);
    if (common::AnfAlgo::HasNodeAttr(kAttrStitch, sub_cnode)) {
      return true;
    }
  }

  return false;
}
}  // namespace mindspore::graphkernel
