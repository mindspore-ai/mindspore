/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/optimizer/trt_pass/graph_converter.h"

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <utility>
#include <string>
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "plugin/device/gpu/optimizer/trt_pass/trt_converter_context.h"
#include "utils/singleton.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "plugin/device/gpu/hal/device/trt_loader.h"

namespace mindspore {
namespace opt {
namespace {
void CopyGraphOutputTypeAndShape(const std::vector<session::KernelWithIndex> &graph_outputs, CNodePtr trt_node) {
  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  for (const auto &item : graph_outputs) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(item.first, item.second));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(item.first, item.second));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, trt_node.get());
  return;
}

CNodePtr NewTrtNode(const FuncGraphPtr &graph, const std::string &model_data, const AnfNodePtrList &graph_inputs,
                    const std::vector<session::KernelWithIndex> &graph_outputs) {
  // Create TrtNode which hold serialzed data.
  auto prim = std::make_shared<Primitive>("TrtNode");
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  std::copy(graph_inputs.begin(), graph_inputs.end(), std::back_inserter(inputs));
  prim->AddAttr("serialize_model", MakeValue(model_data));
  auto trt_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(trt_node);

  // Update output shape and type
  CopyGraphOutputTypeAndShape(graph_outputs, trt_node);
  return trt_node;
}

CNodePtr BuildMakeTupleNode(const FuncGraphPtr root, const std::map<size_t, size_t> &anf_trt_index_map,
                            CNodePtr trt_node) {
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<TypeId> make_tuple_types;
  std::vector<ShapeVector> make_tuple_shapes;

  for (size_t out_idx = 0; out_idx < anf_trt_index_map.size(); out_idx++) {
    // Get TrtNode output index
    auto iter = anf_trt_index_map.find(out_idx);
    if (iter == anf_trt_index_map.end()) {
      MS_LOG(WARNING) << "Output node found: " << out_idx;
      return nullptr;
    }
    size_t trt_index = iter->second;

    // create tuple_getitem_cnode
    std::vector<AnfNodePtr> tuple_getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem), trt_node,
                                                    NewValueNode(MakeValue(SizeToLong(trt_index)))};
    const CNodePtr &tuple_getitem_cnode = root->NewCNode(tuple_getitem_inputs);
    MS_EXCEPTION_IF_NULL(tuple_getitem_cnode);

    // Set tuple_getitem_cnode abstract.
    std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(trt_node, trt_index)};
    std::vector<ShapeVector> shapes = {common::AnfAlgo::GetOutputInferShape(trt_node, trt_index)};
    common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, tuple_getitem_cnode.get());

    // Build make tuple inputs.
    make_tuple_inputs.push_back(tuple_getitem_cnode);
    make_tuple_types.push_back(types[0]);
    make_tuple_shapes.push_back(shapes[0]);
  }

  const CNodePtr &make_tuple_cnode = root->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  common::AnfAlgo::SetOutputInferTypeAndShape(make_tuple_types, make_tuple_shapes, make_tuple_cnode.get());

  return make_tuple_cnode;
}
}  // namespace

AnfNodePtrList GraphConverter::GetUsefulArguments(const AnfNodePtrList &arguments, const AnfNodePtrList &parameters,
                                                  const AnfNodePtrList &useful_parameters) {
  // Present map between formal parameter and actual argument.
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> args_map;
  for (size_t i = 0; i < parameters.size(); i++) {
    (void)args_map.emplace(parameters[i], arguments[i]);
  }

  AnfNodePtrList useful_arguments;
  for (size_t j = 0; j < useful_parameters.size(); j++) {
    auto iter = args_map.find(useful_parameters[j]);
    if (iter == args_map.end() || iter->second == nullptr) {
      MS_LOG(WARNING) << "Argument not found. Arg: " << useful_parameters[j]->DebugString();
      return {};
    }
    useful_arguments.push_back(iter->second);
  }

  return useful_arguments;
}

std::tuple<std::map<size_t, size_t>, CNodePtr> GraphConverter::BuildTrtNode(const FuncGraphPtr &root_graph,
                                                                            const FuncGraphPtr &sub_graph,
                                                                            const AnfNodePtrList &arguments) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(sub_graph);
  auto converter = std::make_shared<TrtConverterContext>(sub_graph);
  bool ret = converter->Init();
  if (!ret) {
    MS_LOG(WARNING) << "Graph convert init failed.";
    return std::make_tuple(std::map<size_t, size_t>(), nullptr);
  }

  ret = converter->Parser();
  if (!ret) {
    MS_LOG(WARNING) << "Graph converter parse failed.";
    return std::make_tuple(std::map<size_t, size_t>(), nullptr);
  }

  std::string model_data;
  ret = converter->Serialize(&model_data);
  if (!ret) {
    MS_LOG(WARNING) << "Graph converte serialize failed.";
    return std::make_tuple(std::map<size_t, size_t>(), nullptr);
  }

  // Get actual arguments by useful formal parameters
  const AnfNodePtrList &parameters = sub_graph->parameters();
  const AnfNodePtrList &useful_parameters = converter->GetGraphInputs();
  const AnfNodePtrList &useful_arguments = GetUsefulArguments(arguments, parameters, useful_parameters);

  // Get outputs by the TensorRT binding order.
  std::map<size_t, size_t> anf_trt_index_map;
  std::vector<session::KernelWithIndex> trt_output_list;
  std::tie(anf_trt_index_map, trt_output_list) = converter->GetGraphOutputs();
  CNodePtr trt_node = NewTrtNode(root_graph, model_data, useful_arguments, trt_output_list);

  return std::make_tuple(anf_trt_index_map, trt_node);
}

void GraphConverter::RemoveParameterWithoutUser(const FuncGraphPtr &graph) {
  std::vector<AnfNodePtr> graph_inputs;

  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const AnfNodePtrList &inputs = kernel_graph->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];

    // Keep inputs of graph.
    if (!input->isa<Parameter>() || !common::AnfAlgo::IsParameterWeight(input->cast<ParameterPtr>())) {
      graph_inputs.push_back(input);
      continue;
    }

    // Remove useless parameters of graph.
    FuncGraphManagerPtr manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const NodeUsersMap &users = manager->node_users();
    const auto &iter = users.find(input);
    if (iter != users.end() && !iter->second.empty()) {
      graph_inputs.push_back(input);
    }
    MS_LOG(INFO) << "Useless input: " << input->DebugString();
  }

  MS_LOG(INFO) << "Graph total inputs num: " << graph_inputs.size();
  kernel_graph->SetGraphInputs(graph_inputs);
  kernel_graph->set_parameters(graph_inputs);
}

bool GraphConverter::ReplaceSubgraphWithTrtNode(const FuncGraphPtr &root, const Subgraph &sub_graph_info) {
  FuncGraphPtr sub_graph;
  AnfNodePtrList args;
  AnfNodePtrList outputs;
  std::tie(sub_graph, args, outputs) = sub_graph_info;

  std::map<size_t, size_t> anf_trt_index_map;
  CNodePtr trt_node;
  std::tie(anf_trt_index_map, trt_node) = BuildTrtNode(root, sub_graph, args);
  if (trt_node == nullptr) {
    MS_LOG(WARNING) << "Convert to Tensor-RT network failed.";
    return false;
  }

  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (outputs.size() == 1) {
    if (common::AnfAlgo::CheckPrimitiveType(outputs[0], prim::kPrimMakeTuple)) {
      const CNodePtr &make_tuple_cnode = BuildMakeTupleNode(root, anf_trt_index_map, trt_node);
      manager->Replace(outputs[0], make_tuple_cnode);
    } else {
      manager->Replace(outputs[0], trt_node);
    }
    return true;
  }

  for (size_t out_idx = 0; out_idx < outputs.size(); out_idx++) {
    size_t trt_index = anf_trt_index_map[out_idx];
    std::vector<AnfNodePtr> fn_inputs = {NewValueNode(prim::kPrimTupleGetItem), trt_node,
                                         NewValueNode(MakeValue(SizeToLong(trt_index)))};
    const CNodePtr &new_out = root->NewCNode(fn_inputs);
    new_out->set_abstract(outputs[out_idx]->abstract());
    manager->Replace(outputs[out_idx], new_out);
  }
  return true;
}

bool GraphConverter::Run(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);

  const auto &context = MsContext::GetInstance();
  if (!context->get_param<bool>(MS_CTX_ENABLE_INFER_OPT)) {
    return false;
  }

  // Set device id before invoke trt api as cudaSetDevice is thread level config.
  const auto &device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  bool ret = device::gpu::CudaDriver::SetDevice(UintToInt(device_id));
  if (!ret) {
    MS_LOG(ERROR) << "Failed to set device id:" << device_id;
    return false;
  }

  const auto &trt_loader = Singleton<device::gpu::TrtLoader>::Instance();
  if (!trt_loader.nvinfer_loaded()) {
    MS_LOG(WARNING) << "Load Tensor-RT so failed. Inference with native backend.";
    return false;
  }

  try {
    auto graph_partition = std::make_shared<GraphPartitioner>();
    const std::map<std::string, AnfNodePtrList> &segments = graph_partition->Partition(fg);
    for (const auto &segment : segments) {
      // Do not fusion when segment only contain 1 node.
      if (segment.second.size() == 1) {
        continue;
      }
      const Subgraph &sub_graph = graph_partition->CreateNewGraph(segment.second);
      ret = ReplaceSubgraphWithTrtNode(fg, sub_graph);
      if (!ret) {
        MS_LOG(WARNING) << "Failed replace sub graph with TrtNode.";
        continue;
      }
      // Remove useless parameters folded in TensorRT network.
      RemoveParameterWithoutUser(fg);
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Convert to Tensor-RT network failed. " << e.what();
    return false;
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
