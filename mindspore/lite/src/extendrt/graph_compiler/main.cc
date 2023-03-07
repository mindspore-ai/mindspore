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

#include <vector>
#include <cstdlib>
#include <iostream>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"
#include "ops/core_ops.h"
#include "include/api/context.h"
#include "src/litert/inner_context.h"
#include "src/litert/cxx_api/converters.h"
#include "backend/graph_compiler/graph_partition.h"
#include "backend/graph_compiler/segment_runner.h"
#include "src/extendrt/graph_compiler/single_graph_compiler.h"

namespace mindspore {
FuncGraphPtr CreateFuncGraph() {
  std::vector<int64_t> shp{2, 2};
  abstract::AbstractTensorPtr abs;

  // Func graph.
  auto func_graph = std::make_shared<FuncGraph>();

  // Parameter.
  auto abstract_x = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_x = func_graph->add_parameter();
  parameter_x->set_name("input_x");
  parameter_x->set_abstract(abstract_x);

  auto abstract_y = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto parameter_y = func_graph->add_parameter();
  parameter_y->set_name("input_y");
  parameter_y->set_abstract(abstract_y);
  auto parameters = func_graph->parameters();

  // Add.
  std::vector<AnfNodePtr> add_inputs{NewValueNode(prim::kPrimAdd), parameters[0], parameters[1]};
  auto add_node = func_graph->NewCNode(add_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  add_node->set_abstract(abs);

  // Reshape.
  std::vector<AnfNodePtr> reshape_inputs{NewValueNode(prim::kPrimReshape), add_node};
  auto reshape_node = func_graph->NewCNode(reshape_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  reshape_node->set_abstract(abs);

  // sub.
  std::vector<AnfNodePtr> sub_inputs{NewValueNode(prim::kPrimSub), reshape_node, parameters[0]};
  auto sub_node = func_graph->NewCNode(sub_inputs);
  abs = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  sub_node->set_abstract(abs);

  // Return.
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), sub_node};
  auto return_node = func_graph->NewCNode(return_inputs);
  func_graph->set_return(return_node);

#ifdef USE_REAL_MODEL
  const std::string model_path = "";  // /path/to/mobilenet.mindir
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(model_path);
#endif
  return func_graph;
}

std::tuple<GraphSegmentPtr, AnfNodePtrList, AnfNodePtrList> CreateSegment() {
  auto func_graph = CreateFuncGraph();
  std::cout << "============================================= func_graph inputs:" << std::endl;
  for (auto &input : func_graph->get_inputs()) {
    std::cout << input << ":" << input->fullname_with_scope() << std::endl;
  }
  auto new_manager = MakeManager({func_graph});
  MS_EXCEPTION_IF_NULL(new_manager);
  new_manager->AddFuncGraph(func_graph);
  func_graph->set_manager(new_manager);

  static const std::vector<PrimitivePtr> ms_nonlinear_ops = {prim::kPrimReturn,   prim::kPrimPartial,
                                                             prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                             prim::kPrimBpropCut, prim::kPrimSwitchLayer};
  auto graph_partition = std::make_shared<compile::GraphPartition>(ms_nonlinear_ops, kMsConvert);
  bool multi_target = false;
  auto segments = graph_partition->Partition(func_graph, &multi_target);
  auto segment = segments[0];
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(segment->nodes_);
  std::cout << "============================================= segment:" << std::endl;
  std::cout << "--------------------------------------------- nodes:" << std::endl;
  for (auto &node : segment->nodes_) {
    std::cout << node << ":" << node->fullname_with_scope() << std::endl;
  }
  std::cout << "--------------------------------------------- inputs:" << std::endl;
  for (auto &input : inputs) {
    std::cout << input << ":" << input->fullname_with_scope() << std::endl;
  }
  std::cout << "--------------------------------------------- outputs:" << std::endl;
  for (auto &output : outputs) {
    std::cout << output << ":" << output->fullname_with_scope() << std::endl;
  }
  return std::make_tuple(segment, inputs, outputs);
}
}  // namespace mindspore

int main() {
  mindspore::GraphSegmentPtr segment;
  mindspore::AnfNodePtrList inputs;
  mindspore::AnfNodePtrList outputs;
  std::tie(segment, inputs, outputs) = mindspore::CreateSegment();
  auto context = std::make_shared<mindspore::Context>();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().emplace_back(device_info);
  auto inner_context = std::shared_ptr<mindspore::lite::InnerContext>(mindspore::ContextUtils::Convert(context.get()));
  auto compiler = std::make_shared<mindspore::infer::SingleGraphCompiler>(inner_context);
  compiler->Compile(segment, inputs, outputs);
  return 0;
}
