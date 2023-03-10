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
#include "tools/converter/parser/onnx/onnx_model_parser.h"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <map>
#include <utility>
#include "include/registry/node_parser_registry.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "tools/common/tensor_util.h"
#include "ops/tensor_list_stack.h"
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/parser/onnx/onnx_inputs_adjust.h"
#include "tools/converter/parser/onnx/onnx_pad_adjust.h"
#include "tools/converter/parser/onnx/onnx_nonzero_adjust.h"
#include "tools/converter/parser/onnx/onnx_einsum_adjust.h"
#include "tools/converter/parser/onnx/onnx_quantize_linear_adjust.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "tools/converter/parser/unify_format.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"

using mindspore::converter::kFmkTypeOnnx;
namespace mindspore {
namespace lite {
namespace {
constexpr int kTensorListDatasize = 3;
constexpr int kTypeIndex = 0;
constexpr int kElementShapeIndex = 1;
constexpr int kTensorsNumIndex = 2;

int Onnx2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (const auto &func_graph : all_func_graphs) {
    MS_ASSERT(func_graph != nullptr);
    if (!OnnxInputAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "onnx adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    if (!OnnxPadAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "onnx pad adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    if (!OnnxNonZeroAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "onnx nonzero adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    if (!OnnxEinsumAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "onnx einsum adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
    if (!OnnxQuantizeLinearAdjust::Adjust(func_graph)) {
      MS_LOG(ERROR) << "onnx quantize linear adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ParameterPtr CreateConstParamter(const FuncGraphPtr &anf_graph, int val) {
  MS_ASSERT(anf_graph != nullptr);
  auto const_node = anf_graph->add_parameter();
  MS_CHECK_TRUE_RET(const_node != nullptr, nullptr);
  auto const_abstract = CreateTensorAbstract({}, kNumberTypeInt32);
  if (const_abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return nullptr;
  }
  const_node->set_abstract(const_abstract);
  int *tensor_data = new (std::nothrow) int[1];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new int[] failed";
    return nullptr;
  }
  tensor_data[0] = val;
  auto tensor_info = CreateTensorInfo(tensor_data, sizeof(int), {1}, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    delete[] tensor_data;
    tensor_data = nullptr;
    return nullptr;
  }
  delete[] tensor_data;
  tensor_data = nullptr;
  const_node->set_default_param(tensor_info);
  return const_node;
}

ValueNodePtr CreateValueNode(const schema::PrimitiveType &op_type) {
  auto node_type = schema::EnumNamePrimitiveType(op_type);
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(node_type) == op_primc_fns.end()) {
    MS_LOG(ERROR) << "have no func to create primitive.";
    return nullptr;
  }
  auto prim = op_primc_fns[node_type]();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "cannot create primitive.";
    return nullptr;
  }
  return NewValueNode(prim);
}

STATUS AddIterNumsUpdateEdge(const FuncGraphPtr &anf_graph, std::vector<AnfNodePtr> *return_new_inputs,
                             const std::unordered_map<std::string, AnfNodePtr> &anf_nodes_map,
                             const std::string &trip_cout_name, const std::string &loop_node_name) {
  MS_ASSERT(anf_graph != nullptr && return_new_inputs != nullptr);
  // trip_cout need -1 after every iteration
  auto sub_value_node = CreateValueNode(schema::PrimitiveType_SubFusion);
  if (sub_value_node == nullptr) {
    MS_LOG(ERROR) << "create sub failed.";
    return RET_NULL_PTR;
  }
  auto trip_cout_paramter_iter = anf_nodes_map.find(trip_cout_name);
  if (trip_cout_paramter_iter == anf_nodes_map.end()) {
    MS_LOG(ERROR) << "can not find " << trip_cout_name;
    return RET_ERROR;
  }
  auto &trip_cout_paramter = trip_cout_paramter_iter->second;
  if (trip_cout_paramter == nullptr) {
    MS_LOG(ERROR) << "trip_cout_paramter found failed";
    return RET_ERROR;
  }
  auto const_one_parameter = CreateConstParamter(anf_graph, 1);
  MS_CHECK_TRUE_MSG(const_one_parameter != nullptr, RET_ERROR, "create const parameter return nullptr");
  const_one_parameter->set_name(loop_node_name + "_index_update_parameter");

  std::vector<AnfNodePtr> sub_inputs = {sub_value_node, trip_cout_paramter, const_one_parameter};
  auto sub_cnode = anf_graph->NewCNode(sub_inputs);
  if (sub_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  sub_cnode->set_fullname_with_scope(loop_node_name + "_sub");
  sub_cnode->set_abstract(trip_cout_paramter->abstract());
  return_new_inputs->insert(return_new_inputs->begin() + 1, sub_cnode);
  return RET_OK;
}

CNodePtr GetCNodeFromControlFlowNodesMap(
  const std::string &loop_node_name,
  const std::unordered_map<std::string, std::unordered_map<std::string, AnfNodePtr> *> &control_nodes_map) {
  auto iter1 = control_nodes_map.find(loop_node_name);
  if (iter1 == control_nodes_map.end()) {
    return nullptr;
  }  // namespace
  auto iter2 = iter1->second->find(loop_node_name);
  if (iter2 == iter1->second->end()) {
    return nullptr;
  }
  return iter2->second->cast<CNodePtr>();
}

STATUS BuildReturnNode(const FuncGraphPtr &anf_graph, const std::vector<AnfNodePtr> &return_inputs) {
  MS_ASSERT(anf_graph != nullptr);
  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "new Return failed";
    return RET_NULL_PTR;
  }
  if (return_inputs.empty()) {
    MS_LOG(ERROR) << "return input is empty";
    return RET_ERROR;
  }
  auto input = return_inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  auto abstract = input->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Input node abstract is null, node: " << input->fullname_with_scope();
    return RET_ERROR;
  }

  auto return_prim_c = return_prim->GetPrim();
  MS_ASSERT(return_prim_c != nullptr);
  auto return_cnode = anf_graph->NewCNode(return_prim_c, return_inputs);
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  return_cnode->set_fullname_with_scope("Return");
  return_cnode->set_abstract(abstract);
  anf_graph->set_return(return_cnode);
  return RET_OK;
}

STATUS BuildParameterNode(const ParameterPtr &parameter_node, const onnx::TensorProto &tensor,
                          const std::string &model_file) {
  MS_ASSERT(parameter_node != nullptr);
  auto data_type = OnnxNodeParser::GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type " << static_cast<onnx::TensorProto_DataType>(tensor.data_type());
    return RET_ERROR;
  }
  std::vector<int64_t> shape_vector(tensor.dims().begin(), tensor.dims().end());
  auto abstract_tensor = CreateTensorAbstract(shape_vector, data_type);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed";
    return RET_ERROR;
  }
  parameter_node->set_abstract(abstract_tensor);
  parameter_node->set_name(tensor.name());

  auto tensor_info = std::make_shared<tensor::Tensor>(data_type, shape_vector);
  MS_CHECK_TRUE_MSG(tensor_info != nullptr, RET_NULL_PTR, "create tensor_info return nullptr");
  std::vector<int> shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                 [](const int64_t &value) { return static_cast<int>(value); });
  if (tensor.data_location() != onnx::TensorProto::EXTERNAL) {
    auto status = OnnxNodeParser::CopyOnnxTensorData(tensor, tensor_info);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "copy data failed.";
      return status;
    }
  } else {
    auto status = OnnxNodeParser::LoadOnnxExternalTensorData(tensor, tensor_info, model_file);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "load external data failed.";
      return status;
    }
  }
  parameter_node->set_default_param(tensor_info);
  return RET_OK;
}

STATUS BuildOpOutputs(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                      std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, const CNodePtr &cnode) {
  MS_ASSERT(anf_graph != nullptr && cnode != nullptr && anf_nodes_map != nullptr);
  if (onnx_node.output_size() == 1) {
    auto abstract_tensor = CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    cnode->set_abstract(abstract_tensor);
    anf_nodes_map->emplace(onnx_node.output(0), cnode);
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (const auto &output_name : onnx_node.output()) {
      auto abstract_tensor = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (abstract_tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      abstract_list.emplace_back(abstract_tensor);
      auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim_c = tuple_get_item_prim_ptr->GetPrim();
      MS_CHECK_TRUE_MSG(tuple_get_item_prim_c != nullptr, RET_NULL_PTR, "create tuple_get_item_prim_c return nullptr");
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
      MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, RET_NULL_PTR, "create ValueNode return nullptr");
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      MS_CHECK_TRUE_MSG(get_item_value != nullptr, RET_NULL_PTR, "create ValueNode return nullptr");
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
      CNodePtr get_item_cnode = anf_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_getitem_" + std::to_string(op_idx));
      auto get_item_abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (get_item_abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      get_item_cnode->set_abstract(get_item_abstract);
      anf_nodes_map->emplace(output_name, get_item_cnode);
      op_idx++;
    }
    auto new_abstract_list = std::make_shared<abstract::AbstractTuple>(abstract_list);
    CHECK_NULL_RETURN(new_abstract_list);
    cnode->set_abstract(new_abstract_list);
  }
  anf_nodes_map->emplace(onnx_node.name(), cnode);
  return RET_OK;
}

STATUS ConvertConstTensors(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                           std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map, const std::string &model_file) {
  MS_ASSERT(func_graph_ptr != nullptr && anf_nodes_map != nullptr);
  for (const auto &onnx_const_value : onnx_graph.initializer()) {
    auto parameter = func_graph_ptr->add_parameter();
    MS_CHECK_TRUE_MSG(parameter != nullptr, RET_NULL_PTR, "create parameter return nullptr");
    auto status = BuildParameterNode(parameter, onnx_const_value, model_file);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "parameter node build failed.";
      return status;
    }
    anf_nodes_map->emplace(onnx_const_value.name(), parameter);
  }
  return RET_OK;
}

STATUS ConvertGraphInputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                          std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map) {
  MS_ASSERT(func_graph_ptr != nullptr && anf_nodes_map != nullptr);
  for (int i = 0; i < onnx_graph.input().size(); ++i) {
    const auto &input_value = onnx_graph.input(i);
    if (anf_nodes_map->find(input_value.name()) != anf_nodes_map->end()) {
      continue;
    }
    auto parameter = func_graph_ptr->add_parameter();
    MS_CHECK_TRUE_MSG(parameter != nullptr, RET_NULL_PTR, "create parameter return nullptr");
    auto data_type = OnnxNodeParser::GetDataTypeFromOnnx(
      static_cast<onnx::TensorProto_DataType>(input_value.type().tensor_type().elem_type()));
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "not support onnx data type "
                    << static_cast<onnx::TensorProto_DataType>(input_value.type().tensor_type().elem_type());
      return RET_ERROR;
    }
    std::vector<int64_t> shape_vector =
      ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(input_value.name());
    if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape_vector.empty()) {
      MS_LOG(WARNING) << "Can not find name in map. name is " << input_value.name();
    }
    if (shape_vector.empty()) {
      auto onnx_shape = input_value.type().tensor_type().shape().dim();
      std::transform(onnx_shape.begin(), onnx_shape.end(), std::back_inserter(shape_vector),
                     [](const onnx::TensorShapeProto_Dimension &val) { return static_cast<int64_t>(val.dim_value()); });
      std::replace(shape_vector.begin(), shape_vector.end(), 0, -1);
    }
    auto abstract_tensor = CreateTensorAbstract(shape_vector, data_type);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(input_value.name());
    anf_nodes_map->emplace(input_value.name(), parameter);
  }
  return RET_OK;
}

STATUS ConvertGraphOutputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                           const std::unordered_map<std::string, AnfNodePtr> &anf_nodes_map) {
  MS_ASSERT(anf_graph != nullptr);
  std::vector<AnfNodePtr> return_inputs;
  if (onnx_graph.output_size() == 0) {
    MS_LOG(ERROR) << "onnx graph has no output";
    return RET_ERROR;
  }
  if (onnx_graph.output_size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    AbstractBasePtrList elem;
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    for (const auto &graph_out : onnx_graph.output()) {
      if (anf_nodes_map.find(graph_out.name()) == anf_nodes_map.end()) {
        MS_LOG(ERROR) << "graph output get failed.";
        return RET_ERROR;
      }
      auto cnode = anf_nodes_map.at(graph_out.name());
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      elem.emplace_back(cnode->abstract());
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    MS_ASSERT(make_tuple_prim_c != nullptr);
    auto make_tuple_cnode = anf_graph->NewCNode(make_tuple_prim_c, make_tuple_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }

    make_tuple_cnode->set_fullname_with_scope("return tuple");
    make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    return_inputs.emplace_back(make_tuple_cnode);
  } else {
    const auto &graph_out = onnx_graph.output(0);
    if (anf_nodes_map.find(graph_out.name()) == anf_nodes_map.end()) {
      MS_LOG(ERROR) << "graph output get failed.";
      return RET_ERROR;
    }
    auto cnode = anf_nodes_map.at(graph_out.name());
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    return_inputs.emplace_back(cnode);
  }
  if (BuildReturnNode(anf_graph, return_inputs) != RET_OK) {
    MS_LOG(ERROR) << "build return node failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

FuncGraphPtr BuildCondGraph(const AnfNodePtr &root_while_node, int inputs_num, const std::string &cond_graph_name) {
  MS_ASSERT(root_while_node != nullptr);
  auto cond_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(cond_graph != nullptr, nullptr, "create cond_graph return nullptr");
  CNodePtr less_cnode = nullptr;
  for (int i = 0; i < inputs_num; i++) {
    auto input_parameter = cond_graph->add_parameter();
    MS_CHECK_TRUE_MSG(input_parameter != nullptr, nullptr, "create input_parameter return nullptr");
    input_parameter->set_name(cond_graph_name + "_input_" + std::to_string(i) + "_parameter");
    auto input_abstract = CreateTensorAbstract({}, kNumberTypeInt32);
    if (input_abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return nullptr;
    }
    input_parameter->set_abstract(input_abstract);
    if (i == 0) {
      auto zero_parameter = CreateConstParamter(cond_graph, 0);
      MS_CHECK_TRUE_MSG(zero_parameter != nullptr, nullptr, "create zero_parameter return nullptr");
      zero_parameter->set_name(root_while_node->fullname_with_scope() + "_const_0");
      auto less_value_node = CreateValueNode(schema::PrimitiveType_Less);
      MS_CHECK_TRUE_MSG(less_value_node != nullptr, nullptr, "create less_value_node return nullptr");
      std::vector<AnfNodePtr> less_inputs = {less_value_node, zero_parameter, input_parameter};
      less_cnode = cond_graph->NewCNode(less_inputs);
      if (less_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return nullptr;
      }
      auto less_abstract = CreateTensorAbstract({}, kNumberTypeBool);
      if (less_abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return nullptr;
      }
      less_cnode->set_abstract(less_abstract);
      less_cnode->set_fullname_with_scope(cond_graph_name + "_less_cnode");
    }
    if (i == 1) {
      auto and_value_node = CreateValueNode(schema::PrimitiveType_LogicalAnd);
      MS_CHECK_TRUE_MSG(and_value_node != nullptr, nullptr, "CreateValueNode failed");
      std::vector<AnfNodePtr> and_inputs = {and_value_node, less_cnode, input_parameter};
      auto and_cnode = cond_graph->NewCNode(and_inputs);
      if (and_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return nullptr;
      }
      and_cnode->set_abstract(less_cnode->abstract());
      and_cnode->set_fullname_with_scope(cond_graph_name + "_output_" + std::to_string(0) + "_cnode");
      auto status = BuildReturnNode(cond_graph, {and_cnode});
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build return node failed: " << status;
        return nullptr;
      }
    }
  }
  cond_graph->set_attr("graph_name", MakeValue(cond_graph_name));
  return cond_graph;
}

FuncGraphPtr ConvertGraph(api::FuncGraphPtr func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}
}  // namespace

FuncGraphPtr OnnxModelParser::BuildBodyGraph(const onnx::NodeProto &loop_node, const onnx::GraphProto &subgraph_proto,
                                             int *cond_graph_input_num) {
  MS_ASSERT(cond_graph_input_num != nullptr);
  auto &loop_node_name = loop_node.name();
  auto node_inputs_num = loop_node.input_size();
  auto node_outputs_num = loop_node.output_size();
  // skip trip_cout and cond input,scan_output nums
  auto act_outputs_num = node_outputs_num - (node_inputs_num - 2);
  auto loop_body_graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(loop_body_graph != nullptr, nullptr, "create loop_body_graph return nullptr");
  std::unordered_map<std::string, AnfNodePtr> anf_nodes_map;
  std::vector<AnfNodePtr> gen_subgraph_inputs;
  auto status = ConvertOnnxGraph(subgraph_proto, loop_body_graph, &anf_nodes_map, &gen_subgraph_inputs, loop_node_name);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert loop OnnxGraph: " << status;
    return nullptr;
  }
  auto return_node = loop_body_graph->get_return();
  MS_CHECK_TRUE_MSG(return_node != nullptr, nullptr, "return node of subgraph is nullptr");
  MS_ASSERT(return_node->inputs().size() == DIMENSION_2D);
  auto return_tuple_cnode = return_node->input(1)->cast<CNodePtr>();
  MS_ASSERT(return_tuple_cnode != nullptr);
  auto return_new_inputs = return_tuple_cnode->inputs();
  return_new_inputs.insert(return_new_inputs.end() - act_outputs_num, gen_subgraph_inputs.begin(),
                           gen_subgraph_inputs.end());

  std::string max_trip_count_name = subgraph_proto.input(0).name();
  status =
    AddIterNumsUpdateEdge(loop_body_graph, &return_new_inputs, anf_nodes_map, max_trip_count_name, loop_node_name);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add iter nums update edge failed: " << status;
    return nullptr;
  }
  auto root_while_node = GetCNodeFromControlFlowNodesMap(loop_node_name, control_nodes_map_);
  MS_CHECK_TRUE_MSG(root_while_node != nullptr, nullptr, "can not find root_while_node is control_nodes_map");
  std::vector<AnfNodePtr> body_graph_inputs;
  body_graph_inputs.reserve(subgraph_proto.input_size());
  for (int j = 0; j < subgraph_proto.input_size(); j++) {
    body_graph_inputs.emplace_back(anf_nodes_map[subgraph_proto.input(j).name()]);
  }
  body_graph_inputs.insert(body_graph_inputs.end(), gen_subgraph_inputs.begin(), gen_subgraph_inputs.end());
  if (act_outputs_num != 0) {
    status =
      AddTensorArrayEdge(loop_body_graph, &return_new_inputs, loop_node_name, &body_graph_inputs, act_outputs_num);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "add tensorarray update edge failed: " << status;
      return nullptr;
    }
    // insert tensorliststack after while output
    status = AddTensorListStackNode(root_while_node, loop_node, act_outputs_num, body_graph_inputs.size());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "add tensorliststack node failed: " << status;
      return nullptr;
    }
  }
  return_tuple_cnode->set_inputs(return_new_inputs);
  auto body_graph_name = loop_node_name + "_body_graph";
  for (size_t j = 0; j < body_graph_inputs.size(); j++) {
    MS_CHECK_TRUE_RET(body_graph_inputs[j] != nullptr, nullptr);
    auto body_input = body_graph_inputs[j]->cast<ParameterPtr>();
    MS_ASSERT(body_input != nullptr);
    body_input->set_name(body_graph_name + "_input_" + std::to_string(j) + "_parameter");
  }
  for (size_t j = 1; j < return_new_inputs.size(); j++) {
    if (utils::isa<CNodePtr>(return_new_inputs[j])) {
      return_new_inputs[j]->cast<CNodePtr>()->set_fullname_with_scope(body_graph_name + "_output_" +
                                                                      std::to_string(j - 1) + "_cnode");
    } else if (utils::isa<ParameterPtr>(return_new_inputs[j])) {
      return_new_inputs[j]->cast<ParameterPtr>()->set_name(body_graph_name + "_output_" + std::to_string(j - 1) +
                                                           "_parameter");
    }
  }
  *cond_graph_input_num = return_new_inputs.size() - 1;
  loop_body_graph->set_attr("graph_name", MakeValue(body_graph_name));
  return loop_body_graph;
}

namespace {
STATUS CheckOnnxModel(const onnx::GraphProto &onnx_graph) {
  // all input should in initialize
  std::set<std::string> providers;
  for (const auto &const_tensor : onnx_graph.initializer()) {
    const auto &name = const_tensor.name();
    if (providers.count(name) != 0) {
      MS_LOG(ERROR) << "const tensor repeated";
      return RET_ERROR;
    }
    providers.insert(name);
  }
  for (int i = 0; i < onnx_graph.input().size(); ++i) {
    providers.insert(onnx_graph.input(i).name());
  }
  for (const auto &onnx_node : onnx_graph.node()) {
    for (int i = 0; i < onnx_node.output_size(); i++) {
      auto &output = onnx_node.output(i);
      if (providers.count(output) != 0) {
        MS_LOG(ERROR) << "Output tensor repeated";
        return RET_ERROR;
      }
      providers.insert(output);
    }
  }
  // all output should find
  for (const auto &onnx_node : onnx_graph.node()) {
    for (int i = 0; i < onnx_node.input_size(); i++) {
      auto &input = onnx_node.input(i);
      if (providers.count(input) == 0) {
        MS_LOG(WARNING) << "Can not find node input: " << input;
      }
    }
  }
  return RET_OK;
}
}  // namespace

api::FuncGraphPtr OnnxModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file;
  NotSupportOp::GetInstance()->set_fmk_type("ONNX");
  auto graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "create FuncGraph failed");
  res_graph_ = api::MakeShared<api::FuncGraph>(graph);
  auto status = InitOriginModel(model_file);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "init origin model failed.";
    return nullptr;
  }
  MS_ASSERT(onnx_root_graph_ != nullptr);

  status = ConvertOnnxGraph(onnx_root_graph_, graph, &anf_nodes_map_, {}, "root_node");
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert onnx graph failed.";
    return nullptr;
  }
  static auto root_func_manager = Manage(graph);
  MS_ASSERT(root_func_manager != nullptr);
  for (auto &subgraph : all_subgraphs_) {
    MS_ASSERT(subgraph != nullptr);
    subgraph->set_manager(root_func_manager);
    subgraph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));
  }
  graph->set_attr("graph_name", MakeValue("main_graph"));
  graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeOnnx)));
  if ((status = CommonAnfAdjust(graph)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(graph, &all_func_graphs);
  if ((status = Onnx2AnfAdjust(all_func_graphs)) != RET_OK) {
    MS_LOG(ERROR) << "Onnx2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(kFmkTypeOnnx, false);
  MS_CHECK_TRUE_MSG(unify_format != nullptr, nullptr, "create unify_format return nullptr");
  if (!unify_format->Run(graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  return res_graph_;
}

STATUS OnnxModelParser::InitOriginModel(const std::string &model_file) {
  MS_ASSERT(res_graph_ != nullptr);
  auto res_graph = ConvertGraph(res_graph_);
  auto status = ValidateFileStr(model_file, ".onnx");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.onnx";
    return status;
  }
  model_file_ = model_file;
  status = ReadProtoFromBinaryFile(model_file, &onnx_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read onnx model file failed, model path: " << model_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return status;
  }
  MS_ASSERT(onnx_model_ != nullptr);
  OnnxNodeParser::set_opset_version(onnx_model_.opset_import().Get(0).version());
  onnx_root_graph_ = onnx_model_.graph();
  auto fmk_value_node = MakeValue(static_cast<int>(converter::kFmkTypeOnnx));
  CHECK_NULL_RETURN(fmk_value_node);
  res_graph->set_attr("fmk", fmk_value_node);
  return RET_OK;
}

STATUS OnnxModelParser::ConvertOnnxGraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                                         std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                         std::vector<AnfNodePtr> *extra_subgraph_inputs,
                                         const std::string &root_node_name) {
  MS_ASSERT(anf_graph != nullptr && anf_nodes_map != nullptr && extra_subgraph_inputs != nullptr);
  STATUS status = RET_OK;
  status = CheckOnnxModel(onnx_graph);
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "input onnx model error: " << status;
    return status;
  }
  status = ConvertConstTensors(onnx_graph, anf_graph, anf_nodes_map, model_file_);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert const nodes failed.";
    return RET_ERROR;
  }

  status = ConvertGraphInputs(onnx_graph, anf_graph, anf_nodes_map);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph inputs failed.";
    return RET_OK;
  }

  status = ConvertNodes(onnx_graph, anf_graph, anf_nodes_map, extra_subgraph_inputs, root_node_name);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert nodes failed.";
    return RET_ERROR;
  }

  status = ConvertGraphOutputs(onnx_graph, anf_graph, *anf_nodes_map);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph outputs failed.";
    return RET_ERROR;
  }
  // save original output tensor names.
  if (root_node_name == "root_node") {
    std::vector<std::string> output_names;
    std::transform(onnx_graph.output().begin(), onnx_graph.output().end(), std::back_inserter(output_names),
                   [](auto &graph_output) { return graph_output.name(); });
    ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(output_names);
  }
  return status;
}

std::vector<int> OnnxModelParser::SortOnnxNodeIndex(const onnx::GraphProto &onnx_graph) {
  std::vector<int> sorted_node_index;
  std::queue<int> onnx_nodes_queue;
  std::set<std::string> node_names;
  // for const tensor
  for (const auto &const_tensor : onnx_graph.initializer()) {
    const auto &name = const_tensor.name();
    node_names.insert(name);
  }
  // for graph input
  for (int i = 0; i < onnx_graph.input().size(); i++) {
    node_names.insert(onnx_graph.input(i).name());
  }
  for (int i = 0; i < onnx_graph.node().size(); i++) {
    auto onnx_node = onnx_graph.node(i);
    if (onnx_node.op_type() == "If" || onnx_node.op_type() == "Loop" || has_subgraph_) {
      sorted_node_index.clear();
      has_subgraph_ = true;
      for (int index = 0; index < onnx_graph.node().size(); index++) {
        sorted_node_index.push_back(index);
      }
      return sorted_node_index;
    }
    if (onnx_node.op_type() == "Constant") {
      sorted_node_index.push_back(i);
      for (auto output_name : onnx_node.output()) {
        node_names.insert(output_name);
      }
    } else {
      onnx_nodes_queue.push(i);
    }
  }
  bool find = false;
  int pre_node_index = -1;
  while (!onnx_nodes_queue.empty()) {
    auto node_index = onnx_nodes_queue.front();
    auto onnx_node = onnx_graph.node(node_index);
    if (std::any_of(onnx_node.input().begin(), onnx_node.input().end(),
                    [&](const string &name) { return node_names.count(name) == 0 && !name.empty(); })) {
      onnx_nodes_queue.pop();
      onnx_nodes_queue.push(node_index);
      if (!find && pre_node_index == node_index) {
        MS_LOG(ERROR) << "sort onnx node failed.";
        return {};
      }
      find = false;
      pre_node_index = pre_node_index == -1 ? node_index : pre_node_index;
    } else {
      find = true;
      pre_node_index = pre_node_index == node_index ? -1 : pre_node_index;
      sorted_node_index.push_back(node_index);
      onnx_nodes_queue.pop();
      for (int i = 0; i < onnx_node.output_size(); i++) {
        node_names.insert(onnx_node.output(i));
      }
    }
  }
  return sorted_node_index;
}

STATUS OnnxModelParser::ConvertNodes(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                                     std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                     std::vector<AnfNodePtr> *graph_inputs, const std::string &root_node_name) {
  MS_ASSERT(anf_graph != nullptr && anf_nodes_map != nullptr && extra_subgraph_inputs != nullptr);
  auto sorted_node_index = SortOnnxNodeIndex(onnx_graph);
  if (sorted_node_index.empty()) {
    MS_LOG(ERROR) << "SortOnnxNodeIndex failed.";
    return RET_ERROR;
  }
  STATUS status = RET_OK;
  for (auto node_index : sorted_node_index) {
    const auto &onnx_node = onnx_graph.node(node_index);
    ops::PrimitiveCPtr primitive_c;
    auto node_parser = registry::NodeParserRegistry::GetNodeParser(kFmkTypeOnnx, onnx_node.op_type());
    if (node_parser != nullptr) {
      primitive_c = node_parser->Parse(onnx_graph, onnx_node)->GetPrim();
    } else {
      auto node_parser_builtin = OnnxNodeParserRegistry::GetInstance().GetNodeParser(onnx_node.op_type());
      if (node_parser_builtin == nullptr) {
        NotSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
        status = status == RET_OK ? RET_NOT_FIND_OP : status;
        MS_LOG(ERROR) << "not support onnx data type " << onnx_node.op_type();
      }
      if (status != RET_OK) {
        continue;
      }
      MS_LOG(INFO) << "parse op:" << onnx_node.op_type();
      primitive_c = node_parser_builtin->Parse(onnx_graph, onnx_node);
    }

    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
      status = RET_ERROR;
      continue;
    }
    if (primitive_c->GetAttr(ops::kOriginalFormat) == nullptr) {
      primitive_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(NCHW));
    }
    status = ConvertOpQuantParams(onnx_node, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert " << onnx_node.op_type() << " quant param failed.";
      continue;
    }
    // build CNode
    status = BuildCNode(onnx_node, anf_graph, anf_nodes_map, graph_inputs, primitive_c, root_node_name);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "build cnode " << onnx_node.op_type() << " failed.";
    }

    if (onnx_node.op_type() == "Loop") {
      child_root_map_[onnx_node.name()] = root_node_name;
      control_nodes_map_[onnx_node.name()] = anf_nodes_map;

      status = ConvertLoopOnnxNode(onnx_node, anf_nodes_map, root_node_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build loop node  failed.";
      }
    }
    if (onnx_node.op_type() == "If") {
      child_root_map_[onnx_node.name()] = root_node_name;
      control_nodes_map_[onnx_node.name()] = anf_nodes_map;

      status = ConvertIfOnnxNode(onnx_node, anf_nodes_map, root_node_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build if node  failed.";
      }
    }
  }
  return status;
}

STATUS OnnxModelParser::ConvertIfSubgraph(const onnx::GraphProto &subgraph_proto, const FuncGraphPtr &subgraph,
                                          const std::string &subgraph_name, const std::string &if_node_name,
                                          const std::string &root_node_name) {
  MS_ASSERT(subgraph != nullptr);
  std::unordered_map<std::string, AnfNodePtr> anf_nodes_map;
  std::vector<AnfNodePtr> subgraph_extra_inputs;
  auto status = ConvertOnnxGraph(subgraph_proto, subgraph, &anf_nodes_map, &subgraph_extra_inputs, if_node_name);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert loop OnnxGraph failed";
    return status;
  }
  subgraph->set_attr("graph_name", MakeValue(subgraph_name));
  // update subgraph in out name
  for (int j = 0; j < subgraph_proto.input_size(); j++) {
    auto input_anode_iter = anf_nodes_map.find(subgraph_proto.input(j).name());
    if (input_anode_iter == anf_nodes_map.end()) {
      MS_LOG(ERROR) << "can not find input anode";
      return RET_ERROR;
    }
    auto input_parameter = input_anode_iter->second->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(input_parameter != nullptr, RET_ERROR, "subgraph input should be a parameter");
    input_parameter->set_name(subgraph_name + "_input_" + std::to_string(j) + "_parameter");
  }
  for (size_t j = 0; j < subgraph_extra_inputs.size(); j++) {
    auto input_parameter = subgraph_extra_inputs[j]->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(input_parameter != nullptr, RET_ERROR, "subgraph input should be a parameter");
    input_parameter->set_name(subgraph_name + "_input_" + std::to_string(j + subgraph_proto.input_size()) +
                              "_parameter");
  }
  auto return_node = subgraph->get_return();
  MS_CHECK_TRUE_MSG(return_node != nullptr, RET_ERROR, "subgraph has no return");
  MS_CHECK_GE(return_node->inputs().size(), kInputSize1, RET_ERROR);
  std::vector<AnfNodePtr> return_act_inputs;
  int start_index = 0;
  if (subgraph_proto.output_size() > 1) {
    auto return_cnode = return_node->input(1)->cast<CNodePtr>();
    MS_ASSERT(return_cnode != nullptr);
    return_act_inputs = return_cnode->inputs();
    start_index = 1;
  } else {
    return_act_inputs = {return_node->input(1)};
  }
  for (size_t j = start_index; j < return_act_inputs.size(); j++) {
    if (utils::isa<CNodePtr>(return_act_inputs[j])) {
      return_act_inputs[j]->cast<CNodePtr>()->set_fullname_with_scope(subgraph_name + "_output_" +
                                                                      std::to_string(j - start_index) + "_cnode");
    } else if (utils::isa<ParameterPtr>(return_act_inputs[j])) {
      return_act_inputs[j]->cast<ParameterPtr>()->set_name(subgraph_name + "_output_" +
                                                           std::to_string(j - start_index) + "_parameter");
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertIfOnnxNode(const onnx::NodeProto &onnx_node,
                                          std::unordered_map<std::string, AnfNodePtr> *anf_root_nodes_map,
                                          const std::string &root_node_name) {
  MS_ASSERT(anf_root_nodes_map != nullptr);
  FuncGraphPtr then_branch_graph = nullptr;
  FuncGraphPtr else_branch_graph = nullptr;
  FuncGraphPtr subgraph = nullptr;
  std::string subgraph_name;
  auto &if_node_name = onnx_node.name();

  for (int i = 0; i < onnx_node.attribute_size(); i++) {
    auto &attr = onnx_node.attribute(i);
    auto &subgraph_proto = attr.g();
    if (attr.name().find("then_branch") != std::string::npos) {
      subgraph_name = if_node_name + "_then_branch";
      then_branch_graph = std::make_shared<FuncGraph>();
      MS_CHECK_TRUE_MSG(then_branch_graph != nullptr, RET_NULL_PTR, "create then_branch_graph return nullptr");
      auto status = ConvertIfSubgraph(subgraph_proto, then_branch_graph, subgraph_name, if_node_name, root_node_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build if node else branch failed.";
      }
    } else if (attr.name().find("else_branch") != std::string::npos) {
      subgraph_name = if_node_name + "_else_branch";
      else_branch_graph = std::make_shared<FuncGraph>();
      MS_CHECK_TRUE_MSG(else_branch_graph != nullptr, RET_NULL_PTR, "create else_branch_graph return nullptr");
      auto status = ConvertIfSubgraph(subgraph_proto, else_branch_graph, subgraph_name, if_node_name, root_node_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build if node else branch failed.";
      }
    } else {
      continue;
    }
  }
  all_subgraphs_.emplace_back(then_branch_graph);
  all_subgraphs_.emplace_back(else_branch_graph);
  auto then_value_node = NewValueNode(then_branch_graph);
  MS_CHECK_TRUE_MSG(then_value_node != nullptr, RET_NULL_PTR, "create then_value_node return nullptr");
  auto else_value_node = NewValueNode(else_branch_graph);
  MS_CHECK_TRUE_MSG(else_value_node != nullptr, RET_NULL_PTR, "create else_value_node return nullptr");
  auto root_if_node = GetCNodeFromControlFlowNodesMap(if_node_name, control_nodes_map_);
  MS_CHECK_TRUE_MSG(root_if_node != nullptr, RET_ERROR, "can not find root_if_node is control_nodes_map");
  auto if_new_inputs = root_if_node->inputs();
  if_new_inputs.insert(if_new_inputs.begin() + 1, {then_value_node, else_value_node});

  std::vector<AnfNodePtr> if_new_input_not_same{};
  std::set<AnfNodePtr> if_set{};
  for (auto &input : if_new_inputs) {
    if (if_set.find(input) != if_set.end()) {
      continue;
    }
    if_new_input_not_same.push_back(input);
    if_set.insert(input);
  }

  root_if_node->set_inputs(if_new_input_not_same);
  return RET_OK;
}

STATUS OnnxModelParser::BuildCNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                   std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                   std::vector<AnfNodePtr> *graph_inputs, PrimitiveCPtr primitive_c,
                                   std::string loop_name) {
  MS_ASSERT(anf_graph != nullptr && anf_nodes_map != nullptr && graph_inputs != nullptr && primitive_c != nullptr);
  std::vector<AnfNodePtr> op_inputs;
  for (const auto &input_name : onnx_node.input()) {
    if (input_name.empty()) {
      continue;
    }

    if (anf_nodes_map->find(input_name) != anf_nodes_map->end()) {
      op_inputs.push_back(anf_nodes_map->at(input_name));
    } else {
      // subgraph may refer root graph nodes
      std::vector<CNodePtr> need_add_input_nodes;
      auto ext_subgraph_input = anf_graph->add_parameter();
      MS_CHECK_TRUE_MSG(ext_subgraph_input != nullptr, RET_NULL_PTR, "create parameter return nullptr");
      ParameterPtr inner_extra_paramter = nullptr;
      while (!loop_name.empty() && child_root_map_.find(loop_name) != child_root_map_.end()) {
        auto cur_node_map = control_nodes_map_[loop_name];
        CHECK_NULL_RETURN(cur_node_map);
        if (cur_node_map->find(input_name) != cur_node_map->end()) {
          auto outside_input_node = cur_node_map->at(input_name);
          CHECK_NULL_RETURN(outside_input_node);
          // copy outside input parameter value to inside subgraph
          ext_subgraph_input->set_abstract(outside_input_node->abstract());
          ext_subgraph_input->set_name(input_name);
          if (outside_input_node->isa<Parameter>()) {
            auto parameter = outside_input_node->cast<ParameterPtr>();
            if (!parameter->has_default()) {
              MS_LOG(ERROR) << "outside_input_node should has data.";
              return RET_ERROR;
            }
            auto tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
            auto copy_tensor_info = CreateTensorInfo(tensor_info->data_c(), tensor_info->Size(), tensor_info->shape(),
                                                     tensor_info->data_type());
            if (copy_tensor_info == nullptr) {
              MS_LOG(ERROR) << "memcpy failed.";
              return RET_ERROR;
            }
            ext_subgraph_input->set_default_param(copy_tensor_info);
          } else {
            // output inside cnode need make extra input
            graph_inputs->emplace_back(ext_subgraph_input);
            if (cur_node_map->find(loop_name) != cur_node_map->end()) {
              CHECK_NULL_RETURN(cur_node_map->at(loop_name));
              auto control_node = cur_node_map->at(loop_name)->cast<CNodePtr>();
              MS_ASSERT(control_node != nullptr);
              control_node->add_input(outside_input_node);
            } else {
              MS_LOG(ERROR) << "loop node: " << loop_name << " not found in cur node map.";
              return RET_ERROR;
            }
            for (auto &control_node : need_add_input_nodes) {
              CHECK_NULL_RETURN(control_node);
              auto func_graph = control_node->func_graph();
              auto extra_input_parameter = func_graph->add_parameter();
              MS_CHECK_TRUE_MSG(extra_input_parameter != nullptr, RET_NULL_PTR, "create parameter return nullptr");
              extra_input_parameter->set_name(input_name);
              extra_input_parameter->set_abstract(outside_input_node->abstract());
              control_node->add_input(extra_input_parameter);
            }
          }
          op_inputs.push_back(ext_subgraph_input);
          anf_nodes_map->emplace(input_name, ext_subgraph_input);
          break;
        } else {
          if (cur_node_map->find(loop_name) != cur_node_map->end()) {
            CHECK_NULL_RETURN(cur_node_map->at(loop_name));
            need_add_input_nodes.emplace_back(cur_node_map->at(loop_name)->cast<CNodePtr>());
          } else {
            MS_LOG(ERROR) << "loop node: " << loop_name << " not found in cur node map.";
            return RET_ERROR;
          }
          loop_name = child_root_map_[loop_name];
        }
      }
    }
  }
  auto new_cnode = anf_graph->NewCNode(primitive_c, op_inputs);
  if (new_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  new_cnode->set_fullname_with_scope(onnx_node.name());
  auto status = BuildOpOutputs(onnx_node, anf_graph, anf_nodes_map, new_cnode);
  return status;
}

STATUS OnnxModelParser::ConvertOpQuantParams(const onnx::NodeProto &onnx_node, ops::PrimitiveCPtr primitive_c) {
  MS_ASSERT(primitive_c != nullptr);
  auto status = ParseQuantParam(onnx_node);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "parse quant param failed.";
    return RET_ERROR;
  }
  // set input tensors
  std::map<int, std::vector<schema::QuantParamT>> input_quant_params;
  size_t idx = 0;
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    const auto &input_name = onnx_node.input(i);
    std::vector<schema::QuantParamT> quant_params;
    status = SetTensorQuantParam(input_name, &quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set input tensor quant param failed.";
      return status;
    }
    if (!quant_params.empty()) {
      input_quant_params.insert({idx, quant_params});
      idx++;
    }
  }
  // set out tensors
  idx = 0;
  std::map<int, std::vector<schema::QuantParamT>> output_quant_params;
  for (int i = 0; i < onnx_node.output_size(); ++i) {
    const auto &output_name = onnx_node.output(i);
    std::vector<schema::QuantParamT> quant_params;
    status = SetTensorQuantParam(output_name, &quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set output tensor quant param failed.";
      return status;
    }
    if (!quant_params.empty()) {
      output_quant_params.insert({idx, quant_params});
      idx++;
    }
  }
  if (!input_quant_params.empty() || !output_quant_params.empty()) {
    auto quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MSLITE_CHECK_PTR(quant_params_holder);
    for (auto &iter : input_quant_params) {
      quant_params_holder->set_input_quant_param(iter.first, iter.second);
    }
    for (auto &iter : output_quant_params) {
      quant_params_holder->set_output_quant_param(iter.first, iter.second);
    }
    primitive_c->AddAttr("quant_params", quant_params_holder);
  }
  return RET_OK;
}

STATUS OnnxModelParser::ParseQuantParam(const onnx::NodeProto &onnx_node) {
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "Y_scale") {
      float scale = onnx_node_attr.f();
      if (BuildParameterNodeForQuantParam(&scale, "scale_" + onnx_node.output(0), kNumberTypeFloat32) != RET_OK) {
        MS_LOG(ERROR) << "parse quant param failed.";
        return RET_ERROR;
      }
    } else if (onnx_node_attr.name() == "Y_zero_point") {
      int64_t zero_point = onnx_node_attr.i();
      if (BuildParameterNodeForQuantParam(&zero_point, "zero_point_" + onnx_node.output(0), kNumberTypeInt64) !=
          RET_OK) {
        MS_LOG(ERROR) << "parse quant param failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::SetTensorQuantParam(const std::string &tensor_name, std::vector<QuantParamT> *quant_params) {
  MS_ASSERT(quant_params != nullptr);
  quant_params->clear();
  auto quant_param = std::make_unique<QuantParamT>();
  MS_CHECK_TRUE_MSG(quant_param != nullptr, RET_NULL_PTR, "create QuantParamT return nullptr");
  for (int i = 0; i < onnx_root_graph_.quantization_annotation_size(); ++i) {
    auto tensor_annotation = onnx_root_graph_.quantization_annotation(i);
    if (!tensor_annotation.has_tensor_name() || tensor_annotation.tensor_name() != tensor_name) {
      continue;
    }
    for (const auto &item : tensor_annotation.quant_parameter_tensor_names()) {
      if (!item.has_key() || !item.has_value()) {
        continue;
      }

      const auto &quant_tensor_name = item.value();
      if (item.key() == "SCALE_TENSOR") {
        auto status = CopyTensorQuantParam(quant_tensor_name, quant_param.get(), true);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "quant param scale get failed";
          return status;
        }
      } else if (item.key() == "ZERO_POINT_TENSOR") {
        auto status = CopyTensorQuantParam(quant_tensor_name, quant_param.get(), false);
        if (status != RET_OK) {
          MS_LOG(ERROR) << "quant param zero_point get failed";
          return status;
        }
      }
    }
    break;
  }
  if (quant_param->inited) {
    quant_params->push_back(*std::move(quant_param));
    return RET_OK;
  }
  return SetTensorQuantParamFromNode(tensor_name, quant_params);
}

STATUS OnnxModelParser::SetTensorQuantParamFromNode(const std::string &tensor_name,
                                                    std::vector<QuantParamT> *quant_params) {
  MS_ASSERT(quant_params != nullptr);
  quant_params->clear();
  auto quant_param = std::make_unique<QuantParamT>();
  MS_CHECK_TRUE_MSG(quant_param != nullptr, RET_NULL_PTR, "create QuantParamT return nullptr");
  if (OnnxNodeParser::opset_version() <= 15) {
    quant_param->multiplier = 0;
  }
  std::string quant_tensor_name = "scale_" + tensor_name;
  auto status = CopyTensorQuantParam(quant_tensor_name, quant_param.get(), true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "quant param scale get failed";
    return status;
  }
  quant_tensor_name = "zero_point_" + tensor_name;
  status = CopyTensorQuantParam(quant_tensor_name, quant_param.get(), false);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "quant param zero_point get failed";
    return status;
  }
  if (quant_param->inited) {
    quant_params->push_back(*std::move(quant_param));
  }
  return RET_OK;
}

STATUS OnnxModelParser::CopyTensorQuantParam(const std::string &tensor_name, QuantParamT *quant_param,
                                             bool scale_or_not) {
  MS_ASSERT(quant_param != nullptr);
  auto iter = anf_nodes_map_.find(tensor_name);
  if (iter == anf_nodes_map_.end()) {
    MS_LOG(DEBUG) << "has no quant param";
    return RET_OK;
  }
  if (!utils::isa<ParameterPtr>(iter->second)) {
    MS_LOG(ERROR) << "quant param get failed";
    return RET_ERROR;
  }
  auto quant_parameter_node = iter->second->cast<ParameterPtr>();
  MS_ASSERT(quant_parameter_node != nullptr);
  if (!quant_parameter_node->has_default()) {
    MS_LOG(ERROR) << "quant param get failed";
    return RET_ERROR;
  }
  auto tensor_info = quant_parameter_node->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "parameterNode's default param is not tensor::TensorPtr";
    return RET_ERROR;
  }
  if (tensor_info->data_c() == nullptr) {
    MS_LOG(ERROR) << "parameterNode's default param has no data";
    return RET_ERROR;
  }
  if (scale_or_not) {
    quant_param->scale = *reinterpret_cast<float *>(tensor_info->data_c());
    quant_param->inited = true;
  } else {
    quant_param->zeroPoint = *reinterpret_cast<int64_t *>(tensor_info->data_c());
    quant_param->inited = true;
  }
  return RET_OK;
}

STATUS OnnxModelParser::AddTensorListStackNode(const AnfNodePtr &root_while_node, const onnx::NodeProto &onnx_node,
                                               int act_outputs_num, int body_output_size) {
  MS_ASSERT(root_while_node != nullptr);
  auto &loop_node_name = onnx_node.name();
  auto root_anf_graph = root_while_node->func_graph();
  auto stack_elem_node = CreateConstParamter(root_anf_graph, -1);
  MS_CHECK_TRUE_MSG(stack_elem_node != nullptr, RET_NULL_PTR, "create const parameter return nullptr");
  stack_elem_node->set_name(loop_node_name + "_element_shape");
  for (int j = 0; j < act_outputs_num; j++) {
    auto output_size = onnx_node.output_size();
    auto &loop_output_name = onnx_node.output(output_size - act_outputs_num + j);
    MS_ASSERT(control_nodes_map_.find(loop_node_name) != control_nodes_map_.end());
    MS_ASSERT(control_nodes_map_[loop_node_name]->find(loop_output_name) != control_nodes_map_[loop_node_name]->end());
    auto &while_output_node = control_nodes_map_[loop_node_name]->at(loop_output_name);
    MS_CHECK_TRUE_MSG(while_output_node != nullptr, RET_ERROR, "can not find while_output_node is control_nodes_map");
    auto tensor_list_stack_prim = std::make_shared<ops::TensorListStack>();
    if (tensor_list_stack_prim == nullptr) {
      MS_LOG(ERROR) << "create stack failed";
      return RET_ERROR;
    }
    tensor_list_stack_prim->set_num_elements(-1);
    auto prim_c = tensor_list_stack_prim->GetPrim();
    MS_CHECK_TRUE_RET(prim_c != nullptr, RET_ERROR);
    auto stack_value_node = NewValueNode(prim_c);
    MS_CHECK_TRUE_MSG(stack_value_node != nullptr, RET_NULL_PTR, "create stack_value_node return nullptr");
    std::vector<AnfNodePtr> stack_inputs = {stack_value_node, while_output_node, stack_elem_node};
    auto tensorlist_stack_cnode = root_anf_graph->NewCNode(stack_inputs);
    if (tensorlist_stack_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    tensorlist_stack_cnode->set_fullname_with_scope(loop_node_name + "_tensorlist_stack_node_" + std::to_string(j));
    tensorlist_stack_cnode->set_abstract(stack_elem_node->abstract());

    // update getitem value output index
    auto new_get_item_value = NewValueNode(MakeValue<int>(body_output_size - act_outputs_num + j));
    MS_CHECK_TRUE_MSG(new_get_item_value != nullptr, RET_NULL_PTR, "create new_get_item_value return nullptr");
    MS_ASSERT(while_output_node->cast<CNodePtr>() != nullptr);
    while_output_node->cast<CNodePtr>()->set_input(2, new_get_item_value);
    // insert tensorliststack after while_output
    (*control_nodes_map_[loop_node_name])[loop_output_name] = tensorlist_stack_cnode;
  }
  return RET_OK;
}

// onnx loop scan_output need through tensorlist op,while node need add new inputs
STATUS OnnxModelParser::AddTensorArrayEdge(const FuncGraphPtr &anf_graph, std::vector<AnfNodePtr> *return_new_inputs,
                                           const std::string &loop_node_name,
                                           std::vector<AnfNodePtr> *body_graph_inputs, int act_output_num) {
  MS_ASSERT(anf_graph != nullptr && return_new_inputs != nullptr && body_graph_inputs != nullptr);
  // body graph output is  trip_count,cond_count,loop_var,placeholder,scan_outputs
  auto root_while_node = GetCNodeFromControlFlowNodesMap(loop_node_name, control_nodes_map_);
  MS_CHECK_TRUE_MSG(root_while_node != nullptr, RET_ERROR, "can not find root_while_node is control_nodes_map");
  if (root_while_node == nullptr) {
    MS_LOG(ERROR) << "anf root node map cannot find loop node" << loop_node_name;
    return RET_ERROR;
  }
  auto anf_root_graph = root_while_node->func_graph();
  auto root_item_index_parameter = CreateConstParamter(anf_root_graph, 0);
  MS_CHECK_TRUE_MSG(root_item_index_parameter != nullptr, RET_NULL_PTR,
                    "create root_item_index_parameter return nullptr");
  root_item_index_parameter->set_name(loop_node_name + "_item_index");
  root_while_node->add_input(root_item_index_parameter);
  // fake parameter need pass by root while node input
  auto item_index_parameter = anf_graph->add_parameter();
  MS_CHECK_TRUE_MSG(item_index_parameter != nullptr, RET_NULL_PTR, "create item_index_parameter return nullptr");
  item_index_parameter->set_name(loop_node_name + "_item_index_2");
  item_index_parameter->set_abstract(root_item_index_parameter->abstract());
  body_graph_inputs->emplace_back(item_index_parameter);
  // item index++ edge
  auto add_value_node = CreateValueNode(schema::PrimitiveType_AddFusion);
  if (add_value_node == nullptr) {
    MS_LOG(ERROR) << "create add failed.";
    return RET_NULL_PTR;
  }
  auto add_one_input = CreateConstParamter(anf_graph, 1);
  MS_CHECK_TRUE_MSG(root_item_index_parameter != nullptr, RET_NULL_PTR, "create add_one_input return nullptr");
  add_one_input->set_name(loop_node_name + "_const_placeholder_1");
  std::vector<AnfNodePtr> add_inputs = {add_value_node, item_index_parameter, add_one_input};
  auto add_cnode = anf_graph->NewCNode(add_inputs);
  if (add_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  add_cnode->set_fullname_with_scope(loop_node_name + "item_index_add_node");
  add_cnode->set_abstract(root_item_index_parameter->abstract());
  // return node inputs will be trip_count,cond_out,loop_var,placeholder,tensorarray...
  if (static_cast<int>(return_new_inputs->size()) < act_output_num || act_output_num < 0) {
    MS_LOG(ERROR) << "act_output_num out of range of return_new_inputs";
    return RET_ERROR;
  }
  return_new_inputs->insert(return_new_inputs->end() - act_output_num, add_cnode);

  for (int i = 0; i < act_output_num; i++) {
    // tensor_array need as root while input
    auto while_tensor_array_input = anf_root_graph->add_parameter();
    MS_CHECK_TRUE_MSG(while_tensor_array_input != nullptr, RET_NULL_PTR,
                      "create while_tensor_array_input return nullptr");
    std::vector<int> tensor_list_data(kTensorListDatasize);
    tensor_list_data[kTypeIndex] = kTypeUnknown;
    tensor_list_data[kElementShapeIndex] = 0;
    tensor_list_data[kTensorsNumIndex] = -1;
    if (INT_MUL_OVERFLOW_THRESHOLD(tensor_list_data.size(), sizeof(int), SIZE_MAX)) {
      MS_LOG(ERROR) << "data_size overflow";
      return RET_ERROR;
    }
    auto tensor_info = CreateTensorInfo(tensor_list_data.data(), tensor_list_data.size() * sizeof(int),
                                        {static_cast<int64_t>(tensor_list_data.size())}, kObjectTypeTensorType);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "Create tensor info failed";
      return RET_ERROR;
    }
    auto abstract_tensor = tensor_info->ToAbstract();
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    while_tensor_array_input->set_abstract(abstract_tensor);
    while_tensor_array_input->set_default_param(tensor_info);
    while_tensor_array_input->set_name(loop_node_name + "_scan_outputs_tensorarray_while_input");
    root_while_node->add_input(while_tensor_array_input);

    auto subgraph_tensor_array_input = anf_graph->add_parameter();
    MS_CHECK_TRUE_MSG(subgraph_tensor_array_input != nullptr, RET_NULL_PTR,
                      "create subgraph_tensor_array_input return nullptr");
    subgraph_tensor_array_input->set_name(loop_node_name + "_scan_outputs_tensorarray_body_fg_input");
    subgraph_tensor_array_input->set_abstract(abstract_tensor);
    body_graph_inputs->emplace_back(subgraph_tensor_array_input);
    // skip trip_count ,cond_out,loop_var,no_loop_var,place_holder, output
    auto loop_output_idx = return_new_inputs->size() - act_output_num + i;
    auto loop_output_node = (*return_new_inputs)[loop_output_idx];
    auto set_item_value_node = CreateValueNode(schema::PrimitiveType_TensorListSetItem);
    if (set_item_value_node == nullptr) {
      MS_LOG(ERROR) << "create tensor list set item failed.";
      return RET_NULL_PTR;
    }
    std::vector<AnfNodePtr> set_item_inputs = {set_item_value_node, subgraph_tensor_array_input, item_index_parameter,
                                               loop_output_node};
    auto tensorlist_setitem_cnode = anf_graph->NewCNode(set_item_inputs);
    if (tensorlist_setitem_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    tensorlist_setitem_cnode->set_fullname_with_scope(loop_node_name + "_tensorlist_setitem_node");
    tensorlist_setitem_cnode->set_abstract(abstract_tensor);
    // loop output need replace by tensorliststack_output
    (*return_new_inputs)[loop_output_idx] = tensorlist_setitem_cnode;
  }

  return RET_OK;
}

STATUS OnnxModelParser::ConvertLoopOnnxNode(const onnx::NodeProto &onnx_node,
                                            std::unordered_map<std::string, AnfNodePtr> *anf_root_nodes_map,
                                            const std::string &root_node_name) {
  MS_ASSERT(anf_root_nodes_map != nullptr);
  for (int i = 0; i < onnx_node.attribute_size(); i++) {
    auto &attr = onnx_node.attribute(i);
    if (attr.name() != "body" || attr.type() != onnx::AttributeProto_AttributeType_GRAPH) {
      continue;
    }
    auto &subgraph_proto = attr.g();
    int cond_graph_input_num = -1;
    auto loop_body_graph = BuildBodyGraph(onnx_node, subgraph_proto, &cond_graph_input_num);
    MS_CHECK_TRUE_MSG(loop_body_graph != nullptr, RET_NULL_PTR, "create loop_body_graph return nullptr");
    auto root_while_node = GetCNodeFromControlFlowNodesMap(onnx_node.name(), control_nodes_map_);
    MS_CHECK_TRUE_MSG(root_while_node != nullptr, RET_ERROR, "can not find root_while_node");
    auto loop_cond_graph = BuildCondGraph(root_while_node, cond_graph_input_num, onnx_node.name() + "_cond_graph");
    MS_CHECK_TRUE_MSG(loop_cond_graph != nullptr, RET_NULL_PTR, "create loop_cond_graph return nullptr");
    all_subgraphs_.emplace_back(loop_body_graph);
    all_subgraphs_.emplace_back(loop_cond_graph);
    auto body_value_node = NewValueNode(loop_body_graph);
    MS_CHECK_TRUE_MSG(body_value_node != nullptr, RET_NULL_PTR, "create body_value_node return nullptr");
    auto inputs = root_while_node->inputs();
    auto cond_value_node = NewValueNode(loop_cond_graph);
    MS_CHECK_TRUE_MSG(cond_value_node != nullptr, RET_NULL_PTR, "create cond_value_node return nullptr");
    inputs.insert(inputs.begin() + 1, {cond_value_node, body_value_node});
    root_while_node->set_inputs(inputs);
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildParameterNodeForQuantParam(const void *data, const std::string &name, TypeId type) {
  MS_ASSERT(data != nullptr);
  if (type != kNumberTypeInt64 && type != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "quant param type don't support.";
    return RET_NOT_SUPPORT;
  }
  auto res_graph = ConvertGraph(res_graph_);
  auto parameter_node = res_graph->add_parameter();
  MS_CHECK_TRUE_MSG(parameter_node != nullptr, RET_NULL_PTR, "create parameter return nullptr");
  auto abstract_tensor = CreateTensorAbstract({}, type);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  parameter_node->set_abstract(abstract_tensor);
  parameter_node->set_name(name);
  int data_size = 0;
  if (type == kNumberTypeFloat32) {
    data_size = sizeof(float);
  } else {
    data_size = sizeof(int64_t);
  }
  auto tensor_info = CreateTensorInfo(data, data_size, {1}, type);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  parameter_node->set_default_param(tensor_info);
  anf_nodes_map_.emplace(name, parameter_node);
  return RET_OK;
}

REG_MODEL_PARSER(kFmkTypeOnnx, LiteModelParserCreator<OnnxModelParser>)
}  // namespace lite
}  // namespace mindspore
