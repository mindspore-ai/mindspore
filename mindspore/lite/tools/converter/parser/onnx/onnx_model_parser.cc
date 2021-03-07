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
#include <unordered_map>
#include <utility>
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "ops/return.h"
#include "ops/make_tuple.h"
#include "ops/tensor_list_stack.h"
#include "ops/tuple_get_item.h"
#include "ir/func_graph.h"
#include "src/param_value_lite.h"
#include "tools/converter/converter_flags.h"

namespace mindspore {
namespace lite {
static const std::unordered_map<int, mindspore::TypeId> TYPE_MAP = {
  {onnx::TensorProto_DataType_INT8, mindspore::kNumberTypeInt8},
  {onnx::TensorProto_DataType_UINT8, mindspore::kNumberTypeUInt8},
  {onnx::TensorProto_DataType_INT16, mindspore::kNumberTypeInt16},
  {onnx::TensorProto_DataType_INT32, mindspore::kNumberTypeInt32},
  {onnx::TensorProto_DataType_UINT32, mindspore::kNumberTypeUInt32},
  {onnx::TensorProto_DataType_INT64, mindspore::kNumberTypeInt64},
  {onnx::TensorProto_DataType_FLOAT16, mindspore::kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, mindspore::kNumberTypeFloat32},
  {onnx::TensorProto_DataType_BOOL, mindspore::kNumberTypeBool}};

std::set<std::string> SPECIAL_NODE = {"Gemm"};
FuncGraphPtr OnnxModelParser::Parse(const std::string &model_file, const std::string &weight_file,
                                    const QuantType &quant_type) {
  NoSupportOp::GetInstance()->SetFmkType("ONNX");
  anf_root_graph_ = std::make_shared<FuncGraph>();
  auto status = InitOriginModel(model_file);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "init origin model failed.";
    return nullptr;
  }

  status = ConvertOnnxGraph(onnx_root_graph_, anf_root_graph_, &anf_nodes_map_, {}, "root_node");
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert onnx graph failed.";
    return nullptr;
  }
  static auto root_func_manager = Manage(anf_root_graph_);
  for (auto &subgraph : all_subgraphs_) {
    subgraph->set_manager(root_func_manager);
    subgraph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_ONNX)));
  }
  anf_root_graph_->set_attr("graph_name", MakeValue("main_graph"));
  anf_root_graph_->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_ONNX)));
  return anf_root_graph_;
}

STATUS OnnxModelParser::InitOriginModel(const std::string &model_file) {
  auto status = ValidateFileStr(model_file, ".onnx");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.onnx";
    return status;
  }

  status = ReadProtoFromBinaryFile((const char *)model_file.c_str(), &onnx_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read onnx model file failed, model path: " << model_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return status;
  }
  OnnxNodeParser::set_opset_version(onnx_model_.opset_import().Get(0).version());
  onnx_root_graph_ = onnx_model_.graph();
  if (OnnxNodeParser::opset_version() > 15) {
    anf_root_graph_->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_ONNX)));
  } else {
    anf_root_graph_->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_ONNX_LOW_VERSION)));
  }
  return RET_OK;
}
STATUS OnnxModelParser::ConvertOnnxGraph(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                                         std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                         std::vector<AnfNodePtr> *extra_subgraph_inputs,
                                         const std::string &root_node_name) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "anf_graph is null";
    return RET_NULL_PTR;
  }
  STATUS status = RET_OK;
  status = ConvertConstTensors(onnx_graph, anf_graph, anf_nodes_map);
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
  return status;
}
STATUS OnnxModelParser::ConvertConstTensors(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                                            std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map) {
  if (func_graph_ptr == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  for (const auto &onnx_const_value : onnx_graph.initializer()) {
    auto parameter = func_graph_ptr->add_parameter();

    auto status = BuildParameterNode(parameter, onnx_const_value);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "parameter node build failed.";
      return status;
    }
    anf_nodes_map->emplace(onnx_const_value.name(), parameter);
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertGraphInputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &func_graph_ptr,
                                           std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map) {
  if (func_graph_ptr == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  for (int i = 0; i < onnx_graph.input().size(); ++i) {
    const auto &input_value = onnx_graph.input(i);
    if (anf_nodes_map->find(input_value.name()) != anf_nodes_map->end()) {
      continue;
    }
    auto parameter = func_graph_ptr->add_parameter();
    auto data_type =
      GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(input_value.type().tensor_type().elem_type()));
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "not support onnx data type "
                    << static_cast<onnx::TensorProto_DataType>(input_value.type().tensor_type().elem_type());
      return RET_ERROR;
    }
    auto type_ptr = TypeIdToType(data_type);
    std::vector<int64_t> shape_vector;
    auto onnx_shape = input_value.type().tensor_type().shape().dim();
    std::transform(onnx_shape.begin(), onnx_shape.end(), std::back_inserter(shape_vector),
                   [](const onnx::TensorShapeProto_Dimension &val) { return static_cast<int64_t>(val.dim_value()); });
    std::replace(shape_vector.begin(), shape_vector.end(), 0, -1);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(input_value.name());
    anf_nodes_map->emplace(input_value.name(), parameter);
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertNodes(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                                     std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                     std::vector<AnfNodePtr> *graph_inputs, const std::string &root_node_name) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  STATUS status = RET_OK;
  for (const auto &onnx_node : onnx_graph.node()) {
    auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser(onnx_node.op_type());
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
      status = status == RET_OK ? RET_NOT_FIND_OP : status;
      MS_LOG(ERROR) << "not support onnx data type " << onnx_node.op_type();
    }
    if (status != RET_OK) {
      continue;
    }

    MS_LOG(INFO) << "parse op:" << onnx_node.op_type();
    auto primitive_c = node_parser->Parse(onnx_graph, onnx_node);
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "parse node " << onnx_node.op_type() << " failed.";
      status = RET_ERROR;
      continue;
    }
    status = ConvertOpQuantParams(onnx_node, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert " << onnx_node.op_type() << " quant param failed.";
      continue;
    }
    if (IsSpecialOnnxNode(onnx_node)) {
      auto status_node = ConvertSpecialOnnxNode(onnx_node, anf_graph, anf_nodes_map, primitive_c);
      status = status == RET_OK ? status_node : status;
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
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
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
    anf_nodes_map[subgraph_proto.input(j).name()]->cast<ParameterPtr>()->set_name(subgraph_name + "_input_" +
                                                                                  std::to_string(j) + "_parameter");
  }
  for (size_t j = 0; j < subgraph_extra_inputs.size(); j++) {
    subgraph_extra_inputs[j]->cast<ParameterPtr>()->set_name(
      subgraph_name + "_input_" + std::to_string(j + subgraph_proto.input_size()) + "_parameter");
  }
  auto return_cnode = subgraph->get_return();
  std::vector<AnfNodePtr> return_act_inputs;
  int start_index = 0;
  if (subgraph_proto.output_size() > 1) {
    return_act_inputs = return_cnode->input(1)->cast<CNodePtr>()->inputs();
    start_index = 1;
  } else {
    return_act_inputs = {return_cnode->input(1)};
  }
  for (size_t j = start_index; j < return_act_inputs.size(); j++) {
    if (utils::isa<CNodePtr>(return_act_inputs[j])) {
      return_act_inputs[start_index]->cast<CNodePtr>()->set_fullname_with_scope(
        subgraph_name + "_output_" + std::to_string(j - start_index) + "_cnode");
    } else if (utils::isa<ParameterPtr>(return_act_inputs[start_index])) {
      return_act_inputs[j]->cast<ParameterPtr>()->set_name(subgraph_name + "_output_" +
                                                           std::to_string(j - start_index) + "_parameter");
    }
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertIfOnnxNode(const onnx::NodeProto &onnx_node,
                                          std::unordered_map<std::string, AnfNodePtr> *anf_root_nodes_map,
                                          const std::string &root_node_name) {
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
      auto status = ConvertIfSubgraph(subgraph_proto, then_branch_graph, subgraph_name, if_node_name, root_node_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build if node else branch failed.";
      }
    } else if (attr.name().find("else_branch") != std::string::npos) {
      subgraph_name = if_node_name + "_else_branch";
      else_branch_graph = std::make_shared<FuncGraph>();
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
  auto else_value_node = NewValueNode(else_branch_graph);
  auto root_if_node = control_nodes_map_.at(if_node_name)->at(if_node_name)->cast<CNodePtr>();
  auto if_new_inputs = root_if_node->inputs();
  if_new_inputs.insert(if_new_inputs.begin() + 1, {then_value_node, else_value_node});
  root_if_node->set_inputs(if_new_inputs);
  return RET_OK;
}

STATUS OnnxModelParser::ConvertGraphOutputs(const onnx::GraphProto &onnx_graph, const FuncGraphPtr &anf_graph,
                                            const std::unordered_map<std::string, AnfNodePtr> &anf_nodes_map) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  std::vector<AnfNodePtr> return_inputs;
  if (onnx_graph.output_size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
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
      if (nullptr == cnode) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_cnode = anf_graph->NewCNode(make_tuple_prim_ptr, make_tuple_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    return_inputs.emplace_back(make_tuple_cnode);
  } else {
    const auto &graph_out = onnx_graph.output(0);
    if (anf_nodes_map.find(graph_out.name()) == anf_nodes_map.end()) {
      MS_LOG(ERROR) << "graph output get failed.";
      return RET_ERROR;
    }
    auto cnode = anf_nodes_map.at(graph_out.name());
    if (nullptr == cnode) {
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

STATUS OnnxModelParser::BuildReturnNode(const FuncGraphPtr &anf_graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  auto returnPrim = std::make_shared<ops::Return>();
  if (returnPrim == nullptr) {
    MS_LOG(ERROR) << "new Return failed";
    return RET_NULL_PTR;
  }
  auto returnCnode = anf_graph->NewCNode(returnPrim, return_inputs);
  if (returnCnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  returnCnode->set_fullname_with_scope("Return");
  anf_graph->set_return(returnCnode);
  return RET_OK;
}
STATUS OnnxModelParser::BuildCNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                   std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                   std::vector<AnfNodePtr> *graph_inputs, ops::PrimitiveC *primitive_c,
                                   std::string loop_name) {
  if (primitive_c == nullptr || anf_graph == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr.";
    return RET_NULL_PTR;
  }
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
      ParameterPtr inner_extra_paramter = nullptr;
      while (!loop_name.empty() && child_root_map_.find(loop_name) != child_root_map_.end()) {
        auto cur_node_map = control_nodes_map_[loop_name];
        if (cur_node_map->find(input_name) != cur_node_map->end()) {
          auto outside_input_node = cur_node_map->at(input_name);
          // copy outside input parameter value to inside subgraph
          ext_subgraph_input->set_abstract(outside_input_node->abstract());
          ext_subgraph_input->set_name(input_name);
          if (outside_input_node->isa<Parameter>()) {
            auto param_value = outside_input_node->cast<ParameterPtr>()->default_param()->cast<ParamValueLitePtr>();
            auto copy_param_value = std::make_shared<ParamValueLite>();
            auto copy_data = new (std::nothrow) char[param_value->tensor_size()];
            if (copy_data == nullptr) {
              MS_LOG(ERROR) << "new char[] failed";
              return RET_MEMORY_FAILED;
            }
            auto ret =
              memcpy_s(copy_data, param_value->tensor_size(), param_value->tensor_addr(), param_value->tensor_size());
            if (ret != EOK) {
              delete[](copy_data);
              MS_LOG(ERROR) << "memcpy error: " << ret;
              return RET_ERROR;
            }
            copy_param_value->set_tensor_shape(param_value->tensor_shape());
            copy_param_value->set_format(param_value->format());
            copy_param_value->set_tensor_type(param_value->tensor_type());
            copy_param_value->SetTensorData(copy_data, param_value->tensor_size());
            ext_subgraph_input->set_default_param(copy_param_value);
          } else {
            // output inside cnode need make extra input
            graph_inputs->emplace_back(ext_subgraph_input);
            if (cur_node_map->find(loop_name) != cur_node_map->end()) {
              auto control_node = cur_node_map->at(loop_name)->cast<CNodePtr>();
              control_node->add_input(outside_input_node);
            } else {
              MS_LOG(ERROR) << "loop node: " << loop_name << " not found in cur node map.";
              return RET_ERROR;
            }
            for (auto &control_node : need_add_input_nodes) {
              auto func_graph = control_node->func_graph();
              auto extra_input_parameter = func_graph->add_parameter();
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
  auto new_cnode = anf_graph->NewCNode(std::shared_ptr<ops::PrimitiveC>(primitive_c), op_inputs);
  if (new_cnode == nullptr) {
    MS_LOG(ERROR) << "new cnode error";
    return RET_ERROR;
  }
  new_cnode->set_fullname_with_scope(onnx_node.op_type() + "_" + onnx_node.output(0));
  auto status = BuildOpOutputs(onnx_node, anf_graph, anf_nodes_map, new_cnode);
  return status;
}

STATUS OnnxModelParser::BuildOpOutputs(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                       std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                       const CNodePtr &cnode) {
  if (anf_graph == nullptr || cnode == nullptr) {
    MS_LOG(ERROR) << "parameter is null, get output tensor failed.";
    return RET_NULL_PTR;
  }
  if (onnx_node.output_size() == 1) {
    auto type_ptr = TypeIdToType(kNumberTypeFloat32);
    std::vector<int64_t> shape_vector;
    cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    anf_nodes_map->emplace(onnx_node.output(0), cnode);
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (const auto &output_name : onnx_node.output()) {
      std::vector<int64_t> shape_vector;
      auto type_ptr = TypeIdToType(kNumberTypeFloat32);
      abstract_list.emplace_back(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
      auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
      CNodePtr get_item_cnode = anf_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_getitem_" + std::to_string(op_idx));
      anf_nodes_map->emplace(output_name, get_item_cnode);
      op_idx++;
    }
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  anf_nodes_map->emplace(onnx_node.name(), cnode);
  return RET_OK;
}

STATUS OnnxModelParser::ConvertOpQuantParams(const onnx::NodeProto &onnx_node, ops::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null, get quant params failed.";
    return RET_NULL_PTR;
  }
  auto status = ParseQuantParam(onnx_node);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "parse quant param failed.";
    return RET_ERROR;
  }
  // set input tensors
  auto quant_params_holder = std::make_shared<QuantParamHolder>();
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    const auto &input_name = onnx_node.input(i);
    std::vector<schema::QuantParamT> quant_params;
    status = SetTensorQuantParam(input_name, &quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set input tensor quant param failed.";
      return status;
    }
    quant_params_holder->AddInputQuantParam(quant_params);
  }
  // set out tensors
  for (int i = 0; i < onnx_node.output_size(); ++i) {
    const auto &output_name = onnx_node.output(i);
    std::vector<schema::QuantParamT> quant_params;
    status = SetTensorQuantParam(output_name, &quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set output tensor quant param failed.";
      return status;
    }
    quant_params_holder->AddOutputQuantParam(quant_params);
  }
  primitive_c->AddAttr("quant_params", quant_params_holder);
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
  quant_params->clear();
  auto quant_param = std::make_unique<QuantParamT>();
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
  quant_params->clear();
  auto quant_param = std::make_unique<QuantParamT>();
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
  } else {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    *quant_params = notinited_quant_params;
  }
  return RET_OK;
}

STATUS OnnxModelParser::CopyTensorQuantParam(const std::string &tensor_name, QuantParamT *quant_param,
                                             bool scale_or_not) {
  if (quant_param == nullptr) {
    MS_LOG(ERROR) << "quant_param is nullptr";
    return RET_NULL_PTR;
  }
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
  if (!quant_parameter_node->has_default()) {
    MS_LOG(ERROR) << "quant param get failed";
    return RET_ERROR;
  }
  auto param_value_lite = quant_parameter_node->default_param()->cast<ParamValueLitePtr>();
  if (param_value_lite == nullptr) {
    MS_LOG(ERROR) << "parameterNode's default param is not paramValueLite";
    return RET_ERROR;
  }
  if (scale_or_not) {
    quant_param->scale = *reinterpret_cast<float *>(param_value_lite->tensor_addr());
    quant_param->inited = true;
  } else {
    quant_param->zeroPoint = *reinterpret_cast<int64_t *>(param_value_lite->tensor_addr());
    quant_param->inited = true;
  }
  return RET_OK;
}

ParameterPtr CreateConstParamter(const FuncGraphPtr &anf_graph, int val) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return nullptr;
  }
  auto const_node = anf_graph->add_parameter();
  auto const_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>());
  const_node->set_abstract(const_abstract);
  int *tensor_data = new (std::nothrow) int[1];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new int[] failed";
    return nullptr;
  }
  tensor_data[0] = val;
  auto param_value = std::make_shared<ParamValueLite>();
  param_value->set_tensor_shape({});
  param_value->SetTensorData(tensor_data, sizeof(int));
  const_node->set_default_param(param_value);
  return const_node;
}

ValueNodePtr CreateValueNode(const PrimitiveType &op_type) {
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
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  // trip_cout need -1 after every iteration
  auto sub_value_node = CreateValueNode(schema::PrimitiveType_SubFusion);
  if (sub_value_node == nullptr) {
    MS_LOG(ERROR) << "create sub failed.";
    return RET_NULL_PTR;
  }
  auto &trip_cout_paramter = anf_nodes_map.at(trip_cout_name);
  if (trip_cout_paramter == nullptr) {
    MS_LOG(ERROR) << "trip_cout_paramter found failed";
    return ERROR;
  }
  auto const_one_parameter = CreateConstParamter(anf_graph, 1);
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

STATUS OnnxModelParser::AddTensorListStackNode(const AnfNodePtr &root_while_node, const onnx::NodeProto &onnx_node,
                                               int act_outputs_num, int body_output_size) {
  auto &loop_node_name = onnx_node.name();
  auto root_anf_graph = root_while_node->func_graph();
  auto stack_elem_node = CreateConstParamter(root_anf_graph, -1);
  stack_elem_node->set_name(loop_node_name + "_element_shape");
  for (int j = 0; j < act_outputs_num; j++) {
    auto output_size = onnx_node.output_size();
    auto &loop_output_name = onnx_node.output(output_size - act_outputs_num + j);
    auto &while_output_node = control_nodes_map_[loop_node_name]->at(loop_output_name);
    auto tensor_list_stack_prim = std::make_shared<ops::TensorListStack>();
    if (tensor_list_stack_prim == nullptr) {
      MS_LOG(ERROR) << "create stack failed";
      return RET_ERROR;
    }
    tensor_list_stack_prim->set_num_elements(-1);
    auto stack_value_node = NewValueNode(tensor_list_stack_prim);
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
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  // body graph output is  trip_count,cond_count,loop_var,placeholder,scan_outputs
  auto root_while_node = control_nodes_map_[loop_node_name]->at(loop_node_name)->cast<CNodePtr>();
  if (root_while_node == nullptr) {
    MS_LOG(ERROR) << "anf root node map cannot find loop node" << loop_node_name;
    return RET_ERROR;
  }
  auto anf_root_graph = root_while_node->func_graph();
  auto root_item_index_parameter = CreateConstParamter(anf_root_graph, 0);
  root_item_index_parameter->set_name(loop_node_name + "_item_index");
  root_while_node->add_input(root_item_index_parameter);
  // fake parameter need pass by root while node input
  auto item_index_parameter = anf_graph->add_parameter();
  item_index_parameter->set_name(loop_node_name + "_item_index");
  item_index_parameter->set_abstract(root_item_index_parameter->abstract());
  body_graph_inputs->emplace_back(item_index_parameter);
  // item index++ edge
  auto add_value_node = CreateValueNode(schema::PrimitiveType_AddFusion);
  if (add_value_node == nullptr) {
    MS_LOG(ERROR) << "create add failed.";
    return RET_NULL_PTR;
  }
  auto add_one_input = CreateConstParamter(anf_graph, 1);
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
  return_new_inputs->insert(return_new_inputs->end() - act_output_num, add_cnode);

  for (int i = 0; i < act_output_num; i++) {
    // tensor_array need as root while input
    auto while_tensor_array_input = anf_root_graph->add_parameter();
    std::vector<int64_t> shape_vector;
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(kTensorType, shape_vector);
    auto param_value = std::make_shared<ParamValueLite>();
    param_value->set_tensor_type(kObjectTypeTensorType);
    while_tensor_array_input->set_abstract(abstract_tensor);
    while_tensor_array_input->set_default_param(param_value);
    while_tensor_array_input->set_name(loop_node_name + "_scan_outputs_tensorarray");
    root_while_node->add_input(while_tensor_array_input);

    auto subgraph_tensor_array_input = anf_graph->add_parameter();
    subgraph_tensor_array_input->set_name(loop_node_name + "_scan_outputs_tensorarray");
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
  auto node_inputs_num = onnx_node.input_size();
  auto node_outputs_num = onnx_node.output_size();
  // skip trip_cout and cond input,scan_output nums
  auto act_outputs_num = node_outputs_num - (node_inputs_num - 2);
  for (int i = 0; i < onnx_node.attribute_size(); i++) {
    auto &attr = onnx_node.attribute(i);
    if (attr.name() != "body" || attr.type() != onnx::AttributeProto_AttributeType_GRAPH) {
      continue;
    }
    auto &subgraph_proto = attr.g();
    auto loop_body_graph = std::make_shared<FuncGraph>();
    std::unordered_map<std::string, AnfNodePtr> anf_nodes_map;
    std::vector<AnfNodePtr> gen_subgraph_inputs;
    auto status =
      ConvertOnnxGraph(subgraph_proto, loop_body_graph, &anf_nodes_map, &gen_subgraph_inputs, onnx_node.name());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert loop OnnxGraph ";
      return status;
    }
    // while node add outside_input
    auto &loop_node_name = onnx_node.name();
    // update body graph input node

    auto return_tuple_cnode = loop_body_graph->get_return()->input(1)->cast<CNodePtr>();
    auto return_new_inputs = return_tuple_cnode->inputs();
    return_new_inputs.insert(return_new_inputs.end() - act_outputs_num, gen_subgraph_inputs.begin(),
                             gen_subgraph_inputs.end());

    std::string max_trip_count_name = subgraph_proto.input(0).name();
    status =
      AddIterNumsUpdateEdge(loop_body_graph, &return_new_inputs, anf_nodes_map, max_trip_count_name, loop_node_name);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "add iter nums update edge failed";
      return status;
    }
    auto root_while_node = control_nodes_map_[loop_node_name]->at(loop_node_name)->cast<CNodePtr>();
    std::vector<AnfNodePtr> body_graph_inputs;
    for (int j = 0; j < subgraph_proto.input_size(); j++) {
      body_graph_inputs.emplace_back(anf_nodes_map[subgraph_proto.input(j).name()]);
    }
    body_graph_inputs.insert(body_graph_inputs.end(), gen_subgraph_inputs.begin(), gen_subgraph_inputs.end());
    if (act_outputs_num != 0) {
      status =
        AddTensorArrayEdge(loop_body_graph, &return_new_inputs, loop_node_name, &body_graph_inputs, act_outputs_num);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "add tensorarray update edge failed";
        return status;
      }
      // insert tensorliststack after while output
      status = AddTensorListStackNode(root_while_node, onnx_node, act_outputs_num, body_graph_inputs.size());
      if (status != RET_OK) {
        MS_LOG(ERROR) << "add tensorliststack node failed";
        return status;
      }
    }
    return_tuple_cnode->set_inputs(return_new_inputs);
    auto loop_cond_graph = std::make_shared<FuncGraph>();
    auto cond_graph_name = loop_node_name + "_cond_graph";
    status = BuildCondGraph(loop_cond_graph, root_while_node, return_new_inputs.size() - 1, cond_graph_name);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "build cond graph failed";
      return status;
    }

    auto body_graph_name = loop_node_name + "_body_graph";
    for (size_t j = 0; j < body_graph_inputs.size(); j++) {
      body_graph_inputs[j]->cast<ParameterPtr>()->set_name(body_graph_name + "_input_" + std::to_string(j) +
                                                           "_parameter");
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
    loop_cond_graph->set_attr("graph_name", MakeValue(cond_graph_name));
    loop_body_graph->set_attr("graph_name", MakeValue(body_graph_name));
    all_subgraphs_.emplace_back(loop_cond_graph);
    all_subgraphs_.emplace_back(loop_body_graph);
    auto cond_value_node = NewValueNode(loop_cond_graph);
    auto body_value_node = NewValueNode(loop_body_graph);
    auto inputs = root_while_node->inputs();
    inputs.insert(inputs.begin() + 1, {cond_value_node, body_value_node});
    root_while_node->set_inputs(inputs);
  }
  return RET_OK;
}
STATUS OnnxModelParser::BuildCondGraph(const FuncGraphPtr &cond_graph, const AnfNodePtr &root_while_node,
                                       int inputs_num, const std::string &cond_graph_name) {
  if (cond_graph == nullptr || root_while_node == nullptr) {
    MS_LOG(ERROR) << "parameter has null.";
    return RET_NULL_PTR;
  }
  STATUS status = RET_OK;
  CNodePtr less_cnode = nullptr;
  for (int i = 0; i < inputs_num; i++) {
    auto input_paramter = cond_graph->add_parameter();
    input_paramter->set_name(cond_graph_name + "_input_" + std::to_string(i) + "_parameter");
    auto root_while_inputs = root_while_node->cast<CNodePtr>()->inputs();
    auto input_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>());
    input_paramter->set_abstract(input_abstract);
    if (i == 0) {
      auto zero_parameter = CreateConstParamter(cond_graph, 0);
      zero_parameter->set_name(root_while_node->fullname_with_scope() + "_const_0");
      auto less_value_node = CreateValueNode(schema::PrimitiveType_Less);
      std::vector<AnfNodePtr> less_inputs = {less_value_node, zero_parameter, input_paramter};
      less_cnode = cond_graph->NewCNode(less_inputs);
      if (less_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return RET_ERROR;
      }
      auto less_abstract = std::make_shared<abstract::AbstractTensor>(kBool, std::vector<int64_t>());
      less_cnode->set_abstract(less_abstract);
      less_cnode->set_fullname_with_scope(cond_graph_name + "_less_cnode");
    }
    if (i == 1) {
      auto and_value_node = CreateValueNode(schema::PrimitiveType_LogicalAnd);
      std::vector<AnfNodePtr> and_inputs = {and_value_node, less_cnode, input_paramter};
      auto and_cnode = cond_graph->NewCNode(and_inputs);
      if (and_cnode == nullptr) {
        MS_LOG(ERROR) << "new cnode error";
        return RET_ERROR;
      }
      and_cnode->set_abstract(less_cnode->abstract());
      and_cnode->set_fullname_with_scope(cond_graph_name + "_output_" + std::to_string(0) + "_cnode");
      status = BuildReturnNode(cond_graph, {and_cnode});
      if (status != RET_OK) {
        MS_LOG(ERROR) << "build return node failed.";
        return status;
      }
    }
  }
  return status;
}

STATUS OnnxModelParser::ConvertSpecialOnnxNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                               std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                               ops::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr || anf_graph == nullptr) {
    MS_LOG(ERROR) << "imitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  STATUS status = RET_OK;
  if (onnx_node.op_type() == "Gemm") {
    status = ConvertOnnxGemmNode(onnx_node, anf_graph, anf_nodes_map, primitive_c);
  } else {
    MS_LOG(ERROR) << "the node is not special node.";
    status = RET_ERROR;
  }
  delete primitive_c;
  return status;
}

STATUS OnnxModelParser::ConvertOnnxGemmNode(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                            std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                            ops::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr || anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has nullptr.";
    return RET_NULL_PTR;
  }
  if (onnx_node.op_type() != "Gemm") {
    MS_LOG(ERROR) << "this op is not gemm, it is " << onnx_node.op_type();
    return RET_ERROR;
  }
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  auto status = BuildCNodeForGemm(onnx_node, anf_graph, anf_nodes_map, primitive_c, "MatMul");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert gemm node failed.";
    return status;
  }
  status = BuildCNodeForGemm(onnx_node, anf_graph, anf_nodes_map, primitive_c, "BiasAdd");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert gemm node failed.";
    return status;
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildCNodeForGemm(const onnx::NodeProto &onnx_node, const FuncGraphPtr &anf_graph,
                                          std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map,
                                          ops::PrimitiveC *primitive_c, const std::string &name) {
  if (primitive_c == nullptr || anf_graph == nullptr) {
    MS_LOG(ERROR) << "parameter has nullptr.";
    return RET_NULL_PTR;
  }
  auto value = primitive_c->GetAttr(name);
  primitive_c->EraseAttr(name);
  if (value == nullptr) {
    MS_LOG(ERROR) << "op parse failed.";
    return RET_NULL_PTR;
  }
  auto prim_ptr = value->cast<std::shared_ptr<ops::PrimitiveC>>();
  if (prim_ptr == nullptr) {
    MS_LOG(ERROR) << "primitive parse failed.";
    return RET_NULL_PTR;
  }
  auto type_ptr = TypeIdToType(kTypeUnknown);
  std::vector<int64_t> shape_vector;
  std::vector<AnfNodePtr> op_inputs;
  auto quant_params_holder = std::make_shared<QuantParamHolder>();
  auto quant_params_holder_origin = primitive_c->GetAttr("quant_params")->cast<QuantParamHolderPtr>();
  if (name == "MatMul") {
    for (int i = 0; i < 2; ++i) {
      if (anf_nodes_map->find(onnx_node.input(i)) == anf_nodes_map->end()) {
        MS_LOG(ERROR) << "op " << onnx_node.op_type() << " inputs get failed.";
        return RET_ERROR;
      } else {
        op_inputs.push_back(anf_nodes_map->at(onnx_node.input(i)));
        quant_params_holder->AddInputQuantParam(quant_params_holder_origin->input_quant_params().at(i));
      }
    }
    quant_params_holder->AddOutputQuantParam(std::vector<schema::QuantParamT>(1));
    auto new_cnode = anf_graph->NewCNode(prim_ptr, op_inputs);
    if (new_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    new_cnode->set_fullname_with_scope("Gemm_MatMul_" + onnx_node.output(0));
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    anf_nodes_map->emplace("Gemm_MatMul_" + onnx_node.output(0), new_cnode);
  } else {
    if (anf_nodes_map->find("Gemm_MatMul_" + onnx_node.output(0)) == anf_nodes_map->end() ||
        anf_nodes_map->find(onnx_node.input(2)) == anf_nodes_map->end()) {
      MS_LOG(ERROR) << "op " << onnx_node.op_type() << " inputs get failed.";
      return RET_ERROR;
    }
    op_inputs.push_back(anf_nodes_map->at("Gemm_MatMul_" + onnx_node.output(0)));
    op_inputs.push_back(anf_nodes_map->at(onnx_node.input(2)));
    quant_params_holder->AddInputQuantParam(std::vector<schema::QuantParamT>(1));
    quant_params_holder->AddInputQuantParam(quant_params_holder_origin->input_quant_params().at(2));
    quant_params_holder->AddOutputQuantParam(quant_params_holder_origin->output_quant_params().front());
    auto new_cnode = anf_graph->NewCNode(prim_ptr, op_inputs);
    if (new_cnode == nullptr) {
      MS_LOG(ERROR) << "new cnode error";
      return RET_ERROR;
    }
    new_cnode->set_fullname_with_scope("Gemm_BiasAdd_" + onnx_node.output(0));
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    anf_nodes_map->emplace(onnx_node.output(0), new_cnode);
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildParameterNodeForQuantParam(const void *data, const std::string &name, TypeId type) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return RET_NULL_PTR;
  }
  if (type != kNumberTypeInt64 && type != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "quant param type don't support.";
    return RET_NOT_SUPPORT;
  }
  std::vector<int64_t> shape_vector;
  auto parameter_node = anf_root_graph_->add_parameter();
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), shape_vector);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "new abstract_tensor failed";
    return RET_MEMORY_FAILED;
  }
  parameter_node->set_abstract(abstract_tensor);
  parameter_node->set_name(name);
  std::vector<int> shape;
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "new param_value failed";
    return RET_MEMORY_FAILED;
  }
  param_value->set_tensor_shape(shape);
  param_value->set_format(schema::Format_NUM_OF_FORMAT);
  param_value->set_tensor_type(type);
  int data_size = 0;
  if (type == kNumberTypeFloat32) {
    data_size = sizeof(float);
  } else {
    data_size = sizeof(int64_t);
  }
  auto *tensor_data = new (std::nothrow) char[data_size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new char[] failed";
    return RET_MEMORY_FAILED;
  }
  if (memcpy_s(tensor_data, data_size, data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] tensor_data;
    return RET_ERROR;
  }
  param_value->SetTensorData(tensor_data, data_size);
  parameter_node->set_default_param(param_value);
  anf_nodes_map_.emplace(name, parameter_node);
  return RET_OK;
}

STATUS OnnxModelParser::BuildParameterNode(const ParameterPtr &parameter_node, const onnx::TensorProto &tensor) {
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "not support onnx data type " << static_cast<onnx::TensorProto_DataType>(tensor.data_type());
    return RET_ERROR;
  }
  auto type_ptr = TypeIdToType(data_type);
  std::vector<int64_t> shape_vector(tensor.dims().begin(), tensor.dims().end());
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  parameter_node->set_abstract(abstract_tensor);
  parameter_node->set_name(tensor.name());

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  std::vector<int> shape;
  std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                 [](const int64_t &value) { return static_cast<int>(value); });
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(data_type);
  param_value->set_format(schema::Format::Format_NCHW);
  auto status = CopyOnnxTensorData(tensor, param_value);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "copy data failed.";
    return status;
  }
  parameter_node->set_default_param(param_value);
  return RET_OK;
}

STATUS OnnxModelParser::CopyOnnxTensorData(const onnx::TensorProto &onnx_const_tensor,
                                           const ParamValueLitePtr &param_value_lite) {
  if (param_value_lite == nullptr) {
    MS_LOG(ERROR) << "param_value_lite is nullptr.";
    return RET_NULL_PTR;
  }
  size_t data_count = 1;
  if (!onnx_const_tensor.dims().empty()) {
    std::for_each(onnx_const_tensor.dims().begin(), onnx_const_tensor.dims().end(),
                  [&data_count](int dim) { data_count *= dim; });
  }
  size_t data_size = 0;
  const void *onnx_data = nullptr;
  auto data_type = GetDataTypeFromOnnx(static_cast<onnx::TensorProto_DataType>(onnx_const_tensor.data_type()));
  switch (data_type) {
    case kNumberTypeFloat32:
      data_size = data_count * sizeof(float);
      if (onnx_const_tensor.float_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.float_data().data();
      }
      break;
    case kNumberTypeInt32:
      data_size = data_count * sizeof(int);
      if (onnx_const_tensor.int32_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.int32_data().data();
      }
      break;
    case kNumberTypeInt64:
      data_size = data_count * sizeof(int64_t);
      if (onnx_const_tensor.int64_data_size() == 0) {
        onnx_data = onnx_const_tensor.raw_data().data();
      } else {
        onnx_data = onnx_const_tensor.int64_data().data();
      }
      break;
    case kNumberTypeUInt8:
    case kNumberTypeInt8:
    case kNumberTypeBool:
      data_size = data_count * sizeof(uint8_t);
      onnx_data = onnx_const_tensor.raw_data().data();
      break;
    default:
      MS_LOG(ERROR) << "unsupported data type " << data_type;
      return RET_ERROR;
  }
  if (data_size == 0) {
    return RET_OK;
  }
  if (onnx_data == nullptr) {
    MS_LOG(ERROR) << "origin data in onnx model is nullptr";
    return RET_MEMORY_FAILED;
  }
  char *param_data = new (std::nothrow) char[data_size];
  if (param_data == nullptr) {
    MS_LOG(ERROR) << "new char[] failed";
    return RET_MEMORY_FAILED;
  }
  if (memcpy_s(static_cast<void *>(param_data), data_size, onnx_data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    delete[] param_data;
    return RET_ERROR;
  }
  param_value_lite->SetTensorData(param_data, data_size);
  return RET_OK;
}

bool OnnxModelParser::IsSpecialOnnxNode(const onnx::NodeProto &onnx_node) {
  return SPECIAL_NODE.find(onnx_node.op_type()) != SPECIAL_NODE.end();
}

TypeId OnnxModelParser::GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type) {
  auto iter = TYPE_MAP.find(onnx_type);
  if (iter == TYPE_MAP.end()) {
    MS_LOG(ERROR) << "unsupported onnx data type: " << onnx_type;
    return kTypeUnknown;
  }
  return iter->second;
}
}  // namespace lite
}  // namespace mindspore
