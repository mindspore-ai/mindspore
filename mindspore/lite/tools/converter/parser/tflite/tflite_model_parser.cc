/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tflite/tflite_model_parser.h"
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <algorithm>
#include <utility>
#include "ops/primitive_c.h"
#include "ir/func_graph.h"
#include "src/common/file_utils.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/common/graph_util.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/parser/tflite/tflite_inputs_adjust.h"
#include "tools/converter/parser/parser_utils.h"

using mindspore::lite::converter::FmkType_TFLITE;
namespace mindspore::lite {
namespace {
constexpr size_t kMainGraphIndex = 0;
constexpr size_t kConvWeightIndex = 2;
}  // namespace
std::unique_ptr<tflite::ModelT> TfliteModelParser::ReadTfliteModel(const std::string &model_path) {
  size_t size = 0;
  tflite_model_buf_ = ReadFile(model_path.c_str(), &size);
  if (tflite_model_buf_ == nullptr) {
    MS_LOG(ERROR) << "the file buffer is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)tflite_model_buf_, size);
  if (!tflite::VerifyModelBuffer(verify)) {
    MS_LOG(ERROR) << "the buffer is invalid and fail to create graph";
    return nullptr;
  }
  return tflite::UnPackModel(tflite_model_buf_);
}

FuncGraphPtr TfliteModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file_;
  quant_type_ = flag.quant_type_;
  // load graph
  tflite_model_ = ReadTfliteModel(model_file);
  if (tflite_model_ == nullptr) {
    MS_LOG(ERROR) << "read tflite model failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  auto status = ConvertTfliteGraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert tflite graph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ControlFlowNodePostProcess();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Control flow node post process failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(res_graph_, &all_func_graphs);

  if ((status = CommonAnfAdjust(all_func_graphs)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  if ((status = Tflite2AnfAdjust(all_func_graphs)) != RET_OK) {
    MS_LOG(ERROR) << "Tflite2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  if ((status = WeightFormatTransform(res_graph_)) != RET_OK) {
    MS_LOG(ERROR) << "WeightFormatTransform failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  return res_graph_;
}
STATUS TfliteModelParser::WeightFormatTransform(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto conv_cnode = node->cast<CNodePtr>();
    if (!opt::CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        !opt::CheckPrimitiveType(node, opt::kPrimConv2DBackpropInputFusion) &&
        !opt::CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      continue;
    }
    MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
    auto weight_node = conv_cnode->input(kConvWeightIndex);
    MS_ASSERT(weight_node != nullptr);
    auto tensor_info = opt::GetTensorInfo(weight_node);
    auto status = HardCodeTflite(conv_cnode, tensor_info, graph);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "Format hard code failed: " << status << ", node: " << node->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::HardCodeTflite(const CNodePtr &conv_node, const tensor::TensorPtr &tensor_info,
                                         const FuncGraphPtr &graph) {
  MS_ASSERT(conv_cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Invalid anfnode, which don't have primitive.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
  schema::Format weight_dst_format = schema::Format::Format_KHWC;
  STATUS status = RET_OK;
  schema::Format weight_src_format = Format_NUM_OF_FORMAT;
  auto weight_node = conv_node->input(kConvWeightIndex);
  int64_t format = prim->GetAttr(ops::kFormat) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kFormat)) : 0;
  switch (quant_type_) {
    case QuantType_AwareTraining:
    case QuantType_PostTraining:
    case QuantType_WeightQuant:
    case QuantType_QUANT_NONE: {
      if (format == KHWC) {
        weight_src_format = schema::Format::Format_KHWC;
      } else if (opt::CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
        if (!is_depth_wise) {
          weight_src_format = schema::Format::Format_KHWC;
        } else {
          weight_src_format = schema::Format::Format_CHWK;
        }
      } else if (opt::CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
        weight_src_format = schema::Format::Format_CHWK;
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported quantType: " << EnumNameQuantType(quant_type_)
                    << ", node: " << conv_node->fullname_with_scope();
      return RET_ERROR;
    }
  }
  status = DoWeightFormatTransform(conv_node, weight_node, graph, weight_src_format, weight_dst_format);
  if (status != RET_OK) {
    return RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS TfliteModelParser::DoWeightFormatTransform(const CNodePtr &conv_node, const AnfNodePtr &weight_node,
                                                  const FuncGraphPtr &graph, schema::Format weight_src_format,
                                                  schema::Format weight_dst_format) {
  if (utils::isa<CNodePtr>(weight_node)) {
    auto status =
      HandleWeightConst(graph, conv_node, weight_node->cast<CNodePtr>(), weight_src_format, weight_dst_format);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "handle weight-const failed.";
      return RET_ERROR;
    }
  }
  auto weight_value = opt::GetTensorInfo(weight_node);
  if (weight_value != nullptr) {
    auto status = opt::TransFilterFormat(weight_value, weight_src_format, weight_dst_format);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(schema::EnumValuesFormat()[weight_dst_format]) << "To"
                    << EnumNameFormat(weight_dst_format) << " failed, node : " << conv_node->fullname_with_scope()
                    << "quant type:" << quant_type_;
      return RET_ERROR;
    }
    auto type_id = static_cast<TypeId>(weight_value->data_type());
    auto shape = weight_value->shape();
    std::vector<int64_t> shape_vector(shape.begin(), shape.end());
    auto abstract = lite::CreateTensorAbstract(shape_vector, type_id);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    weight_node->set_abstract(abstract);
  }
  if (utils::isa<ParameterPtr>(weight_node)) {
    auto status =
      HandleWeightSharing(graph, KHWC, weight_node->cast<ParameterPtr>(), weight_src_format, weight_dst_format);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "handle weight-sharing failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertTfliteGraph() {
  auto subgraph_num = tflite_model_->subgraphs.size();
  for (size_t idx = 0; idx < subgraph_num; idx++) {
    std::unordered_map<int, AnfNodePtr> anf_node_map;
    const auto &tflite_subgraph = tflite_model_->subgraphs.at(idx);
    const auto subgraph_name = tflite_subgraph->name;
    // build function graph
    FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
    auto type_value = MakeValue(static_cast<int>(converter::FmkType_TFLITE));
    MS_ASSERT(type_value != nullptr);
    func_graph->set_attr("fmk", type_value);
    auto attr_value = MakeValue(subgraph_name);
    MS_ASSERT(attr_value != nullptr);
    func_graph->set_attr("graph_name", attr_value);

    auto status = ConvertGraphInputs(tflite_subgraph, func_graph, &anf_node_map);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert subgraph inputs failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }

    status = ConvertOps(tflite_subgraph, func_graph, &anf_node_map);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert ops failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }

    status = ConvertGraphOutputs(tflite_subgraph, func_graph, &anf_node_map);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert graph outputs failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }

    // record the function graph
    if (idx == kMainGraphIndex) {
      res_graph_ = func_graph;
    } else {
      status = BuildSubFuncGraphMap(idx, func_graph, subgraph_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Fail to build the map from CNode to FuncGraph.";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

std::string GetTensorName(size_t index, const tflite::BuiltinOperator &op_type, const std::string &op_name) {
  std::string tensor_name = op_name + "/input-" + std::to_string(index);
  if (op_type == tflite::BuiltinOperator_CONV_2D || op_type == tflite::BuiltinOperator_TRANSPOSE_CONV ||
      op_type == tflite::BuiltinOperator_DEPTHWISE_CONV_2D || op_type == tflite::BuiltinOperator_FULLY_CONNECTED) {
    if (index == 1) {
      tensor_name = op_name + "/weight";
    }
    if (index == 2) {
      tensor_name = op_name + "/bias";
    }
  }
  return tensor_name;
}

STATUS TfliteModelParser::ConvertOps(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const FuncGraphPtr &func_graph,
                                     std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  NotSupportOp::GetInstance()->set_fmk_type("TFLITE");
  STATUS status = RET_OK;
  int op_idx = 0;
  for (auto &op : tflite_subgraph->operators) {
    auto tflite_op_type = (tflite_model_->operator_codes[op->opcode_index])->builtin_code;
    std::string op_type = tflite::EnumNameBuiltinOperator(tflite_op_type);
    std::string op_name = op_type + "-" + std::to_string(op_idx);
    op_idx++;
    // parse primitive
    MS_LOG(INFO) << "parse node :" << op_name;
    auto node_parser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(tflite_op_type);
    if (node_parser == nullptr) {
      NotSupportOp::GetInstance()->InsertOp(op_type);
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      MS_LOG(ERROR) << "Can not find " << op_type << " op parser.";
      continue;
    }
    if (status != RET_OK) {
      continue;
    }

    std::vector<AnfNodePtr> op_inputs;
    auto primitive_c = node_parser->Parse(op, tflite_subgraph, tflite_model_);
    if (primitive_c != nullptr) {
      op_inputs = {NewValueNode(std::shared_ptr<ops::PrimitiveC>(primitive_c))};
    } else {
      MS_LOG(ERROR) << "parse failed for node: " << op_name;
      return RET_ERROR;
    }

    status = ConvertOpQuantParams(op, tflite_subgraph, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert " << op_name << " quant param failed.";
      continue;
    }

    // parse inputs
    for (int i = 0; i < static_cast<int>(op->inputs.size()); i++) {
      auto input_idx = op->inputs.at(i);
      if (tflite_op_type == tflite::BuiltinOperator_FULLY_CONNECTED && input_idx == -1) {
        continue;
      }
      if (input_idx < 0) {
        input_idx += tflite_subgraph->tensors.size();
      }
      const auto &input_tensor = tflite_subgraph->tensors[input_idx];
      if (anf_node_map->find(input_idx) != anf_node_map->end()) {
        op_inputs.emplace_back(anf_node_map->at(input_idx));
        continue;
      }

      // const tensor
      std::string tensor_name;
      if (!input_tensor->name.empty()) {
        tensor_name = input_tensor->name;
      } else {
        tensor_name = GetTensorName(i, tflite_op_type, op_name);
      }
      auto parameter = func_graph->add_parameter();
      status = ConvertConstTensor(input_tensor, parameter, tensor_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "convert " << op_name << " node: " << input_idx << " const node failed.";
        continue;
      }
      parameter->set_name(tensor_name);
      op_inputs.emplace_back(parameter);
      anf_node_map->insert(std::pair(input_idx, parameter));
    }
    auto new_cnode = func_graph->NewCNode(op_inputs);
    if (new_cnode == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    new_cnode->set_fullname_with_scope(op_name);
    status = ProcessControlFlowOp(op, new_cnode, op_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "ProcessControlFlowOp failed.";
      return RET_ERROR;
    }

    // parse outputs
    status = ConvertOutputTensor(tflite_subgraph, func_graph, op, new_cnode, anf_node_map);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert output tensors for " << new_cnode->fullname_with_scope() << " failed.";
      continue;
    }
  }
  return status;
}

STATUS TfliteModelParser::ProcessControlFlowOp(const std::unique_ptr<tflite::OperatorT> &op, const CNodePtr &anf_node,
                                               const std::string &op_type) {
  if (op_type == "WHILE") {
    const auto &tflite_attr = op->builtin_options.AsWhileOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get While attr failed";
      return RET_ERROR;
    }
    auto cnode = anf_node->cast<CNodePtr>();
    control_flow_nodes_[tflite_attr->cond_subgraph_index] = cnode;
    control_flow_nodes_[tflite_attr->body_subgraph_index] = cnode;
  } else if (op_type == "IF") {
    const auto &tflite_attr = op->builtin_options.AsIfOptions();
    if (tflite_attr == nullptr) {
      MS_LOG(ERROR) << "get If attr failed";
      return RET_ERROR;
    }
    auto cnode = anf_node->cast<CNodePtr>();
    control_flow_nodes_[tflite_attr->then_subgraph_index] = cnode;
    control_flow_nodes_[tflite_attr->else_subgraph_index] = cnode;
  }
  return RET_OK;
}

STATUS TfliteModelParser::SetTensorQuantParam(const std::unique_ptr<tflite::TensorT> &tflite_tensor,
                                              std::vector<QuantParamT> *quant_params, int round_type) {
  if (tflite_tensor == nullptr) {
    MS_LOG(ERROR) << "tflite_tensor is null, set tensor quant params failed.";
    return RET_NULL_PTR;
  }
  quant_params->clear();

  if (tflite_tensor->quantization == nullptr ||
      (tflite_tensor->quantization->scale.empty() && tflite_tensor->quantization->zero_point.empty() &&
       tflite_tensor->quantization->min.empty() && tflite_tensor->quantization->max.empty())) {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    *quant_params = notinited_quant_params;
    return RET_OK;
  }

  for (size_t i = 0; i < tflite_tensor->quantization->scale.size(); i++) {
    std::unique_ptr<schema::QuantParamT> quant_param = std::make_unique<QuantParamT>();
    if (quant_param == nullptr) {
      MS_LOG(ERROR) << "new quant_param failed";
      return RET_NULL_PTR;
    }

    if (!tflite_tensor->quantization->scale.empty()) {
      quant_param->scale = tflite_tensor->quantization->scale[i];
    }

    if (!tflite_tensor->quantization->zero_point.empty()) {
      quant_param->zeroPoint = tflite_tensor->quantization->zero_point[i];
    }

    if (!tflite_tensor->quantization->min.empty()) {
      quant_param->min = tflite_tensor->quantization->min[i];
    }

    if (!tflite_tensor->quantization->max.empty()) {
      quant_param->max = tflite_tensor->quantization->max[i];
    }
    quant_param->dstDtype = GetTfliteDataType(tflite_tensor->type);
    quant_param->inited = true;
    quant_param->roundType = round_type;
    quant_param->multiplier = 1;
    quant_params->emplace_back(*std::move(quant_param));
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertOpQuantParams(const std::unique_ptr<tflite::OperatorT> &op,
                                               const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                               ops::PrimitiveC *primitive_c) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "tflite op is null, get quant params failed.";
    return RET_NULL_PTR;
  }
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null, get quant params failed.";
    return RET_NULL_PTR;
  }
  int round_type = 1;
  if (primitive_c->name() == "Conv2D" || primitive_c->name() == "Conv2DFusion") {
    round_type = 2;
  }
  auto quant_params_holder = std::make_shared<QuantParamHolder>(op->inputs.size(), op->outputs.size());
  if (quant_params_holder == nullptr) {
    MS_LOG(ERROR) << "create QuantParamHolder failed";
    return RET_ERROR;
  }
  size_t idx = 0;
  for (auto input_idx : op->inputs) {
    if (input_idx < 0) {
      input_idx += tflite_subgraph->tensors.size();
    }
    const auto &input_tensor = tflite_subgraph->tensors[input_idx];
    std::vector<schema::QuantParamT> quant_params;
    auto status = SetTensorQuantParam(input_tensor, &quant_params, round_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set input tensor quant param failed.";
      return status;
    }
    quant_params_holder->set_input_quant_param(idx, quant_params);
    idx++;
  }
  idx = 0;
  for (auto output_idx : op->outputs) {
    if (output_idx < 0) {
      output_idx += tflite_subgraph->tensors.size();
    }
    const auto &output_tensor = tflite_subgraph->tensors.at(output_idx);
    std::vector<schema::QuantParamT> quant_params;
    auto status = SetTensorQuantParam(output_tensor, &quant_params, round_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set output tensor quant param failed.";
      return status;
    }
    quant_params_holder->set_output_quant_param(idx, quant_params);
    idx++;
  }
  primitive_c->AddAttr("quant_params", quant_params_holder);
  return RET_OK;
}

STATUS TfliteModelParser::ConvertGraphInputs(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                             const FuncGraphPtr &func_graph,
                                             std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  for (size_t i = 0; i < tflite_subgraph->inputs.size(); i++) {
    auto tflite_graph_input = tflite_subgraph->inputs.at(i);
    if (tflite_graph_input < 0) {
      tflite_graph_input = tflite_graph_input + tflite_subgraph->tensors.size();
    }
    auto parameter = func_graph->add_parameter();
    const auto &tensor = tflite_subgraph->tensors.at(tflite_graph_input);
    std::vector<int64_t> shape_vector = ConverterContext::GetInstance()->GetGraphInputTensorShape(tensor->name);
    if (ConverterContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape_vector.empty()) {
      MS_LOG(WARNING) << "Can not find name in map. name is " << tensor->name;
    }
    if (shape_vector.empty()) {
      (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                           [](const int32_t &value) { return static_cast<int64_t>(value); });
    }
    auto dtype = GetTfliteDataType(tensor->type);
    auto abstract_tensor = CreateTensorAbstract(shape_vector, dtype);
    if (abstract_tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    parameter->set_abstract(abstract_tensor);
    if (tflite_subgraph == tflite_model_->subgraphs.front()) {
      parameter->set_name(tensor->name);
      ConverterContext::GetInstance()->AddGraphInputTensorNames(tensor->name);
    } else {
      parameter->set_name(tflite_subgraph->name + "_input_" + std::to_string(i) + "_parameter");
    }
    anf_node_map->insert(std::pair(tflite_graph_input, parameter));
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertGraphOutputs(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                              const FuncGraphPtr &func_graph,
                                              std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  std::vector<AnfNodePtr> output_nodes;
  if (tflite_subgraph->outputs.size() > 1) {
    for (auto output_node : tflite_subgraph->outputs) {
      auto output_idx = output_node < 0 ? output_node + tflite_subgraph->tensors.size() : output_node;
      auto cnode = anf_node_map->at(output_idx);
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      output_nodes.emplace_back(cnode);
    }
    auto make_tuple_prim_ptr = std::make_shared<lite::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    std::vector<AnfNodePtr> make_tuple_inputs = output_nodes;
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = func_graph->NewCNode(make_tuple_inputs);
    if (make_tuple_cnode == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    make_tuple_cnode->set_fullname_with_scope("return_tuple");
    auto return_prim_ptr = std::make_shared<lite::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs{value_node};
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = func_graph->NewCNode(op_inputs);
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    cnode->set_fullname_with_scope("Return");
    func_graph->set_return(cnode);
  } else {
    auto returnPrim = std::make_shared<lite::Return>();
    if (returnPrim == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    int output_idx = tflite_subgraph->outputs.front() < 0
                       ? static_cast<int>(tflite_subgraph->outputs.front() + tflite_subgraph->tensors.size())
                       : static_cast<int>(tflite_subgraph->outputs.front());
    auto valueNode = NewValueNode(returnPrim);
    std::vector<AnfNodePtr> op_inputs{valueNode};
    auto cnode = anf_node_map->at(output_idx);
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    output_nodes.emplace_back(cnode);
    op_inputs.emplace_back(cnode);
    auto returnCnode = func_graph->NewCNode(op_inputs);
    if (returnCnode == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    returnCnode->set_fullname_with_scope("Return");
    func_graph->set_return(returnCnode);
  }

  if (tflite_subgraph == tflite_model_->subgraphs.front()) {
    // save original output tensor names.
    std::vector<std::string> output_names;
    auto output_idx = tflite_subgraph->outputs;
    std::transform(output_idx.begin(), output_idx.end(), std::back_inserter(output_names),
                   [&](auto out_idx) { return tflite_subgraph->tensors.at(out_idx)->name; });
    ConverterContext::GetInstance()->SetGraphOutputTensorNames(output_names);
  } else {
    // set output cnode name for subgraph
    for (size_t i = 0; i < output_nodes.size(); i++) {
      auto output_node = output_nodes.at(i);
      auto subgraph_name = tflite_subgraph->name;
      if (utils::isa<CNodePtr>(output_node)) {
        output_node->cast<CNodePtr>()->set_fullname_with_scope(subgraph_name + "_output_" + std::to_string(i) +
                                                               "_cnode");
      } else if (utils::isa<ParameterPtr>(output_node)) {
        output_node->cast<ParameterPtr>()->set_name(subgraph_name + "_output_" + std::to_string(i) + "_parameter");
      }
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::BuildSubFuncGraphMap(size_t subgraph_idx, const FuncGraphPtr &sub_func_graph,
                                               const std::string &subgraph_name) {
  auto control_flow_node = control_flow_nodes_.at(subgraph_idx);
  if (opt::CheckPrimitiveType(control_flow_node, prim::kPrimWhile)) {
    if (subgraph_name.find("cond") != std::string::npos) {
      control_flow_map_[control_flow_node].first = sub_func_graph;
    } else if (subgraph_name.find("body") != std::string::npos) {
      control_flow_map_[control_flow_node].second = sub_func_graph;
    }
  } else if (opt::CheckPrimitiveType(control_flow_node, prim::kPrimIf)) {
    if (subgraph_name.find("then") != std::string::npos) {
      control_flow_map_[control_flow_node].first = sub_func_graph;
    } else if (subgraph_name.find("else") != std::string::npos) {
      control_flow_map_[control_flow_node].second = sub_func_graph;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported control flow subgraph type, name: " << subgraph_name;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TfliteModelParser::ControlFlowNodePostProcess() {
  if (control_flow_map_.empty()) {
    return RET_OK;
  }
  static auto root_func_manager = Manage(res_graph_);
  for (auto &node_vs_graph : control_flow_map_) {
    auto control_flow_node = node_vs_graph.first;
    auto sub_graphs = node_vs_graph.second;
    auto &first_sub_graph = sub_graphs.first;
    auto &second_sub_graph = sub_graphs.second;
    if (first_sub_graph == nullptr || second_sub_graph == nullptr) {
      MS_LOG(ERROR) << "Incomplete subgraph for op: " << control_flow_node->fullname_with_scope();
      return RET_ERROR;
    }
    first_sub_graph->set_manager(root_func_manager);
    second_sub_graph->set_manager(root_func_manager);
    auto first_value_node = NewValueNode(first_sub_graph);
    auto second_value_node = NewValueNode(second_sub_graph);
    auto inputs = control_flow_node->inputs();
    inputs.insert(inputs.begin() + 1, {first_value_node, second_value_node});
    auto new_node = res_graph_->NewCNode(inputs);  // must create new node, otherwise node_users won't update
    if (new_node == nullptr) {
      MS_LOG(ERROR) << "new node failed";
      return RET_ERROR;
    }
    new_node->set_abstract(control_flow_node->abstract()->Clone());
    new_node->set_fullname_with_scope(control_flow_node->fullname_with_scope());
    if (!root_func_manager->Replace(control_flow_node, new_node)) {
      MS_LOG(ERROR) << "replace new node failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertConstTensor(const std::unique_ptr<tflite::TensorT> &tensor,
                                             const ParameterPtr &parameter, const std::string &tensor_name) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is null, get const tensor failed.";
    return RET_NULL_PTR;
  }

  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is null, get const tensor failed.";
    return RET_NULL_PTR;
  }
  const auto &tflite_model_buffers = tflite_model_->buffers;
  auto type_id = GetTfliteDataType(tensor->type);
  std::vector<int64_t> shape_vector;

  const auto &data = tflite_model_buffers.at(tensor->buffer)->data;
  std::string shape_str;
  if (type_id == kObjectTypeString) {
    shape_str += std::to_string(tensor->shape.size()) + ",";
    for (auto &dim : tensor->shape) {
      shape_str += std::to_string(dim) + ",";
    }
    shape_vector = {static_cast<int64_t>(shape_str.size() + data.size())};
  } else {
    (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
  }

  auto tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, type_id);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "init tensor info failed";
    return RET_NULL_PTR;
  }
  if (!data.empty()) {
    auto tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    if (type_id == kObjectTypeString) {
      if (memcpy_s(tensor_data, shape_str.size(), shape_str.data(), shape_str.size()) != EOK) {
        MS_LOG(ERROR) << "memcpy failed.";
        return RET_ERROR;
      }
      if (memcpy_s(tensor_data + shape_str.size(), data.size(), data.data(), data.size()) != EOK) {
        MS_LOG(ERROR) << "memcpy failed.";
        return RET_ERROR;
      }
    } else {
      if (memcpy_s(tensor_data, tensor_info->Size(), data.data(), data.size()) != EOK) {
        MS_LOG(ERROR) << "memcpy failed.";
        return RET_ERROR;
      }
    }
  }
  auto status = InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed.";
    return RET_ERROR;
  }
  parameter->set_name(tensor_name);
  return RET_OK;
}

STATUS TfliteModelParser::ConvertOutputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                              const FuncGraphPtr &func_graph,
                                              const std::unique_ptr<tflite::OperatorT> &op, const CNodePtr &dst_cnode,
                                              std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null, get output tensor failed.";
    return RET_NULL_PTR;
  }
  if (dst_cnode == nullptr) {
    MS_LOG(ERROR) << "parameter is null, get output tensor failed.";
    return RET_NULL_PTR;
  }
  if (op->outputs.size() == 1) {
    int output_idx =
      op->outputs.front() < 0 ? tflite_subgraph->tensors.size() + op->outputs.front() : op->outputs.front();
    const auto &tensor = tflite_subgraph->tensors.at(output_idx);
    std::vector<int64_t> shape_vector;
    (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto type_ptr = TypeIdToType(GetTfliteDataType(tensor->type));
    dst_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    anf_node_map->insert(std::pair(op->outputs.front(), dst_cnode));
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (auto output_idx : op->outputs) {
      if (output_idx < 0) {
        output_idx = output_idx + tflite_subgraph->tensors.size();
      }
      const auto &tensor = tflite_subgraph->tensors.at(output_idx);
      std::vector<int64_t> shape_vector;
      (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                           [](const int32_t &value) { return static_cast<int64_t>(value); });
      auto abstract_tensor = CreateTensorAbstract(shape_vector, GetTfliteDataType(tensor->type));
      if (abstract_tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      abstract_list.emplace_back(abstract_tensor);
      auto tuple_get_item_prim_ptr = std::make_shared<lite::TupleGetItem>();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, dst_cnode, get_item_value};
      CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "new node failed";
        return RET_ERROR;
      }
      std::string output_item_name = dst_cnode->fullname_with_scope() + "_getitem_" + std::to_string(op_idx);
      auto get_item_abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
      if (get_item_abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      get_item_cnode->set_fullname_with_scope(output_item_name);
      get_item_cnode->set_abstract(get_item_abstract);
      anf_node_map->insert(std::pair(output_idx, get_item_cnode));
      op_idx++;
    }
    dst_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

int TfliteModelParser::Tflite2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (const auto &func_graph : all_func_graphs) {
    auto tflite_inputs_adjust = std::make_shared<TfliteInputsAdjust>();
    if (!tflite_inputs_adjust->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_MODEL_PARSER(FmkType_TFLITE, LiteModelParserCreator<TfliteModelParser>)
}  // namespace mindspore::lite
