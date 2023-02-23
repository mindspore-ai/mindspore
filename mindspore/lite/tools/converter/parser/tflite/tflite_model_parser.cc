/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "include/registry/node_parser_registry.h"
#include "ops/primitive_c.h"
#include "ir/func_graph.h"
#include "src/common/file_utils.h"
#include "tools/common/graph_util.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/parser/tflite/tflite_inputs_adjust.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "tools/converter/parser/unify_format.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"

using mindspore::converter::kFmkTypeTflite;
namespace mindspore::lite {
namespace {
constexpr size_t kMainGraphIndex = 0;
constexpr size_t kConvWeightIndex = 2;

FuncGraphPtr ConvertGraph(const api::FuncGraphPtr &func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}
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

STATUS TfliteModelParser::TfliteOpVerify(const std::unique_ptr<tflite::SubGraphT> &subgraph,
                                         const size_t operator_codes_size, const size_t all_tensor_size) {
  int32_t all_tensor_num = static_cast<int32_t>(all_tensor_size);
  for (auto &op : subgraph->operators) {
    if (op == nullptr) {
      MS_LOG(ERROR) << "tflite contain nullptr op.";
      return RET_ERROR;
    }
    if (op->opcode_index >= operator_codes_size) {
      MS_LOG(ERROR) << "op is not a tflite opcode";
      return RET_ERROR;
    }
    if (std::any_of(op->inputs.begin(), op->inputs.end(), [&all_tensor_num](int32_t index) {
          return index >= all_tensor_num || index + all_tensor_num < 0;
        })) {
      MS_LOG(ERROR) << "op input illegal.";
      return RET_ERROR;
    }
    if (std::any_of(op->outputs.begin(), op->outputs.end(), [&all_tensor_num](int32_t index) {
          return index >= all_tensor_num || index + all_tensor_num < 0;
        })) {
      MS_LOG(ERROR) << "op output illegal.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::TfliteTensorVerify(const std::unique_ptr<tflite::SubGraphT> &subgraph,
                                             const size_t model_buffers_size) {
  for (auto &tensor : subgraph->tensors) {
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "tflite model contain nullptr tensor.";
      return RET_ERROR;
    }
    if (tensor->buffer >= model_buffers_size) {
      MS_LOG(ERROR) << "tflite tensor buffer index beyond upper limit.";
      return RET_ERROR;
    }
    if (tensor->quantization != nullptr && !tensor->quantization->scale.empty()) {
      auto scale_size = tensor->quantization->scale.size();
      auto zp_size = tensor->quantization->zero_point.size();
      auto min_size = tensor->quantization->min.size();
      auto max_size = tensor->quantization->max.size();
      if ((zp_size != 0 && zp_size != scale_size) || (min_size != 0 && min_size != scale_size) ||
          (max_size != 0 && max_size != scale_size)) {
        MS_LOG(ERROR) << "The element numbers of non-empty quantization parameters must be same.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS TfliteModelParser::TfliteModelVerify() {
  if (tflite_model_->subgraphs.empty()) {
    MS_LOG(ERROR) << "tflite model does not has a main graph.";
    return RET_ERROR;
  }
  const auto tflite_model_buffers_size = tflite_model_->buffers.size();
  const auto tflite_model_operator_codes_size = tflite_model_->operator_codes.size();

  for (auto &subgraph : tflite_model_->subgraphs) {
    if (subgraph == nullptr) {
      MS_LOG(ERROR) << "tflite contain nullptr subgraph.";
      return RET_ERROR;
    }
    auto all_subgraph_tensor_size = subgraph->tensors.size();
    if (subgraph->inputs.empty() || subgraph->outputs.empty()) {
      MS_LOG(ERROR) << "tflite subgraph inputs or outputs is empty.";
      return RET_ERROR;
    }
    if (std::any_of(subgraph->inputs.begin(), subgraph->inputs.end(), [&all_subgraph_tensor_size](int32_t index) {
          return index >= static_cast<int32_t>(all_subgraph_tensor_size) || index < 0;
        })) {
      MS_LOG(ERROR) << "tflite input illegal.";
      return RET_ERROR;
    }
    if (std::any_of(subgraph->outputs.begin(), subgraph->outputs.end(), [&all_subgraph_tensor_size](int32_t index) {
          return index >= static_cast<int32_t>(all_subgraph_tensor_size) || index < 0;
        })) {
      MS_LOG(ERROR) << "tflite output illegal.";
      return RET_ERROR;
    }
    auto ret = TfliteOpVerify(subgraph, tflite_model_operator_codes_size, all_subgraph_tensor_size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Tflite op verification dose not pass.";
      return RET_ERROR;
    }
    ret = TfliteTensorVerify(subgraph, tflite_model_buffers_size);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Tflite Tensor verification dose not pass.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

api::FuncGraphPtr TfliteModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file;
  // load graph
  tflite_model_ = ReadTfliteModel(model_file);
  if (tflite_model_ == nullptr) {
    MS_LOG(ERROR) << "read tflite model failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  auto status = TfliteModelVerify();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "tflite model verify failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertTfliteGraph();
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

  auto func_graph = ConvertGraph(res_graph_);
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  if ((status = CommonAnfAdjust(func_graph)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  if ((status = Tflite2AnfAdjust(all_func_graphs)) != RET_OK) {
    MS_LOG(ERROR) << "Tflite2AnfAdjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(kFmkTypeTflite, false, flag.save_type);
  MS_CHECK_TRUE_RET(unify_format != nullptr, nullptr);
  if (!unify_format->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  return res_graph_;
}

STATUS TfliteModelParser::ConvertTfliteGraph() {
  auto subgraph_num = tflite_model_->subgraphs.size();
  for (size_t idx = 0; idx < subgraph_num; idx++) {
    std::unordered_map<int, AnfNodePtr> anf_node_map;
    const auto &tflite_subgraph = tflite_model_->subgraphs.at(idx);
    const auto subgraph_name = tflite_subgraph->name;
    // build function graph
    FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
    MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
    auto type_value = MakeValue(static_cast<int>(converter::kFmkTypeTflite));
    MS_CHECK_TRUE_RET(type_value != nullptr, RET_ERROR);
    func_graph->set_attr("fmk", type_value);
    auto attr_value = MakeValue(subgraph_name);
    MS_CHECK_TRUE_RET(attr_value != nullptr, RET_ERROR);
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
      res_graph_ = api::MakeShared<api::FuncGraph>(func_graph);
      MS_CHECK_TRUE_MSG(res_graph_ != nullptr, RET_ERROR, "create FuncGraph failed");
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
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr && tflite_subgraph != nullptr);
  NotSupportOp::GetInstance()->set_fmk_type("TFLITE");
  STATUS status = RET_OK;
  int op_idx = 0;
  for (auto &op : tflite_subgraph->operators) {
    auto &opcode = tflite_model_->operator_codes[op->opcode_index];
    if (opcode == nullptr) {
      MS_LOG(ERROR) << "opcode is nullptr";
      return RET_ERROR;
    }
    auto tflite_op_type = opcode->builtin_code;
    std::string op_type = tflite::EnumNameBuiltinOperator(tflite_op_type);
    std::string op_name = op_type + "-" + std::to_string(op_idx);
    op_idx++;
    // parse primitive
    MS_LOG(INFO) << "parse node :" << op_name;
    ops::PrimitiveCPtr primitive_c;
    auto node_parser = registry::NodeParserRegistry::GetNodeParser(kFmkTypeTflite, op_type);
    if (node_parser != nullptr) {
      primitive_c = node_parser->Parse(op, tflite_subgraph, tflite_model_)->GetPrim();
    } else {
      auto node_parser_builtin = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(tflite_op_type);
      if (node_parser_builtin == nullptr) {
        NotSupportOp::GetInstance()->InsertOp(op_type);
        status = (status == RET_OK ? RET_NOT_FIND_OP : status);
        MS_LOG(ERROR) << "Can not find " << op_type << " op parser.";
        continue;
      }
      if (status != RET_OK) {
        continue;
      }
      primitive_c = node_parser_builtin->Parse(op, tflite_subgraph, tflite_model_);
    }

    std::vector<AnfNodePtr> op_inputs;
    if (primitive_c != nullptr) {
      auto value_node = NewValueNode(primitive_c);
      MSLITE_CHECK_PTR(value_node);
      op_inputs = {value_node};
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
    ConvertInputTensor(tflite_subgraph, func_graph, op, tflite_op_type, anf_node_map, op_name, &op_inputs);
    auto new_cnode = func_graph->NewCNode(op_inputs);
    MSLITE_CHECK_PTR(new_cnode);
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
  MS_ASSERT(op != nullptr && anf_node != nullptr);
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
  MS_ASSERT(quant_params != nullptr);
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

    quant_param->scale = tflite_tensor->quantization->scale[i];
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
                                               PrimitiveCPtr primitive_c) {
  MS_ASSERT(tflite_subgraph != nullptr);
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

  std::map<int, std::vector<schema::QuantParamT>> in_quant_param;
  size_t idx = 0;
  for (auto input_idx : op->inputs) {
    if (input_idx < 0) {
      if (primitive_c->name() == "FullConnection") {
        continue;
      }
      input_idx += tflite_subgraph->tensors.size();
    }
    MS_CHECK_TRUE_RET(static_cast<size_t>(input_idx) < tflite_subgraph->tensors.size(), RET_ERROR);
    const auto &input_tensor = tflite_subgraph->tensors[input_idx];
    MS_CHECK_TRUE_MSG(input_tensor != nullptr, RET_NULL_PTR, "input_tensor is nullptr.");
    std::vector<schema::QuantParamT> quant_params;
    auto status = SetTensorQuantParam(input_tensor, &quant_params, round_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set input tensor quant param failed.";
      return status;
    }
    in_quant_param.insert({idx, quant_params});
    idx++;
  }
  std::map<size_t, std::vector<schema::QuantParamT>> out_quant_param;
  idx = 0;
  for (auto output_idx : op->outputs) {
    if (output_idx < 0) {
      if (primitive_c->name() == "FullConnection") {
        continue;
      }
      output_idx += tflite_subgraph->tensors.size();
    }
    MS_CHECK_TRUE_RET(static_cast<size_t>(output_idx) < tflite_subgraph->tensors.size(), RET_ERROR);
    const auto &output_tensor = tflite_subgraph->tensors.at(output_idx);
    MS_CHECK_TRUE_MSG(output_tensor != nullptr, RET_NULL_PTR, "output_tensor is nullptr.");
    std::vector<schema::QuantParamT> quant_params;
    auto status = SetTensorQuantParam(output_tensor, &quant_params, round_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set output tensor quant param failed.";
      return status;
    }
    out_quant_param.insert({idx, quant_params});
    idx++;
  }
  if (!in_quant_param.empty() || !out_quant_param.empty()) {
    auto quant_params_holder = std::make_shared<QuantParamHolder>(0, 0);
    MSLITE_CHECK_PTR(quant_params_holder);
    for (auto &iter : in_quant_param) {
      quant_params_holder->set_input_quant_param(iter.first, iter.second);
    }
    for (auto &iter : out_quant_param) {
      quant_params_holder->set_output_quant_param(iter.first, iter.second);
    }
    primitive_c->AddAttr("quant_params", quant_params_holder);
  }
  return RET_OK;
}

STATUS TfliteModelParser::ConvertGraphInputs(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                             const FuncGraphPtr &func_graph,
                                             std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr && tflite_subgraph != nullptr);
  for (size_t i = 0; i < tflite_subgraph->inputs.size(); i++) {
    auto tflite_graph_input = tflite_subgraph->inputs.at(i);
    if (tflite_graph_input < 0) {
      tflite_graph_input = tflite_graph_input + tflite_subgraph->tensors.size();
    }
    auto parameter = func_graph->add_parameter();
    MSLITE_CHECK_PTR(parameter);
    const auto &tensor = tflite_subgraph->tensors.at(tflite_graph_input);
    MS_CHECK_TRUE_MSG(tensor != nullptr, RET_NULL_PTR, "tensor is nullptr.");
    std::vector<int64_t> shape_vector = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(tensor->name);
    if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape_vector.empty()) {
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
  MS_ASSERT(anf_node_map != nullptr && func_graph != nullptr && tflite_subgraph != nullptr);
  std::vector<AnfNodePtr> output_nodes;
  if (tflite_subgraph->outputs.size() > 1) {
    for (auto output_node : tflite_subgraph->outputs) {
      auto output_idx = output_node < 0 ? output_node + tflite_subgraph->tensors.size() : output_node;
      MS_CHECK_TRUE_RET(anf_node_map->find(output_idx) != anf_node_map->end(), RET_NOT_FIND_OP);
      auto cnode = anf_node_map->at(output_idx);
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      output_nodes.emplace_back(cnode);
    }
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(make_tuple_prim_c);
    auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
    MSLITE_CHECK_PTR(make_tuple_prim);
    std::vector<AnfNodePtr> make_tuple_inputs = output_nodes;
    make_tuple_inputs.insert(make_tuple_inputs.begin(), make_tuple_prim);
    auto make_tuple_cnode = func_graph->NewCNode(make_tuple_inputs);
    MSLITE_CHECK_PTR(make_tuple_cnode);
    make_tuple_cnode->set_fullname_with_scope("return_tuple");

    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto return_prim_c = return_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(return_prim_c);
    auto value_node = NewValueNode(return_prim_c);
    MSLITE_CHECK_PTR(value_node);
    std::vector<AnfNodePtr> op_inputs{value_node};
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = func_graph->NewCNode(op_inputs);
    MSLITE_CHECK_PTR(cnode);
    cnode->set_fullname_with_scope("Return");
    func_graph->set_return(cnode);
  } else {
    auto returnPrim = std::make_shared<ops::Return>();
    if (returnPrim == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    int output_idx = tflite_subgraph->outputs.front() < 0
                       ? static_cast<int>(tflite_subgraph->outputs.front() + tflite_subgraph->tensors.size())
                       : static_cast<int>(tflite_subgraph->outputs.front());
    auto return_prim_c = returnPrim->GetPrim();
    MSLITE_CHECK_PTR(return_prim_c);
    auto value_node = NewValueNode(return_prim_c);
    MSLITE_CHECK_PTR(value_node);
    std::vector<AnfNodePtr> op_inputs{value_node};
    MS_CHECK_TRUE_RET(anf_node_map->find(output_idx) != anf_node_map->end(), RET_NOT_FIND_OP);
    auto cnode = anf_node_map->at(output_idx);
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    output_nodes.emplace_back(cnode);
    op_inputs.emplace_back(cnode);
    auto returnCnode = func_graph->NewCNode(op_inputs);
    MSLITE_CHECK_PTR(returnCnode);
    returnCnode->set_fullname_with_scope("Return");
    func_graph->set_return(returnCnode);
  }

  if (tflite_subgraph == tflite_model_->subgraphs.front()) {
    // save original output tensor names.
    std::vector<std::string> output_names;
    auto output_idx = tflite_subgraph->outputs;
    std::transform(output_idx.begin(), output_idx.end(), std::back_inserter(output_names),
                   [&](auto out_idx) { return tflite_subgraph->tensors.at(out_idx)->name; });
    ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(output_names);
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
  MS_ASSERT(sub_func_graph != nullptr);
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
  auto func_graph = ConvertGraph(res_graph_);
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  static auto root_func_manager = Manage(func_graph);
  MS_CHECK_TRUE_RET(root_func_manager != nullptr, RET_ERROR);
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
    MSLITE_CHECK_PTR(first_value_node);
    auto second_value_node = NewValueNode(second_sub_graph);
    MSLITE_CHECK_PTR(second_value_node);
    auto inputs = control_flow_node->inputs();
    inputs.insert(inputs.begin() + 1, {first_value_node, second_value_node});
    auto new_node = func_graph->NewCNode(inputs);  // must create new node, otherwise node_users won't update
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
                                             const ParameterPtr &parameter, const std::string &tensor_name,
                                             bool is_uint8_weight_quant) {
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
      MS_CHECK_TRUE_MSG(tensor_info->Size() == data.size(), RET_ERROR, "invalid const tensor data.");
      if (memcpy_s(tensor_data, tensor_info->Size(), data.data(), data.size()) != EOK) {
        MS_LOG(ERROR) << "memcpy failed.";
        return RET_ERROR;
      }
      if (is_uint8_weight_quant && type_id == kNumberTypeUInt8) {
        int64_t shape_size = 1;
        for (size_t i = 0; i < shape_vector.size(); i++) {
          shape_size *= shape_vector[i];
        }
        auto uint8_tensor_data = reinterpret_cast<uint8_t *>(tensor_info->data_c());
        for (int64_t i = 0; i < shape_size; i++) {
          uint8_tensor_data[i] = static_cast<int8_t>(uint8_tensor_data[i]);
        }
        tensor_info->set_data_type(kNumberTypeInt8);
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

void TfliteModelParser::ConvertInputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const FuncGraphPtr &func_graph, const std::unique_ptr<tflite::OperatorT> &op,
                                           tflite::BuiltinOperator tflite_op_type,
                                           std::unordered_map<int, AnfNodePtr> *anf_node_map, std::string op_name,
                                           std::vector<AnfNodePtr> *op_inputs) {
  MS_ASSERT(anf_node_map != nullptr && tflite_subgraph != nullptr && func_graph != nullptr && op_inputs != nullptr);
  bool is_uint8_weight_quant = false;
  for (int i = 0; i < static_cast<int>(op->inputs.size()); i++) {
    auto input_idx = op->inputs.at(i);
    if (tflite_op_type == tflite::BuiltinOperator_FULLY_CONNECTED && input_idx == -1) {
      continue;
    }
    if (input_idx < 0) {
      input_idx += tflite_subgraph->tensors.size();
    }
    const auto &input_tensor = tflite_subgraph->tensors[input_idx];
    MS_CHECK_PTR_IF_NULL(input_tensor);
    auto type_id = GetTfliteDataType(input_tensor->type);
    if (anf_node_map->find(input_idx) != anf_node_map->end()) {
      if (utils::isa<CNodePtr>(anf_node_map->at(input_idx)) && type_id != kNumberTypeUInt8) {
        is_uint8_weight_quant = true;
      }
      op_inputs->emplace_back(anf_node_map->at(input_idx));
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
    MS_CHECK_PTR_IF_NULL(parameter);
    auto status = ConvertConstTensor(input_tensor, parameter, tensor_name, is_uint8_weight_quant);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "convert " << op_name << " node: " << input_idx << " const node failed.";
      continue;
    }
    parameter->set_name(tensor_name);
    op_inputs->emplace_back(parameter);
    anf_node_map->insert(std::pair(input_idx, parameter));
  }
  return;
}

STATUS TfliteModelParser::ConvertOutputTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                              const FuncGraphPtr &func_graph,
                                              const std::unique_ptr<tflite::OperatorT> &op, const CNodePtr &dst_cnode,
                                              std::unordered_map<int, AnfNodePtr> *anf_node_map) {
  MS_ASSERT(anf_node_map != nullptr && tflite_subgraph != nullptr && func_graph != nullptr);
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
    MSLITE_CHECK_PTR(tensor);
    std::vector<int64_t> shape_vector;
    (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto type_ptr = TypeIdToType(GetTfliteDataType(tensor->type));
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    MSLITE_CHECK_PTR(abstract_tensor);
    dst_cnode->set_abstract(abstract_tensor);
    anf_node_map->insert(std::pair(op->outputs.front(), dst_cnode));
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (auto output_idx : op->outputs) {
      if (output_idx < 0) {
        output_idx = output_idx + tflite_subgraph->tensors.size();
      }
      const auto &tensor = tflite_subgraph->tensors.at(output_idx);
      MSLITE_CHECK_PTR(tensor);
      std::vector<int64_t> shape_vector;
      (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                           [](const int32_t &value) { return static_cast<int64_t>(value); });
      auto abstract_tensor = CreateTensorAbstract(shape_vector, GetTfliteDataType(tensor->type));
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
      MSLITE_CHECK_PTR(tuple_get_item_prim_c);
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
      MSLITE_CHECK_PTR(tuple_get_item_prim);
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      MSLITE_CHECK_PTR(get_item_value);
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, dst_cnode, get_item_value};
      CNodePtr get_item_cnode = func_graph->NewCNode(inputs);
      MSLITE_CHECK_PTR(get_item_cnode);
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
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    MSLITE_CHECK_PTR(abstract_tuple);
    dst_cnode->set_abstract(abstract_tuple);
  }
  return RET_OK;
}

int TfliteModelParser::Tflite2AnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (const auto &func_graph : all_func_graphs) {
    auto tflite_inputs_adjust = std::make_shared<TfliteInputsAdjust>();
    MSLITE_CHECK_PTR(tflite_inputs_adjust);
    if (!tflite_inputs_adjust->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REG_MODEL_PARSER(kFmkTypeTflite, LiteModelParserCreator<TfliteModelParser>)
}  // namespace mindspore::lite
