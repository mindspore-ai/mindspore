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
#include "tools/converter/parser/tflite/model_parser_for_tflite.h"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "src/param_value_lite.h"

namespace mindspore::lite {

FuncGraphPtr ModelParserForTflite::Parse(const std::string &modelFile, const std::string &weightFile,
                                         const QuantType &quantType) {
  // load graph
  tfliteModel = ReadTfliteModel(modelFile.c_str());
  if (tfliteModel == nullptr) {
    MS_LOG(ERROR) << "read tflite model failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }

  if (tfliteModel->subgraphs.size() != 1) {
    MS_LOG(ERROR) << "read tflite model subgraphs failed";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return nullptr;
  }
  funcGraphPtr = std::make_shared<FuncGraph>();

  auto status = ConvertGraphInputs();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert graph inputs failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertOps();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert ops failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertGraphOutputs();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert graph outputs failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  return funcGraphPtr;
}

STATUS ModelParserForTflite::ConvertOps() {
  const auto &tfliteSubgraph = tfliteModel->subgraphs.front();
  const auto &tfliteModelBuffers = tfliteModel->buffers;
  NoSupportOp::GetInstance()->SetFmkType("TFLITE");
  STATUS status = RET_OK;
  int opIdx = 0;
  for (auto &op : tfliteSubgraph->operators) {
    auto tfliteOpType = (tfliteModel->operator_codes[op->opcode_index])->builtin_code;
    auto opType = GetMSOpType(tfliteOpType);

    // parse primitive
    auto nodeParser = TfliteNodeParserRegistry::GetInstance()->GetNodeParser(opType);
    if (nodeParser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(opType);
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      continue;
    }
    PrimitiveC *primitiveC = nullptr;
    if (status == RET_OK) {
      status = nodeParser->Parse(op, tfliteModel, primitiveC);
      if (status != RET_OK) {
        if (status == RET_NOT_FIND_OP) {
          opType = (opType != "Custom" ? opType : (tfliteModel->operator_codes[op->opcode_index])->custom_code);
          NoSupportOp::GetInstance()->InsertOp(opType);
        } else {
          MS_LOG(ERROR) << "node " << opType.c_str() << " parser failed";
        }
        continue;
      }

      std::vector<AnfNodePtr> opInputs = {NewValueNode(std::shared_ptr<PrimitiveC>(primitiveC))};
      // parse inputs
      for (auto inputIdx : op->inputs) {
        const auto &inputTensor = tfliteSubgraph->tensors[inputIdx];
        if (nodes.find(inputIdx) != nodes.end()) {
          opInputs.emplace_back(nodes.at(inputIdx));
          continue;
        }
        // const tensor
        if (tfliteModelBuffers.at(inputTensor->buffer)->data.empty()) {
          ParameterPtr parameter;
          ConvertConstTensor(inputTensor.get(), parameter);
          opInputs.emplace_back(parameter);
          nodes.insert(std::pair(inputIdx, parameter));
          continue;
        }
        MS_LOG(ERROR) << "tensor" << inputIdx << " is neither a node output nor a weight tensor.";
        return RET_ERROR;
      }
      auto newCNode = funcGraphPtr->NewCNode(opInputs);
      newCNode->set_fullname_with_scope(opType + "-" + std::to_string(opIdx++));

      // parse outputs
      status = ConvertOutputTensor(op.get(), newCNode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Convert output tensors for " << newCNode->fullname_with_scope() << " failed.";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        return status;
      }
    }
  }
  return status;
}

STATUS ModelParserForTflite::ConvertGraphInputs() {
  const auto &tfliteSubgraph = tfliteModel->subgraphs.front();
  for (auto tfliteGraphInput : tfliteSubgraph->inputs) {
    if (tfliteGraphInput < 0) {
      tfliteGraphInput = tfliteGraphInput + tfliteSubgraph->tensors.size();
    }
    auto parameter = funcGraphPtr->add_parameter();
    const auto &tensor = tfliteSubgraph->tensors.at(tfliteGraphInput);
    std::vector<int64_t> shape_vector;
    (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto type_ptr = TypeIdToType(GetTfliteDataType(tensor->type));
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    parameter->set_abstract(abstract_tensor);
    parameter->set_name("graph_input_" + std::to_string(tfliteGraphInput) + "_parameter");
    nodes.insert(std::pair(tfliteGraphInput, parameter));
  }
  return RET_OK;
}
STATUS ModelParserForTflite::ConvertGraphOutputs() {
  const auto &tfliteSubgraph = tfliteModel->subgraphs.front();
  if (tfliteSubgraph->outputs.size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = GetMakeTuplePrim();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetMakeTuplePrim return nullptr";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.emplace_back(make_tuple_prim);
    for (auto outputNode : tfliteSubgraph->outputs) {
      auto cnode = nodes.at(outputNode);
      if (nullptr == cnode) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_cnode = funcGraphPtr->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    std::vector<AnfNodePtr> op_inputs;
    auto return_prim_ptr = GetReturnPrim();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    op_inputs.emplace_back(value_node);
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = funcGraphPtr->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("return");
    funcGraphPtr->set_return(cnode);
  } else {
    auto returnPrim = GetReturnPrim();
    if (returnPrim == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto valueNode = NewValueNode(returnPrim);
    std::vector<AnfNodePtr> opInputs{valueNode};
    auto cnode = nodes.at(tfliteSubgraph->outputs.front());
    if (nullptr == cnode) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    opInputs.emplace_back(cnode);
    auto returnCnode = funcGraphPtr->NewCNode(opInputs);
    returnCnode->set_fullname_with_scope("return");
    funcGraphPtr->set_return(returnCnode);
  }
  return RET_OK;
}

STATUS ModelParserForTflite::ConvertConstTensor(const tflite::TensorT *tensor, ParameterPtr parameter) {
  parameter = funcGraphPtr->add_parameter();
  const auto &tfliteModelBuffers = tfliteModel->buffers;
  auto type_id = static_cast<TypeId>(tensor->type);
  auto type_ptr = TypeIdToType(type_id);
  std::vector<int64_t> shape_vector;
  (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  parameter->set_abstract(abstract_tensor);
  parameter->set_name("const_" + std::to_string(nodes.size()) + "_parameter");

  ParamValueLitePtr paramValue = std::make_shared<ParamValueLite>();
  MS_ASSERT(paramValue != nullptr);
  paramValue->set_tensor_shape(tensor->shape);
  paramValue->set_tensor_type(GetTfliteDataType(tensor->type));
  paramValue->set_format(schema::Format::Format_NHWC);
  const auto &data = tfliteModelBuffers.at(tensor->buffer)->data;
  if (!data.empty()) {
    auto size = data.size();
    char *tensor_data = new (std::nothrow) char[size];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "new char[] failed";
      return RET_MEMORY_FAILED;
    }
    std::memcpy(tensor_data, data.data(), size);
    paramValue->set_tensor_addr(tensor_data);
    paramValue->set_tensor_size(size);
    parameter->set_default_param(paramValue);
  }
  return RET_OK;
}

STATUS ModelParserForTflite::ConvertOutputTensor(const tflite::OperatorT *op, CNodePtr dstCNode) {
  MS_ASSERT(op != nullptr);
  MS_ASSERT(dstCNode != nullptr);
  const auto &tfliteSubgraph = tfliteModel->subgraphs.front();
  if (op->outputs.size() == 1) {
    const auto &tensor = tfliteSubgraph->tensors.at(op->outputs.front());
    std::vector<int64_t> shape_vector;
    (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto typePtr = TypeIdToType(GetTfliteDataType(tensor->type));
    dstCNode->set_abstract(std::make_shared<abstract::AbstractTensor>(typePtr, shape_vector));
    nodes.insert(std::pair(op->outputs.front(), dstCNode));
  } else {
    AbstractBasePtrList abstractList;
    for (auto outputIdx : op->outputs) {
      const auto &tensor = tfliteSubgraph->tensors.at(outputIdx);
      std::vector<int64_t> shape_vector;
      (void)std::transform(tensor->shape.begin(), tensor->shape.end(), std::back_inserter(shape_vector),
                           [](const int32_t &value) { return static_cast<int64_t>(value); });
      auto typePtr = TypeIdToType(GetTfliteDataType(tensor->type));
      abstractList.emplace_back(std::make_shared<abstract::AbstractTensor>(typePtr, shape_vector));
      auto tupleGetItemPrimPtr = GetTupleGetItemPrim();
      if (tupleGetItemPrimPtr == nullptr) {
        MS_LOG(ERROR) << "GetTupleGetItemPrim return nullptr";
        return RET_NULL_PTR;
      }
      auto tupleGetItemPrim = NewValueNode(tupleGetItemPrimPtr);
      auto getItemValue = NewValueNode(MakeValue<int>(outputIdx));
      std::vector<AnfNodePtr> inputs{tupleGetItemPrim, dstCNode, getItemValue};
      CNodePtr getItemCNode = funcGraphPtr->NewCNode(inputs);
      getItemCNode->set_fullname_with_scope(dstCNode->fullname_with_scope() + "_getitem_" + std::to_string(outputIdx));
      nodes.insert(std::pair(outputIdx, getItemCNode));
    }
    dstCNode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstractList));
  }
  return RET_OK;
}
}  // namespace mindspore::lite
