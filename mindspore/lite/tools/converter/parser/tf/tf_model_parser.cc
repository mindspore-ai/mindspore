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
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/parser/tf/tf_model_parser.h"
#include <map>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "tools/common/graph_util.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {
FuncGraphPtr TFModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                  const QuantType &quantType) {
  auto status = ValidateFileStr(modelFile, ".prototxt");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.prototxt";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  if (!TensorFlowUtils::TfReadProtoFromBinary(modelFile.c_str(), tf_graph_def.get())) {
    MS_LOG(ERROR) << "Open modelFile for TF converter failed!";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  funcGraphPtr = std::make_shared<FuncGraph>();
  status = ConvertGraphInputs();
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
STATUS TFModelParser::ConvertConstTensor(const tensorflow::NodeDef *node, ParameterPtr parameter) {
  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(node, "value", &attr_value)) {
    tensorflow::AttrValue data_type;
    tensorflow::DataType type = tensorflow::DT_FLOAT;
    // datatype
    if (TensorFlowUtils::FindAttrValue(node, "dtype", &data_type)) {
      type = data_type.type();
    }
    const tensorflow::TensorProto &tensorProto = attr_value.tensor();
    const tensorflow::TensorShapeProto &tensorShape = tensorProto.tensor_shape();
    parameter = funcGraphPtr->add_parameter();
    std::vector<int64_t> shape_vector;
    int shape_size = 1;
    shape_vector.resize(tensorShape.dim_size());
    for (int i = 0; i < tensorShape.dim_size(); i++) {
      shape_vector[i] = tensorShape.dim(i).size();
      shape_size *= shape_vector[i];
    }
    // convert const to paramter
    TypePtr ms_data_ype;
    auto paramValue = std::make_shared<ParamValueLite>();
    if (type == tensorflow::DT_FLOAT) {
      ms_data_ype = kFloat32;
      auto tensor_data = new (std::nothrow) float[shape_size];
      if (tensorProto.float_val_size() == 1) {
        float value = tensorProto.float_val(0);
        for (int i = 0; i < shape_size; i++) {
          tensor_data[i] = value;
        }
      }
      if (tensorProto.tensor_content().size() == shape_size * sizeof(float)) {
        const auto addr = reinterpret_cast<const float *>(tensorProto.tensor_content().data());
        auto ret = ::memcpy_s(tensor_data, shape_size * sizeof(float), addr, shape_size * sizeof(float));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return RET_ERROR;
        }
      }
      paramValue->set_tensor_addr(tensor_data);
      paramValue->set_tensor_size(shape_size * sizeof(float));
    } else if (type == tensorflow::DT_INT32) {
      ms_data_ype = kInt32;
      auto tensor_data = new (std::nothrow) int[shape_size];
      if (tensorProto.int_val_size() == 1) {
        int value = tensorProto.int_val(0);
        for (int i = 0; i < shape_size; i++) {
          tensor_data[i] = value;
        }
      }
      if (tensorProto.tensor_content().size() == shape_size * sizeof(int32_t)) {
        const auto addr = reinterpret_cast<const int32_t *>(tensorProto.tensor_content().data());
        auto ret = ::memcpy_s(tensor_data, shape_size * sizeof(int32_t), addr, shape_size * sizeof(int32_t));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed";
          return RET_ERROR;
        }
      }
      paramValue->set_tensor_addr(tensor_data);
      paramValue->set_tensor_size(shape_size * sizeof(int));
    } else if (type == tensorflow::DT_BOOL) {
      ms_data_ype = kFloat32;
      auto tensor_data = new (std::nothrow) int[shape_size];
      if (tensorProto.bool_val_size() == 1) {
        int value = tensorProto.bool_val(0);
        for (int i = 0; i < shape_size; i++) {
          tensor_data[i] = value;
        }
      }
      paramValue->set_tensor_addr(tensor_data);
      paramValue->set_tensor_size(shape_size * sizeof(int));
    } else {
      MS_LOG(ERROR) << "Unsupport dataType," << node->name();
      return RET_ERROR;
    }
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(ms_data_ype, shape_vector);
    parameter->set_abstract(abstract_tensor);
    parameter->set_name("const_" + std::to_string(anf_node_map.size()) + "_parameter");

    std::vector<int> param_shape;
    (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(param_shape),
                         [](const int64_t &value) { return static_cast<int>(value); });

    MS_ASSERT(paramValue != nullptr);
    paramValue->set_tensor_shape(param_shape);
    paramValue->set_tensor_type(ms_data_ype->type_id());
    paramValue->set_format(schema::Format::Format_NHWC);
    paramValue->set_tensor_size(shape_size * sizeof(int));
    parameter->set_default_param(paramValue);
  }
  return RET_OK;
}
STATUS TFModelParser::ConvertOutputTensor(const tensorflow::NodeDef *op, const CNodePtr &anf_node, int output_size) {
  if (output_size == 1) {
    std::vector<int64_t> shape_vector;
    anf_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector));
    anf_node_map.insert(std::pair(op->name(), anf_node));
  } else {
    AbstractBasePtrList abstractList;
    for (int output_idx = 0; output_idx < output_size; output_idx++) {
      std::vector<int64_t> shape_vector;
      abstractList.emplace_back(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector));
      auto tupleGetItemPrimPtr = GetTupleGetItemPrim();
      if (tupleGetItemPrimPtr == nullptr) {
        MS_LOG(ERROR) << "GetTupleGetItemPrim return nullptr";
        return RET_NULL_PTR;
      }
      auto tupleGetItemPrim = NewValueNode(tupleGetItemPrimPtr);
      auto getItemValue = NewValueNode(MakeValue<int>(output_idx));
      std::vector<AnfNodePtr> inputs{tupleGetItemPrim, anf_node, getItemValue};
      CNodePtr getItemCNode = funcGraphPtr->NewCNode(inputs);
      std::string output_item_name = anf_node->fullname_with_scope() + "_getitem_" + std::to_string(output_idx);
      getItemCNode->set_fullname_with_scope(output_item_name);
      anf_node_map.insert(std::pair(output_item_name, getItemCNode));
    }
    anf_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstractList));
  }
  return RET_OK;
}
STATUS TFModelParser::ConvertOps() {
  NoSupportOp::GetInstance()->SetFmkType("TENSORFLOW");
  STATUS status = RET_OK;

  // redirect identity to it's input0
  ClipIdentityAndStopGradient();
  int op_idx = 0;
  for (int i = 0; i < tf_graph_def->node_size(); i++) {
    auto node_def = tf_graph_def->mutable_node(i);
    tf_node_map[node_def->name()] = node_def;
    auto tf_op_type = node_def->op();
    if (tf_op_type == "Placeholder" || tf_op_type == "Const") {
      continue;
    }
    auto node_parser = TFNodeParserRegistry::GetInstance()->GetNodeParser(tf_op_type);
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(tf_op_type);
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      MS_LOG(ERROR) << "cannot find node parser:" << tf_op_type;
      continue;
    }
    PrimitiveC *primitiveC = nullptr;
    if (status == RET_OK) {
      int output_size = 1;
      status = node_parser->Parse(node_def, tf_graph_def, primitiveC, &output_size);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "node " << tf_op_type.c_str() << " parser failed";
        continue;
      }
      std::vector<AnfNodePtr> opInputs = {NewValueNode(std::shared_ptr<PrimitiveC>(primitiveC))};
      // parse inputs
      for (int j = 0; j < node_def->input_size(); j++) {
        auto input_node = tf_node_map[node_def->input(i)];
        // last node output
        if (anf_node_map.find(input_node->name()) != anf_node_map.end()) {
          opInputs.emplace_back(anf_node_map[input_node->name()]);
          continue;
        }
        // const tensor
        if (input_node->op() == "Const") {
          ParameterPtr parameter;
          if (ConvertConstTensor(input_node, parameter) != RET_OK) {
            MS_LOG(ERROR) << "convert const tensor failed," << input_node->name();
            return RET_ERROR;
          }
          opInputs.emplace_back(parameter);
          anf_node_map[parameter->fullname_with_scope()] = parameter;
          continue;
        }
        MS_LOG(ERROR) << "node" << node_def->name() << "has inputs neither a node output nor a weight tensor.";
        return RET_ERROR;
      }
      auto anf_node = funcGraphPtr->NewCNode(opInputs);
      anf_node->set_fullname_with_scope(tf_op_type + "-" + std::to_string(op_idx++));

      // parse outputs
      status = ConvertOutputTensor(node_def, anf_node, output_size);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Convert output tensors for " << anf_node->fullname_with_scope() << " failed.";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        return status;
      }
    }
    // redirect identity to it's input0
    ClipIdentityAndStopGradient();
  }
  return RET_OK;
}
STATUS TFModelParser::ConvertGraphInputs() {
  for (int i = 0; i < tf_graph_def->node_size(); i++) {
    auto node_def = tf_graph_def->mutable_node(i);
    tf_node_map[node_def->name()] = node_def;
    if (node_def->op() == "Placeholder") {
      auto parameter = funcGraphPtr->add_parameter();
      if (ConvertConstTensor(node_def, parameter) != RET_OK) {
        MS_LOG(ERROR) << "convert const tensor failed";
        return RET_ERROR;
      }
      anf_node_map[node_def->name()] = parameter;
      graph_input_names.emplace_back(node_def->name());
    }
  }
  return RET_OK;
}
STATUS TFModelParser::ConvertGraphOutputs() { return RET_OK; }

std::string TFModelParser::GetOriginInputName(const tensorflow::NodeDef &node) {
  if (node.op() != "Identity" && node.op() != "StopGradient") {
    return node.name();
  }
  auto tmpNode = node;
  while (tmpNode.op() == "Identity" || tmpNode.op() == "StopGradient") {
    tmpNode = *tf_node_map[tmpNode.input(0)];
  }
  return tmpNode.name();
}

void TFModelParser::ClipIdentityAndStopGradient() {
  for (auto &pair : tf_node_map) {
    pair.second = tf_node_map[GetOriginInputName(*pair.second)];
  }
}
}  // namespace lite
}  // namespace mindspore
