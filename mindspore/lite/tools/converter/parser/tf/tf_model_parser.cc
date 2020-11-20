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
#include <functional>
#include <set>
#include "src/common/utils.h"
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "src/param_value_lite.h"
#include "tools/common/protobuf_utils.h"

namespace mindspore {
namespace lite {
static const std::unordered_map<int, mindspore::TypeId> TF_TYPE_MAP = {
  {tensorflow::DT_INT8, mindspore::kNumberTypeInt8},      {tensorflow::DT_UINT8, mindspore::kNumberTypeUInt8},
  {tensorflow::DT_INT16, mindspore::kNumberTypeInt16},    {tensorflow::DT_UINT16, mindspore::kNumberTypeUInt16},
  {tensorflow::DT_INT32, mindspore::kNumberTypeInt32},    {tensorflow::DT_INT64, mindspore::kNumberTypeInt64},
  {tensorflow::DT_HALF, mindspore::kNumberTypeFloat16},   {tensorflow::DT_FLOAT, mindspore::kNumberTypeFloat32},
  {tensorflow::DT_DOUBLE, mindspore::kNumberTypeFloat64}, {tensorflow::DT_COMPLEX64, mindspore::kNumberTypeComplex64},
  {tensorflow::DT_BOOL, mindspore::kNumberTypeBool}};

TypeId GetTFDataType(const tensorflow::DataType &tf_data_type) {
  auto iter = TF_TYPE_MAP.find(tf_data_type);
  if (iter == TF_TYPE_MAP.end()) {
    MS_LOG(ERROR) << "unsupported TF data type: " << tf_data_type;
    return kTypeUnknown;
  }
  return iter->second;
}

AnfNodePtr TFModelParser::GetAnfNode(const std::string &name) {
  AnfNodePtr ret = nullptr;
  if (anf_node_map.find(name) != anf_node_map.end()) {
    ret = anf_node_map[name];
  } else if (anf_node_map.find(name + ":0") != anf_node_map.end()) {
    ret = anf_node_map[name + ":0"];
  }
  return ret;
}

std::string TFModelParser::GetOriginInputName(const tensorflow::NodeDef &node) {
  if (node.op() != "Identity" && node.op() != "StopGradient") {
    return node.name();
  }
  auto tmp_node = &node;
  while (tmp_node->op() == "Identity" || tmp_node->op() == "StopGradient") {
    tmp_node = tf_node_map[tmp_node->input(0)];
  }
  return tmp_node->name();
}

STATUS TFModelParser::ConvertConstTensor(const tensorflow::AttrValue &attr_value, const TypeId &type,
                                         const ParameterPtr &parameter, std::vector<int64_t> *shape_vector) {
  MS_ASSERT(parameter != nullptr);
  MS_ASSERT(shape_vector != nullptr);
  const tensorflow::TensorProto &tensor_proto = attr_value.tensor();
  const tensorflow::TensorShapeProto &tensor_shape = tensor_proto.tensor_shape();
  int shape_size = 1;
  shape_vector->clear();
  for (int i = 0; i < tensor_shape.dim_size(); i++) {
    shape_vector->push_back(tensor_shape.dim(i).size());
    shape_size *= tensor_shape.dim(i).size();
  }

  int tensor_size;
  auto param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "param_value is nullptr";
    return RET_ERROR;
  }
  if (type == kNumberTypeFloat32 || type == kNumberTypeFloat) {
    auto tensor_data = new (std::nothrow) float[shape_size];
    if (tensor_proto.float_val_size() == 1) {
      float value = tensor_proto.float_val(0);
      for (int i = 0; i < shape_size; i++) {
        tensor_data[i] = value;
      }
    }
    if (tensor_proto.tensor_content().size() == shape_size * sizeof(float)) {
      const auto addr = reinterpret_cast<const float *>(tensor_proto.tensor_content().data());
      auto ret = ::memcpy_s(tensor_data, shape_size * sizeof(float), addr, shape_size * sizeof(float));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        return RET_ERROR;
      }
    }
    param_value->set_tensor_addr(tensor_data);
    tensor_size = shape_size * sizeof(float);
  } else if (type == kNumberTypeInt32) {
    auto tensor_data = new (std::nothrow) int[shape_size];
    if (tensor_proto.int_val_size() == 1) {
      int value = tensor_proto.int_val(0);
      for (int i = 0; i < shape_size; i++) {
        tensor_data[i] = value;
      }
    }
    if (tensor_proto.tensor_content().size() == shape_size * sizeof(int32_t)) {
      const auto addr = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
      auto ret = ::memcpy_s(tensor_data, shape_size * sizeof(int32_t), addr, shape_size * sizeof(int32_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        return RET_ERROR;
      }
    }
    param_value->set_tensor_addr(tensor_data);
    tensor_size = shape_size * sizeof(int);
  } else if (type == kNumberTypeBool) {
    auto tensor_data = new (std::nothrow) int[shape_size];
    if (tensor_proto.bool_val_size() == 1) {
      int value = tensor_proto.bool_val(0);
      for (int i = 0; i < shape_size; i++) {
        tensor_data[i] = value;
      }
    }
    param_value->set_tensor_addr(tensor_data);
    tensor_size = shape_size * sizeof(int);
  } else {
    MS_LOG(ERROR) << "Unsupport dataType: " << type;
    return RET_ERROR;
  }

  std::vector<int> param_shape(shape_vector->begin(), shape_vector->end());
  param_value->set_tensor_shape(param_shape);
  param_value->set_tensor_type(type);
  param_value->set_tensor_size(tensor_size);
  param_value->set_format(schema::Format::Format_NHWC);
  parameter->set_default_param(param_value);
  parameter->set_name("const_" + std::to_string(anf_node_map.size()) + "_parameter");
  return RET_OK;
}

STATUS TFModelParser::ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(parameter != nullptr);

  tensorflow::AttrValue attr_value;
  TypeId type = kNumberTypeFloat32;
  if (TensorFlowUtils::FindAttrValue(node, "dtype", &attr_value)) {
    type = GetTFDataType(attr_value.type());
  }
  auto type_ptr = TypeIdToType(type);

  std::vector<int> shape;
  if (TensorFlowUtils::FindAttrValue(node, "shape", &attr_value)) {
    auto &shape_attr = attr_value.shape();
    for (int i = 0; i < shape_attr.dim_size(); ++i) {
      shape.push_back(shape_attr.dim(i).size());
    }
  }
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());

  if (TensorFlowUtils::FindAttrValue(node, "value", &attr_value)) {
    MS_LOG(INFO) << "Found value attr, means it has default value";
    auto status = ConvertConstTensor(attr_value, type, parameter, &shape_vector);
    if (status != RET_OK) {
      return status;
    }
  } else {
    parameter->set_name("placeholder_" + std::to_string(anf_node_map.size()));
    graph_input_names.emplace_back(parameter->name());
  }

  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor is nullptr";
    return RET_ERROR;
  }
  parameter->set_abstract(abstract_tensor);

  anf_node_map[node.name()] = parameter;
  return RET_OK;
}

STATUS TFModelParser::ConvertGraphInputsAndConsts() {
  for (auto &pair : tf_node_map) {
    bool have_data_depend = false;
    for (int i = 0; i < pair.second->input_size(); ++i) {
      auto name = pair.second->input(i);
      if (!name.empty() && name[0] != '^') {  // control_depend input start with "^"
        have_data_depend = true;
        break;
      }
    }
    if (!have_data_depend) {
      auto parameter = funcGraphPtr->add_parameter();
      if (ConvertParameter(*pair.second, parameter) != RET_OK) {
        MS_LOG(ERROR) << "convert Parameter Node failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

FuncGraphPtr TFModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                  const QuantType &quantType) {
  auto status = ValidateFileStr(modelFile, ".pb");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.pb";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  tf_graph_def = std::make_unique<tensorflow::GraphDef>();
  if (tf_graph_def == nullptr) {
    MS_LOG(ERROR) << "tf_graph_def is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  status = ReadProtoFromBinaryFile((const char *)modelFile.c_str(), tf_graph_def.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Open modelFile for TF converter failed!";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  funcGraphPtr = std::make_shared<FuncGraph>();
  if (funcGraphPtr == nullptr) {
    MS_LOG(ERROR) << "funGraphPtr is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }

  for (int i = 0; i < tf_graph_def->node_size(); i++) {
    auto &node_def = tf_graph_def->node(i);
    tf_node_map[node_def.name()] = &node_def;
  }

  status = ConvertGraphInputsAndConsts();
  if (status != RET_OK) {
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
schema::MetaGraphT *TFModelParser::ParseToFb(const std::string &modelFile, const std::string &weightFile,
                                             const QuantType &quantType) {
  MS_LOG(ERROR) << "TF Model Parser not return MetaGraph, use TFModelParser::Parse instead";
  return nullptr;
}

STATUS TFModelParser::ConvertInputNodes(const tensorflow::NodeDef &node_def,
                                        const std::vector<std::string> &input_names, std::vector<AnfNodePtr> *inputs) {
  // parse inputs
  for (size_t j = 0; j < input_names.size(); j++) {
    std::string input_name = input_names[j];  // input may be produced by multi-outputs node
    if (tf_node_map.find(input_name) != tf_node_map.end()) {
      auto input_node = tf_node_map[input_name];
      input_name = GetOriginInputName(*input_node);
    }
    auto input = GetAnfNode(input_name);
    if (input == nullptr) {
      MS_LOG(ERROR) << node_def.name() << " input " << j << ": " << input_name << " can't find parsed in_nodes";
      return RET_ERROR;
    }
    inputs->emplace_back(input);
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOutputTensor(const tensorflow::NodeDef &op, const CNodePtr &anf_node, int output_size) {
  if (output_size == 1) {
    std::vector<int64_t> shape_vector;
    anf_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector));
    anf_node_map.insert(std::pair(op.name(), anf_node));
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
      anf_node_map.insert(std::pair(op.name() + ":" + std::to_string(output_idx), getItemCNode));
    }
    anf_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstractList));
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOps() {
  NoSupportOp::GetInstance()->SetFmkType("TF");
  STATUS status = RET_OK;
  int op_idx = 0;
  for (int i = 0; i < tf_graph_def->node_size(); i++) {
    auto &node_def = tf_graph_def->node(i);
    const auto &op_type = node_def.op();
    if (op_type == "Placeholder" || op_type == "Const" || op_type == "Identity" || op_type == "StopGradient") {
      continue;
    }
    auto node_parser = TFNodeParserRegistry::GetInstance()->GetNodeParser(op_type);
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(op_type);
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      MS_LOG(ERROR) << "cannot find node parser:" << op_type;
      continue;
    }
    if (status != RET_OK) {
      continue;
    }
    PrimitiveC *primitiveC = nullptr;
    int output_size;
    std::vector<std::string> input_names;
    status = node_parser->Parse(node_def, tf_node_map, &primitiveC, &input_names, &output_size);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "node " << op_type << " parser failed";
      continue;
    }

    auto value_node = NewValueNode(std::shared_ptr<PrimitiveC>(primitiveC));
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "value_node is nullptr";
      status = RET_ERROR;
      continue;
    }
    std::vector<AnfNodePtr> inputs = {value_node};
    status = ConvertInputNodes(node_def, input_names, &inputs);
    if (status != RET_OK) {
      continue;
    }
    // control_depends are not processed currently
    auto anf_node = funcGraphPtr->NewCNode(inputs);
    anf_node->set_fullname_with_scope(op_type + "-" + std::to_string(op_idx++));

    status = ConvertOutputTensor(node_def, anf_node, output_size);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert output tensors for " << anf_node->fullname_with_scope() << " failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      continue;
    }
  }
  return status;
}

STATUS TFModelParser::ConvertGraphOutputs() {
  // because output of intermediate node in anf graph may also be output tensors, we search output tensors in
  // tf_node_map but not anf_node_map
  std::set<std::string> all_node_inputs;
  std::vector<AnfNodePtr> output_nodes;
  for (auto &pair : tf_node_map) {
    for (int i = 0; i < pair.second->input_size(); ++i) {
      all_node_inputs.insert(pair.second->input(i));
    }
  }
  for (auto &pair : tf_node_map) {
    auto it = all_node_inputs.find(pair.first);
    if (it == all_node_inputs.end() && pair.second->input_size() > 0) {  // output node not constraint to Identity
      auto origin_name = GetOriginInputName(*(pair.second));
      auto anf_node = GetAnfNode(origin_name);
      if (anf_node == nullptr) {
        MS_LOG(ERROR) << "can't find anf node";
        return RET_ERROR;
      }
      output_nodes.push_back(anf_node);
      graph_output_names.push_back(anf_node->fullname_with_scope());
    }
  }

  if (output_nodes.size() > 1) {
    std::vector<AnfNodePtr> &make_tuple_inputs = output_nodes;
    auto make_tuple_prim_ptr = GetMakeTuplePrim();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetMakeTuplePrim return nullptr";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.insert(output_nodes.begin(), make_tuple_prim);
    auto make_tuple_cnode = funcGraphPtr->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    auto return_prim_ptr = GetReturnPrim();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = funcGraphPtr->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("return");
    funcGraphPtr->set_return(cnode);
  } else {
    auto return_prim_ptr = GetReturnPrim();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetReturnPrim return nullptr";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs{value_node, output_nodes.front()};
    auto return_cnode = funcGraphPtr->NewCNode(op_inputs);
    return_cnode->set_fullname_with_scope("return");
    funcGraphPtr->set_return(return_cnode);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
