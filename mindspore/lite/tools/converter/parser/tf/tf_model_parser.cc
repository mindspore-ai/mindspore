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
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "src/param_value_lite.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/return.h"
#include "ops/make_tuple.h"
#include "ops/tuple_get_item.h"
#include "ir/anf.h"
#include "tools/converter/converter_flags.h"

namespace mindspore {
namespace lite {
namespace {
bool IsTensorListOp(const AnfNodePtr &anf_node) {
  return opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListFromTensor) ||
         opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListSetItem) ||
         opt::CheckPrimitiveType(anf_node, prim::kPrimTensorListReserve);
}

AnfNodePtr GetAnfNode(const std::string &name, const std::unordered_map<std::string, AnfNodePtr> &anf_node_map) {
  AnfNodePtr ret = nullptr;
  auto flat_anf_name = TensorFlowUtils::GetFlattenNodeName(name);
  if (anf_node_map.find(flat_anf_name) != anf_node_map.end()) {
    ret = anf_node_map.at(flat_anf_name);
  } else if (anf_node_map.find(name + ":0") != anf_node_map.end()) {
    ret = anf_node_map.at(flat_anf_name + ":0");
  }
  return ret;
}

std::string GetOriginInputName(const tensorflow::NodeDef &node,
                               const std::map<std::string, const tensorflow::NodeDef *> &tf_graph_nodes) {
  if (node.op() != "Identity" && node.op() != "StopGradient") {
    return node.name();
  }
  auto tmp_node = &node;
  while (tmp_node->op() == "Identity" || tmp_node->op() == "StopGradient") {
    auto flatten_input_name = TensorFlowUtils::GetFlattenNodeName(tmp_node->input(0));
    if (tf_graph_nodes.find(flatten_input_name) == tf_graph_nodes.end()) {
      return flatten_input_name;
    }
    tmp_node = tf_graph_nodes.at(flatten_input_name);
  }
  return tmp_node->name();
}

STATUS CheckStrView(std::string_view str_view, uint64_t *scratch) {
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  for (size_t i = 0; i < static_cast<size_t>(*scratch); ++i) {
    if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
      return RET_ERROR;
    }
  }
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  if (!TensorFlowUtils::DecodeInt64(&str_view, scratch)) {
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS GetFloatValue(const tensorflow::TensorProto &tensor_proto, const tensorflow::TensorShapeProto &tensor_shape,
                     ParamValueLitePtr param_value, int shape_size) {
  auto tensor_data = new (std::nothrow) float[shape_size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    delete[] tensor_data;
    return RET_ERROR;
  }

  if (tensor_proto.float_val_size() == 1) {
    for (int i = 0; i < shape_size; i++) {
      tensor_data[i] = tensor_proto.float_val(0);
    }
  }
  if (tensor_proto.tensor_content().size() == shape_size * sizeof(float)) {
    const auto addr = reinterpret_cast<const float *>(tensor_proto.tensor_content().data());
    if (::memcpy_s(tensor_data, shape_size * sizeof(float), addr, shape_size * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      delete[] tensor_data;
      return RET_ERROR;
    }
  }
  auto tensor_size = shape_size * sizeof(float);
  param_value->SetTensorData(tensor_data, tensor_size);
  return RET_OK;
}

STATUS GetInt32Value(const tensorflow::TensorProto &tensor_proto, const tensorflow::TensorShapeProto &tensor_shape,
                     ParamValueLitePtr param_value, int shape_size) {
  auto tensor_data = new (std::nothrow) int[shape_size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    delete[] tensor_data;
    return RET_ERROR;
  }

  if (tensor_proto.int_val_size() == 1) {
    for (int i = 0; i < shape_size; i++) {
      tensor_data[i] = tensor_proto.int_val(0);
    }
  }
  if (shape_size != 0 && tensor_proto.tensor_content().size() == shape_size * sizeof(int32_t)) {
    const auto addr = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
    if (::memcpy_s(tensor_data, shape_size * sizeof(int32_t), addr, shape_size * sizeof(int32_t)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed";
      delete[] tensor_data;
      return RET_ERROR;
    }
  }
  auto tensor_size = shape_size * sizeof(int);
  param_value->SetTensorData(tensor_data, tensor_size);
  return RET_OK;
}

STATUS GetInt64Value(const tensorflow::TensorProto &tensor_proto, const tensorflow::TensorShapeProto &tensor_shape,
                     ParamValueLitePtr param_value, int shape_size) {
  param_value->set_tensor_type(kNumberTypeInt32);
  auto *tensor_data = new (std::nothrow) int[shape_size];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "new data failed";
    delete[] tensor_data;
    return RET_ERROR;
  }
  if (tensor_shape.dim_size() == 0) {  // scalar
    const auto &origin_data = tensor_proto.int64_val();
    for (int i = 0; i < tensor_proto.int64_val_size(); ++i) {
      if (origin_data[i] > static_cast<int64_t>(INT32_MAX) || origin_data[i] < static_cast<int64_t>(INT32_MIN)) {
        MS_LOG(ERROR) << "int64 data " << origin_data[i] << "too big to fit into int32";
        delete[] tensor_data;
        return RET_ERROR;
      } else {
        tensor_data[i] = static_cast<int>(origin_data[i]);
      }
    }
  } else {
    const auto origin_data = reinterpret_cast<const int64_t *>(tensor_proto.tensor_content().data());
    for (int i = 0; i < shape_size; ++i) {
      if (origin_data[i] > static_cast<int64_t>(INT32_MAX) || origin_data[i] < static_cast<int64_t>(INT32_MIN)) {
        MS_LOG(WARNING) << "int64 data " << origin_data[i] << "too big to fit into int32";
        tensor_data[i] = origin_data[i] > 0 ? INT32_MAX : INT32_MIN;
      } else {
        tensor_data[i] = static_cast<int>(origin_data[i]);
      }
    }
  }
  param_value->SetTensorData(tensor_data, shape_size * sizeof(int32_t));
  return RET_OK;
}

}  // namespace

STATUS TFModelParser::ConvertConstVariant(const tensorflow::TensorProto &tensor_proto,
                                          const ParamValueLitePtr &param_value) {
  if (tensor_proto.variant_val_size() != 1) {
    MS_LOG(ERROR) << "only support variant_val_size == 1 now";
    return RET_ERROR;
  }
  auto &variant = tensor_proto.variant_val(0);
  if (variant.type_name() != "tensorflow::TensorList" || variant.tensors_size() <= 0) {
    MS_LOG(DEBUG) << "Only nonempty TensorList type is supported now";
  }
  auto descriptor = variant.GetMetadata().descriptor;
  auto reflection = variant.GetMetadata().reflection;
  if (descriptor == nullptr || reflection == nullptr) {
    MS_LOG(ERROR) << "descriptor or reflection is nullptr";
    return RET_ERROR;
  }
  auto field_descriptor = descriptor->field(1);
  if (field_descriptor == nullptr) {
    MS_LOG(ERROR) << "field_descriptor is nullptr";
    return RET_ERROR;
  }
  if (field_descriptor->type() != google::protobuf::FieldDescriptor::TYPE_BYTES) {
    MS_LOG(ERROR) << "metadata type is not TYPE_BYTES";
    return RET_ERROR;
  }
  std::string_view str_view(reflection->GetString(variant, field_descriptor));
  uint64_t scratch;
  if (CheckStrView(str_view, &scratch) != RET_OK) {
    return RET_ERROR;
  }
  auto element_dtype = static_cast<size_t>(scratch);

  tensorflow::TensorShapeProto element_shape_proto;
  element_shape_proto.ParseFromString(std::string(str_view.data(), str_view.size()));
  auto dim_size = element_shape_proto.dim_size();
  auto tensor_data = new (std::nothrow) int[dim_size + 2];  // encode element_dtype,shape.size,shape[i]... into data
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_ERROR;
  }
  tensor_data[0] = TensorFlowUtils::GetTFDataType(tensorflow::DataType(element_dtype));
  tensor_data[1] = element_shape_proto.dim_size();
  for (int i = 0; i < dim_size; ++i) {
    auto dim = element_shape_proto.dim(i).size();
    if (dim > static_cast<int64_t>(INT32_MAX) || dim < static_cast<int64_t>(INT32_MIN)) {
      MS_LOG(ERROR) << "int64 data " << dim << " too big to fit into int32";
      delete[] tensor_data;
      return RET_ERROR;
    } else {
      tensor_data[i + 2] = static_cast<int>(dim);
    }
  }
  std::vector<int> tensor_list_data(dim_size + 2);
  tensor_list_data[0] = TensorFlowUtils::GetTFDataType(tensorflow::DataType(element_dtype));
  tensor_list_data[1] = element_shape_proto.dim_size();
  for (int i = 0; i < dim_size; i++) {
    auto dim = element_shape_proto.dim(i).size();
    if (dim > static_cast<int64_t>(INT32_MAX) || dim < static_cast<int64_t>(INT32_MIN)) {
      MS_LOG(ERROR) << "int64 data " << dim << " too big to fit into int32";
      delete[] tensor_data;
      return RET_ERROR;
    } else {
      tensor_list_data[i + 2] = static_cast<int>(dim);
    }
  }
  tensor_list_data.emplace_back(variant.tensors_size());
  for (const auto &tensor : variant.tensors()) {
    std::vector<int> single_tensor_data;
    single_tensor_data.emplace_back(tensor.tensor_shape().dim_size());
    for (int i = 0; i < tensor.tensor_shape().dim_size(); i++) {
      single_tensor_data.emplace_back(tensor.tensor_shape().dim(i).size());
    }
    tensor_list_data.insert(tensor_list_data.end(), single_tensor_data.begin(), single_tensor_data.end());
  }
  auto tensor_data_ptr = new (std::nothrow) int[tensor_list_data.size()];
  if (tensor_data_ptr == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return RET_NULL_PTR;
  }
  if (EOK != ::memcpy_s(tensor_data_ptr, tensor_list_data.size() * sizeof(int), tensor_list_data.data(),
                        tensor_list_data.size() * sizeof(int))) {
    MS_LOG(ERROR) << "memcpy_s failed";
    return RET_NULL_PTR;
  }
  param_value->SetTensorData(tensor_data_ptr, tensor_list_data.size() * sizeof(int));
  return RET_OK;
}

STATUS TFModelParser::GetValueFromType(const tensorflow::TensorProto &tensor_proto,
                                       const tensorflow::TensorShapeProto &tensor_shape, ParamValueLitePtr param_value,
                                       const TypeId &type, int shape_size) {
  if (type == kNumberTypeFloat32 || type == kNumberTypeFloat) {
    return GetFloatValue(tensor_proto, tensor_shape, param_value, shape_size);
  } else if (type == kNumberTypeInt32 || type == kNumberTypeInt) {
    return GetInt32Value(tensor_proto, tensor_shape, param_value, shape_size);
  } else if (type == kNumberTypeInt64) {
    return GetInt64Value(tensor_proto, tensor_shape, param_value, shape_size);
  } else if (type == kNumberTypeBool) {
    auto tensor_data = new (std::nothrow) int[shape_size];
    if (tensor_proto.bool_val_size() == 1) {
      int value = tensor_proto.bool_val(0);
      for (int i = 0; i < shape_size; i++) {
        tensor_data[i] = value;
      }
    }
    auto tensor_size = shape_size * sizeof(int);
    param_value->SetTensorData(tensor_data, tensor_size);
  } else if (type == kObjectTypeTensorType) {
    return ConvertConstVariant(tensor_proto, param_value);
  } else if (type == kObjectTypeString) {
    auto tensor_data = new (std::nothrow) string;
    if (tensor_proto.string_val_size() == 1) {
      *tensor_data = tensor_proto.string_val(0);
    } else {
      MS_LOG(ERROR) << "string size bigger than one, not support.";
      return RET_ERROR;
    }
    auto tensor_size = (*tensor_data).size();
    param_value->SetTensorData(tensor_data, tensor_size);
  } else {
    MS_LOG(ERROR) << "Unsupported dataType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertConstTensor(const tensorflow::NodeDef &node_def, const tensorflow::AttrValue &attr_value,
                                         const TypeId &type, const ParameterPtr &parameter,
                                         std::vector<int64_t> *shape_vector) {
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

  auto param_value = std::make_shared<ParamValueLite>();
  if (param_value == nullptr) {
    MS_LOG(ERROR) << "param_value is nullptr";
    return RET_ERROR;
  }
  param_value->set_tensor_type(type);
  if (GetValueFromType(tensor_proto, tensor_shape, param_value, type, shape_size) != RET_OK) {
    MS_LOG(ERROR) << "get value from type failed.";
    return RET_ERROR;
  }
  std::vector<int> param_shape(shape_vector->begin(), shape_vector->end());
  param_value->set_tensor_shape(param_shape);
  if (TensorFlowUtils::FindAttrValue(node_def, "data_format", const_cast<tensorflow::AttrValue *>(&attr_value))) {
    auto format = mindspore::lite::TensorFlowUtils::ParseNodeFormat(node_def);
    if (format == mindspore::Format::NUM_OF_FORMAT) {
      MS_LOG(ERROR) << "Do not support data format: " << attr_value.s();
    }
    param_value->set_format(format);
  } else {
    param_value->set_format(schema::Format::Format_NHWC);
  }
  parameter->set_default_param(param_value);
  return RET_OK;
}

STATUS TFModelParser::ConvertParameter(const tensorflow::NodeDef &node, const ParameterPtr &parameter,
                                       std::unordered_map<std::string, AnfNodePtr> *anf_node_map) {
  MS_ASSERT(node != nullptr);
  MS_ASSERT(parameter != nullptr);

  tensorflow::AttrValue attr_value;
  TypeId type = kNumberTypeFloat32;
  if (TensorFlowUtils::FindAttrValue(node, "dtype", &attr_value)) {
    type = TensorFlowUtils::GetTFDataType(attr_value.type());
  }

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
    auto status = ConvertConstTensor(node, attr_value, type, parameter, &shape_vector);
    if (status != RET_OK) {
      return status;
    }
  } else {
    graph_input_names_.emplace_back(node.name());  // only root graph need set graph input names
  }

  auto type_ptr = TypeIdToType(type == kNumberTypeInt64 ? kNumberTypeInt32 : type);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor is nullptr";
    return RET_ERROR;
  }
  parameter->set_name(node.name());
  parameter->set_abstract(abstract_tensor);

  (*anf_node_map)[node.name()] = parameter;
  (*anf_node_map)[node.name() + ":0"] = parameter;
  return RET_OK;
}

STATUS TFModelParser::ConvertGraphInputsAndConsts(
  const std::map<std::string, const tensorflow::NodeDef *> &tf_graph_nodes, const FuncGraphPtr &anf_graph,
  std::unordered_map<std::string, AnfNodePtr> *anf_node_map) {
  for (auto &pair : tf_graph_nodes) {
    bool have_data_depend = false;
    for (int i = 0; i < pair.second->input_size(); ++i) {
      auto name = pair.second->input(i);
      if (!name.empty() && name[0] != '^') {  // control_depend input start with "^"
        have_data_depend = true;
        break;
      }
    }
    if (!have_data_depend) {
      auto parameter = anf_graph->add_parameter();
      if (ConvertParameter(*pair.second, parameter, anf_node_map) != RET_OK) {
        MS_LOG(ERROR) << "convert Parameter Node failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
FuncGraphPtr paserTfFuction() { return nullptr; }
FuncGraphPtr TFModelParser::Parse(const std::string &modelFile, const std::string &weightFile,
                                  const QuantType &quantType) {
  NoSupportOp::GetInstance()->SetFmkType("TF");
  auto status = ValidateFileStr(modelFile, ".pb");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.pb";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  tf_root_graph_ = std::make_unique<tensorflow::GraphDef>();
  if (tf_root_graph_ == nullptr) {
    MS_LOG(ERROR) << "tf_root_graph_ is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  status = ReadProtoFromBinaryFile((const char *)modelFile.c_str(), tf_root_graph_.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Open modelFile for TF converter failed!";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  anf_root_graph_ = std::make_shared<FuncGraph>();
  if (anf_root_graph_ == nullptr) {
    MS_LOG(ERROR) << "funGraphPtr is nullptr";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }
  anf_root_graph_->set_attr("graph_name", MakeValue("main_graph"));
  anf_root_graph_->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_TF)));

  for (int i = 0; i < tf_root_graph_->node_size(); i++) {
    auto &node_def = tf_root_graph_->node(i);
    tf_root_graph_nodes_[node_def.name()] = &node_def;
  }

  status = ConvertGraphInputsAndConsts(tf_root_graph_nodes_, anf_root_graph_, &anf_root_node_map_);
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  bool success_flag = true;
  for (int i = 0; i < tf_root_graph_->node_size(); i++) {
    auto &node_def = tf_root_graph_->node(i);
    status = ConvertOps(node_def, tf_root_graph_nodes_, anf_root_graph_, &anf_root_node_map_);
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    if (status != RET_OK) {
      success_flag = false;
    }
  }
  if (!success_flag) {
    MS_LOG(ERROR) << "Convert ops failed.";
    return nullptr;
  }

  if (!nodes_with_null_input_.empty()) {
    status = ConnectNullInput();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Connect null inputs failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return nullptr;
    }
  }

  status = ConvertRootGraphOutputs();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert graph outputs failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertSubgraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert subgraph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  return anf_root_graph_;
}

STATUS TFModelParser::ConvertSubgraphInputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                            std::unordered_map<std::string, AnfNodePtr> *anf_sub_node_map,
                                            const tensorflow::FunctionDef &tf_sub_fuction, CNodePtr cnode,
                                            FuncGraphPtr sub_func_graph) {
  std::vector<ParameterPtr> sub_graph_inputs;
  auto &tf_sub_signature = tf_sub_fuction.signature();
  auto &sub_graph_name = tf_sub_signature.name();
  auto input_arg_size = tf_sub_signature.input_arg_size();
  for (int j = 0; j < input_arg_size; j++) {
    auto &input_arg = tf_sub_signature.input_arg(j);
    auto parameter = sub_func_graph->add_parameter();
    parameter->set_name(input_arg.name());
    (*anf_sub_node_map)[input_arg.name()] = parameter;
    auto root_inputs = cnode->inputs();
    if (opt::CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      parameter->set_abstract(root_inputs[j + 1]->abstract());
    } else {
      parameter->set_abstract(root_inputs[j + 2]->abstract());
    }
    sub_graph_inputs.emplace_back(parameter);
  }
  for (int j = 0; j < tf_sub_fuction.node_def_size(); j++) {
    auto &node_def = tf_sub_fuction.node_def(j);
    (*tf_sub_node_map)[node_def.name()] = &node_def;
  }
  if (ConvertGraphInputsAndConsts(*tf_sub_node_map, sub_func_graph, anf_sub_node_map) != RET_OK) {
    MS_LOG(ERROR) << "Convert subgraph consts failed";
    return RET_ERROR;
  }

  // hardcode subgraph inputs name
  for (size_t j = 0; j < sub_graph_inputs.size(); j++) {
    sub_graph_inputs[j]->set_name(sub_graph_name + "_input_" + std::to_string(j) + "_parameter");
  }

  return RET_OK;
}

STATUS TFModelParser::ConvertSubgraphOutputs(std::map<std::string, const tensorflow::NodeDef *> *tf_sub_node_map,
                                             const std::unordered_map<std::string, AnfNodePtr> &anf_sub_node_map,
                                             const tensorflow::FunctionDef &tf_sub_fuction,
                                             FuncGraphPtr sub_func_graph) {
  auto &tf_sub_signature = tf_sub_fuction.signature();
  auto &sub_graph_name = tf_sub_signature.name();

  std::vector<AnfNodePtr> sub_output_nodes;
  auto &subgraph_ret = tf_sub_fuction.ret();
  for (auto &output_arg : tf_sub_signature.output_arg()) {
    auto &signature_name = output_arg.name();
    if (subgraph_ret.find(signature_name) == subgraph_ret.end()) {
      MS_LOG(ERROR) << "can't find signature_name: " << signature_name;
      return RET_ERROR;
    }
    auto t = subgraph_ret.find(signature_name);
    MS_LOG(INFO) << "subret " << t->first << " " << t->second;
    auto tf_output_name = TensorFlowUtils::GetFlattenNodeName(t->second);
    AnfNodePtr anf_node = nullptr;
    if (tf_sub_node_map->find(tf_output_name) == tf_sub_node_map->end()) {
      anf_node = GetAnfNode(tf_output_name, anf_sub_node_map);
    } else {
      auto tf_real_name = GetOriginInputName(*tf_sub_node_map->at(tf_output_name), *tf_sub_node_map);
      anf_node = GetAnfNode(tf_real_name, anf_sub_node_map);
    }
    if (anf_node == nullptr) {
      MS_LOG(ERROR) << "can't find anf node,tf node flatten name" << tf_output_name;
      return RET_ERROR;
    }
    sub_output_nodes.push_back(anf_node);
  }
  if (MakeAnfGraphOutputs(&sub_output_nodes, sub_func_graph) != RET_OK) {
    MS_LOG(ERROR) << "cmake anf graph outputs node error";
    return RET_ERROR;
  }

  // hardcode subgraph outputs name
  if (sub_output_nodes.size() == 1) {
    if (utils::isa<CNodePtr>(sub_output_nodes[0])) {
      sub_output_nodes[0]->cast<CNodePtr>()->set_fullname_with_scope(sub_graph_name + "_output_0_cnode");
    } else if (utils::isa<ParameterPtr>(sub_output_nodes[0])) {
      sub_output_nodes[0]->cast<ParameterPtr>()->set_name(sub_graph_name + "_output_0_parameter");
    }
  } else {
    for (size_t j = 1; j < sub_output_nodes.size(); j++) {
      if (utils::isa<CNodePtr>(sub_output_nodes[j])) {
        sub_output_nodes[j]->cast<CNodePtr>()->set_fullname_with_scope(sub_graph_name + "_output_" +
                                                                       std::to_string(j - 1) + "_cnode");
      } else if (utils::isa<ParameterPtr>(sub_output_nodes[j])) {
        sub_output_nodes[j]->cast<ParameterPtr>()->set_name(sub_graph_name + "_output_" + std::to_string(j - 1) +
                                                            "_parameter");
      }
    }
  }

  return RET_OK;
}

STATUS TFModelParser::ConvertSubgraph() {
  std::map<CNodePtr, FuncGraphPtr> while_cond_map, while_body_map, if_then_map, if_else_map;
  bool success_flag = true;
  for (int i = 0; i < tf_root_graph_->library().function_size(); i++) {
    auto &tf_sub_fuction = tf_root_graph_->library().function(i);
    auto &tf_sub_signature = tf_sub_fuction.signature();
    auto input_arg_size = tf_sub_signature.input_arg_size();
    auto &sub_graph_name = tf_sub_signature.name();
    CNodePtr cnode = nullptr;
    if (function_while_map_.count(sub_graph_name)) {
      cnode = function_while_map_[sub_graph_name]->cast<CNodePtr>();
      if (cnode == nullptr || static_cast<int>(cnode->inputs().size()) != input_arg_size + 1) {
        MS_LOG(ERROR) << "while cnode  not equal input arg size";
        return RET_ERROR;
      }
    } else if (function_if_map_.count(sub_graph_name)) {
      cnode = function_if_map_[sub_graph_name]->cast<CNodePtr>();
      if (cnode == nullptr || static_cast<int>(cnode->inputs().size()) != input_arg_size + 2) {
        MS_LOG(ERROR) << "if cnode  not equal input arg size";
        return RET_ERROR;
      }
    } else {
      continue;
    }

    FuncGraphPtr sub_func_graph = std::make_shared<FuncGraph>();
    sub_func_graph->set_attr("graph_name", MakeValue(sub_graph_name));
    sub_func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_TF)));
    std::unordered_map<std::string, AnfNodePtr> anf_sub_node_map;
    std::map<std::string, const tensorflow::NodeDef *> tf_sub_node_map;

    if (ConvertSubgraphInputs(&tf_sub_node_map, &anf_sub_node_map, tf_sub_fuction, cnode, sub_func_graph) != RET_OK) {
      MS_LOG(ERROR) << "Convert subgraph inputs failed.";
      return RET_ERROR;
    }

    // convert sub graph ops
    STATUS status = RET_OK;
    for (int j = 0; j < tf_sub_fuction.node_def_size(); j++) {
      auto &node_def = tf_sub_fuction.node_def(j);
      status = ConvertOps(node_def, tf_sub_node_map, sub_func_graph, &anf_sub_node_map);
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Convert subgraph ops failed.";
        success_flag = false;
      }
    }
    if (!success_flag) {
      MS_LOG(ERROR) << "Convert subgraph is failed.";
      continue;
    }

    if (ConvertSubgraphOutputs(&tf_sub_node_map, anf_sub_node_map, tf_sub_fuction, sub_func_graph) != RET_OK) {
      MS_LOG(ERROR) << "Convert subgraph outputs failed.";
      return RET_ERROR;
    }

    // add while cond body function to while node input
    if (opt::CheckPrimitiveType(cnode, prim::kPrimWhile)) {
      if (sub_graph_name.find("cond") != std::string::npos) {
        while_cond_map[cnode] = sub_func_graph;
      } else {
        while_body_map[cnode] = sub_func_graph;
      }
    } else {
      if (sub_graph_name.find("true") != std::string::npos) {
        if_then_map[cnode] = sub_func_graph;
      } else {
        if_else_map[cnode] = sub_func_graph;
      }
    }
  }
  if (!success_flag) {
    MS_LOG(ERROR) << "Convert subgraph is failed.";
    return RET_ERROR;
  }
  if (ControlFlowNodePostProcess(while_cond_map, while_body_map) != RET_OK) {
    MS_LOG(ERROR) << "while node post process failed";
    return RET_ERROR;
  }
  if (ControlFlowNodePostProcess(if_then_map, if_else_map) != RET_OK) {
    MS_LOG(ERROR) << "if node post process failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TFModelParser::ControlFlowNodePostProcess(const std::map<CNodePtr, FuncGraphPtr> &first_func_map,
                                                 const std::map<CNodePtr, FuncGraphPtr> &second_func_map) {
  if (first_func_map.size() != second_func_map.size()) {
    MS_LOG(ERROR) << "while cond body size error";
    return RET_ERROR;
  }
  static auto root_func_manager = Manage(anf_root_graph_);

  for (auto &kv : first_func_map) {
    auto control_flow_node = kv.first;
    auto &first_sub_graph = kv.second;
    auto &second_sub_graph = second_func_map.at(control_flow_node);
    first_sub_graph->set_manager(root_func_manager);
    second_sub_graph->set_manager(root_func_manager);
    auto first_value_node = NewValueNode(first_sub_graph);
    auto second_value_node = NewValueNode(second_sub_graph);
    auto inputs = control_flow_node->inputs();
    inputs.insert(inputs.begin() + 1, {first_value_node, second_value_node});
    auto new_node = anf_root_graph_->NewCNode(inputs);  // must create new node, otherwise node_users won't update
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

STATUS TFModelParser::ConvertInputNodes(const tensorflow::NodeDef &node_def,
                                        const std::vector<std::string> &input_names,
                                        const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                        const std::unordered_map<std::string, AnfNodePtr> &anf_node_map,
                                        std::vector<AnfNodePtr> *inputs,
                                        std::vector<std::string> *input_name_not_found) {
  MS_ASSERT(node_def != nullptr);
  // parse inputs
  for (size_t j = 0; j < input_names.size(); j++) {
    std::string input_name = input_names[j];  // input may be produced by multi-outputs node
    // subgraph input name x:output:index,need flatten
    auto flatten_input_name = TensorFlowUtils::GetFlattenNodeName(input_name);
    if (tf_node_map.find(flatten_input_name) != tf_node_map.end()) {
      auto input_node = tf_node_map.at(flatten_input_name);
      flatten_input_name = GetOriginInputName(*input_node, tf_node_map);
    }
    auto input = GetAnfNode(flatten_input_name, anf_node_map);
    if (input == nullptr) {
      MS_LOG(WARNING) << node_def.name() << " input " << j << ": " << input_name << " can't find parsed in_nodes";
      (*input_name_not_found).push_back(flatten_input_name);
    }
    inputs->emplace_back(input);
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOutputTensor(const tensorflow::NodeDef &op, const CNodePtr &anf_node,
                                          std::unordered_map<std::string, AnfNodePtr> *anf_node_map,
                                          const FuncGraphPtr &anf_graph, int output_size) {
  MS_ASSERT(op != nullptr);
  MS_ASSERT(anf_node != nullptr);
  MS_ASSERT(anf_graph != nullptr);
  if (IsTensorListOp(anf_node) && output_size != 1) {
    MS_LOG(ERROR) << "tensorlist output op output_size !=1";
    return RET_ERROR;
  }
  if (output_size == 0) {
    return RET_OK;
  } else if (output_size == 1) {
    auto type = kFloat32;
    std::vector<int64_t> shape_vector;
    if (IsTensorListOp(anf_node)) {
      type = TypeIdToType(kObjectTypeTensorType);
    }
    auto abstract = std::make_shared<abstract::AbstractTensor>(type, shape_vector);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "create AbstractTensor failed";
      return RET_ERROR;
    }
    anf_node->set_abstract(abstract);
    anf_node_map->insert(std::pair(op.name(), anf_node));
  } else {
    AbstractBasePtrList abstractList;
    for (int output_idx = 0; output_idx < output_size; output_idx++) {
      std::vector<int64_t> shape_vector;
      abstractList.emplace_back(std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector));
      auto tupleGetItemPrimPtr = std::make_shared<ops::TupleGetItem>();
      if (tupleGetItemPrimPtr == nullptr) {
        MS_LOG(ERROR) << "new TupleGetItem failed";
        return RET_NULL_PTR;
      }
      auto tupleGetItemPrim = NewValueNode(tupleGetItemPrimPtr);
      auto getItemValue = NewValueNode(MakeValue<int>(output_idx));
      std::vector<AnfNodePtr> inputs{tupleGetItemPrim, anf_node, getItemValue};
      CNodePtr getItemCNode = anf_graph->NewCNode(inputs);
      std::string output_item_name = anf_node->fullname_with_scope() + "_getitem_" + std::to_string(output_idx);
      auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape_vector);
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "create AbstractTensor failed";
        return RET_ERROR;
      }
      getItemCNode->set_abstract(abstract);
      getItemCNode->set_fullname_with_scope(output_item_name);
      anf_node_map->insert(std::pair(op.name() + ":" + std::to_string(output_idx), getItemCNode));
    }
    anf_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstractList));
  }
  return RET_OK;
}

STATUS TFModelParser::RecordNullInput(const CNodePtr &node, const std::vector<std::string> &input_name_not_found) {
  nodes_with_null_input_.emplace_back(node, input_name_not_found);
  return RET_OK;
}

STATUS TFModelParser::ConnectNullInput() {
  for (auto &it : nodes_with_null_input_) {
    auto &cnode = it.first;
    auto &input_name_not_found = it.second;
    auto &inputs = cnode->inputs();
    int i = 0;
    for (size_t j = 0; j < inputs.size(); ++j) {
      if (inputs[j] == nullptr) {
        cnode->set_input(j, GetAnfNode(input_name_not_found[i], anf_root_node_map_));
        ++i;
      }
    }
  }
  return RET_OK;
}

STATUS TFModelParser::ConvertOps(const tensorflow::NodeDef &node_def,
                                 const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                 const FuncGraphPtr &func_graph_ptr,
                                 std::unordered_map<std::string, AnfNodePtr> *anf_node_map) {
  MS_ASSERT(node_def != nullptr);
  MS_ASSERT(func_graph_ptr != nullptr);
  STATUS status = RET_OK;
  const auto &op_type = node_def.op();
  if (op_type == "Placeholder" || op_type == "Const" || op_type == "Identity" || op_type == "StopGradient") {
    return RET_OK;
  }

  MS_LOG(INFO) << "parse op : " << op_type;
  auto node_parser = TFNodeParserRegistry::GetInstance()->GetNodeParser(op_type);
  if (node_parser == nullptr) {
    NoSupportOp::GetInstance()->InsertOp(op_type);
    MS_LOG(ERROR) << "cannot find node parser: " << node_def.name() << " in "
                  << func_graph_ptr->get_attr("graph_name")->ToString();
    return RET_NOT_FIND_OP;
  }

  int output_size;
  std::vector<std::string> input_names;
  auto primitiveC = node_parser->Parse(node_def, tf_node_map, &input_names, &output_size);
  if (primitiveC == nullptr) {
    MS_LOG(ERROR) << "node " << op_type << " parser failed";
    return RET_ERROR;
  }
  auto value_node = NewValueNode(std::shared_ptr<ops::PrimitiveC>(primitiveC));
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "value_node is nullptr";
    return RET_ERROR;
  }
  std::vector<AnfNodePtr> inputs = {value_node};
  std::vector<std::string> input_name_not_found{};
  status = ConvertInputNodes(node_def, input_names, tf_node_map, *anf_node_map, &inputs, &input_name_not_found);
  if (status != RET_OK) {
    return status;
  }
  // control_depends are not processed currently
  auto anf_node = func_graph_ptr->NewCNode(inputs);
  anf_node->set_fullname_with_scope(node_def.name());
  if (op_type == "StatelessWhile" || op_type == "While") {
    MS_LOG(INFO) << "find while node:" << node_def.name();
    tensorflow::AttrValue attr_value;
    if (TensorFlowUtils::FindAttrValue(node_def, "body", &attr_value)) {
      auto body_name = attr_value.func().name();
      function_while_map_[body_name] = anf_node;
      MS_LOG(DEBUG) << "parse body name:" << body_name;
    }
    if (TensorFlowUtils::FindAttrValue(node_def, "cond", &attr_value)) {
      auto cond_name = attr_value.func().name();
      function_while_map_[cond_name] = anf_node;
      MS_LOG(DEBUG) << "parse cond name:" << cond_name;
    }
  } else if (op_type == "StatelessIf") {
    MS_LOG(INFO) << "find if node:" << node_def.name();
    tensorflow::AttrValue attr_value;
    if (TensorFlowUtils::FindAttrValue(node_def, "then_branch", &attr_value)) {
      auto then_name = attr_value.func().name();
      function_if_map_[then_name] = anf_node;
      MS_LOG(DEBUG) << "parse then name:" << then_name;
    }
    if (TensorFlowUtils::FindAttrValue(node_def, "else_branch", &attr_value)) {
      auto else_name = attr_value.func().name();
      function_if_map_[else_name] = anf_node;
      MS_LOG(DEBUG) << "parse else name:" << else_name;
    }
  }

  if (!input_name_not_found.empty()) {
    RecordNullInput(anf_node, input_name_not_found);
  }

  status = ConvertOutputTensor(node_def, anf_node, anf_node_map, func_graph_ptr, output_size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert output tensors for " << anf_node->fullname_with_scope() << " failed.";
  }
  return status;
}

STATUS TFModelParser::ConvertRootGraphOutputs() {
  // because output of intermediate node in anf graph may also be output tensors, we search output tensors in
  // tf_root_graph_nodes_ but not anf_root_node_map_
  std::set<std::string> all_node_inputs;
  std::vector<AnfNodePtr> output_nodes;
  for (auto &pair : tf_root_graph_nodes_) {
    for (int i = 0; i < pair.second->input_size(); ++i) {
      all_node_inputs.insert(TensorFlowUtils::GetNodeName(pair.second->input(i)));
      auto input_name = pair.second->input(i);
      if (input_name[0] == '^') {
        input_name.erase(0, 1);
      }
      all_node_inputs.insert(input_name);
    }
  }
  for (auto &pair : tf_root_graph_nodes_) {
    if (pair.second->op() == "Assert") {
      continue;
    }
    auto it = all_node_inputs.find(pair.first);
    if (it == all_node_inputs.end() && pair.second->input_size() > 0) {  // output node not constraint to Identity
      auto origin_name = GetOriginInputName(*(pair.second), tf_root_graph_nodes_);
      auto anf_node = GetAnfNode(origin_name, anf_root_node_map_);
      if (anf_node == nullptr) {
        MS_LOG(ERROR) << "can't find anf node: " << origin_name;
        return RET_ERROR;
      }
      output_nodes.push_back(anf_node);
      graph_output_names_.push_back(anf_node->fullname_with_scope());
    }
  }
  auto status = MakeAnfGraphOutputs(&output_nodes, anf_root_graph_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "make anf graph outputs node error";
    return status;
  }
  return RET_OK;
}
STATUS TFModelParser::MakeAnfGraphOutputs(std::vector<AnfNodePtr> *output_nodes, const FuncGraphPtr &anf_graph) {
  if (output_nodes->empty() || anf_graph == nullptr) {
    MS_LOG(ERROR) << "anf output nodes empty or  null anf graph";
    return RET_ERROR;
  }
  if (output_nodes->size() > 1) {
    std::vector<AnfNodePtr> *make_tuple_inputs = output_nodes;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs->insert(make_tuple_inputs->begin(), make_tuple_prim);
    auto make_tuple_cnode = anf_graph->NewCNode(*make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs = {value_node, make_tuple_cnode};
    auto cnode = anf_graph->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("Return");
    anf_graph->set_return(cnode);
  } else {
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    std::vector<AnfNodePtr> op_inputs{value_node, output_nodes->front()};
    auto return_cnode = anf_graph->NewCNode(op_inputs);
    return_cnode->set_fullname_with_scope("Return");
    anf_graph->set_return(return_cnode);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
