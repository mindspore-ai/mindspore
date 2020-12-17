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
#include <set>
#include <utility>
#include <unordered_map>
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"

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

std::set<std::string> SPECIAL_NODE = {"Gemm", "Loop"};
FuncGraphPtr OnnxModelParser::Parse(const std::string &model_file, const std::string &weight_file,
                                    const QuantType &quant_type) {
  NoSupportOp::GetInstance()->SetFmkType("ONNX");
  auto status = InitOriginModel(model_file);
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "init origin model failed.";
    return nullptr;
  }

  func_graph_ptr_ = std::make_shared<FuncGraph>();
  if (func_graph_ptr_ == nullptr) {
    MS_LOG(ERROR) << "funcgraph is nullptr.";
    return nullptr;
  }

  status = ConvertConstTensors();
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert const nodes failed.";
    return nullptr;
  }

  status = ConvertGraphInputs();
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph inputs failed.";
    return nullptr;
  }

  status = ConvertNodes();
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert nodes failed.";
    return nullptr;
  }

  status = ConvertGraphOutputs();
  if (RET_OK != status) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    MS_LOG(ERROR) << "convert graph outputs failed.";
    return nullptr;
  }
  func_graph_ptr_->set_attr("graph_name", MakeValue("main_graph"));
  return func_graph_ptr_;
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
  onnx_graph_ = onnx_model_.graph();
  return RET_OK;
}

STATUS OnnxModelParser::ConvertConstTensors() {
  for (const auto &onnx_const_value : onnx_graph_.initializer()) {
    auto parameter = func_graph_ptr_->add_parameter();
    auto status = BuildParameterNode(parameter, onnx_const_value);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "parameter node build failed.";
      return status;
    }
    nodes_.emplace(onnx_const_value.name(), parameter);
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertGraphInputs() {
  for (int i = 0; i < onnx_graph_.input().size(); ++i) {
    const auto &input_value = onnx_graph_.input(i);
    if (nodes_.find(input_value.name()) != nodes_.end()) {
      continue;
    }
    auto parameter = func_graph_ptr_->add_parameter();
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
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    parameter->set_abstract(abstract_tensor);
    parameter->set_name(input_value.name());
    nodes_.emplace(input_value.name(), parameter);
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertNodes() {
  STATUS status = RET_OK;
  for (const auto &onnx_node : onnx_graph_.node()) {
    auto node_parser = OnnxNodeParserRegistry::GetInstance()->GetNodeParser(onnx_node.op_type());
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(onnx_node.op_type());
      status = status == RET_OK ? RET_NOT_FIND_OP : status;
    }
    if (status != RET_OK) {
      continue;
    }
    auto primitive_c = node_parser->ParseLitePrimitive(onnx_graph_, onnx_node);
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
      auto status_node = ConvertSpecialOnnxNode(onnx_node, primitive_c);
      status = status == RET_OK ? status_node : status;
      continue;
    }
    // build CNode
    status = BuildCNode(onnx_node, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "build cnode " << onnx_node.op_type() << " failed.";
    }
  }
  return status;
}

STATUS OnnxModelParser::ConvertGraphOutputs() {
  std::vector<AnfNodePtr> return_inputs;
  if (onnx_graph_.output_size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = GetMakeTuplePrim();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "GetMakeTuplePrim return nullptr";
      return RET_NULL_PTR;
    }
    for (const auto &graph_out : onnx_graph_.output()) {
      if (nodes_.find(graph_out.name()) == nodes_.end()) {
        MS_LOG(ERROR) << "graph output get failed.";
        return RET_ERROR;
      }
      auto cnode = nodes_[graph_out.name()];
      if (nullptr == cnode) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_cnode = func_graph_ptr_->NewCNode(make_tuple_prim_ptr, make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");
    return_inputs.emplace_back(make_tuple_cnode);
  } else {
    const auto &graph_out = onnx_graph_.output(0);
    if (nodes_.find(graph_out.name()) == nodes_.end()) {
      MS_LOG(ERROR) << "graph output get failed.";
      return RET_ERROR;
    }
    auto cnode = nodes_[graph_out.name()];
    if (nullptr == cnode) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    return_inputs.emplace_back(cnode);
  }
  if (BuildReturnNode(return_inputs) != RET_OK) {
    MS_LOG(ERROR) << "build return node failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildReturnNode(const std::vector<AnfNodePtr> &return_inputs) {
  auto returnPrim = GetReturnPrim();
  if (returnPrim == nullptr) {
    MS_LOG(ERROR) << "GetReturnPrim return nullptr";
    return RET_NULL_PTR;
  }
  auto returnCnode = func_graph_ptr_->NewCNode(returnPrim, return_inputs);
  returnCnode->set_fullname_with_scope("return");
  func_graph_ptr_->set_return(returnCnode);
  return RET_OK;
}

STATUS OnnxModelParser::BuildCNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  std::vector<AnfNodePtr> op_inputs;
  for (const auto &input_name : onnx_node.input()) {
    if (input_name.empty()) {
      continue;
    }
    if (nodes_.find(input_name) == nodes_.end()) {
      MS_LOG(ERROR) << "op " << onnx_node.op_type() << " inputs get failed.";
      return RET_ERROR;
    } else {
      op_inputs.push_back(nodes_[input_name]);
    }
  }
  auto new_cnode = func_graph_ptr_->NewCNode(std::shared_ptr<lite::PrimitiveC>(primitive_c), op_inputs);
  new_cnode->set_fullname_with_scope(onnx_node.op_type() + "_" + onnx_node.output(0));
  auto status = BuildOpOutputs(onnx_node, new_cnode);
  return status;
}

STATUS OnnxModelParser::BuildOpOutputs(const onnx::NodeProto &onnx_node, const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "parameter is null, get output tensor failed.";
    return RET_NULL_PTR;
  }
  if (onnx_node.output_size() == 1) {
    auto type_ptr = TypeIdToType(kTypeUnknown);
    std::vector<int64_t> shape_vector;
    cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    nodes_.emplace(onnx_node.output(0), cnode);
  } else {
    AbstractBasePtrList abstract_list;
    int op_idx = 0;
    for (const auto &output_name : onnx_node.output()) {
      std::vector<int64_t> shape_vector;
      auto type_ptr = TypeIdToType(kTypeUnknown);
      abstract_list.emplace_back(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
      auto tuple_get_item_prim_ptr = GetTupleGetItemPrim();
      if (tuple_get_item_prim_ptr == nullptr) {
        MS_LOG(ERROR) << "GetTupleGetItemPrim return nullptr";
        return RET_NULL_PTR;
      }
      auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
      auto get_item_value = NewValueNode(MakeValue<int>(op_idx));
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
      CNodePtr get_item_cnode = func_graph_ptr_->NewCNode(inputs);
      get_item_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_getitem_" + std::to_string(op_idx));
      nodes_.emplace(output_name, get_item_cnode);
      op_idx++;
    }
    cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
  return RET_OK;
}

STATUS OnnxModelParser::ConvertOpQuantParams(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c) {
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
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    const auto &input_name = onnx_node.input(i);
    std::vector<schema::QuantParamT> quant_params;
    status = SetTensorQuantParam(input_name, &quant_params);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "set input tensor quant param failed.";
      return status;
    }
    primitive_c->AddInputQuantParam(quant_params);
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
    primitive_c->AddOutputQuantParam(quant_params);
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
  quant_params->clear();
  auto quant_param = std::make_unique<QuantParamT>();
  for (int i = 0; i < onnx_graph_.quantization_annotation_size(); ++i) {
    auto tensor_annotation = onnx_graph_.quantization_annotation(i);
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
  auto iter = nodes_.find(tensor_name);
  if (iter == nodes_.end()) {
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

STATUS OnnxModelParser::ConvertSpecialOnnxNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "imitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  STATUS status = RET_OK;
  if (onnx_node.op_type() == "Loop") {
    MS_LOG(ERROR) << "loop hasn't supported.";
    return RET_NOT_FIND_OP;
  } else if (onnx_node.op_type() == "Gemm") {
    status = ConvertOnnxGemmNode(onnx_node, primitive_c);
  } else {
    MS_LOG(ERROR) << "the node is not special node.";
    status = RET_ERROR;
  }
  delete primitive_c;
  return status;
}

STATUS OnnxModelParser::ConvertOnnxGemmNode(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c) {
  if (onnx_node.op_type() != "Gemm") {
    MS_LOG(ERROR) << "this op is not gemm, it is " << onnx_node.op_type();
    return RET_ERROR;
  }
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  auto status = BuildCNodeForGemm(onnx_node, primitive_c, "MatMul");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert gemm node failed.";
    return status;
  }
  status = BuildCNodeForGemm(onnx_node, primitive_c, "BiasAdd");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert gemm node failed.";
    return status;
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildCNodeForGemm(const onnx::NodeProto &onnx_node, lite::PrimitiveC *primitive_c,
                                          const std::string &name) {
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr.";
    return RET_NULL_PTR;
  }
  auto value = primitive_c->GetAttr(name);
  primitive_c->EraseAttr(name);
  if (value == nullptr) {
    MS_LOG(ERROR) << "op parse failed.";
    return RET_NULL_PTR;
  }
  auto prim_ptr = value->cast<std::shared_ptr<lite::PrimitiveC>>();
  if (prim_ptr == nullptr) {
    MS_LOG(ERROR) << "p parse failed.";
    return RET_NULL_PTR;
  }
  auto type_ptr = TypeIdToType(kTypeUnknown);
  std::vector<int64_t> shape_vector;
  std::vector<AnfNodePtr> op_inputs;
  if (name == "MatMul") {
    for (int i = 0; i < 2; ++i) {
      if (nodes_.find(onnx_node.input(i)) == nodes_.end()) {
        MS_LOG(ERROR) << "op " << onnx_node.op_type() << " inputs get failed.";
        return RET_ERROR;
      } else {
        op_inputs.push_back(nodes_[onnx_node.input(i)]);
        prim_ptr->AddInputQuantParam(primitive_c->input_quant_params().at(i));
      }
    }
    prim_ptr->AddOutputQuantParam(std::vector<schema::QuantParamT>(1));
    auto new_cnode = func_graph_ptr_->NewCNode(prim_ptr, op_inputs);
    new_cnode->set_fullname_with_scope("Gemm_MatMul_" + onnx_node.output(0));
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    nodes_.emplace("Gemm_MatMul_" + onnx_node.output(0), new_cnode);
  } else {
    if (nodes_.find("Gemm_MatMul_" + onnx_node.output(0)) == nodes_.end() ||
        nodes_.find(onnx_node.input(2)) == nodes_.end()) {
      MS_LOG(ERROR) << "op " << onnx_node.op_type() << " inputs get failed.";
      return RET_ERROR;
    }
    op_inputs.push_back(nodes_["Gemm_MatMul_" + onnx_node.output(0)]);
    op_inputs.push_back(nodes_[onnx_node.input(2)]);
    prim_ptr->AddInputQuantParam(std::vector<schema::QuantParamT>(1));
    prim_ptr->AddInputQuantParam(primitive_c->input_quant_params().at(2));
    prim_ptr->AddOutputQuantParam(primitive_c->output_quant_params().front());
    auto new_cnode = func_graph_ptr_->NewCNode(prim_ptr, op_inputs);
    new_cnode->set_fullname_with_scope("Gemm_BiasAdd_" + onnx_node.output(0));
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    nodes_.emplace(onnx_node.output(0), new_cnode);
  }
  return RET_OK;
}

STATUS OnnxModelParser::BuildParameterNodeForQuantParam(void *data, const std::string &name, TypeId type) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "value is nullptr.";
    return RET_NULL_PTR;
  }
  if (type != kNumberTypeInt64 && type != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "quant param type don't support.";
    return RET_NOT_SUPPORT;
  }
  std::vector<int64_t> shape_vector;
  auto parameter_node = func_graph_ptr_->add_parameter();
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), shape_vector);
  parameter_node->set_abstract(abstract_tensor);
  parameter_node->set_name(name);
  std::vector<int> shape;
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
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
  if (memcpy_s(tensor_data, data_size, data, data_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    delete[] tensor_data;
    return RET_ERROR;
  }
  param_value->SetTensorData(tensor_data, data_size);
  parameter_node->set_default_param(param_value);
  nodes_.emplace(name, parameter_node);
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
  MS_ASSERT(param_value != nullptr);
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
  std::for_each(onnx_const_tensor.dims().begin(), onnx_const_tensor.dims().end(),
                [&data_count](int dim) { data_count *= dim; });
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
