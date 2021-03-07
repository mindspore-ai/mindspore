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
#include "tools/converter/parser/caffe/caffe_model_parser.h"
#include <vector>
#include <iostream>
#include <map>
#include <memory>
#include <algorithm>
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"
#include "tools/converter/parser/caffe/caffe_inspector.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "src/param_value_lite.h"
#include "ops/return.h"
#include "ops/make_tuple.h"
#include "ops/tuple_get_item.h"
#include "ir/func_graph.h"
#include "tools/converter/converter_flags.h"

namespace mindspore::lite {
bool IsSkipedLayer(const caffe::LayerParameter &layer) {
  if (layer.type() == "Input" || layer.type() == "Dropout" || layer.type() == "Split") {
    return true;
  }
  return layer.include_size() == 1 && layer.include(0).phase() == caffe::TRAIN;
}

CaffeModelParser::CaffeModelParser() = default;

CaffeModelParser::~CaffeModelParser() = default;

FuncGraphPtr CaffeModelParser::Parse(const std::string &model_file, const std::string &weight_file,
                                     const QuantType &quant_type) {
  STATUS status = InitOriginModel(model_file, weight_file);
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  func_graph_ptr_ = std::make_shared<FuncGraph>();
  status = ConvertGraphInputs();
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertLayers();
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }

  status = ConvertGraphOutputs();
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  func_graph_ptr_->set_attr("graph_name", MakeValue("main_graph"));
  func_graph_ptr_->set_attr("fmk", MakeValue(static_cast<int>(converter::FmkType_CAFFE)));
  return func_graph_ptr_;
}

STATUS CaffeModelParser::ConvertLayers() {
  STATUS status = RET_OK;
  std::map<std::string, caffe::LayerParameter> weight_layers;
  for (int i = 0; i < caffe_weight_.layer_size(); i++) {
    auto weight_layer = caffe_weight_.layer(i);
    weight_layers[weight_layer.name()] = weight_layer;
  }
  for (int i = 0; i < caffe_model_.layer_size(); i++) {
    auto layer = caffe_model_.layer(i);

    // save caffe layers
    for (int top_idx = 0; top_idx < layer.top_size(); top_idx++) {
      caffe_layers_[layer.top(top_idx)] = layer;
    }
    caffe::LayerParameter weight;
    if (weight_layers.find(layer.name()) != weight_layers.end()) {
      weight = weight_layers.find(layer.name())->second;
    }

    if (IsSkipedLayer(layer)) {
      continue;
    }

    // parse primitive
    MS_LOG(INFO) << "parse op : " << layer.type();
    auto node_parser = CaffeNodeParserRegistry::GetInstance()->GetNodeParser(layer.type());
    if (node_parser == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(layer.type());
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      continue;
    }

    if (status != RET_OK) {
      continue;
    }

    auto primitive_c = node_parser->Parse(layer, weight);
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "parse node " << layer.name() << " failed.";
      status = RET_ERROR;
      continue;
    }

    // build inputs
    std::vector<AnfNodePtr> input_nodes;
    status = ConvertBottom(layer, &input_nodes);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert layer bottom for " << layer.name() << " failed.";
      continue;
    }

    // build weights
    std::vector<ParameterPtr> const_parameters;
    status = ConvertBlobs(weight, &const_parameters);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert blobs for " << layer.name() << " failed.";
      continue;
    }

    // build cnode
    std::vector<AnfNodePtr> op_inputs = {NewValueNode(std::shared_ptr<ops::PrimitiveC>(primitive_c))};
    op_inputs.insert(op_inputs.end(), input_nodes.begin(), input_nodes.end());
    op_inputs.insert(op_inputs.end(), const_parameters.begin(), const_parameters.end());
    auto new_cnode = func_graph_ptr_->NewCNode(op_inputs);
    new_cnode->set_fullname_with_scope(layer.name());

    // convert outputs
    status = ConvertTop(layer, new_cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert outputs for " << layer.name() << " failed.";
      continue;
    }

    status = ConvertLayerQuantParams(layer, weight, primitive_c);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert quant params for " << layer.name() << " failed.";
      continue;
    }
  }
  return status;
}

STATUS CaffeModelParser::InitOriginModel(const std::string &model_file, const std::string &weight_file) {
  int status = ValidateFileStr(model_file, ".prototxt");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.prototxt";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_INPUT_PARAM_INVALID;
  }

  if (weight_file.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: weightFile is necessary";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_GRAPH_FILE_ERR);
    return RET_INPUT_PARAM_INVALID;
  }

  status = ValidateFileStr(weight_file, ".caffemodel");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: weightFile must be *.caffemodel";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_INPUT_PARAM_INVALID;
  }

  status = ReadProtoFromText((const char *)model_file.c_str(), &caffe_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read prototxt file failed, model path: " << model_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  status = ReadProtoFromBinaryFile((const char *)weight_file.c_str(), &caffe_weight_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read caffemodel file failed, model path: " << weight_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputs() {
  for (int i = 0; i < caffe_model_.layer_size(); i++) {
    auto layer = caffe_model_.layer(i);
    if (layer.type() == "Input") {
      auto parameter = func_graph_ptr_->add_parameter();
      std::vector<int64_t> shape;
      for (int j = 0; j < layer.input_param().shape(0).dim_size(); j++) {
        shape.push_back(layer.input_param().shape(0).dim(j));
      }
      auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
      auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
      parameter->set_abstract(abstract_tensor);
      parameter->set_name("graph_input-" + std::to_string(i));
      nodes_.insert(std::pair(layer.top(0), parameter));
    }
  }

  if (caffe_model_.input_dim_size() > 0) {
    for (int i = 0; i < caffe_model_.input_size(); i++) {
      std::vector<int64_t> shape;
      if (caffe_model_.input_dim_size() > 4) {
        int step = caffe_model_.input_dim_size() / caffe_model_.input_size();
        for (int j = i * step; j < (i + 1) * step; j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      } else {
        for (int j = 0; j < caffe_model_.input_dim_size(); j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      }
      auto parameter = func_graph_ptr_->add_parameter();
      auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
      auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
      parameter->set_abstract(abstract_tensor);
      parameter->set_name("graph_input-" + caffe_model_.input(i));
      nodes_.insert(std::pair(caffe_model_.input(i), parameter));
    }
  } else {
    for (int i = 0; i < caffe_model_.input_shape_size(); i++) {
      auto shape = caffe_model_.input_shape(i);
      std::vector<int64_t> shape_vector;
      for (int j = 0; j < shape.dim_size(); j++) {
        shape_vector.push_back(shape.dim(j));
      }
      auto parameter = func_graph_ptr_->add_parameter();
      auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
      auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
      parameter->set_abstract(abstract_tensor);
      parameter->set_name("graph_input-" + caffe_model_.input(i));
      nodes_.insert(std::pair(caffe_model_.input(i), parameter));
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphOutputs() {
  CaffeInspector caffeInspector;
  caffeInspector.InspectModel(caffe_model_);
  if (caffeInspector.GetGraphOutput().size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = NewValueNode(make_tuple_prim_ptr);
    make_tuple_inputs.emplace_back(make_tuple_prim);
    for (const auto &output_node : caffeInspector.GetGraphOutput()) {
      if (nodes_.find(output_node) == nodes_.end()) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      auto cnode = nodes_.find(output_node)->second;
      make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_cnode = func_graph_ptr_->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    std::vector<AnfNodePtr> op_inputs;
    auto return_prim_ptr = std::make_shared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = NewValueNode(return_prim_ptr);
    op_inputs.emplace_back(value_node);
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = func_graph_ptr_->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("Return");
    func_graph_ptr_->set_return(cnode);
  } else {
    auto returnPrim = std::make_shared<ops::Return>();
    if (returnPrim == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto valueNode = NewValueNode(returnPrim);
    std::vector<AnfNodePtr> opInputs{valueNode};
    if (nodes_.find(*caffeInspector.GetGraphOutput().begin()) == nodes_.end()) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    auto cnode = nodes_.find(*caffeInspector.GetGraphOutput().begin())->second;
    if (nullptr == cnode) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    opInputs.emplace_back(cnode);
    auto returnCnode = func_graph_ptr_->NewCNode(opInputs);
    returnCnode->set_fullname_with_scope("Return");
    func_graph_ptr_->set_return(returnCnode);
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertLayerQuantParams(const caffe::LayerParameter &layer,
                                                 const caffe::LayerParameter &weight, ops::PrimitiveC *primitive_c) {
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null, get quant params failed.";
    return RET_NULL_PTR;
  }
  auto quant_params_holder = std::make_shared<QuantParamHolder>();
  for (auto input_idx : layer.bottom()) {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    quant_params_holder->AddInputQuantParam(notinited_quant_params);
  }
  for (auto input_idx : weight.blobs()) {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    quant_params_holder->AddInputQuantParam(notinited_quant_params);
  }
  for (auto output_idx : layer.top()) {
    std::vector<schema::QuantParamT> notinited_quant_params(1);
    quant_params_holder->AddOutputQuantParam(notinited_quant_params);
  }
  primitive_c->AddAttr("quant_params", quant_params_holder);
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBlobs(const caffe::LayerParameter &layer, std::vector<ParameterPtr> *const_parameters) {
  if (const_parameters == nullptr) {
    MS_LOG(ERROR) << "const parameters are null";
    return RET_NULL_PTR;
  }

  // Layer must have Filter
  if (layer.blobs_size() == 0) {
    MS_LOG(INFO) << "No filter data in layer " << layer.name().c_str();
    return RET_OK;
  }
  for (int i = 0; i < layer.blobs_size(); i++) {
    std::vector<int32_t> shape;
    ConvertShape(layer.blobs(i), &shape);

    // cal Weight num
    auto parameter = func_graph_ptr_->add_parameter();
    auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    parameter->set_abstract(abstract_tensor);
    if (layer.type() == "Convolution" || layer.type() == "Deconvolution") {
      if (i == 0) {
        parameter->set_name(layer.name() + "/weight");
      } else if (i == 1) {
        parameter->set_name(layer.name() + "/bias");
      }
    } else {
      parameter->set_name(layer.name() + "/input-" + std::to_string(i + layer.top_size()));
    }
    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    MS_ASSERT(param_value != nullptr);
    param_value->set_tensor_shape(shape);
    param_value->set_tensor_type(TypeId::kNumberTypeFloat32);
    param_value->set_format(schema::Format::Format_NCHW);

    int count = 0;
    if (layer.blobs(i).double_data_size() > 0) {
      count = layer.blobs(i).double_data_size();
      auto buf = std::make_unique<float[]>(count);
      for (int j = 0; j < count; ++j) {
        buf[j] = layer.blobs(j).double_data(j);
      }
      param_value->set_tensor_addr(buf.release());
    } else {
      count = layer.blobs(i).data_size();
      auto buf = std::make_unique<float[]>(count);
      if (buf == nullptr) {
        MS_LOG(INFO) << "new buffer failed";
        return RET_NULL_PTR;
      }
      const float *data_ptr = layer.blobs(i).data().data();
      if (data_ptr == nullptr) {
        MS_LOG(INFO) << "data of origin layer is nullptr";
        return RET_NULL_PTR;
      }
      if (EOK != ::memcpy_s(buf.get(), count * sizeof(float), data_ptr, count * sizeof(float))) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return RET_ERROR;
      }
      param_value->set_tensor_addr(buf.release());
    }
    param_value->set_tensor_size(count * sizeof(float));
    parameter->set_default_param(param_value);
    const_parameters->emplace_back(parameter);
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBottom(const caffe::LayerParameter &layer, std::vector<AnfNodePtr> *input_nodes) {
  if (input_nodes == nullptr) {
    MS_LOG(ERROR) << "input_nodes is null";
    return RET_NULL_PTR;
  }
  for (int i = 0; i < layer.bottom_size(); i++) {
    string origin_layer = GetOriginLayerName(layer.bottom(i));
    if (origin_layer.empty()) {
      MS_LOG(ERROR) << "layer not found";
      return RET_ERROR;
    }

    if (nodes_.find(origin_layer) == nodes_.end()) {
      MS_LOG(ERROR) << "layer bottom " << layer.bottom(i) << " is not found";
      return RET_NOT_FIND_OP;
    }
    input_nodes->emplace_back(nodes_.find(origin_layer)->second);
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertTop(const caffe::LayerParameter &layer, const CNodePtr &cnode) {
  auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
  std::vector<int64_t> shape_vector;
  if (layer.top_size() == 1) {
    cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    nodes_[layer.top(0)] = cnode;
    return RET_OK;
  }

  AbstractBasePtrList abstract_list;
  for (int i = 0; i < layer.top_size(); i++) {
    abstract_list.emplace_back(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new TupleGetItem failed";
      return RET_NULL_PTR;
    }
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_ptr);
    auto get_item_value = NewValueNode(MakeValue<int>(i));
    std::vector<AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
    CNodePtr get_item_cnode = func_graph_ptr_->NewCNode(inputs);
    get_item_cnode->set_fullname_with_scope(layer.top(i));
    nodes_[layer.top(i)] = get_item_cnode;
  }
  cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return RET_OK;
}

std::string CaffeModelParser::GetOriginLayerName(const std::string &layer_name) {
  if (caffe_layers_.find(layer_name) == caffe_layers_.end()) {
    return layer_name;
  }
  auto layer = caffe_layers_.at(layer_name);
  if (layer.type() != "Split") {
    return layer_name;
  }
  while (layer.type() == "Split") {
    string input_name = layer.bottom(0);
    if (caffe_layers_.find(input_name) == caffe_layers_.end()) {
      return input_name;
    }
    layer = caffe_layers_.at(input_name);
  }
  return layer.name();
}
}  // namespace mindspore::lite
