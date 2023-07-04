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
#include "parser/caffe/caffe_model_parser.h"
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <utility>
#include "parser/caffe/caffe_inspector.h"
#include "parser/caffe/caffe_node_parser_registry.h"
#include "common/anf_util.h"
#include "parser/parser_utils.h"
#include "common/op_enum.h"
#include "parser/unify_format.h"
#include "mindapi/ir/func_graph.h"
#include "include/registry/converter_context.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"

using mindspore::converter::kFmkTypeCaffe;
namespace mindspore {
namespace lite {
namespace {
bool IsSkipedLayer(const caffe::LayerParameter &layer) {
  if (layer.type() == "Input" || layer.type() == "Dropout" || layer.type() == "Split") {
    return true;
  }
  return layer.include_size() == 1 && layer.include(0).phase() == caffe::TRAIN;
}

void FcSqueezeWeightBias(const caffe::LayerParameter &layer, int blob_index, std::vector<int32_t> *shape) {
  if (layer.type() == "InnerProduct") {
    if (blob_index == 0) {
      if (shape->size() == dpico::kDims4 && shape->at(0) == 1 && shape->at(1) == 1) {
        (void)shape->erase(shape->begin());
        (void)shape->erase(shape->begin());
      }
    } else if (blob_index == 1) {
      if (shape->size() == dpico::kDims4 && shape->at(0) == 1 && shape->at(1) == 1 && shape->at(dpico::kAxis2) == 1) {
        (void)shape->erase(shape->begin());
        (void)shape->erase(shape->begin());
        (void)shape->erase(shape->begin());
      }
    }
  }
}
}  // namespace

CaffeModelParser::CaffeModelParser() = default;

CaffeModelParser::~CaffeModelParser() = default;

api::FuncGraphPtr CaffeModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file;
  auto weight_file = flag.weight_file;
  STATUS status = InitOriginModel(model_file, weight_file);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init origin model failed.";
    return nullptr;
  }
  res_graph_ = api::FuncGraph::Create();
  if (res_graph_ == nullptr) {
    return nullptr;
  }
  status = ConvertGraphInputs();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert graph inputs failed.";
    return nullptr;
  }

  status = ConvertLayers();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert layers failed.";
    return nullptr;
  }

  status = ConvertGraphOutputs();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "convert graph outputs failed.";
    return nullptr;
  }
  res_graph_->set_attr("graph_name", api::MakeValue("main_graph"));
  res_graph_->set_attr("fmk", api::MakeValue(static_cast<int64_t>(kFmkTypeCaffe)));
  std::set<api::FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(res_graph_, &all_func_graphs);
  if (PostAdjust(all_func_graphs) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>();
  if (unify_format == nullptr) {
    MS_LOG(ERROR) << "unify format is nullptr.";
    return nullptr;
  }
  if (!unify_format->Run(res_graph_)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
    return nullptr;
  }
  return res_graph_;
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
    // eliminate _cpu mark
    auto layer_name = layer.mutable_name();
    std::string suffix = "_cpu";
    auto pos = layer_name->rfind(suffix);
    if (pos != std::string::npos && pos == layer_name->size() - suffix.length()) {
      MS_LOG(WARNING) << "Don't support \"_cpu\" mark for now, this mark will be eliminated.";
      (void)layer_name->replace(pos, suffix.length(), "");
    }

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
      MS_LOG(ERROR) << "not support op: " << layer.type();
      status = (status == RET_OK ? RET_NOT_FIND_OP : status);
      continue;
    }

    if (status != RET_OK) {
      continue;
    }

    auto base_operator_ptr = node_parser->Parse(layer, weight);
    if (base_operator_ptr == nullptr) {
      MS_LOG(ERROR) << "parse node " << layer.name() << " failed.";
      status = RET_ERROR;
      continue;
    }

    if (layer.top_size() == 1) {
      auto top_name = layer.top(0);
      if (top_name.size() > lite::kTopNameMaxSize) {  // mapper don't support node name length > 31
        top_name = top_name.substr(top_name.size() - lite::kTopNameMaxSize, lite::kTopNameMaxSize);
      }
      (void)base_operator_ptr->AddAttr(lite::kTopName, mindspore::api::MakeValue(top_name));
    }

    // build inputs
    std::vector<api::AnfNodePtr> input_nodes;
    status = ConvertBottom(layer, &input_nodes);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert layer bottom for " << layer.name() << " failed.";
      continue;
    }

    // build weights
    std::vector<api::ParameterPtr> const_parameters;
    status = ConvertBlobs(weight, &const_parameters);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert blobs for " << layer.name() << " failed.";
      continue;
    }

    // build cnode
    api::SharedPtr<ops::BaseOperator> primitive(std::move(base_operator_ptr));
    std::vector<api::AnfNodePtr> op_inputs = {api::NewValueNode(primitive)};
    (void)op_inputs.insert(op_inputs.end(), input_nodes.begin(), input_nodes.end());
    (void)op_inputs.insert(op_inputs.end(), const_parameters.begin(), const_parameters.end());
    auto new_cnode = res_graph_->NewCNode(op_inputs);
    new_cnode->set_fullname_with_scope(layer.name());

    // convert outputs
    status = ConvertTop(layer, new_cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Convert outputs for " << layer.name() << " failed.";
      continue;
    }
  }
  return status;
}

STATUS CaffeModelParser::InitOriginModel(const std::string &model_file, const std::string &weight_file) {
  int status = ValidateFileStr(model_file, ".prototxt");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.prototxt";
    return RET_INPUT_PARAM_INVALID;
  }

  if (weight_file.empty()) {
    MS_LOG(ERROR) << "INPUT MISSING: weightFile is necessary";
    return RET_INPUT_PARAM_INVALID;
  }

  status = ValidateFileStr(weight_file, ".caffemodel");
  if (status != RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: weightFile must be *.caffemodel";
    return RET_INPUT_PARAM_INVALID;
  }

  status = ReadProtoFromText(model_file, &caffe_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read prototxt file failed, model path: " << model_file;
    return RET_ERROR;
  }

  status = ReadProtoFromBinaryFile(weight_file, &caffe_weight_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read caffemodel file failed, model path: " << weight_file;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertInputLayers() {
  for (int i = 0; i < caffe_model_.layer_size(); i++) {
    auto layer = caffe_model_.layer(i);
    if (layer.type() == "Input") {
      auto parameter = res_graph_->add_parameter();
      std::vector<int64_t> shape;
      for (int j = 0; j < layer.input_param().shape(0).dim_size(); j++) {
        shape.push_back(layer.input_param().shape(0).dim(j));
      }
      auto abstract = dpico::CreateTensorAbstract(shape, kNumberTypeFloat32);
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      parameter->set_abstract(abstract);
      parameter->set_name(layer.name());
      (void)nodes_.emplace(std::pair(layer.top(0), parameter));
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputs() {
  if (ConvertInputLayers() != RET_OK) {
    MS_LOG(ERROR) << "Convert input layers failed.";
    return RET_ERROR;
  }

  if (caffe_model_.input_dim_size() > 0) {
    for (int i = 0; i < caffe_model_.input_size(); i++) {
      std::vector<int64_t> shape;
      if (static_cast<size_t>(caffe_model_.input_dim_size()) > dpico::kDims4) {
        int step = caffe_model_.input_dim_size() / caffe_model_.input_size();
        for (int j = i * step; j < (i + 1) * step; j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      } else {
        for (int j = 0; j < caffe_model_.input_dim_size(); j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      }
      auto parameter = res_graph_->add_parameter();
      auto abstract = dpico::CreateTensorAbstract(shape, kNumberTypeFloat32);
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      parameter->set_abstract(abstract);
      parameter->set_name(caffe_model_.input(i));
      (void)nodes_.emplace(std::pair(caffe_model_.input(i), parameter));
    }
  } else {
    for (int i = 0; i < caffe_model_.input_shape_size(); i++) {
      auto shape = caffe_model_.input_shape(i);
      std::vector<int64_t> shape_vector;
      for (int j = 0; j < shape.dim_size(); j++) {
        shape_vector.push_back(shape.dim(j));
      }
      auto parameter = res_graph_->add_parameter();
      auto tensor_info = dpico::CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeFloat32);
      if (tensor_info == nullptr) {
        MS_LOG(ERROR) << "Create tensor info failed";
        return RET_ERROR;
      }
      auto abstract = tensor_info->ToAbstract();
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      parameter->set_abstract(abstract);
      parameter->set_name(caffe_model_.input(i));
      (void)nodes_.emplace(std::pair(caffe_model_.input(i), parameter));
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphOutputs() {
  CaffeInspector caffeInspector;
  (void)caffeInspector.InspectModel(caffe_model_);
  if (caffeInspector.GetGraphOutput().size() > 1) {
    std::vector<api::AnfNodePtr> make_tuple_inputs;
    auto make_tuple_prim_ptr = api::MakeShared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return RET_NULL_PTR;
    }
    auto make_tuple_prim = api::NewValueNode(make_tuple_prim_ptr);
    (void)make_tuple_inputs.emplace_back(make_tuple_prim);
    for (const auto &output_node : caffeInspector.GetGraphOutput()) {
      if (nodes_.find(output_node) == nodes_.end()) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      auto cnode = nodes_.find(output_node)->second;
      (void)make_tuple_inputs.emplace_back(cnode);
    }
    auto make_tuple_cnode = res_graph_->NewCNode(make_tuple_inputs);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    std::vector<api::AnfNodePtr> op_inputs;
    auto return_prim_ptr = api::MakeShared<ops::Return>();
    if (return_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto value_node = api::NewValueNode(return_prim_ptr);
    (void)op_inputs.emplace_back(value_node);
    (void)op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = res_graph_->NewCNode(op_inputs);
    cnode->set_fullname_with_scope("Return");
    res_graph_->set_return(cnode);
  } else {
    auto return_prim = api::MakeShared<ops::Return>();
    if (return_prim == nullptr) {
      MS_LOG(ERROR) << "new Return failed";
      return RET_NULL_PTR;
    }
    auto valueNode = api::NewValueNode(return_prim);
    std::vector<api::AnfNodePtr> opInputs{valueNode};
    std::string top_name = *caffeInspector.GetGraphOutput().begin();
    if (nodes_.find(top_name) == nodes_.end()) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    auto cnode = nodes_.find(top_name)->second;
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    (void)opInputs.emplace_back(cnode);
    auto returnCnode = res_graph_->NewCNode(opInputs);
    returnCnode->set_fullname_with_scope("Return");
    res_graph_->set_return(returnCnode);

    const std::string top_name_suffix = "duplicate";
    const size_t max_loop = 1000;
    for (size_t i = 0; i < max_loop; i++) {
      std::string top_name_tmp = top_name + "_" + top_name_suffix + std::to_string(i);
      if (nodes_.find(top_name_tmp) != nodes_.end()) {
        auto cnode_tmp = nodes_[top_name_tmp];
        if (cnode_tmp == nullptr) {
          MS_LOG(ERROR) << "Can't find input node.";
          return RET_NOT_FIND_OP;
        }
        res_graph_->set_attr(top_name_tmp, api::MakeValue(cnode_tmp->fullname_with_scope()));
      } else {
        break;
      }
    }
  }
  // save original output tensor names.
  converter::ConverterContext::SetGraphOutputTensorNames(caffeInspector.GetGraphOutput());
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBlobs(const caffe::LayerParameter &layer,
                                      std::vector<api::ParameterPtr> *const_parameters) {
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
    (void)ConvertShape(layer.blobs(i), &shape);

    FcSqueezeWeightBias(layer, i, &shape);

    // cal Weight num
    auto parameter = res_graph_->add_parameter();
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    if (layer.type() == "Convolution" || layer.type() == "Deconvolution") {
      if (i == 0) {
        parameter->set_name(layer.name() + "/weight");
      } else if (i == 1) {
        parameter->set_name(layer.name() + "/bias");
      }
    } else {
      parameter->set_name(layer.name() + "/input-" + std::to_string(i + layer.top_size()));
    }

    int count = 0;
    api::TensorPtr tensor_info = nullptr;
    if (layer.blobs(i).double_data_size() > 0) {
      count = layer.blobs(i).double_data_size();
      auto buf = std::make_unique<float[]>(count);
      if (buf == nullptr) {
        MS_LOG(ERROR) << "buf is nullptr.";
        return RET_NULL_PTR;
      }
      for (int j = 0; j < count; ++j) {
        buf[j] = layer.blobs(j).double_data(j);
      }
      tensor_info = dpico::CreateTensorInfo(buf.get(), static_cast<size_t>(count) * sizeof(float), shape_vector,
                                            TypeId::kNumberTypeFloat32);
    } else {
      count = layer.blobs(i).data_size();
      const float *data_ptr = layer.blobs(i).data().data();
      if (data_ptr == nullptr) {
        MS_LOG(INFO) << "data of origin layer is nullptr";
        return RET_NULL_PTR;
      }
      tensor_info = dpico::CreateTensorInfo(data_ptr, static_cast<size_t>(count) * sizeof(float), shape_vector,
                                            TypeId::kNumberTypeFloat32);
    }
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "create tensor info failed";
      return RET_NULL_PTR;
    }
    auto status = dpico::InitParameterFromTensorInfo(parameter, tensor_info);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return RET_ERROR;
    }
    (void)const_parameters->emplace_back(parameter);
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBottom(const caffe::LayerParameter &layer, std::vector<api::AnfNodePtr> *input_nodes) {
  if (input_nodes == nullptr) {
    MS_LOG(ERROR) << "input_nodes is null";
    return RET_NULL_PTR;
  }
  for (int i = 0; i < layer.bottom_size(); i++) {
    std::string origin_layer = GetOriginLayerName(layer.bottom(i));
    if (origin_layer.empty()) {
      MS_LOG(ERROR) << "layer not found";
      return RET_ERROR;
    }

    if (nodes_.find(origin_layer) == nodes_.end()) {
      if (nodes_.find(origin_layer + "_report") == nodes_.end()) {
        MS_LOG(ERROR) << "layer bottom " << layer.bottom(i) << " is not found";
        return RET_NOT_FIND_OP;
      } else {
        (void)input_nodes->emplace_back(nodes_.find(origin_layer + "_report")->second);
      }
    } else {
      (void)input_nodes->emplace_back(nodes_.find(origin_layer)->second);
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertTop(const caffe::LayerParameter &layer, const api::CNodePtr &cnode) {
  if (layer.top_size() == 1) {
    const std::string top_name_suffix = "duplicate";
    const size_t max_loop = 1000;
    std::string top_name = layer.top(0);
    if (nodes_.find(top_name) != nodes_.end()) {
      std::string top_name_new = "";
      for (size_t i = 0; i < max_loop; i++) {
        std::string top_name_tmp = top_name + "_" + top_name_suffix + std::to_string(i);
        if (nodes_.find(top_name_tmp) == nodes_.end()) {
          top_name_new = top_name_tmp;
          break;
        }
      }
      if (top_name_new.empty()) {
        MS_LOG(ERROR) << "Create new top name failed";
        return RET_ERROR;
      }
      nodes_[top_name_new] = nodes_[top_name];
    }
    auto abstract = dpico::CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    cnode->set_abstract(abstract);
    nodes_[layer.top(0)] = cnode;
    return RET_OK;
  }

  api::AbstractBasePtrList abstract_list;
  for (int i = 0; i < layer.top_size(); i++) {
    auto abstract = dpico::CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    (void)abstract_list.emplace_back(abstract);
    auto tuple_get_item_prim_ptr = api::MakeShared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new TupleGetItem failed";
      return RET_NULL_PTR;
    }
    auto tuple_get_item_prim = api::NewValueNode(tuple_get_item_prim_ptr);
    auto get_item_value = api::NewValueNode(api::MakeValue<int64_t>(static_cast<int64_t>(i)));
    std::vector<api::AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
    api::CNodePtr get_item_cnode = res_graph_->NewCNode(inputs);
    get_item_cnode->set_fullname_with_scope(layer.top(i));
    auto top_name = layer.top(i);
    if (top_name.size() > lite::kTopNameMaxSize) {
      top_name = top_name.substr(top_name.size() - lite::kTopNameMaxSize, lite::kTopNameMaxSize);
    }
    (void)tuple_get_item_prim_ptr->AddAttr(lite::kTopName, api::MakeValue(top_name));
    nodes_[layer.top(i)] = get_item_cnode;
  }
  auto abstract_tuple = api::MakeShared<api::AbstractTuple>(abstract_list);
  if (abstract_tuple == nullptr) {
    MS_LOG(ERROR) << "abstract_tuple is nullptr.";
    return RET_ERROR;
  }
  cnode->set_abstract(abstract_tuple);
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
    std::string input_name = layer.bottom(0);
    if (caffe_layers_.find(input_name) == caffe_layers_.end()) {
      return input_name;
    }
    layer = caffe_layers_.at(input_name);
  }
  return layer.name();
}

converter::ModelParser *CaffeModelParserCreator() {
  auto *parser = new (std::nothrow) CaffeModelParser();
  if (parser == nullptr) {
    MS_LOG(ERROR) << "caffe model parser failed";
    return nullptr;
  }
  return parser;
}
REG_MODEL_PARSER(kFmkTypeCaffe, CaffeModelParserCreator)
}  // namespace lite
}  // namespace mindspore
