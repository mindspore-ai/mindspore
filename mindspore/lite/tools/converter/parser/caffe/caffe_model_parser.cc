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
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include "include/registry/node_parser_registry.h"
#include "tools/converter/parser/caffe/caffe_node_parser_registry.h"
#include "tools/converter/parser/caffe/caffe_inspector.h"
#include "tools/common/graph_util.h"
#include "tools/common/protobuf_utils.h"
#include "tools/common/tensor_util.h"
#include "ir/func_graph.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/lite_model_parser_creator.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/parser/unify_format.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/make_tuple.h"
#include "ops/return.h"
#include "ops/tuple_get_item.h"

using mindspore::converter::kFmkTypeCaffe;
namespace mindspore::lite {
namespace {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvWeightShapeSize = 4;
constexpr size_t kFcWeightFirstShapeIndex = 0;
constexpr size_t kFcWeightSecondShapeIndex = 1;
constexpr size_t kFcBiasFirstShapeIndex = 0;
constexpr size_t kFcBiasSecondShapeIndex = 1;
constexpr size_t kFcBiasThirdShapeIndex = 2;

STATUS CheckCaffeModel(const caffe::NetParameter &caffe_model, const caffe::NetParameter &caffe_weight) {
  std::set<std::string> providers;
  std::set<std::string> consumers;
  for (int i = 0; i < caffe_model.input_size(); i++) {
    const auto &input = caffe_model.input(i);
    if (providers.count(input) != 0) {
      MS_LOG(ERROR) << "Top repeated";
      return RET_ERROR;
    }
    providers.insert(input);
  }
  for (const auto &layer : caffe_model.layers()) {
    for (const auto &top : layer.top()) {
      if (providers.count(top) != 0) {
        MS_LOG(ERROR) << "Top repeated";
        return RET_ERROR;
      }
      providers.insert(top);
    }
    for (const auto &bottom : layer.bottom()) {
      if (consumers.count(bottom) != 0) {
        MS_LOG(ERROR) << "Bottom repeated";
        return RET_ERROR;
      }
      consumers.insert(bottom);
    }
  }
  for (const auto &consumer : consumers) {
    if (providers.count(consumer) == 0) {
      MS_LOG(ERROR) << "Bottom and top mismatch";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

FuncGraphPtr ConvertGraph(api::FuncGraphPtr func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}
}  // namespace
bool IsSkipedLayer(const caffe::LayerParameter &layer) {
  if (layer.type() == "Input" || layer.type() == "Dropout" || layer.type() == "Split") {
    return true;
  }
  return layer.include_size() == 1 && layer.include(0).phase() == caffe::TRAIN;
}

STATUS FcSqueezeWeightBias(const caffe::LayerParameter &layer, int blob_index, std::vector<int32_t> *shape) {
  MSLITE_CHECK_PTR(shape);
  if (layer.type() == "InnerProduct") {
    if (blob_index == 0) {
      if (shape->size() == kConvWeightShapeSize && shape->at(kFcWeightFirstShapeIndex) == 1 &&
          shape->at(kFcWeightSecondShapeIndex) == 1) {
        shape->erase(shape->begin());
        shape->erase(shape->begin());
      }
    } else if (blob_index == 1) {
      if (shape->size() == kConvWeightShapeSize && shape->at(kFcBiasFirstShapeIndex) == 1 &&
          shape->at(kFcBiasSecondShapeIndex) == 1 && shape->at(kFcBiasThirdShapeIndex) == 1) {
        shape->erase(shape->begin());
        shape->erase(shape->begin());
        shape->erase(shape->begin());
      }
    }
  }

  return RET_OK;
}
}  // namespace

CaffeModelParser::CaffeModelParser() = default;

CaffeModelParser::~CaffeModelParser() = default;

api::FuncGraphPtr CaffeModelParser::Parse(const converter::ConverterParameters &flag) {
  auto model_file = flag.model_file;
  auto weight_file = flag.weight_file;
  STATUS status = InitOriginModel(model_file, weight_file);
  if (status != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  status = CheckCaffeModel(caffe_model_, caffe_weight_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Input caffe model error: " << status;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto graph = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "create FuncGraph failed");
  res_graph_ = api::MakeShared<api::FuncGraph>(graph);
  MS_CHECK_TRUE_RET(res_graph_ != nullptr, nullptr);
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
  graph->set_attr("graph_name", MakeValue("main_graph"));
  auto value_ptr = MakeValue(static_cast<int>(converter::kFmkTypeCaffe));
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  graph->set_attr("fmk", value_ptr);
  if ((status = CommonAnfAdjust(graph)) != RET_OK) {
    MS_LOG(ERROR) << "AdjustForAnf failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return nullptr;
  }
  auto unify_format = std::make_shared<UnifyFormatToNHWC>(kFmkTypeCaffe, false, flag.save_type);
  MS_CHECK_TRUE_RET(unify_format != nullptr, nullptr);
  if (!unify_format->Run(graph)) {
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
    ops::PrimitiveCPtr primitive_c;
    auto node_parser = registry::NodeParserRegistry::GetNodeParser(kFmkTypeCaffe, layer.type());
    if (node_parser != nullptr) {
      primitive_c = node_parser->Parse(layer, weight)->GetPrim();
    } else {
      auto node_parser_builtin = CaffeNodeParserRegistry::GetInstance()->GetNodeParser(layer.type());
      if (node_parser_builtin == nullptr) {
        NotSupportOp::GetInstance()->InsertOp(layer.type());
        status = (status == RET_OK ? RET_NOT_FIND_OP : status);
        continue;
      }
      if (status != RET_OK) {
        continue;
      }
      primitive_c = node_parser_builtin->Parse(layer, weight);
    }
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
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto value_node = NewValueNode(primitive_c);
    MSLITE_CHECK_PTR(value_node);
    std::vector<AnfNodePtr> op_inputs = {value_node};
    op_inputs.insert(op_inputs.end(), input_nodes.begin(), input_nodes.end());
    op_inputs.insert(op_inputs.end(), const_parameters.begin(), const_parameters.end());
    auto new_cnode = graph->NewCNode(op_inputs);
    MSLITE_CHECK_PTR(new_cnode);
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

  status = ReadProtoFromText(model_file, &caffe_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read prototxt file failed, model path: " << model_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  status = ReadProtoFromBinaryFile(weight_file, &caffe_weight_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read caffemodel file failed, model path: " << weight_file;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputsOfLayer() {
  for (int i = 0; i < caffe_model_.layer_size(); i++) {
    auto layer = caffe_model_.layer(i);
    if (layer.type() == "Input") {
      if (layer.bottom_size() != 0) {
        MS_LOG(ERROR) << "The input layer should not have inputs";
        return RET_ERROR;
      }
      auto graph = ConvertGraph(res_graph_);
      MSLITE_CHECK_PTR(graph);
      auto parameter = graph->add_parameter();
      MSLITE_CHECK_PTR(parameter);
      std::vector<int64_t> shape = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(layer.name());
      if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape.empty()) {
        MS_LOG(WARNING) << "Can not find name in map. name is " << layer.name();
      }
      if (shape.empty()) {
        for (int j = 0; j < layer.input_param().shape(0).dim_size(); j++) {
          shape.push_back(layer.input_param().shape(0).dim(j));
        }
      }
      auto abstract = CreateTensorAbstract(shape, kNumberTypeFloat32);
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstarct failed";
        return RET_ERROR;
      }
      parameter->set_abstract(abstract);
      parameter->set_name(layer.name());
      nodes_.insert(std::pair(layer.top(0), parameter));
    }
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputsOfShape() {
  for (int i = 0; i < caffe_model_.input_shape_size(); i++) {
    auto shape = caffe_model_.input_shape(i);
    std::vector<int64_t> shape_vector =
      ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(caffe_model_.input(i));
    if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape_vector.empty()) {
      MS_LOG(WARNING) << "Can not find name in map. name is " << caffe_model_.input(i);
    }
    if (shape_vector.empty()) {
      for (int j = 0; j < shape.dim_size(); j++) {
        shape_vector.push_back(shape.dim(j));
      }
    }
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto parameter = graph->add_parameter();
    MSLITE_CHECK_PTR(parameter);
    auto tensor_info = CreateTensorInfo(nullptr, 0, shape_vector, kNumberTypeFloat32);
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
    nodes_.insert(std::pair(caffe_model_.input(i), parameter));
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputsOfDim() {
  const int default_input_dim_size = 4;
  for (int i = 0; i < caffe_model_.input_size(); i++) {
    std::vector<int64_t> shape = ConverterInnerContext::GetInstance()->GetGraphInputTensorShape(caffe_model_.input(i));
    if (ConverterInnerContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 && shape.empty()) {
      MS_LOG(WARNING) << "Can not find name in map. name is " << caffe_model_.input(i);
    }
    if (shape.empty()) {
      if (caffe_model_.input_dim_size() > default_input_dim_size) {
        int step = caffe_model_.input_dim_size() / caffe_model_.input_size();
        for (int j = i * step; j < (i + 1) * step; j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      } else {
        for (int j = 0; j < caffe_model_.input_dim_size(); j++) {
          shape.push_back(caffe_model_.input_dim(j));
        }
      }
    }
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto parameter = graph->add_parameter();
    MSLITE_CHECK_PTR(parameter);
    auto abstract = CreateTensorAbstract(shape, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    parameter->set_abstract(abstract);
    parameter->set_name(caffe_model_.input(i));
    nodes_.insert(std::pair(caffe_model_.input(i), parameter));
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertGraphInputs() {
  auto ret = ConvertGraphInputsOfLayer();
  if (ret != RET_OK) {
    return ret;
  }
  ret = ConvertGraphInputsOfShape();
  if (ret != RET_OK) {
    return ret;
  }
  if (caffe_model_.input_dim_size() > 0) {
    ret = ConvertGraphInputsOfDim();
    if (ret != RET_OK) {
      return ret;
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
    MSLITE_CHECK_PTR(make_tuple_prim_ptr);
    auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(make_tuple_prim_c);
    auto make_tuple_prim = NewValueNode(make_tuple_prim_c);
    MSLITE_CHECK_PTR(make_tuple_prim);
    make_tuple_inputs.emplace_back(make_tuple_prim);
    for (const auto &output_node : caffeInspector.GetGraphOutput()) {
      if (nodes_.find(output_node) == nodes_.end()) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_NOT_FIND_OP;
      }
      auto cnode = nodes_.find(output_node)->second;
      make_tuple_inputs.emplace_back(cnode);
    }
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto make_tuple_cnode = graph->NewCNode(make_tuple_inputs);
    MSLITE_CHECK_PTR(make_tuple_cnode);
    make_tuple_cnode->set_fullname_with_scope("return tuple");

    std::vector<AnfNodePtr> op_inputs;
    auto return_prim_ptr = std::make_shared<ops::Return>();
    MSLITE_CHECK_PTR(return_prim_ptr);
    auto return_prim_c = return_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(return_prim_c);
    auto value_node = NewValueNode(return_prim_c);
    MSLITE_CHECK_PTR(value_node);
    op_inputs.emplace_back(value_node);
    op_inputs.emplace_back(make_tuple_cnode);
    auto cnode = graph->NewCNode(op_inputs);
    MSLITE_CHECK_PTR(cnode);
    cnode->set_fullname_with_scope("Return");
    graph->set_return(cnode);
  } else {
    auto returnPrim = std::make_shared<ops::Return>();
    MSLITE_CHECK_PTR(returnPrim);
    auto return_prim_c = returnPrim->GetPrim();
    MSLITE_CHECK_PTR(return_prim_c);
    auto valueNode = NewValueNode(return_prim_c);
    MSLITE_CHECK_PTR(valueNode);
    std::vector<AnfNodePtr> opInputs{valueNode};
    if (nodes_.find(caffeInspector.GetGraphOutput().front()) == nodes_.end()) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    auto cnode = nodes_.find(caffeInspector.GetGraphOutput().front())->second;
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_NOT_FIND_OP;
    }
    opInputs.emplace_back(cnode);
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto returnCnode = graph->NewCNode(opInputs);
    MSLITE_CHECK_PTR(returnCnode);
    returnCnode->set_fullname_with_scope("Return");
    graph->set_return(returnCnode);
  }
  // save original output tensor names.
  ConverterInnerContext::GetInstance()->SetGraphOutputTensorNames(caffeInspector.GetGraphOutput());
  return RET_OK;
}

STATUS CaffeModelParser::ConvertLayerQuantParams(const caffe::LayerParameter &layer,
                                                 const caffe::LayerParameter &weight, PrimitiveCPtr primitive_c) {
  MSLITE_CHECK_PTR(primitive_c);
  auto quant_params_holder =
    std::make_shared<QuantParamHolder>(layer.bottom_size() + weight.blobs_size(), layer.top_size());
  MSLITE_CHECK_PTR(quant_params_holder);
#ifdef ENABLE_ACL_QUANT_PARAM
  // set quant parameter to output tensor of quant.
  if (layer.type() == "Quant") {
    QuantParamT quant_param;
    const caffe::QuantParameter &layer_quant_param = layer.quant_param();
    MS_CHECK_TRUE_RET(layer_quant_param.has_scale(), RET_ERROR);
    quant_param.scale = layer_quant_param.scale();
    MS_CHECK_TRUE_RET(layer_quant_param.has_offset(), RET_ERROR);
    quant_param.zeroPoint = *(reinterpret_cast<int8_t *>(const_cast<char *>(layer_quant_param.offset().c_str())));
    quant_param.inited = true;
    quant_params_holder->set_output_quant_param(0, {quant_param});
  }
#endif
  primitive_c->AddAttr("quant_params", quant_params_holder);
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBlobs(const caffe::LayerParameter &layer, std::vector<ParameterPtr> *const_parameters) {
  MSLITE_CHECK_PTR(const_parameters);

  // Layer must have Filter
  if (layer.blobs_size() == 0) {
    MS_LOG(INFO) << "No filter data in layer " << layer.name().c_str();
    return RET_OK;
  }
  for (int i = 0; i < layer.blobs_size(); i++) {
    std::vector<int32_t> shape;
    auto ret = ConvertShape(layer.blobs(i), &shape);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertShape failed.";
      return ret;
    }

    ret = FcSqueezeWeightBias(layer, i, &shape);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "FcSqueezeWeightBias failed.";
      return ret;
    }

    // cal Weight num
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    auto parameter = graph->add_parameter();
    MSLITE_CHECK_PTR(parameter);
    auto type_ptr = TypeIdToType(TypeId::kNumberTypeFloat32);
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
    tensor::TensorPtr tensor_info = nullptr;
    if (layer.blobs(i).double_data_size() > 0) {
      count = layer.blobs(i).double_data_size();
      auto buf = std::make_unique<float[]>(count);
      MSLITE_CHECK_PTR(buf);
      for (int j = 0; j < count; ++j) {
        buf[j] = layer.blobs(i).double_data(j);
      }
      tensor_info = CreateTensorInfo(buf.get(), count * sizeof(float), shape_vector, TypeId::kNumberTypeFloat32);
#ifdef ENABLE_ACL_QUANT_PARAM
    } else if (layer.blobs(i).has_int8_data()) {
      const int8_t *data_ptr = reinterpret_cast<int8_t *>(const_cast<char *>(layer.blobs(i).int8_data().c_str()));
      MSLITE_CHECK_PTR(data_ptr);
      count = std::accumulate(shape_vector.begin(), shape_vector.end(), 1, std::multiplies<int>());
      tensor_info = CreateTensorInfo(data_ptr, count * sizeof(int8_t), shape_vector, TypeId::kNumberTypeInt8);
    } else if (layer.blobs(i).int32_data_size() > 0) {
      count = layer.blobs(i).int32_data_size();
      const int *data_ptr = layer.blobs(i).int32_data().data();
      MSLITE_CHECK_PTR(data_ptr);
      tensor_info = CreateTensorInfo(data_ptr, count * sizeof(int), shape_vector, TypeId::kNumberTypeInt32);
    } else if (layer.blobs(i).uint64_data_size() > 0) {
      count = layer.blobs(i).uint64_data_size();
      const size_t *data_ptr = layer.blobs(i).uint64_data().data();
      MSLITE_CHECK_PTR(data_ptr);
      tensor_info = CreateTensorInfo(data_ptr, count * sizeof(size_t), shape_vector, TypeId::kNumberTypeUInt64);
#endif
    } else {
      count = layer.blobs(i).data_size();
      const float *data_ptr = layer.blobs(i).data().data();
      MSLITE_CHECK_PTR(data_ptr);
      tensor_info = CreateTensorInfo(data_ptr, count * sizeof(float), shape_vector, TypeId::kNumberTypeFloat32);
    }
    MSLITE_CHECK_PTR(tensor_info);
    auto status = InitParameterFromTensorInfo(parameter, tensor_info);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "init parameter from tensor info failed";
      return RET_ERROR;
    }
    const_parameters->emplace_back(parameter);
  }
  return RET_OK;
}

STATUS CaffeModelParser::ConvertBottom(const caffe::LayerParameter &layer, std::vector<AnfNodePtr> *input_nodes) {
  MSLITE_CHECK_PTR(input_nodes);
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
  MSLITE_CHECK_PTR(cnode);
  if (layer.top_size() == 1) {
    auto abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    cnode->set_abstract(abstract);
    nodes_[layer.top(0)] = cnode;
    return RET_OK;
  }

  AbstractBasePtrList abstract_list;
  for (int i = 0; i < layer.top_size(); i++) {
    auto abstract = CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstarct failed";
      return RET_ERROR;
    }
    abstract_list.emplace_back(abstract);
    auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
    if (tuple_get_item_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new TupleGetItem failed";
      return RET_NULL_PTR;
    }
    auto tuple_get_item_prim_c = tuple_get_item_prim_ptr->GetPrim();
    MSLITE_CHECK_PTR(tuple_get_item_prim_c);
    auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
    MSLITE_CHECK_PTR(tuple_get_item_prim);
    auto get_item_value = NewValueNode(MakeValue<int>(i));
    MSLITE_CHECK_PTR(get_item_value);
    std::vector<AnfNodePtr> inputs{tuple_get_item_prim, cnode, get_item_value};
    auto graph = ConvertGraph(res_graph_);
    MSLITE_CHECK_PTR(graph);
    CNodePtr get_item_cnode = graph->NewCNode(inputs);
    MSLITE_CHECK_PTR(get_item_cnode);
    get_item_cnode->set_fullname_with_scope(layer.top(i));
    nodes_[layer.top(i)] = get_item_cnode;
  }
  auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MSLITE_CHECK_PTR(abstract);
  cnode->set_abstract(abstract);
  return RET_OK;
}

std::string CaffeModelParser::GetOriginLayerName(const std::string &layer_name) {
  if (caffe_layers_.find(layer_name) == caffe_layers_.end()) {
    return layer_name;
  }
  auto layer = caffe_layers_.at(layer_name);
  if (layer.type() != "Split" && layer.type() != "Dropout") {
    return layer_name;
  }
  if (layer.type() == "Dropout" && layer.bottom(0) == layer.top(0)) {
    return layer_name;
  }
  while (layer.type() == "Split" || layer.type() == "Dropout") {
    string input_name = layer.bottom(0);
    if (caffe_layers_.find(input_name) == caffe_layers_.end()) {
      return input_name;
    }
    layer = caffe_layers_.at(input_name);
  }
  return layer.name();
}
REG_MODEL_PARSER(kFmkTypeCaffe, LiteModelParserCreator<CaffeModelParser>)
}  // namespace mindspore::lite
