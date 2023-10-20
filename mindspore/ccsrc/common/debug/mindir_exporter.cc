/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "include/common/utils/compile_cache_context.h"

namespace {
using mindspore::CNodePtr;
using mindspore::FileUtils;
using mindspore::FuncGraph;
using mindspore::FuncGraphPtr;
using mindspore::ValueNode;
using mindspore::ValueNodePtr;

void GetAllFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(all_func_graphs != nullptr);
  MS_ASSERT(func_graph != nullptr);
  if (all_func_graphs->find(func_graph) == all_func_graphs->end()) {
    (void)(all_func_graphs->insert(func_graph));
  } else {
    return;
  }
  auto nodes = mindspore::TopoSort(func_graph->get_return());
  for (auto &node : nodes) {
    if (mindspore::IsValueNode<FuncGraph>(node)) {
      MS_ASSERT(node->cast<ValueNodePtr>() != nullptr);
      MS_ASSERT(node->cast<ValueNodePtr>()->value() != nullptr);
      MS_ASSERT((node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>() != nullptr);
      auto new_fg = (node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
      GetAllFuncGraphs(new_fg, all_func_graphs);
    }
    if (mindspore::utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_ASSERT(cnode != nullptr);
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          if (mindspore::IsValueNode<FuncGraph>(input)) {
            MS_ASSERT(input->cast<ValueNodePtr>() != nullptr);
            MS_ASSERT(input->cast<ValueNodePtr>()->value() != nullptr);
            MS_ASSERT((input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>() != nullptr);
            auto new_fg = (input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
            GetAllFuncGraphs(new_fg, all_func_graphs);
          }
        }
      }
    }
  }
}

bool DeleteDirRecursively(const std::string &dir_name) {
  DIR *dir = opendir(dir_name.c_str());
  dirent *dirent = nullptr;
  std::vector<std::string> file_names{};
  while ((dirent = readdir(dir)) != nullptr) {
    if (strcmp(dirent->d_name, ".") != 0 && strcmp(dirent->d_name, "..") != 0) {
      (void)(file_names.emplace_back(dirent->d_name));
    }
  }
  for (auto &file_name : file_names) {
    auto file_path = dir_name + "/" + file_name;
    auto real_file_path = FileUtils::GetRealPath(file_path.c_str());
    if (!real_file_path.has_value()) {
      MS_LOG(ERROR) << "Cannot get pwd path";
      return false;
    }
    auto result = unlink(real_file_path.value().c_str());
    if (result != 0) {
      (void)(closedir(dir));
      MS_LOG(ERROR) << "Delete the file(" << real_file_path.value() << ") failed." << mindspore::ErrnoToString(errno);
      return false;
    }
  }
  (void)(closedir(dir));
  return true;
}
};  // namespace

namespace mindspore {
bool IrExportBuilder::SetAbstractFuncToAttributeProto(const abstract::AbstractBasePtr &abstract,
                                                      mind_ir::AttributeProto *const attr_proto) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(attr_proto);
  if (abstract->isa<abstract::FuncGraphAbstractClosure>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FUNCGRAPHCLOSURE);
    auto func_name = abstract->cast<abstract::FuncGraphAbstractClosurePtr>()->func_graph()->ToString();
    attr_proto->set_s(func_name);
  } else if (abstract->isa<abstract::PrimitiveAbstractClosure>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_PRIMITIVECLOSURE);
    auto prim = abstract->cast<abstract::PrimitiveAbstractClosurePtr>()->prim();
    attr_proto->set_s(GetPrimitiveUniqueName(prim));
  } else if (abstract->isa<abstract::PartialAbstractClosure>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_PARTIALCLOSURE);
    auto node_ptr = abstract->cast<abstract::PartialAbstractClosurePtr>()->node();
    MS_EXCEPTION_IF_NULL(node_ptr);
    attr_proto->set_s(GetUniqueNodeName(node_ptr));
  } else if (abstract->isa<abstract::AbstractFuncUnion>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UNIONFUNCCLOSURE);
    auto visit_func = [this, &attr_proto](const abstract::AbstractFuncAtomPtr &poss) {
      auto element_attr_proto = attr_proto->add_values();
      if (!this->SetAbstractFuncToAttributeProto(poss, element_attr_proto)) {
        MS_LOG(EXCEPTION) << "Set union function abstract to proto error." << poss->ToString();
      }
    };
    abstract->cast<abstract::AbstractFunctionPtr>()->Visit(visit_func);
  } else {
    MS_LOG(ERROR) << "The parameter abstract is not an abstractFunction: " << abstract->ToString();
    return false;
  }
  return true;
}

std::string IrExportBuilder::GetPrimitiveUniqueName(const PrimitivePtr &primitive_ptr) {
  auto it = primitive_name_map_.find(primitive_ptr);
  if (it != primitive_name_map_.end()) {
    return it->second;
  }
  // Remove this check if we find a way to handle save/load training model with flattened parameters.
  if (IsPrimitiveEquals(primitive_ptr, prim::kPrimFlattenConcat)) {
    MS_LOG(EXCEPTION) << "Export model with operator '" << primitive_ptr->name() << "' is not supported yet.\n"
                      << "Please remove 'net.flatten_weights()' in your script and try again.";
  }
  auto answer = primitive_ptr->name() + ":" + std::to_string(GetUniqueID());
  primitive_name_map_[primitive_ptr] = answer;
  return answer;
}

bool IrExportBuilder::BuildPrimitives() {
  for (auto it = primitive_name_map_.begin(); it != primitive_name_map_.end(); ++it) {
    auto prim = it->first;
    if (prim->name() == prim::kPrimPyExecute->name()) {
      MS_LOG(EXCEPTION) << "Cannot export a PyExecute CNode in MindIR.";
    }
    auto prim_proto = model_->add_primitives();

    prim_proto->set_name(it->second);
    prim_proto->set_op_type(prim->name());

    auto real_prim = GetValueWithoutDoSignature(prim)->cast<PrimitivePtr>();
    if (real_prim != nullptr) {
      prim = real_prim;
    }

    prim_proto->set_instance_name(prim->instance_name());

    // Set primitive attributes
    for (const auto &attr : prim->attrs()) {
      MS_LOG(DEBUG) << "attr: " << attr.first << " " << attr.second->DumpText() << " " << attr.second->type_name();
      auto iter = g_export_attr_blacklist.find(attr.first);
      if (iter != g_export_attr_blacklist.end()) {
        continue;
      }
      mind_ir::AttributeProto *attr_proto = prim_proto->add_attribute();
      attr_proto->set_name(attr.first);
      auto attr_value = attr.second;
      if (!is_kernel_graph_) {
        CheckAndConvertUtils::ConvertAttrValueInExport(prim->name(), attr.first, &attr_value);
      }
      if (!SetValueToAttributeProto(attr_value, attr_proto)) {
        MS_LOG(ERROR) << "Set value to AttributeProto failed.";
        return false;
      }
    }  // Loop of attrs
  }    // Loop of primitives
  return true;
}

std::string IrExporter::GetDumpString(const FuncGraphPtr &func_graph) {
  auto dump_proto = GetDumpProto(func_graph);
  if (dump_proto == nullptr) {
    MS_LOG(EXCEPTION) << "Get dump proto for graph " << func_graph->ToString() << " failed.";
  }
  return builder_->GetProtoString();
}

ModelProtoPtr IrExporter::GetDumpProto(const FuncGraphPtr &func_graph) {
  if ((builder_ == nullptr) || (func_graph == nullptr)) {
    MS_LOG(EXCEPTION) << "Input params is null.";
  }

  // Export model info
  builder_->BuildModelInfo();

  // Export model and return string
  if (!builder_->BuildModel(func_graph)) {
    return nullptr;
  }
  return builder_->Model();
}

ModelProtoPtr IrExporter::GetDumpProto(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                                       const std::vector<AnfNodePtr> &isolated_nodes) {
  // Export model info
  builder_->BuildModelInfo();
  // Export model and return string
  if (!builder_->BuildModel(root_graph, child_graphs, isolated_nodes)) {
    return nullptr;
  }
  return builder_->Model();
}

std::string IrExportBuilder::GetProtoString() const {
  MS_LOG(DEBUG) << "BuildModel complete!";
  return model_->SerializeAsString();
}

void IrExportBuilder::BuildModelInfo() {
  constexpr auto ir_version = "0.1.1";
  constexpr auto mindspore_name = "MindSpore";
  model_->set_ir_version(ir_version);
  model_->set_producer_name(mindspore_name);
  model_->set_model_version(VERSION);
  model_->set_little_endian(common::IsLittleByteOrder());
  model_->set_mind_ir_version(mind_ir::Version_MAX);
}

// build model for kernel graph
bool IrExportBuilder::BuildModel(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                                 const std::vector<AnfNodePtr> &isolated_nodes) {
  MS_EXCEPTION_IF_NULL(root_graph);
  is_kernel_graph_ = root_graph->type_name() == kKernelGraphTypeName;
  nodeName_.clear();
  node_name_map_.clear();
  primitive_name_map_.clear();

  // Because param may be called across graphs, build params of all graphs first.
  auto build_params_attrs = [this](const FuncGraphPtr &graph, mind_ir::GraphProto *const proto) {
    if (!BuildParameters(graph, proto)) {
      MS_LOG(ERROR) << "Build graph parameters failed.";
      return false;
    }
    if (!BuildFuncGraphAttrs(graph, proto)) {
      MS_LOG(ERROR) << "Build graph parameters attrs failed.";
      return false;
    }
    return true;
  };

  (void)nodeName_.insert(root_graph->ToString());
  auto root_graph_proto = model_->mutable_graph();
  // build root graph params
  top_graph = true;
  if (!(build_params_attrs(root_graph, root_graph_proto))) {
    return false;
  }
  root_graph_proto->set_name(root_graph->ToString());
  graph_protos_[root_graph] = root_graph_proto;
  // build child graph params
  top_graph = false;
  for (const auto &graph : child_graphs) {
    auto func_proto = model_->add_functions();
    func_proto->set_name(graph->ToString());
    (void)nodeName_.insert(graph->ToString());
    if (!(build_params_attrs(graph, func_proto))) {
      return false;
    }
    graph_protos_[graph] = func_proto;
  }
  // build nodes for root_graph, then child_graph
  if (!BuildNodes(root_graph, root_graph_proto)) {
    return false;
  }
  std::map<std::string, FuncGraphPtr> sorted_graphs;
  std::for_each(child_graphs.begin(), child_graphs.end(),
                [&sorted_graphs](const auto &iter) { sorted_graphs[iter->ToString()] = iter; });
  for (const auto &iter : sorted_graphs) {
    const auto &graph = iter.second;
    if (!BuildNodes(graph, graph_protos_[graph])) {
      return false;
    }
  }
  if (!BuildIsolatedNodes(isolated_nodes)) {
    return false;
  }
  // build primitives
  if (!BuildPrimitives()) {
    return false;
  }
  // Release resource
  nodeName_.clear();
  node_name_map_.clear();
  primitive_name_map_.clear();
  graph_protos_.clear();
  return true;
}

bool IrExportBuilder::BuildModel(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  mind_ir::GraphProto *graph_proto = model_->mutable_graph();
  graph_proto->set_name(func_graph->ToString());
  graph_proto->set_bprop_hash(func_graph->bprop_hash());
  graph_proto->set_bprop_filepath(func_graph->bprop_filepath());
  todo_.clear();
  nodeName_.clear();
  primitive_name_map_.clear();
  // Build the main funcGraph
  (void)nodeName_.insert(func_graph->ToString());
  top_graph = true;

  if (!BuildFuncGraph(func_graph, graph_proto)) {
    MS_LOG(ERROR) << "Build func_graph " << func_graph->ToString() << " failed.";
    return false;
  }

  // Build child funcGraphs
  std::set<FuncGraphPtr> graphVisited;
  (void)graphVisited.insert(func_graph);
  top_graph = false;

  auto &context = CompileCacheContext::GetInstance();
  const auto &child_graphs = context.GetChileGraphs();
  (void)(std::transform(child_graphs.begin(), child_graphs.end(), std::back_inserter(todo_),
                        [](const FuncGraphPtr &g) { return g; }));
  while (!todo_.empty()) {
    FuncGraphPtr fg = todo_.back();
    todo_.pop_back();
    if (graphVisited.count(fg) > 0) {
      continue;
    }
    if (nodeName_.count(fg->ToString()) > 0) {
      MS_LOG(ERROR) << "There is a duplicate name: " << fg->ToString();
      return false;
    }
    (void)nodeName_.insert(fg->ToString());
    (void)graphVisited.insert(fg);
    auto graph = model_->add_functions();
    if (!BuildFuncGraph(fg, graph)) {
      MS_LOG(ERROR) << "Build func_graph " << fg->ToString() << " failed.";
      return false;
    }
  }

  if (!BuildPrimitives()) {
    return false;
  }
  // Release resource
  nodeName_.clear();
  node_name_map_.clear();
  primitive_name_map_.clear();
  graph_protos_.clear();
  MS_LOG(INFO) << "BuildModel end.";
  return true;
}

bool IrExportBuilder::BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  graph_protos_[func_graph] = graph_proto;
  // Export funcGraph name.
  graph_proto->set_name(func_graph->ToString());
  // Export parameters
  // 1. parameters should be mapped to ValueInfoProto
  // 2. parameters with default value should be mapped to Initializer
  if (!BuildParameters(func_graph, graph_proto)) {
    MS_LOG(ERROR) << "Build parameters failed.";
    return false;
  }

  // Export graph attributes
  if (!BuildFuncGraphAttrs(func_graph, graph_proto)) {
    MS_LOG(ERROR) << "Build attributes for graph failed.";
    return false;
  }

  // Export operator nodes(include output)
  return BuildNodes(func_graph, graph_proto);
}

bool IrExportBuilder::BuildFuncGraphAttrs(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_proto);
  for (const auto &attr : func_graph->attrs()) {
    MS_LOG(DEBUG) << "attr: " << attr.first << " " << attr.second->DumpText() << " " << attr.second->type_name();
    auto iter = g_export_attr_blacklist.find(attr.first);
    if (iter != g_export_attr_blacklist.end()) {
      continue;
    }
    mind_ir::AttributeProto *attr_proto = graph_proto->add_attribute();
    attr_proto->set_name(attr.first);
    if (!SetValueToAttributeProto(attr.second, attr_proto)) {
      MS_LOG(ERROR) << "Set value to AttributeProto for GraphProto failed.";
      return false;
    }
  }
  return true;
}

bool IrExportBuilder::ExportWeight(const ParameterPtr &param, const std::string &param_name,
                                   mind_ir::GraphProto *const graph_proto) {
  MS_LOG(DEBUG) << "Parameter: '" << param->DebugString();
  auto param_abs = param->abstract();
  MS_EXCEPTION_IF_NULL(param_abs);
  if (param_abs->isa<abstract::AbstractMapTensor>()) {
    auto *map_parameter_proto = graph_proto->add_map_parameter();
    if (!ConvertMapParameterToMapTensorProto(param, map_parameter_proto)) {
      MS_LOG(ERROR) << "Convert MapParameter " << param->ToString() << " to MapTensorProto failed.";
      return false;
    }
    return true;
  }
  if (param_abs->isa<abstract::AbstractTensor>()) {
    mind_ir::TensorProto *parameter_proto = graph_proto->add_parameter();
    parameter_proto->set_name(param_name);
    if (!SetParamToTensorProto(param, parameter_proto)) {
      MS_LOG(ERROR) << "Set parameter " << param->DebugString() << " to TensorProto failed.";
      return false;
    }
    return true;
  }
  MS_LOG(ERROR) << "Only support MapTensor or Tensor as default param of Parameter, got: "
                << param->default_param()->ToString();
  return false;
}

bool IrExportBuilder::BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_proto);
  auto &context = CompileCacheContext::GetInstance();
  auto param_size = func_graph->parameters().size();
  MS_LOG(DEBUG) << "func graph: " << func_graph->ToString() << " parameter num:" << param_size
                << ", fv param num:" << func_graph->fv_param_count();
  for (size_t param_counter = 0; param_counter < param_size; ++param_counter) {
    auto &item = func_graph->parameters()[param_counter];
    MS_EXCEPTION_IF_NULL(item);
    auto param = item->cast<ParameterPtr>();
    if (param == nullptr) {
      MS_LOG(ERROR) << "Parameter: '" << item->ToString() << "' could not cast to parameter.";
      return false;
    }
    if (is_kernel_graph_ && (node_name_map_.find(param) != node_name_map_.end() || param->func_graph() != func_graph)) {
      continue;
    }
    std::string param_name = GetUniqueNodeName(param);
    param->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(param_name));
    if (is_kernel_graph_ && context.IsBackendParamGenFromFrontendParam(param)) {
      (void)nodeName_.insert(param_name);
      continue;
    }
    if (top_graph &&
        (param_counter >= param_size - func_graph->fv_param_count() || (is_kernel_graph_ && param->has_default()))) {
      if (!ExportWeight(param, param_name, graph_proto)) {
        MS_LOG(ERROR) << "Failed to export parameter weight:" << param->DebugString();
        return false;
      }
    } else {
      // export graph input
      mind_ir::ValueInfoProto *input_proto = graph_proto->add_input();
      input_proto->set_name(param_name);
      if (!SetValueInfoProto(param, input_proto)) {
        MS_LOG(ERROR) << "Set parameter " << param->DebugString() << " to TensorProto failed.";
        return false;
      }
    }
    if (nodeName_.count(param_name) > 0) {
      MS_LOG(ERROR) << "parameter name is duplicate:" << param_name;
      return false;
    }
    (void)nodeName_.insert(param_name);
  }
  return true;
}

bool IrExportBuilder::SetQuantizationParamToAttrProto(const std::shared_ptr<QuantizationParam> &quantization_param,
                                                      mind_ir::TensorProto_QuantParamProto *const quant_param_proto) {
  quant_param_proto->set_quant_algo_name(quantization_param->quant_algo_name());
  auto quant_param_attrs = quantization_param->attrs();
  for (auto &quant_param_attr : quant_param_attrs) {
    auto attr_proto = quant_param_proto->add_attribute();
    attr_proto->set_name(quant_param_attr.first);
    auto value_ptr = quant_param_attr.second;
    auto ret = SetValueToAttributeProto(value_ptr, attr_proto);
    if (!ret) {
      MS_LOG(ERROR) << "QuantizationParam Set Value to AttributeProto Error";
      return false;
    }
  }
  return true;
}

bool IrExportBuilder::SetFunctorToAttrProto(const FunctorPtr &func, mind_ir::AttributeProto *const attr_proto) {
  auto *functor_proto = attr_proto->mutable_functor();
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FUNCTOR);
  if (func->isa<ShapeCalcFunctor>()) {
    functor_proto->set_type(mind_ir::FunctorProto_FunctorType_SHAPE_CALC_FUNCTOR);
  } else {
    MS_LOG(ERROR) << "Unknown functor: " << func->ToString();
    return false;
  }
  functor_proto->set_name(func->name());
  auto values = func->ToValue();
  if (values == nullptr) {
    values = kNone;
  }
  if (!SetValueToAttributeProto(values, functor_proto->add_values())) {
    return false;
  }
  return true;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataType(TypeId type_id) const {
  auto iter = g_data_type_map.find(type_id);
  if (iter == g_data_type_map.end()) {
    MS_LOG(ERROR) << "Convert type error, unsupported type! " << type_id;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsIntType(int bits) const {
  auto iter = g_data_bits_int_map.find(bits);
  if (iter == g_data_bits_int_map.end()) {
    MS_LOG(ERROR) << "Convert bits int error, unsupported bits! " << bits;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsUIntType(int bits) const {
  auto iter = g_data_bits_uint_map.find(bits);
  if (iter == g_data_bits_uint_map.end()) {
    MS_LOG(ERROR) << "Convert bits uint error, unsupported bits! " << bits;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsFloatType(int bits) const {
  auto iter = g_data_bits_float_map.find(bits);
  if (iter == g_data_bits_float_map.end()) {
    MS_LOG(ERROR) << "Convert bits float error, unsupported bits! " << bits;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsBFloatType(int bits) const {
  auto iter = g_data_bits_bfloat_map.find(bits);
  if (iter == g_data_bits_bfloat_map.end()) {
    MS_LOG(ERROR) << "Convert bits bfloat error, unsupported bits! " << bits;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsComplexType(int bits) const {
  auto iter = g_data_bits_complex_map.find(bits);
  if (iter == g_data_bits_complex_map.end()) {
    MS_LOG(ERROR) << "Convert bits float error, unsupported bits! " << bits;
    return mind_ir::TensorProto_DataType_UNDEFINED;
  }
  return iter->second;
}

bool IrExportBuilder::SetValueInfoProto(const AnfNodePtr &node, mind_ir::ValueInfoProto *const value_proto) {
  if (node == nullptr || value_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or ValueInfo is null!";
  }
  MS_LOG(DEBUG) << "SetValueInfoProto: " << node->DebugString();
  const TypePtr &type = node->Type();
  const BaseShapePtr &shape = node->Shape();
  // For the bprop fg which has not been renormalized.
  if (type == nullptr || shape == nullptr) {
    return true;
  }
  if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    mind_ir::TensorProto *tensor_proto = value_proto->add_tensor();
    if (!SetTensorProto(node->abstract(), tensor_proto)) {
      return false;
    }
  } else {
    mind_ir::AttributeProto *attribute = value_proto->mutable_attr_info();
    if (!SetAbstractToNodeProto(node->abstract(), attribute)) {
      MS_LOG(ERROR) << "Set shape to Proto for " << node->DebugString() << " failed.";
      return false;
    }
    value_proto->set_denotation(type->type_name());
  }
  MS_LOG(DEBUG) << "Value type: " << type->type_name();
  return true;
}

bool IrExportBuilder::SetTensorToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
  mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
  tensor_proto->set_name("value0");
  auto data = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(data);
  tensor_proto->set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
  auto dtype = data->data_type();
  auto shape = data->shape_c();
  auto data_type = GetMindirDataType(dtype);
  if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
    return false;
  }
  tensor_proto->set_data_type(data_type);
  for (const auto &dim : shape) {
    tensor_proto->add_dims(dim);
  }
  return true;
}

bool IrExportBuilder::SetCSRTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto) {
  abstract::AbstractCSRTensorPtr csr_tensor_abs = abstract->cast<abstract::AbstractCSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor_abs);
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_CSR_TENSOR);
  mind_ir::AttributeProto *indptr = attr_proto->add_values();
  bool res = SetAbstractToNodeProto(csr_tensor_abs->indptr(), indptr);
  mind_ir::AttributeProto *indices = attr_proto->add_values();
  res = res && SetAbstractToNodeProto(csr_tensor_abs->indices(), indices);
  mind_ir::AttributeProto *values = attr_proto->add_values();
  res = res && SetAbstractToNodeProto(csr_tensor_abs->values(), values);
  mind_ir::AttributeProto *shape = attr_proto->add_values();
  res = res && SetAbstractToNodeProto(csr_tensor_abs->shape(), shape);
  return res;
}

bool IrExportBuilder::SetCOOTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto) {
  abstract::AbstractCOOTensorPtr coo_tensor_abs = abstract->cast<abstract::AbstractCOOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor_abs);
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_COO_TENSOR);
  mind_ir::AttributeProto *indices = attr_proto->add_values();
  bool res = SetAbstractToNodeProto(coo_tensor_abs->indices(), indices);
  mind_ir::AttributeProto *values = attr_proto->add_values();
  res = res && SetAbstractToNodeProto(coo_tensor_abs->values(), values);
  mind_ir::AttributeProto *shape = attr_proto->add_values();
  res = res && SetAbstractToNodeProto(coo_tensor_abs->shape(), shape);
  return res;
}

bool IrExportBuilder::SetTensorProto(const AbstractBasePtr &abstract, mind_ir::TensorProto *const tensor_proto) {
  auto type = abstract->BuildType();
  auto shape = abstract->BuildShape();
  if (!type->isa<TensorType>() || !shape->isa<abstract::Shape>()) {
    MS_LOG(ERROR) << "Type or shape is not supported! " << type->ToString();
    return false;
  }
  auto tensor = type->cast<TensorTypePtr>();
  auto tensor_shape = shape->cast<abstract::ShapePtr>();
  const auto &dims = tensor_shape->shape();
  auto data_type = GetMindirDataType(tensor->element()->type_id());
  if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
    return false;
  }
  tensor_proto->set_data_type(data_type);
  for (const auto &dim : dims) {
    tensor_proto->add_dims(dim);
  }

  if (!abstract->name().empty()) {
    tensor_proto->set_name(abstract->name());
  }
  // Deal Ref
  if (!type->isa<RefType>()) {
    return true;
  }

  auto abs_ref = abstract->cast<abstract::AbstractRefPtr>();
  if (abs_ref == nullptr) {
    MS_LOG(ERROR) << "The abstract " << abstract->ToString() << " should be AbstractRefTensor.";
    return false;
  }
  auto ref_key_value = abs_ref->ref_key_value()->cast<StringImmPtr>();
  if (ref_key_value == nullptr) {
    MS_LOG(INFO) << "The ref_key_value of abstract ref " << abstract->ToString() << " is nullptr";
    return true;
  }
  tensor_proto->set_ref_key(ref_key_value->value());
  return true;
}

bool IrExportBuilder::SetParamToTensorProto(const ParameterPtr &param, mind_ir::TensorProto *const tensor_proto) {
  if (param == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter or TensorProto is null!";
  }
  MS_LOG(DEBUG) << "SetParamToTensorProto: " << param->DebugString();
  if (!SetTensorProto(param->abstract(), tensor_proto)) {
    MS_LOG(ERROR) << "Export Parameter to tensor proto failed.";
    return false;
  }
  // export quant parameter info
  auto tensor = param->default_param()->cast<tensor::TensorPtr>();
  if (tensor != nullptr) {
    tensor_proto->set_compression_type(static_cast<mind_ir::TensorProto_CompressionType>(tensor->compression_type()));
    auto quant_params = tensor->quant_params();
    for (const auto &quant_param : quant_params) {
      auto quant_param_proto = tensor_proto->add_quant_params();
      auto ret = SetQuantizationParamToAttrProto(quant_param, quant_param_proto);
      if (ret != true) {
        MS_LOG(ERROR) << "QuantizationParam Set Value to AttributeProto Error";
        return false;
      }
    }
  }
  return true;
}

bool IrExportBuilder::ConvertMapParameterToMapTensorProto(const ParameterPtr &map_parameter,
                                                          mind_ir::MapTensorProto *const map_tensor_proto) {
  if (map_parameter == nullptr || map_tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "MapParameter or MapTensorProto is null!";
  }
  MS_LOG(DEBUG) << "ConvertMapParameterToMapTensorProto: " << map_parameter->ToString();

  // parameter name
  map_tensor_proto->set_name(GetUniqueNodeName(map_parameter));

  auto param_default = map_parameter->default_param();
  MS_EXCEPTION_IF_NULL(param_default);
  auto map_tensor = param_default->cast<tensor::MapTensorPtr>();
  MS_EXCEPTION_IF_NULL(map_tensor);
  // default value
  auto default_value = map_tensor->default_value();
  MS_EXCEPTION_IF_NULL(default_value);
  auto *default_value_proto = map_tensor_proto->mutable_default_value();
  MS_EXCEPTION_IF_NULL(default_value_proto);
  if (!SetValueToAttributeProto(default_value, default_value_proto)) {
    MS_LOG(ERROR) << "Export default value of MapTensor failed, default_value: " << default_value->ToString();
    return false;
  }
  tensor::MapTensor::ExportData export_data = map_tensor->Export(this->incremental_);
  // key_tensor
  auto *key_tensor_proto = map_tensor_proto->mutable_key_tensor();
  MS_EXCEPTION_IF_NULL(key_tensor_proto);
  auto &key_tensor = export_data.key_tensor;
  MS_EXCEPTION_IF_NULL(key_tensor);
  if (!SetTensorProto(key_tensor->ToAbstract(), key_tensor_proto)) {
    MS_LOG(ERROR) << "Export key tensor of MapTensor failed, key_tensor: " << key_tensor->ToString();
    return false;
  }
  // value_tensor
  auto *value_tensor_proto = map_tensor_proto->mutable_value_tensor();
  MS_EXCEPTION_IF_NULL(value_tensor_proto);
  auto &value_tensor = export_data.value_tensor;
  MS_EXCEPTION_IF_NULL(value_tensor);
  if (!SetTensorProto(value_tensor->ToAbstract(), value_tensor_proto)) {
    MS_LOG(ERROR) << "Export value tensor of MapTensor failed, value_tensor: " << value_tensor->ToString();
    return false;
  }
  // status_tensor
  auto *status_tensor_proto = map_tensor_proto->mutable_status_tensor();
  MS_EXCEPTION_IF_NULL(status_tensor_proto);
  auto &status_tensor = export_data.status_tensor;
  MS_EXCEPTION_IF_NULL(status_tensor);
  if (!SetTensorProto(status_tensor->ToAbstract(), status_tensor_proto)) {
    MS_LOG(ERROR) << "Export status tensor of MapTensor failed, status_tensor: " << status_tensor->ToString();
    return false;
  }
  return true;
}

bool IrExportBuilder::ConvertAbstractMapTensorToAttrProto(const AbstractBasePtr &abstract,
                                                          mind_ir::AttributeProto *const attr_proto) {
  auto map_tensor_abs = abstract->cast<abstract::AbstractMapTensorPtr>();
  MS_EXCEPTION_IF_NULL(map_tensor_abs);

  auto map_tensor_type = map_tensor_abs->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_MAP_TENSOR);
  // key_tensor
  auto key_dtype = map_tensor_type->key_dtype();
  auto key_shape = {abstract::Shape::kShapeDimAny};
  auto key_tensor_abs = std::make_shared<abstract::AbstractTensor>(key_dtype, key_shape);
  auto *key_tensor_proto = attr_proto->add_tensors();
  MS_EXCEPTION_IF_NULL(key_tensor_proto);
  if (!SetTensorProto(key_tensor_abs, key_tensor_proto)) {
    MS_LOG(ERROR) << "Export key tensor abstract of AbstractMapTensor failed, abstract_map_tensor: "
                  << abstract->ToString();
    return false;
  }
  // value_dtype value_shape
  auto value_dtype = map_tensor_type->key_dtype();
  auto value_shape = map_tensor_abs->value_shape()->shape();
  auto value_tensor_abs = std::make_shared<abstract::AbstractTensor>(value_dtype, value_shape);
  auto *value_tensor_proto = attr_proto->add_tensors();
  MS_EXCEPTION_IF_NULL(value_tensor_proto);
  if (!SetTensorProto(value_tensor_abs, value_tensor_proto)) {
    MS_LOG(ERROR) << "Export value tensor abstract of AbstractMapTensor failed, abstract_map_tensor: "
                  << abstract->ToString();
    return false;
  }
  // default_value
  auto default_value = map_tensor_abs->default_value();
  auto *default_value_proto = attr_proto->add_values();
  MS_EXCEPTION_IF_NULL(default_value_proto);
  if (!SetValueToAttributeProto(default_value, default_value_proto)) {
    MS_LOG(ERROR) << "Export default value of AbstractMapTensor failed, abstract_map_tensor: " << abstract->ToString();
    return false;
  }
  return true;
}

bool IrExportBuilder::BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Node: '" << node->ToString() << "' is not cnode";
      continue;
    }
    if (is_kernel_graph_ && (node_name_map_.find(node) != node_name_map_.end() || node->func_graph() != func_graph)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      if (!BuildOutput(cnode, graph_proto)) {
        MS_LOG(ERROR) << "Build output for graph " << func_graph->ToString() << " failed.";
        return false;
      }
    } else {
      auto iter = graph_protos_.find(node->func_graph());
      if (iter == graph_protos_.end()) {
        MS_LOG(ERROR) << "Can not find the graph proto of func_graph " << node->func_graph()->ToString();
        return false;
      }
      auto owner_graph_proto = iter->second;
      if (!BuildCNode(cnode, owner_graph_proto)) {
        MS_LOG(ERROR) << "Build proto for cnode " << cnode->DebugString() << " failed.";
        return false;
      }
    }
  }
  return true;
}

bool IrExportBuilder::BuildIsolatedCNode(const AnfNodePtr &node, std::set<AnfNodePtr> *visited) {
  MS_EXCEPTION_IF_NULL(node);
  auto iter = node_name_map_.find(node);
  if (iter != node_name_map_.end()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(visited);
  if (visited->find(node) != visited->end()) {
    MS_LOG(ERROR) << "There is a cycle when build node " << node->DebugString();
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &graph = cnode->func_graph();
  if (!graph) {
    MS_LOG(ERROR) << "The isolated node " << node->DebugString() << " is not belongs to any graph.";
    return false;
  }
  auto graph_proto = graph_protos_[graph];
  auto input_size = cnode->size();
  std::vector<string> input_names;
  // build input nodes
  for (size_t i = 1; i < input_size; i++) {
    auto input = cnode->input(i);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<Parameter>()) {
      MS_LOG(ERROR) << "Only support that the isolated node's input is cnode or value_node, but the input is "
                    << input->DebugString();
      return false;
    }
    std::string node_name;
    if (input->isa<ValueNode>()) {
      auto input_graph = input->func_graph();
      auto input_proto = input_graph ? graph_protos_[input_graph] : graph_proto;
      MS_EXCEPTION_IF_NULL(input_proto);
      node_name = BuildInputNode(input, input_proto);
    } else {
      if (!BuildIsolatedCNode(input, visited)) {
        MS_LOG(ERROR) << "Build input node for " << input->DebugString() << " failed.";
        return false;
      }
      node_name = GetUniqueNodeName(input);
    }
    if (node_name.empty()) {
      MS_LOG(ERROR) << "Build input node for " << input->DebugString() << " failed.";
      return false;
    }
    input->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(node_name));
    input_names.push_back(node_name);
  }
  // build cnode
  auto output_name = GetUniqueNodeName(cnode);
  cnode->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(output_name));
  if (nodeName_.count(output_name) > 0) {
    MS_LOG(INFO) << "There is a duplicate name: " << output_name;
    return true;
  }
  mind_ir::NodeProto *node_proto = graph_proto->add_node();
  (void)nodeName_.insert(output_name);
  node_proto->add_output(output_name);
  node_proto->set_name(output_name);
  node_proto->set_domain(cnode->fullname_with_scope());
  AnfNodePtr op = cnode->input(0);
  std::string type_name = GetOpTypeName(op);
  if (type_name.empty()) {
    MS_LOG(ERROR) << "Get op type name for " << op->DebugString() << " failed.";
    return false;
  }
  node_proto->set_op_type(type_name);
  if (!SetAbstractToNodeProto(cnode, node_proto)) {
    MS_LOG(DEBUG) << "Fail to export abstract of the node: " << node->DebugString();
  }
  (void)std::for_each(input_names.begin(), input_names.end(),
                      [&node_proto](const string &name) { node_proto->add_input(name); });
  if (!BuildCNodeAttr(cnode, node_proto)) {
    MS_LOG(ERROR) << "Set value to node attr to node proto failed.";
    return false;
  }
  (void)(visited->insert(node));
  return true;
}

bool IrExportBuilder::BuildIsolatedNodes(const std::vector<AnfNodePtr> &isolated_nodes) {
  for (const auto &node : isolated_nodes) {
    if (!node->isa<CNode>()) {
      MS_LOG(ERROR) << "Only support that the isolated node is cnode, but the node is " << node->DebugString();
      return false;
    }
    if (mindspore::IsPrimitiveCNode(node, mindspore::prim::kPrimReturn)) {
      MS_LOG(ERROR) << "Only support that the isolated node is not return node, but the node is "
                    << node->DebugString();
      return false;
    }
    std::set<AnfNodePtr> visited;
    if (!BuildIsolatedCNode(node, &visited)) {
      MS_LOG(ERROR) << "Build isolated node " << node->DebugString() << " failed.";
      return false;
    }
  }

  return true;
}

bool IrExportBuilder::BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph_proto);
  const int OutputSize = 2;
  if (node->size() != OutputSize) {
    MS_LOG(ERROR) << "Number of inputs of return node is not equal to 2.";
    return false;
  }
  auto graph_name = graph_proto->name();
  node->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(graph_name + kReturnNode));
  AnfNodePtr arg = node->input(1);
  auto node_name = BuildInputNode(arg, graph_proto);
  arg->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(node_name));
  if (node_name.empty()) {
    MS_LOG(ERROR) << "Build input node failed for arg " << arg->DebugString();
    return false;
  }
  mind_ir::ValueInfoProto *output_proto = graph_proto->add_output();
  output_proto->set_name(node_name);
  // for return node primitive export
  AnfNodePtr op = node->input(0);
  op->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(graph_name + kReturnPrimNode));
  return SetValueInfoProto(arg, output_proto);
}

std::string IrExportBuilder::GetOpTypeName(const AnfNodePtr &node) {
  // May be ValueNode/CNode/Parameter
  std::string type_name = "";
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto do_sign_prim = prim->cast_ptr<prim::DoSignaturePrimitive>();
    if (do_sign_prim != nullptr && do_sign_prim->function() != nullptr &&
        do_sign_prim->function()->isa<MetaFuncGraph>()) {
      type_name = "REF::MetaFuncGraph::" + do_sign_prim->function()->cast_ptr<MetaFuncGraph>()->name();
    } else {
      const auto &unique_name = GetPrimitiveUniqueName(prim);
      type_name = "REF::" + unique_name;
      // for valuenode export
      node->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(unique_name));
    }
  } else if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(fg);
    todo_.push_back(fg);
    type_name = "REF::" + fg->ToString();
  } else if (node->isa<CNode>() || node->isa<Parameter>()) {
    auto nodeName = GetUniqueNodeName(node);
    type_name = "REF::" + nodeName;
    if (nodeName_.count(nodeName) == 0) {
      MS_LOG(ERROR) << "There is not the name: " << nodeName;
      return "";
    }
  } else if (IsValueNode<MindIRClassType>(node)) {
    auto class_type = GetValueNode<MindIRClassTypePtr>(node)->name();
    // class 'XXX' -> XXX
    constexpr int64_t path_begin_index = 7;
    auto str = std::string(class_type.begin() + path_begin_index, class_type.end() - 1);
    type_name = "REF::ClassType::" + str;
  } else if (IsValueNode<MetaFuncGraph>(node)) {
    auto meta_fg = GetValueNode<MetaFuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(meta_fg);
    type_name = "REF::MetaFuncGraph::" + meta_fg->name();
  } else {
    MS_LOG(ERROR) << "Need to support op type: " << node->DebugString();
    return "";
  }
  MS_LOG(DEBUG) << "ExportType: " << type_name;
  return type_name;
}

bool IrExportBuilder::ExportSequence(const abstract::AbstractSequencePtr &seq_abs,
                                     mind_ir::AttributeProto *const attr_proto) {
  if (seq_abs->isa<abstract::AbstractTuple>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TUPLE);
  } else {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_LIST);
  }
  auto seq_info_proto = attr_proto->mutable_seq_info();
  seq_info_proto->set_is_dyn_len(seq_abs->dynamic_len());

  auto elem_abs = seq_abs->dynamic_len_element_abs();
  if (elem_abs != nullptr) {
    mind_ir::AttributeProto *tuple_elem_proto = seq_info_proto->mutable_tuple_elem_item();
    if (!SetAbstractToNodeProto(elem_abs, tuple_elem_proto)) {
      return false;
    }
  }

  const auto &elems = seq_abs->elements();
  for (const auto &item : elems) {
    mind_ir::AttributeProto *attr_values = attr_proto->add_values();
    if (!SetAbstractToNodeProto(item, attr_values)) {
      return false;
    }
  }
  return true;
}

bool IrExportBuilder::SetAbstractToNodeProto(const AbstractBasePtr &abs, mind_ir::AttributeProto *const attr_proto) {
  auto type = abs->BuildType();
  auto shape = abs->BuildShape();
  // Not use abstract because the abstract of csr tensor is a subclass of AbstractTuple
  if (type->isa<Tuple>() || type->isa<List>()) {
    return ExportSequence(abs->cast<abstract::AbstractSequencePtr>(), attr_proto);
  } else if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    return SetTensorProto(abs, tensor_proto);
  } else if (type->isa<Number>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_SCALAR);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto data_type = GetMindirDataType(type->type_id());
    tensor_proto->set_data_type(data_type);
    tensor_proto->add_dims(0);
  } else if (type->isa<Function>()) {
    if (!SetAbstractFuncToAttributeProto(abs, attr_proto)) {
      return false;
    }
  } else if (type->isa<String>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_STRING);
  } else if (type->isa<UMonadType>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UMONAD);
  } else if (type->isa<IOMonadType>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_IOMONAD);
  } else if (type->isa<CSRTensorType>()) {
    auto csr_tensor_abs = abs->cast<abstract::AbstractCSRTensorPtr>();
    if (!SetCSRTensorToProto(csr_tensor_abs, attr_proto)) {
      return false;
    }
  } else if (type->isa<COOTensorType>()) {
    auto coo_tensor_abs = abs->cast<abstract::AbstractCOOTensorPtr>();
    if (!SetCOOTensorToProto(coo_tensor_abs, attr_proto)) {
      return false;
    }
  } else if (type->isa<TypeNone>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_NONE);
  } else if (type->isa<MapTensorType>()) {
    return ConvertAbstractMapTensorToAttrProto(abs, attr_proto);
  } else {
    MS_LOG(ERROR) << "Type of cnode need to be supported: " << type->type_name();
    return false;
  }

  return true;
}

bool IrExportBuilder::SetAbstractToNodeProto(const CNodePtr &node, mind_ir::NodeProto *const node_proto) {
  // Get shape of cnode
  // 1. need to get shape from tuple element
  // 2. save shape in TensorProto
  MS_EXCEPTION_IF_NULL(node);
  auto type = node->Type();
  auto shape = node->Shape();
  auto abs = node->abstract();
  // For the bprop fg which has not been renormalized.
  if (type == nullptr || shape == nullptr) {
    return true;
  }
  mind_ir::AttributeProto *attr_proto = node_proto->add_attribute();
  if (!SetAbstractToNodeProto(abs, attr_proto)) {
    MS_LOG(WARNING) << "Set shape to NodeProto for " << node->DebugString() << " failed. abs: " << abs->ToString();
    return false;
  }
  attr_proto->set_name("shape");
  return true;
}

bool IrExportBuilder::BuildCNode(const CNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  auto inputs_size = node->size();
  if (inputs_size < 1) {
    MS_LOG(ERROR) << "Inputs of node " << node->DebugString() << " is empty";
    return false;
  }

  // Need to build input node before dealing with cnode
  std::vector<string> input_names;
  for (size_t i = 1; i < inputs_size; i++) {
    auto input = node->input(i);
    std::string node_name = BuildInputNode(input, graph_proto);
    input->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(node_name));
    if (node_name.empty()) {
      MS_LOG(ERROR) << "Build input node for " << input->DebugString() << " failed.";
      return false;
    }
    input_names.push_back(node_name);
  }

  // Build cnode
  std::string output_name = GetUniqueNodeName(node);
  node->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(output_name));
  if (nodeName_.count(output_name) > 0) {
    MS_LOG(INFO) << "There is a duplicate name: " << output_name;
    return true;
  }

  mind_ir::NodeProto *node_proto = graph_proto->add_node();
  (void)nodeName_.insert(output_name);
  node_proto->add_output(output_name);
  node_proto->set_name(output_name);
  node_proto->set_domain(node->fullname_with_scope());
  AnfNodePtr op = node->input(0);
  std::string type_name = GetOpTypeName(op);
  if (type_name.empty()) {
    MS_LOG(ERROR) << "Get op type name for " << op->DebugString() << " failed.";
    return false;
  }
  node_proto->set_op_type(type_name);
  last_node_ = node_proto;
  if (!SetAbstractToNodeProto(node, node_proto)) {
    MS_LOG(DEBUG) << "Fail to export abstract of the node: " << node->DebugString();
  }

  (void)std::for_each(input_names.begin(), input_names.end(),
                      [&node_proto](const string &name) { node_proto->add_input(name); });

  if (!BuildCNodeAttr(node, node_proto)) {
    MS_LOG(ERROR) << "Set value to node attr to node proto failed.";
    return false;
  }
  return true;
}

bool IrExportBuilder::BuildValueNode(const ValueNodePtr &node, const string &node_name,
                                     mind_ir::GraphProto *const graph_proto) {
  // FuncGraphNode don't need to be exported to the proto in this step
  // check the node has been exported before
  if (IsValueNode<FuncGraph>(node) || nodeName_.count(node_name) > 0) {
    return true;
  }
  (void)nodeName_.insert(node_name);
  // When node input is a ValueNode, need to create a Constant Node
  mind_ir::NodeProto *node_proto = graph_proto->add_node();
  node_proto->set_name(node_name);
  node_proto->add_output(node_name);
  if (!SetAttributeProto(node, node_proto)) {
    return false;
  }
  return true;
}

std::string IrExportBuilder::BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  std::string node_name = GetUniqueNodeName(node);
  if (node->isa<ValueNode>()) {
    if (!BuildValueNode(node->cast<ValueNodePtr>(), node_name, graph_proto)) {
      MS_LOG(ERROR) << "Export ValueNode Failed";
      return "";
    }
    MS_LOG(DEBUG) << "Export ValueNode " << node->DebugString() << " success";
  }
  return node_name;
}

std::string IrExportBuilder::GetUniqueNodeName(const AnfNodePtr &node) {
  // Naming anfnode
  // 1. parameter is unique in one func_graph
  // 2. cnode and valuenode may be reduplicative, so add index to identify.
  auto iter = node_name_map_.find(node);
  if (iter != node_name_map_.end()) {
    return iter->second;
  }
  // FuncGraph will be added to functions and the input name is the function name.
  if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    todo_.push_back(fg);
    auto name = fg->ToString();
    node->set_user_data<std::string>(kUniqueCacheName, std::make_shared<std::string>(name));
    return name;
  }

  std::string node_name = GetNodeName(node);
  // Compatible before. CNode = FuncGraphName:CNodeName:index ,Parameter = FuncGraphName:ParameterName
  if (node->isa<CNode>()) {
    node_name = node_name + ":" + std::to_string(GetUniqueID());
  }
  // Avoid duplicate name.
  while (nodeName_.count(node_name) > 0) {
    node_name = node_name + "_" + std::to_string(GetUniqueID());
  }
  node_name_map_[node] = node_name;
  return node_name;
}

std::string IrExportBuilder::GetNodeName(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  std::string node_name = "";
  if (node->func_graph() != nullptr) {
    node_name = node->func_graph()->ToString() + ":";
  }
  if (node->isa<ValueNode>()) {
    // Needn't value
    node_name += node->AnfNode::ToString();
  } else {
    node_name += node->ToString();
  }
  MS_LOG(DEBUG) << "GetNodeName: " << node_name;
  return node_name;
}

bool IrExportBuilder::SetAttributeProto(const AnfNodePtr &node, mind_ir::NodeProto *const node_proto) {
  if (node == nullptr || node_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or NodeProto is null!";
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  node_proto->set_op_type("Constant");
  mind_ir::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("value");
  MS_LOG(DEBUG) << "Set Constant attribute: " << value->ToString();
  return SetValueToAttributeProto(value, attr_proto);
}

bool IrExportBuilder::SetTensorTypeToAttributeProto(const ValuePtr &value, mind_ir::TensorProto *tensor_proto) {
  tensor_proto->set_name("tensor0");
  auto elem_type = value->cast<TensorTypePtr>()->element();
  if (elem_type->isa<Int>()) {
    auto int_value = elem_type->cast<IntPtr>();
    auto data_type = GetMindirDataBitsIntType(int_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (elem_type->isa<Float>()) {
    auto float_value = elem_type->cast<FloatPtr>();
    auto data_type = GetMindirDataBitsFloatType(float_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else {
    MS_LOG(ERROR) << "Unsupported type " << elem_type->type_name();
    return false;
  }
  return true;
}

bool IrExportBuilder::SetTypeToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
  mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
  if (value->isa<Int>()) {
    tensor_proto->set_name("value0");
    auto int_value = value->cast<IntPtr>();
    auto data_type = GetMindirDataBitsIntType(int_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<UInt>()) {
    tensor_proto->set_name("value0");
    auto float_value = value->cast<UIntPtr>();
    auto data_type = GetMindirDataBitsUIntType(float_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<Float>()) {
    tensor_proto->set_name("value0");
    auto float_value = value->cast<FloatPtr>();
    auto data_type = GetMindirDataBitsFloatType(float_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<BFloat>()) {
    tensor_proto->set_name("value0");
    auto bfloat_value = value->cast<BFloatPtr>();
    auto data_type = GetMindirDataBitsBFloatType(bfloat_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<Complex>()) {
    tensor_proto->set_name("value0");
    auto complex_value = value->cast<ComplexPtr>();
    auto data_type = GetMindirDataBitsComplexType(complex_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<Bool>()) {
    tensor_proto->set_name("value0");
    tensor_proto->set_data_type(mind_ir::TensorProto_DataType_BOOL);
  } else if (value->isa<TensorType>()) {
    return SetTensorTypeToAttributeProto(value, tensor_proto);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
  return true;
}

bool IrExportBuilder::SetNamedValueToAttributeProto(const ValuePtr &value,
                                                    mind_ir::AttributeProto *const attr_proto) const {
  if (value->isa<None>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_NONE);
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else if (value->isa<MindIRClassType>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_CLASS_TYPE);
    auto class_type = GetValue<MindIRClassTypePtr>(value)->name();
    // class 'XXX' -> XXX
    constexpr int64_t path_begin_index = 7;
    auto str = std::string(class_type.begin() + path_begin_index, class_type.end() - 1);
    attr_proto->set_s(str);
  } else if (value->isa<MindIRNameSpace>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_NAME_SPACE);
    attr_proto->set_s(GetValue<MindIRNameSpacePtr>(value)->name_space());
  } else if (value->isa<MindIRSymbol>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_SYMBOL);
    attr_proto->set_s(GetValue<MindIRSymbolPtr>(value)->symbol());
  } else {
    MS_LOG(ERROR) << "Unsupported named type: " << value->type_name();
    return false;
  }
  return true;
}

bool IrExportBuilder::SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(attr_proto);
  if (value->isa<StringImm>() || value->isa<Scalar>()) {
    return SetScalarToAttributeProto_ir(value, attr_proto);
  } else if (value->isa<Number>() || value->isa<TensorType>()) {
    return SetTypeToAttributeProto(value, attr_proto);
  } else if (value->isa<ValueSequence>()) {
    if (!SetSequenceToAttributeProto(value->cast<ValueSequencePtr>(), attr_proto)) {
      MS_LOG(ERROR) << "Set sequence to AttributeProto failed.";
      return false;
    }
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else if (value->isa<ValueDictionary>()) {
    if (!SetDictToAttributeProto(value->cast<ValueDictionaryPtr>(), attr_proto)) {
      MS_LOG(ERROR) << "Set dictionary to AttributeProto failed.";
      return false;
    }
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else if (value->isa<tensor::Tensor>()) {
    return SetTensorToAttributeProto(value, attr_proto);
  } else if (value->isa<Named>()) {
    return SetNamedValueToAttributeProto(value, attr_proto);
  } else if (value->isa<TypeNull>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TYPE_NULL);
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else if (value->isa<Monad>()) {
    if (value->isa<UMonad>()) {
      attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UMONAD);
    } else if (value->isa<IOMonad>()) {
      attr_proto->set_type(mind_ir::AttributeProto_AttributeType_IOMONAD);
    } else {
      MS_LOG(ERROR) << "Unsupported Monad type: " << value->type_name();
      return false;
    }
  } else if (value->isa<QuantizationParam>()) {
    auto quantization_param = value->cast<std::shared_ptr<QuantizationParam>>();
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    auto tensor_proto = attr_proto->add_tensors();
    tensor_proto->set_name(attr_proto->name());
    auto quant_param_proto = tensor_proto->add_quant_params();
    auto ret = SetQuantizationParamToAttrProto(quantization_param, quant_param_proto);
    if (ret != true) {
      MS_LOG(ERROR) << "QuantizationParam Set Value to AttributeProto Error";
      return false;
    }
  } else if (value->isa<Functor>()) {
    return SetFunctorToAttrProto(value->cast<FunctorPtr>(), attr_proto);
  } else {
    MS_LOG(ERROR) << "Unsupported type: " << value->type_name();
    return false;
  }
  return true;
}

bool IrExportBuilder::SetScalarToAttributeProto_ir(const ValuePtr &value,
                                                   mind_ir::AttributeProto *const attr_proto) const {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  if (value->isa<StringImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_STRING);
    attr_proto->set_s(GetValue<std::string>(value));
  } else if (value->isa<BoolImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    int64_t attr_value = GetValue<bool>(value) ? 1 : 0;
    attr_proto->set_i(attr_value);
  } else if (SetScalarToAttributeProtoForInt_ir(value, attr_proto)) {
    return true;
  } else if (value->isa<FP32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FLOAT);
    attr_proto->set_f(GetValue<float>(value));
  } else if (value->isa<FP64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_DOUBLE);
    attr_proto->set_d(GetValue<double>(value));
  } else {
    MS_LOG(ERROR) << "Unsupported scalar type: " << value->type_name();
    return false;
  }
  return true;
}

bool IrExportBuilder::SetScalarToAttributeProtoForInt_ir(const ValuePtr &value,
                                                         mind_ir::AttributeProto *const attr_proto) const {
  if (value->isa<Int8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT8);
    attr_proto->set_i(value->cast<Int8ImmPtr>()->value());
  } else if (value->isa<Int16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT16);
    attr_proto->set_i(value->cast<Int16ImmPtr>()->value());
  } else if (value->isa<Int32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT32);
    attr_proto->set_i(value->cast<Int32ImmPtr>()->value());
  } else if (value->isa<Int64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT64);
    attr_proto->set_i(value->cast<Int64ImmPtr>()->value());
  } else if (value->isa<UInt8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT8);
    attr_proto->set_i(value->cast<UInt8ImmPtr>()->value());
  } else if (value->isa<UInt16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT16);
    attr_proto->set_i(value->cast<UInt16ImmPtr>()->value());
  } else if (value->isa<UInt32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT32);
    attr_proto->set_i(value->cast<UInt32ImmPtr>()->value());
  } else if (value->isa<UInt64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT64);
    attr_proto->set_i(UlongToLong(value->cast<UInt64ImmPtr>()->value()));
  } else {
    return false;
  }
  return true;
}

bool IrExportBuilder::SetTypeToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AttributeProto is null!";
  }
  if (value->isa<Int>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto int_value = value->cast<IntPtr>();
    auto data_type = GetMindirDataBitsIntType(int_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<Float>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto float_value = value->cast<FloatPtr>();
    auto data_type = GetMindirDataBitsFloatType(float_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<UInt>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto uint_value = value->cast<UIntPtr>();
    auto data_type = GetMindirDataBitsUIntType(uint_value->nbits());
    if (data_type == mind_ir::TensorProto_DataType_UNDEFINED) {
      return false;
    }
    tensor_proto->set_data_type(data_type);
  } else if (value->isa<Bool>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    tensor_proto->set_data_type(mind_ir::TensorProto_DataType_BOOL);
  } else if (value->isa<tensor::Tensor>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    return SetTensorToAttributeProto(value, attr_proto);
  } else if (value->isa<QuantizationParam>()) {
    auto quantization_param = value->cast<std::shared_ptr<QuantizationParam>>();
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    auto tensor_proto = attr_proto->add_tensors();
    tensor_proto->set_name("quant_param");
    auto quant_param_proto = tensor_proto->add_quant_params();
    auto ret = SetQuantizationParamToAttrProto(quantization_param, quant_param_proto);
    if (ret != true) {
      MS_LOG(ERROR) << "QuantizationParam Set Value to AttributeProto Error";
      return false;
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
  return true;
}

bool IrExportBuilder::SetScalarToAttributeProto_irs(const ValuePtr &value,
                                                    mind_ir::AttributeProto *const attr_proto) const {
  if (attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AttributeProto is null!";
  }
  if (value->isa<StringImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_STRING);
    attr_proto->add_strings(GetValue<std::string>(value));
  } else if (value->isa<BoolImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    attr_proto->add_ints(GetValue<bool>(value));
  } else if (SetScalarToAttributeProtoForInt_irs(value, attr_proto)) {
    return true;
  } else if (value->isa<FP32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FLOAT);
    attr_proto->add_floats(GetValue<float>(value));
  } else if (value->isa<FP64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_DOUBLE);
    attr_proto->add_doubles(GetValue<double>(value));
  } else {
    MS_LOG(ERROR) << "Unsupported scalar type: " << value->type_name();
    return false;
  }
  return true;
}

bool IrExportBuilder::SetScalarToAttributeProtoForInt_irs(const ValuePtr &value,
                                                          mind_ir::AttributeProto *const attr_proto) const {
  if (value->isa<Int8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT8);
    attr_proto->add_ints(value->cast<Int8ImmPtr>()->value());
  } else if (value->isa<Int16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT16);
    attr_proto->add_ints(value->cast<Int16ImmPtr>()->value());
  } else if (value->isa<Int32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT32);
    attr_proto->add_ints(value->cast<Int32ImmPtr>()->value());
  } else if (value->isa<Int64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT64);
    attr_proto->add_ints(value->cast<Int64ImmPtr>()->value());
  } else if (value->isa<UInt8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT8);
    attr_proto->add_ints(value->cast<UInt8ImmPtr>()->value());
  } else if (value->isa<UInt16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT16);
    attr_proto->add_ints(value->cast<UInt16ImmPtr>()->value());
  } else if (value->isa<UInt32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT32);
    attr_proto->add_ints(value->cast<UInt32ImmPtr>()->value());
  } else if (value->isa<UInt64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT64);
    attr_proto->add_ints(SizeToInt(value->cast<UInt64ImmPtr>()->value()));
  } else {
    return false;
  }
  return true;
}

bool IrExportBuilder::SetSeqElemToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "Value is nullptr";
    return false;
  }
  if (value->isa<StringImm>() || value->isa<Scalar>()) {
    return SetScalarToAttributeProto_irs(value, attr_proto);
  }
  return SetTypeToAttributeProto_irs(value, attr_proto);
}

bool IrExportBuilder::SetSequenceToAttributeProto(const ValueSequencePtr &value,
                                                  mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValueSequencePtr or AttributeProto is null!";
  }
  if (value->isa<ValueTuple>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TUPLE);
  } else if (value->isa<ValueList>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_LIST);
  } else {
    MS_LOG(EXCEPTION) << "The sequance value should be ValueTuple or ValueList, but it is " << value->ToString();
  }
  auto value_sequence = value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_sequence);
  const auto &values = value_sequence->value();
  if (values.empty()) {
    MS_LOG(DEBUG) << "SetSequenceToAttributeProto sequence size is 0";
    return true;
  }
  for (const auto &item : values) {
    mind_ir::AttributeProto *attr_values = attr_proto->add_values();
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<ValueSequence>()) {
      if (!SetSequenceToAttributeProto(item->cast<ValueSequencePtr>(), attr_values)) {
        MS_LOG(ERROR) << "Set sequence to AttributeProto failed.";
        return false;
      }
    } else {
      if (!SetSeqElemToAttributeProto(item, attr_values)) {
        MS_LOG(ERROR) << "Set seq elem to AttributeProto failed.";
        return false;
      }
    }
  }
  return true;
}

bool IrExportBuilder::SetDictToAttributeProto(const ValueDictionaryPtr &value_dict,
                                              mind_ir::AttributeProto *const attr_proto) {
  if (value_dict == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValueDictionaryPtr or AttributeProto is null!";
  }
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_DICT);
  const auto &values = value_dict->value();
  if (values.empty()) {
    MS_LOG(DEBUG) << "SetDictToAttributeProto dictionary size is 0";
    return true;
  }
  for (const auto &item : values) {
    mind_ir::AttributeProto *dict_item_proto = attr_proto->add_values();
    const auto &key = item.first;
    dict_item_proto->set_name(GetValue<std::string>(key));
    const auto &value = item.second;
    MS_EXCEPTION_IF_NULL(value);
    mind_ir::AttributeProto *dict_item_value = dict_item_proto->add_values();
    if (value->isa<ValueSequence>()) {
      if (!SetSequenceToAttributeProto(value->cast<ValueSequencePtr>(), dict_item_value)) {
        MS_LOG(ERROR) << "Set sequence to AttributeProto failed.";
        return false;
      }
    } else if (value->isa<ValueDictionary>()) {
      if (!SetDictToAttributeProto(value->cast<ValueDictionaryPtr>(), dict_item_value)) {
        MS_LOG(ERROR) << "Set dictionary to AttributeProto failed.";
        return false;
      }
    } else if (value->isa<StringImm>() || value->isa<Scalar>()) {
      if (!SetScalarToAttributeProto_irs(value, dict_item_value)) {
        MS_LOG(ERROR) << "Set StringImm or Scalar to AttributeProto failed.";
        return false;
      }
    } else if (value->isa<Number>() || value->isa<tensor::Tensor>()) {
      if (!SetTypeToAttributeProto_irs(value, dict_item_value)) {
        MS_LOG(ERROR) << "Set Number or Tensor to AttributeProto failed.";
        return false;
      }
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type while converting ValueDictionary to AttributeProto: "
                        << value->type_name();
    }
  }
  return true;
}

bool IrExportBuilder::BuildCNodeAttr(const CNodePtr &node, mind_ir::NodeProto *const node_proto) {
  for (const auto &attr : node->attrs()) {
    mind_ir::AttributeProto *attr_proto = node_proto->add_node_attr();
    attr_proto->set_name(attr.first);
    if (!SetValueToAttributeProto(attr.second, attr_proto)) {
      MS_LOG(ERROR) << "Set value to node attr to node proto failed.";
      MS_LOG(ERROR) << "node :" << node->DebugString() << "attr:{" << attr.first << "," << attr.second << "}";
      return false;
    }
  }

  for (const auto &attr : node->primal_attrs()) {
    mind_ir::AttributeProto *attr_proto = node_proto->add_primal_attr();
    attr_proto->set_name(attr.first);
    if (!SetValueToAttributeProto(attr.second, attr_proto)) {
      MS_LOG(ERROR) << "Set value to node primal attr to node proto failed.";
      MS_LOG(ERROR) << "node :" << node->DebugString() << "attr:{" << attr.first << "," << attr.second << "}";
      return false;
    }
  }
  return true;
}

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph, const bool &incremental) {
  auto builder = std::make_shared<IrExportBuilder>(incremental);
  if (builder == nullptr) {
    MS_LOG(ERROR) << "Create ir exporter failed!";
    return "";
  }
  auto exporter = std::make_shared<IrExporter>(builder);
  if (exporter == nullptr) {
    return "";
  }
  auto ret = exporter->GetDumpString(func_graph);
  return ret;
}

ModelProtoPtr GenBinaryProto(const FuncGraphPtr &func_graph) {
  auto exporter = std::make_shared<IrExporter>(std::make_shared<IrExportBuilder>());
  return exporter->GetDumpProto(func_graph);
}

bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path) {
  auto exporter = std::make_shared<IrExporter>(std::make_shared<IrExportBuilder>());
  auto proto = exporter->GetDumpProto(func_graph);
  MindIRExporter mindir_exporter;
  if (proto == nullptr) {
    MS_LOG(ERROR) << "Get binary proto for graph " << func_graph->ToString() << " failed.";
    return false;
  }
  return mindir_exporter.SaveProtoToFile(proto.get(), file_path);
}

bool DumpBinaryProto(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                     const std::vector<AnfNodePtr> &isolated_nodes, const std::string &file_path) {
  auto exporter = std::make_shared<IrExporter>(std::make_shared<IrExportBuilder>());
  auto proto = exporter->GetDumpProto(root_graph, child_graphs, isolated_nodes);
  if (proto == nullptr) {
    MS_LOG(ERROR) << "Get binary proto for graph " << root_graph->ToString() << " failed.";
    return false;
  }
  auto realpath = Common::CreatePrefixPath(file_path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << file_path << " failed.";
    return false;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open the file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return false;
  }
  if (!proto->SerializeToOstream(&fout)) {
    MS_LOG(ERROR) << "Failed to write the mindir proto to file " << realpath.value();
    fout.close();
    return false;
  }
  fout.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
  return true;
}

bool MindIRExporter::ParserPath(const std::string &output_path) {
  if (!FileUtils::ParserPathAndModelName(output_path, &save_path_, &model_name_)) {
    MS_LOG(ERROR) << "parser save path and model name from output_path failed.";
    return false;
  }
#ifdef _WIN32
  save_model_path_ = save_path_ + "\\" + model_name_ + ".mindir";
#else
  save_model_path_ = save_path_ + "/" + model_name_ + ".mindir";
#endif
  return true;
}

bool MindIRExporter::ExportProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                                 const FuncGraphPtr &param_layout_fg) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return false;
  }

  if (!ParserPath(file_path)) {
    MS_LOG(ERROR) << "parse path failed.";
    return false;
  }

  // Serialize to protobuf using unique parameter name label.
  // Do preprocess on func_graph and check conditions for saving together.
  bool ret = PreProcSaveTogether(func_graph);
  if (!ret) {
    MS_LOG(ERROR) << "PreProcSaveTogether failed";
    return ret;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("PreProcSaveTogether.ir", func_graph);
  }
#endif

  if (save_together_) {
    MS_LOG(INFO) << "SaveMindIRTogether";
    ret = SaveMindIRTogether();
  } else {
    MS_LOG(INFO) << "SplitSave";
    ret = SplitSave();
  }
  if (!ret) {
    MS_LOG(ERROR) << "save mindir weight failed.";
    return ret;
  }
  return true;
}

bool MindIRExporter::SaveMindIRTogether() {
  for (auto &param_proto : *(model_proto_.mutable_graph()->mutable_parameter())) {
    std::string proto_name = param_proto.name();
    auto para = GetFgParaAccordingToProtoName(proto_name);
    if (para == nullptr) {
      return false;
    }
    if (!para->has_default()) {
      continue;
    }
    auto data = para->default_param()->cast<tensor::TensorPtr>();
    param_proto.clear_raw_data();
    param_proto.set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
  }
  return SaveProtoToFile(&model_proto_, save_model_path_);
}

bool MindIRExporter::CreateParameterDir() {
#ifdef _WIN32
  dir_name_ = save_path_ + "\\" + model_name_ + "_variables";
#else
  dir_name_ = save_path_ + "/" + model_name_ + "_variables";
#endif
  fs_ = system::Env::GetFileSystem();
  if (fs_ == nullptr) {
    MS_LOG(ERROR) << "create file system failed.";
    return false;
  }

  if (fs_->FileExist(dir_name_)) {
    if (!DeleteDirRecursively(dir_name_)) {
      return false;
    }
  }

  if (!fs_->CreateDir(dir_name_)) {
    MS_LOG(ERROR) << "create dir failed.";
    return false;
  }

  ChangeFileMode(dir_name_, S_IWUSR | S_IRUSR | S_IXUSR);
  return true;
}

std::string MindIRExporter::CreateExternalPath(const std::string &external_file) {
  dir_path_ = FileUtils::GetRealPath(dir_name_.c_str()).value();
  std::string external_local_path{};
#ifdef _WIN32
  external_local_path = dir_path_ + "\\" + external_file;
#else
  external_local_path = dir_path_ + "/" + external_file;
#endif
  return external_local_path;
}

bool MindIRExporter::SplitSave() {
  MS_LOG(DEBUG) << "Parameters in the net capacity exceeds 1G, save MindIR model and parameters separately.";
  if (!CreateParameterDir()) {
    MS_LOG(ERROR) << "create parameter dir failed.";
    return false;
  }

  int index = 0;
  std::string external_local = "data_" + std::to_string(index);
  auto external_local_path = CreateExternalPath(external_local);
  if (fs_->FileExist(external_local_path)) {
    if (!fs_->DeleteFile(external_local_path)) {
      MS_LOG(ERROR) << "delete file failed.";
      return false;
    }
  }
  int64_t parameter_size = 0;
  int64_t offset = OFFSET;

  data_fs_ = FileUtils::OpenFile(external_local_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Open " << external_local_path << " failed";
    return false;
  }
  if (!ChangeParaDataFile(external_local)) {
    MS_LOG(ERROR) << "change parameter data file failed.";
    return false;
  }

  for (auto &param_proto : *(model_proto_.mutable_graph()->mutable_parameter())) {
    std::string proto_name = param_proto.name();
    auto para = GetFgParaAccordingToProtoName(proto_name);
    if (para == nullptr) {
      return false;
    }
    if (!para->has_default()) {
      continue;
    }
    auto data = para->default_param()->cast<tensor::TensorPtr>();
    int64_t data_length = static_cast<int64_t>(data->data().nbytes());
    int64_t append_size = 0;
    if (data_length % OFFSET != 0) {
      append_size = OFFSET - (data_length % OFFSET);
    }
    parameter_size += ((append_size + data_length) / PARA_ROUND);
    if (parameter_size > static_cast<int64_t>(TOTAL_SAVE)) {
      index++;
      external_local = "data_" + std::to_string(index);
      data_fs_->close();
      delete data_fs_;
      data_fs_ = nullptr;

      if (!ChangeParaDataFile(external_local)) {
        MS_LOG(ERROR) << "change parameter data file failed.";
        return false;
      }
      parameter_size = OFFSET / PARA_ROUND;
    }
    std::string external_local_data = model_name_ + "_variables/" + external_local;
    param_proto.mutable_external_data()->set_location(external_local_data);
    param_proto.mutable_external_data()->set_length(data_length);
    param_proto.mutable_external_data()->set_offset(offset);

    data_fs_->write(static_cast<const char *>(data->data_c()), data_length);
    auto append_data = new char[append_size];
    if (append_data == nullptr) {
      return false;
    }
    data_fs_->write(append_data, append_size);
    offset += (data_length + append_size);
    delete[] append_data;
  }
  std::string split_model_file_name = "";
#ifdef _WIN32
  split_model_file_name = save_path_ + "\\" + model_name_ + "_graph.mindir";
#else
  split_model_file_name = save_path_ + "/" + model_name_ + "_graph.mindir";
#endif
  return SaveProtoToFile(&model_proto_, split_model_file_name);
}

bool MindIRExporter::SaveProtoToFile(mind_ir::ModelProto *model_proto, const std::string &output_file) {
  auto realpath = Common::CreatePrefixPath(output_file, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path of file " << output_file << " failed.";
    return false;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open the file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return false;
  }

  if (!model_proto->SerializeToOstream(&fout)) {
    MS_LOG(ERROR) << "Failed to write the mindir proto to file " << realpath.value();
    fout.close();
    return false;
  }

  fout.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
  return true;
}

bool MindIRExporter::ChangeParaDataFile(const std::string &file) {
  auto real_path = CreateExternalPath(file);
  if (fs_->FileExist(real_path)) {
    if (!fs_->DeleteFile(real_path)) {
      MS_LOG(ERROR) << "delete file failed.";
      return false;
    }
  }
  ChangeFileMode(real_path, S_IWUSR);
  data_fs_ = FileUtils::OpenFile(real_path, std::ios::app);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "data_fs_ is nullptr.";
    return false;
  }
  char front_info[OFFSET]{0};
  front_info[0] = IsSystemLittleEndidan();
  (void)data_fs_->write(front_info, OFFSET);
  return true;
}

bool MindIRExporter::IsSystemLittleEndidan() const {
  int check = 0x01;
  auto address = reinterpret_cast<char *>(&check);
  return *address == 0x01;
}

bool MindIRExporter::PreProcSaveTogether(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return false;
  }

  if (!UpdateParamCount(func_graph)) {
    MS_LOG(ERROR) << "Update parameter count failed.";
    return false;
  }

  // Parse func_graph as model proto
  std::string proto_string = GetBinaryProtoString(func_graph);
  if (proto_string.empty()) {
    MS_LOG(ERROR) << "parse proto string failed.";
    return false;
  }

  if (!model_proto_.ParseFromString(proto_string)) {
    MS_LOG(ERROR) << "parse model proto from string failed.";
    return false;
  }

  if (!ParamDict(func_graph)) {
    MS_LOG(ERROR) << "parse param form funcgraph failed.";
    return false;
  }

  if (!IfSaveTogether(&save_together_)) {
    MS_LOG(ERROR) << "error occur when check condition of saving together.";
    return false;
  }

  return true;
}

bool MindIRExporter::IfSaveTogether(bool *save_together) {
  size_t data_total = model_proto_.ByteSizeLong();
  for (auto &param_proto : model_proto_.graph().parameter()) {
    std::string proto_name = param_proto.name();
    auto para = GetFgParaAccordingToProtoName(proto_name);
    if (para == nullptr) {
      return false;
    }
    if (!para->has_default()) {
      continue;
    }
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(para->default_param());
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "param node default_param is not tensor.";
      return false;
    }
    data_total += tensor->Size();
  }
  if (data_total > TOTAL_SAVE) {
    *save_together = false;
  } else {
    *save_together = false;
  }
  return true;
}

std::shared_ptr<Parameter> MindIRExporter::GetFgParaAccordingToProtoName(const std::string &proto_name) {
  auto beg_pos = proto_name.find_first_of(':') + 1;
  if (beg_pos >= proto_name.size()) {
    MS_LOG(ERROR) << "begin pos exceed proto name length.";
    return nullptr;
  }
  auto name = proto_name.substr(beg_pos);
  if (param_dict_.find(name) == param_dict_.end()) {
    MS_LOG(ERROR) << "param proto name: " << name << " is not in param dict.";
    return nullptr;
  }
  return param_dict_.at(name);
}

bool MindIRExporter::UpdateParamCount(const FuncGraphPtr &func_graph) {
  auto fv_count = 0;
  std::vector<AnfNodePtr> params;
  std::vector<AnfNodePtr> reorder_param;
  reorder_param.reserve(func_graph->parameters().size());
  for (const auto &node : func_graph->parameters()) {
    auto param_node = node->cast<ParameterPtr>();
    if (param_node == nullptr) {
      MS_LOG(ERROR) << "The parameters() in func graph should be all Parameter Node. but got " << node->DebugString();
      return false;
    }
    if (param_node->has_default()) {
      (void)params.emplace_back(param_node);
      ++fv_count;
      continue;
    }
    (void)reorder_param.emplace_back(param_node);
  }

  std::copy(params.begin(), params.end(), std::back_inserter(reorder_param));
  func_graph->set_parameters(reorder_param);
  func_graph->set_fv_param_count(fv_count);
  return true;
}

bool MindIRExporter::ParamDict(const FuncGraphPtr &func_graph) {
  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraphs(func_graph, &all_func_graphs);
  for (auto &fg : all_func_graphs) {
    for (auto &para : fg->parameters()) {
      if (!para->isa<Parameter>()) {
        MS_LOG(ERROR) << "fg parameters contains non-parameter type node.";
        return false;
      }
      auto para_node = para->cast<ParameterPtr>();
      param_dict_[para->ToString()] = para_node;
    }
  }
  return true;
}
}  // namespace mindspore
