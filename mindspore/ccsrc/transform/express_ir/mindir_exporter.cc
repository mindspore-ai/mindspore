/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <fstream>

#include "utils/hash_map.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/func_graph.h"
#include "mindspore/core/ops/core_ops.h"
#include "proto/mind_ir.pb.h"
#include "utils/check_convert_utils.h"
#include "include/common/debug/dump_proto.h"
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"
#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#endif
#include "abstract/abstract_function.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
using FloatPtr = std::shared_ptr<Float>;
using IntPtr = std::shared_ptr<Int>;
using UIntPtr = std::shared_ptr<UInt>;
using ComplexPtr = std::shared_ptr<Complex>;
using ModelProtoPtr = std::shared_ptr<mind_ir::ModelProto>;

// anf type to mindir type map
static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_type_map = {
  {kNumberTypeBool, mind_ir::TensorProto_DataType_BOOL},
  {kNumberTypeInt8, mind_ir::TensorProto_DataType_INT8},
  {kNumberTypeInt16, mind_ir::TensorProto_DataType_INT16},
  {kNumberTypeInt, mind_ir::TensorProto_DataType_INT32},
  {kNumberTypeInt32, mind_ir::TensorProto_DataType_INT32},
  {kNumberTypeInt64, mind_ir::TensorProto_DataType_INT64},
  {kNumberTypeUInt8, mind_ir::TensorProto_DataType_UINT8},
  {kNumberTypeUInt16, mind_ir::TensorProto_DataType_UINT16},
  {kNumberTypeUInt32, mind_ir::TensorProto_DataType_UINT32},
  {kNumberTypeUInt64, mind_ir::TensorProto_DataType_UINT64},
  {kNumberTypeFloat16, mind_ir::TensorProto_DataType_FLOAT16},
  {kNumberTypeFloat, mind_ir::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat32, mind_ir::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat64, mind_ir::TensorProto_DataType_DOUBLE},
  {kObjectTypeString, mind_ir::TensorProto_DataType_STRING},
  {kNumberTypeComplex64, mind_ir::TensorProto_DataType_COMPLEX64},
  {kNumberTypeComplex128, mind_ir::TensorProto_DataType_COMPLEX128}};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_int_map = {
  {8, mind_ir::TensorProto_DataType_INT8},
  {16, mind_ir::TensorProto_DataType_INT16},
  {32, mind_ir::TensorProto_DataType_INT32},
  {64, mind_ir::TensorProto_DataType_INT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_uint_map = {
  {8, mind_ir::TensorProto_DataType_UINT8},
  {16, mind_ir::TensorProto_DataType_UINT16},
  {32, mind_ir::TensorProto_DataType_UINT32},
  {64, mind_ir::TensorProto_DataType_UINT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_float_map = {
  {16, mind_ir::TensorProto_DataType_FLOAT16},
  {32, mind_ir::TensorProto_DataType_FLOAT},
  {64, mind_ir::TensorProto_DataType_FLOAT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_complex_map = {
  {64, mind_ir::TensorProto_DataType_COMPLEX64},
  {128, mind_ir::TensorProto_DataType_COMPLEX128},
};

static std::set<std::string> g_export_attr_blacklist = {kAttrDump};

// Can build different builder according to format
class IrExportBuilder;
using IrExportBuilderPtr = std::shared_ptr<IrExportBuilder>;

class IrExporter {
 public:
  explicit IrExporter(IrExportBuilderPtr builder) : builder_(std::move(builder)) {}
  virtual ~IrExporter() = default;
  std::string GetDumpString(const FuncGraphPtr &func_graph);
  ModelProtoPtr GetDumpProto(const FuncGraphPtr &func_graph, const FuncGraphPtr &param_layout_fg = nullptr);

 private:
  IrExportBuilderPtr builder_;
};
using IrExporterPtr = std::shared_ptr<IrExporter>;

class IrExportBuilder {
 public:
  IrExportBuilder() : model_(std::make_shared<mind_ir::ModelProto>()) {}
  ~IrExportBuilder() = default;
  std::string GetProtoString() const;
  void BuildModelInfo();
  bool BuildModel(const FuncGraphPtr &func_graph);
  ModelProtoPtr Model() { return model_; }

#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
  void BuildLayout(const FuncGraphPtr &func_graph);
#endif

  bool BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildFuncGraphAttrs(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  bool BuildCNode(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  std::string BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto);

  bool SetValueInfoProto(const AnfNodePtr &node, mind_ir::ValueInfoProto *const value_proto);
  bool SetParamToTensorProto(const ParameterPtr &param, mind_ir::TensorProto *const tensor_proto);
  bool SetTensorProto(const AbstractBasePtr &abstract, mind_ir::TensorProto *const tensor_proto);
  bool SetCSRTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetCOOTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetAttributeProto(const AnfNodePtr &node, mind_ir::NodeProto *const node_proto);
  bool SetAbstractToNodeProto(const CNodePtr &node, mind_ir::NodeProto *const node_proto);
  bool SetAbstractToNodeProto(const abstract::AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetTypeToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetScalarToAttributeProto_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProtoForInt_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProtoForInt_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetTypeToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetTensorToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetSequenceToAttributeProto(const ValueSequencePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetSeqElemToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);

  mind_ir::TensorProto_DataType GetMindirDataType(TypeId type_id) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsIntType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsFloatType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsUIntType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsComplexType(int bits) const;
  std::string GetNodeName(const AnfNodePtr &node) const;
  std::string GetUniqueNodeName(const AnfNodePtr &node);
  std::string GetOpTypeName(const AnfNodePtr &node);
  size_t GetUniqueID() { return ++unique_id_; }

 private:
  bool SetAbstractFuncToAttributeProto(const abstract::AbstractBasePtr &abstract,
                                       mind_ir::AttributeProto *const attr_proto);
  std::string GetPrimitiveUniqueName(const PrimitivePtr &primitive_ptr);
  bool BuildPrimitives();

  ModelProtoPtr model_;
  mind_ir::NodeProto *last_node_{nullptr};
  std::list<FuncGraphPtr> todo_;
  std::map<AnfNodePtr, std::string> node_name_map_;
  std::map<PrimitivePtr, std::string> primitive_name_map_;
  std::set<std::string> nodeName_;
  size_t unique_id_{0};
  bool top_graph{true};
};

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
    auto prim_proto = model_->add_primitives();
    auto prim = it->first;
    prim_proto->set_name(it->second);
    prim_proto->set_op_type(prim->name());

    auto real_prim = GetValueWithoutDoSignature(prim)->cast<PrimitivePtr>();
    if (real_prim != nullptr) {
      prim = real_prim;
    }

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
      CheckAndConvertUtils::ConvertAttrValueInExport(prim->name(), attr.first, &attr_value);
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

ModelProtoPtr IrExporter::GetDumpProto(const FuncGraphPtr &func_graph, const FuncGraphPtr &param_layout_fg) {
  if ((builder_ == nullptr) || (func_graph == nullptr)) {
    MS_LOG(EXCEPTION) << "Input params is null.";
  }

  // Export model info
  builder_->BuildModelInfo();

  // Export model and return string
  if (!builder_->BuildModel(func_graph)) {
    return nullptr;
  }

#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
  // Export layout information
  if (param_layout_fg) {
    builder_->BuildLayout(param_layout_fg);
  }
#endif
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

#ifndef MINDIR_EXPORT_TENSOR_LAYOUT_CLIP
void IrExportBuilder::BuildLayout(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  mind_ir::ParallelProto *parallel_proto = model_->mutable_parallel();
  for (auto para : graph_params) {
    std::string name = std::static_pointer_cast<Parameter>(para)->name();
    auto tensor_layout = para->user_data<parallel::TensorLayout>();
    if (tensor_layout == nullptr) {
      MS_LOG(INFO) << "GetParameterLayout nullptr name = " << name;
    } else {
      mind_ir::LayoutProto *layoutProto = parallel_proto->add_layout();

      // Get all the information for layput
      auto device_arrangement = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();
      auto slice_shape = tensor_layout->slice_shape().array();
      int64_t field_size = tensor_layout->get_field_size();
      bool uniform_split = tensor_layout->uniform_split();
      std::string opt_shard_group = tensor_layout->opt_shard_group();

      // Save all information to Layout Proto
      layoutProto->set_name(name);
      for (auto device_arrangement_element : device_arrangement) {
        layoutProto->add_device_arrangement_int(device_arrangement_element);
      }
      for (auto tensor_map_element : tensor_map) {
        layoutProto->add_tensor_map_int(tensor_map_element);
      }
      for (auto slice_shape_element : slice_shape) {
        layoutProto->add_slice_shape_int(slice_shape_element);
      }
      layoutProto->set_field_size(field_size);
      layoutProto->set_uniform_split(uniform_split);
      layoutProto->set_opt_shard_group(opt_shard_group);
    }
  }
}
#endif

bool IrExportBuilder::BuildModel(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  mind_ir::GraphProto *graph_proto = model_->mutable_graph();
  graph_proto->set_name(func_graph->ToString());
  graph_proto->set_bprop_hash(func_graph->bprop_hash());
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
  return true;
}

bool IrExportBuilder::BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
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

bool IrExportBuilder::BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_proto);
  for (auto &item : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(item);
    auto param = item->cast<ParameterPtr>();
    if (param == nullptr) {
      MS_LOG(ERROR) << "Parameter: '" << item->ToString() << "' could not cast to parameter.";
      return false;
    }
    std::string param_name = GetUniqueNodeName(param);
    if (top_graph && param->has_default()) {
      MS_LOG(DEBUG) << "Parameter: '" << item->DebugString();
      mind_ir::TensorProto *parameter_proto = graph_proto->add_parameter();
      parameter_proto->set_name(param_name);
      if (!SetParamToTensorProto(param, parameter_proto)) {
        MS_LOG(ERROR) << "Set parameter " << param->DebugString() << " to TensorProto failed.";
        return false;
      }
    } else {
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

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsComplexType(int bits) const {
  auto iter = g_data_bits_complex_map.find(bits);
  if (iter == g_data_bits_complex_map.end()) {
    MS_LOG(ERROR) << "Convert bits complex error, unsupported bits:" << bits << "!";
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
  } else if (type->isa<Tuple>()) {
    mind_ir::AttributeProto *attribute = value_proto->mutable_attr_info();
    if (!SetAbstractToNodeProto(node->abstract(), attribute)) {
      MS_LOG(ERROR) << "Set shape to Proto for " << node->DebugString() << " failed.";
      return false;
    }
    attribute->set_name("shape");
  } else {
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
  if (tensor_shape->IsDynamic()) {
    auto min_shape = tensor_shape->min_shape();
    auto max_shape = tensor_shape->max_shape();
    for (auto item : min_shape) {
      tensor_proto->add_min_dims(item);
    }
    for (auto item : max_shape) {
      tensor_proto->add_max_dims(item);
    }
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
  return SetTensorProto(param->abstract(), tensor_proto);
}

bool IrExportBuilder::BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Node: '" << node->ToString() << "' is not cnode";
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      if (!BuildOutput(cnode, graph_proto)) {
        MS_LOG(ERROR) << "Build output for graph " << func_graph->ToString() << " failed.";
        return false;
      }
    } else {
      if (!BuildCNode(cnode, graph_proto)) {
        MS_LOG(ERROR) << "Build proto for cnode " << cnode->DebugString() << " failed.";
        return false;
      }
    }
  }
  return true;
}

bool IrExportBuilder::BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(node);
  const int OutputSize = 2;
  if (node->size() != OutputSize) {
    MS_LOG(ERROR) << "Number of inputs of return node is not equal to 2.";
    return false;
  }
  AnfNodePtr arg = node->input(1);
  std::string node_name = BuildInputNode(arg, graph_proto);
  if (node_name.empty()) {
    MS_LOG(ERROR) << "Build input node failed for arg " << arg->DebugString();
    return false;
  }
  mind_ir::ValueInfoProto *output_proto = graph_proto->add_output();
  output_proto->set_name(node_name);
  return SetValueInfoProto(arg, output_proto);
}

std::string IrExportBuilder::GetOpTypeName(const AnfNodePtr &node) {
  // May be ValueNode/CNode/Parameter
  std::string type_name = "";
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(prim);
    type_name = "REF::" + GetPrimitiveUniqueName(prim);
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
  } else {
    MS_LOG(ERROR) << "Need to support op type: " << node->type_name();
    return "";
  }
  MS_LOG(DEBUG) << "ExportType: " << type_name;
  return type_name;
}

bool IrExportBuilder::SetAbstractToNodeProto(const AbstractBasePtr &abs, mind_ir::AttributeProto *const attr_proto) {
  auto type = abs->BuildType();
  auto shape = abs->BuildShape();
  if (type->isa<Tuple>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TUPLE);
    auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
    for (size_t i = 0; i < tuple_abs->size(); i++) {
      mind_ir::AttributeProto *attr_values = attr_proto->add_values();
      if (!SetAbstractToNodeProto((*tuple_abs)[i], attr_values)) {
        return false;
      }
    }
  } else if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    return SetTensorProto(abs, tensor_proto);
  } else if (type->isa<Number>()) {
    if (type->isa<Bool>()) {
      attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    } else {
      attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
      mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
      auto data_type = GetMindirDataType(type->type_id());
      tensor_proto->set_data_type(data_type);
      tensor_proto->add_dims(1);
    }
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
    return true;
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
    if (node_name.empty()) {
      MS_LOG(ERROR) << "Build input node for " << input->DebugString() << " failed.";
      return false;
    }
    input_names.push_back(node_name);
  }

  // Build cnode
  mind_ir::NodeProto *node_proto = graph_proto->add_node();
  std::string output_name = GetUniqueNodeName(node);
  if (nodeName_.count(output_name) > 0) {
    MS_LOG(EXCEPTION) << "There is a duplicate name: " << output_name;
  }
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
  // Maybe Tensor or Function or nullptr
  if (!SetAbstractToNodeProto(node, node_proto)) {
    return false;
  }

  (void)std::for_each(input_names.begin(), input_names.end(),
                      [&node_proto](const string &name) { node_proto->add_input(name); });
  return true;
}

std::string IrExportBuilder::BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  // Return the NodeName that the node has been processed.
  auto iter = node_name_map_.find(node);
  if (iter != node_name_map_.end()) {
    return iter->second;
  }

  std::string node_name = GetUniqueNodeName(node);
  // FuncGraph will be added to functions and the input name is the function name.
  if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    todo_.push_back(fg);
    return fg->ToString();
  }
  if (node->isa<ValueNode>()) {
    (void)nodeName_.insert(node_name);
    // When node input is a ValueNode, need to create a Constant Node
    mind_ir::NodeProto *node_proto = graph_proto->add_node();
    node_proto->set_name(node_name);
    node_proto->add_output(node_name);
    if (!SetAttributeProto(node, node_proto)) {
      return "";
    }
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
  } else {
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
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
  return true;
}

bool IrExportBuilder::SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
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
  } else if (value->isa<tensor::Tensor>()) {
    return SetTensorToAttributeProto(value, attr_proto);
  } else if (value->isa<None>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_NONE);
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

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph) {
  auto builder = std::make_shared<IrExportBuilder>();
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

bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                     const FuncGraphPtr &param_layout_fg) {
  auto exporter = std::make_shared<IrExporter>(std::make_shared<IrExportBuilder>());
  auto proto = exporter->GetDumpProto(func_graph, param_layout_fg);
  if (proto == nullptr) {
    MS_LOG(ERROR) << "Get binary proto for graph " << func_graph->ToString() << " failed.";
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
}  // namespace mindspore
