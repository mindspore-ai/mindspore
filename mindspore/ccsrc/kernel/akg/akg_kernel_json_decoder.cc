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
#include "kernel/akg/akg_kernel_json_decoder.h"

#include <memory>
#include "kernel/akg/akg_kernel_json_generator.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/adapter/fake_abstract_shape.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/operator/ops.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "ir/meta_tensor.h"
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/python_adapter.h"
#include "include/backend/kernel_info.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace kernel {
namespace {
using graphkernel::kJsonKeyAttr;
using graphkernel::kJsonKeyDataType;
using graphkernel::kJsonKeyFormat;
using graphkernel::kJsonKeyInputDesc;
using graphkernel::kJsonKeyName;
using graphkernel::kJsonKeyOpDesc;
using graphkernel::kJsonKeyOutputDesc;
using graphkernel::kJsonKeyProcess;
using graphkernel::kJsonKeyShape;
using graphkernel::kJsonKeyTensorName;
using graphkernel::kJsonKeyValue;

constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";

class CNodeDecoder {
 public:
  explicit CNodeDecoder(std::map<std::string, AnfNodePtr> *nodes_map) : nodes_map_(*nodes_map) {}
  ~CNodeDecoder() = default;
  CNodePtr DecodeCNode(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph, kernel::Processor processor) {
    op_name_ = cnode_json[kJsonKeyName];
    MS_LOG(DEBUG) << "Start decode cnode " << op_name_ << ", json: " << cnode_json;
    // decode attrs.
    if (!DecodeAttrs(cnode_json)) {
      MS_LOG(ERROR) << "Decode attrs failed. op: " << op_name_ << ", json: " << cnode_json;
      return nullptr;
    }
    if (!DecodeInputDesc(cnode_json, func_graph) || cnode_ == nullptr) {
      MS_LOG(ERROR) << "Decode inputs failed. op: " << op_name_ << ", json: " << cnode_json;
      return nullptr;
    }
    if (!DecodeOutputDesc(cnode_json, func_graph)) {
      MS_LOG(ERROR) << "Decode outputs failed. op: " << op_name_ << ", json: " << cnode_json;
      return nullptr;
    }
    CreateKernelInfo(processor);
    CreateAbstract();
    InitIOName(cnode_);
    return cnode_;
  }

 private:
  ValuePtr ParseValue(const nlohmann::json &attr_json, const std::string &type) const {
    if (type == "str") {
      std::string value = attr_json[kJsonKeyValue];
      if (op_name_ == "Cast" && attr_json[kJsonKeyName] == kAttrDstType) {
        return StringToType(value);
      }
      return MakeValue(value);
    } else if (type == "int") {
      int64_t value = attr_json[kJsonKeyValue];
      return MakeValue(value);
    } else if (type == "bool") {
      bool value = attr_json[kJsonKeyValue];
      return MakeValue(value);
    } else if (type == "float") {
      float value = attr_json[kJsonKeyValue];
      return MakeValue(value);
    } else if (type == "listInt") {
      std::vector<int64_t> value = attr_json[kJsonKeyValue];
      return MakeValue(value);
    } else if (type == "listStr") {
      std::vector<std::string> value = attr_json[kJsonKeyValue];
      return MakeValue(value);
    } else {
      MS_LOG(ERROR) << "Fail to parse attr " << attr_json[kJsonKeyName] << " in json, because its type: " << type
                    << " is not in supported list: [str, int, bool, float, listInt, listStr]. json is: " << attr_json;
      return nullptr;
    }
  }

  void InitIOName(const CNodePtr &cnode) {
    auto primitive = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(primitive);
    const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
    auto const iter = op_primc_fns.find(primitive->name());
    if (iter == op_primc_fns.end()) {
      return;
    }
    auto prim = iter->second();
    if (prim != nullptr) {
      (void)primitive->AddAttr(kAttrInputNames, prim->GetAttr(kAttrInputNames));
      (void)primitive->AddAttr(kAttrOutputNames, prim->GetAttr(kAttrOutputNames));
    }
  }

  bool DecodeAttrs(const nlohmann::json &attrs_json) {
    MS_LOG(DEBUG) << "start decode attrs, " << attrs_json;
    // attrs maybe empty
    if (attrs_json.find(kJsonKeyAttr) == attrs_json.end() || attrs_json[kJsonKeyAttr].is_null()) {
      return true;
    }

    std::vector<nlohmann::json> attr_descs = attrs_json[kJsonKeyAttr];
    for (const auto &attr_desc : attr_descs) {
      std::string name = attr_desc[kJsonKeyName];
      std::string type = attr_desc[kJsonKeyDataType];
      auto value = ParseValue(attr_desc, type);
      if (value == nullptr) {
        return false;
      }
      cnode_attrs_[name] = value;
    }
    return true;
  }

  bool DecodeInputDesc(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph) {
    auto primitive = CreatePrimitiveWithAttrs();
    MS_EXCEPTION_IF_NULL(primitive);

    // collect inputs.
    auto primitive_v = NewValueNode(primitive);
    func_graph->AddValueNode(primitive_v);
    std::vector<AnfNodePtr> inputs{primitive_v};
    std::vector<nlohmann::json> input_descs = cnode_json[kJsonKeyInputDesc];
    for (size_t i = 0; i < input_descs.size(); ++i) {
      nlohmann::json input_desc = input_descs[i][0];
      std::string name = input_desc[kJsonKeyTensorName];
      if (input_desc.find(kJsonKeyValue) != input_desc.end()) {
        auto value = DecodeValueNode(input_desc, func_graph);
        if (value == nullptr) {
          return false;
        }
        inputs.push_back(value);
      } else if (nodes_map_.count(name) == 0) {
        MS_LOG(ERROR) << "Input: " << name << " of: " << op_name_ << " not found.";
        return false;
      } else {
        inputs.push_back(nodes_map_[name]);
      }
      input_formats_.push_back(input_desc[kJsonKeyFormat]);
      input_types_.push_back(StringToTypeId(input_desc[kJsonKeyDataType]));
      input_shapes_.push_back(input_desc[kJsonKeyShape]);
    }
    // new cnode.
    cnode_ = func_graph->NewCNode(inputs);
    func_graph->AddNode(cnode_);
    return true;
  }

  bool DecodeOutputDesc(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph) {
    std::vector<nlohmann::json> output_descs = cnode_json[kJsonKeyOutputDesc];
    if (output_descs.empty()) {
      MS_LOG(ERROR) << "No outputs found in json: " << cnode_json << ", " << kJsonKeyOutputDesc << " is empty.";
      return false;
    } else if (output_descs.size() == 1) {
      // single output.
      nlohmann::json output_desc = output_descs[0];
      output_formats_.push_back(output_desc[kJsonKeyFormat]);
      output_types_.push_back(StringToTypeId(output_desc[kJsonKeyDataType]));
      output_shapes_.push_back(output_desc[kJsonKeyShape]);
      nodes_map_[output_desc[kJsonKeyTensorName]] = cnode_;
    } else {
      // multi outputs.
      for (size_t j = 0; j < output_descs.size(); ++j) {
        nlohmann::json output_desc = output_descs[j];
        output_formats_.push_back(output_desc[kJsonKeyFormat]);
        output_types_.push_back(StringToTypeId(output_desc[kJsonKeyDataType]));
        output_shapes_.push_back(output_desc[kJsonKeyShape]);
        auto get_item =
          func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode_, NewValueNode(SizeToLong(j))});
        func_graph->AddNode(get_item);
        nodes_map_[output_desc[kJsonKeyTensorName]] = get_item;
      }
    }
    return true;
  }

  void CreateKernelInfo(kernel::Processor processor) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    std::vector<size_t> feature_map_input_indexs;
    // if the node only has the primitive(such as getNext) or the node's input has a feature map input
    // then the node's output is a feature map output
    const auto &inputs = cnode_->inputs();
    for (size_t index = 1; index < inputs.size(); ++index) {
      auto node = common::AnfAlgo::VisitKernel(inputs[index], 0);
      if ((node.first)->isa<Parameter>()) {
        auto parameter = (node.first)->cast<ParameterPtr>();
        bool is_weight = common::AnfAlgo::IsParameterWeight(parameter);
        kernel_info->set_feature_map_flag(!is_weight);
        if (!is_weight) {
          feature_map_input_indexs.push_back(index - 1);
        }
      }
      if (AnfAlgo::IsFeatureMapOutput(node.first)) {
        feature_map_input_indexs.push_back(index - 1);
      }
    }
    if (common::AnfAlgo::GetCNodeName(cnode_) == prim::kPrimCast->name()) {
      common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(false), cnode_);
    }
    if (inputs.size() == 1 || !feature_map_input_indexs.empty()) {
      kernel_info->set_feature_map_flag(true);
    }
    if (AnfUtils::IsRealCNodeKernel(cnode_)) {
      common::AnfAlgo::SetNodeAttr(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), cnode_);
      common::AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), cnode_);
    }
    cnode_->set_kernel_info(kernel_info);
    // create kernel_build_info.
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetInputsFormat(input_formats_);
    builder->SetInputsDeviceType(input_types_);
    builder->SetOutputsFormat(output_formats_);
    builder->SetOutputsDeviceType(output_types_);
    builder->SetProcessor(processor);
    builder->SetKernelType(KernelType::AKG_KERNEL);
    builder->SetFusionType(kernel::kPatternOpaque);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode_.get());
  }

  void CreateAbstract() const {
    auto shape = graphkernel::GetFakeAbstractShape(output_shapes_[0], output_formats_[0]);
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(output_types_[0]), shape);
    cnode_->set_abstract(abstract);
  }

  PrimitivePtr CreatePrimitiveWithAttrs() const {
    auto primitive = std::make_shared<Primitive>(op_name_);
    for (const auto &attr : cnode_attrs_) {
      (void)primitive->AddAttr(attr.first, attr.second);
    }
    return primitive;
  }

  tensor::TensorPtr DecodeScalar(const nlohmann::json &scalar_json) const {
    auto type_id = StringToTypeId(scalar_json[kJsonKeyDataType]);
    if (type_id == TypeId::kNumberTypeFloat16) {
      return std::make_shared<tensor::Tensor>(static_cast<float>(scalar_json[kJsonKeyValue]), kFloat16);
    } else if (type_id == TypeId::kNumberTypeFloat32) {
      return std::make_shared<tensor::Tensor>(static_cast<float>(scalar_json[kJsonKeyValue]), kFloat32);
    } else if (type_id == TypeId::kNumberTypeInt32) {
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(scalar_json[kJsonKeyValue]), kInt32);
    }
    MS_LOG(ERROR) << "Fail to parse scalar " << scalar_json[kJsonKeyValue]
                  << " in json, because its type: " << scalar_json[kJsonKeyDataType]
                  << " is not in supported list: [float16, float32, int32]. json is: " << scalar_json;
    return nullptr;
  }

  ValueNodePtr DecodeValueNode(const nlohmann::json &value_json, const FuncGraphPtr &func_graph) const {
    MS_LOG(DEBUG) << "start decode value node, " << value_json;
    auto tensor = DecodeScalar(value_json);
    if (tensor == nullptr) {
      return nullptr;
    }
    auto value_node = std::make_shared<ValueNode>(tensor);
    value_node->set_abstract(tensor->ToAbstract());
    // create kernel_info fo new value node.
    auto kernel_info = std::make_shared<device::KernelInfo>();
    value_node->set_kernel_info(kernel_info);
    // create kernel_build_info for new value node.
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    // layout info.
    builder->SetOutputsFormat(std::vector<std::string>{value_json[kJsonKeyFormat]});
    builder->SetOutputsDeviceType(std::vector<TypeId>{StringToTypeId(value_json[kJsonKeyDataType])});
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), value_node.get());
    func_graph->AddValueNode(value_node);
    MS_LOG(DEBUG) << "decode value node success, " << value_node->DebugString(2);
    return value_node;
  }

  std::map<std::string, AnfNodePtr> &nodes_map_;
  std::map<std::string, ValuePtr> cnode_attrs_;
  std::vector<std::string> input_formats_;
  std::vector<std::string> output_formats_;
  std::vector<TypeId> input_types_;
  std::vector<TypeId> output_types_;
  std::vector<ShapeVector> input_shapes_;
  std::vector<ShapeVector> output_shapes_;
  CNodePtr cnode_{nullptr};
  std::string op_name_;
};
}  // namespace

ParameterPtr AkgKernelJsonDecoder::DecodeParameter(const nlohmann::json &parameter_json,
                                                   const FuncGraphPtr &func_graph) {
  MS_LOG(DEBUG) << "start decode parameter, " << parameter_json;
  ParameterPtr new_parameter = func_graph->add_parameter();
  std::string name = parameter_json[kJsonKeyTensorName];
  new_parameter->set_name(name);
  std::string format = parameter_json[kJsonKeyFormat];
  TypeId dtype = StringToTypeId(parameter_json[kJsonKeyDataType]);
  ShapeVector shape = graphkernel::GetFakeAbstractShape(parameter_json[kJsonKeyShape], format);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(dtype), shape);
  new_parameter->set_abstract(abstract);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_parameter->set_kernel_info(kernel_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetOutputsFormat(std::vector<std::string>{format});
  builder->SetOutputsDeviceType(std::vector<TypeId>{dtype});
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), new_parameter.get());
  nodes_map_[name] = new_parameter;
  return new_parameter;
}

CNodePtr AkgKernelJsonDecoder::DecodeCNode(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph,
                                           const std::string &processor) {
  CNodeDecoder decoder(&nodes_map_);
  Processor p = kernel::GetProcessor(processor);
  return decoder.DecodeCNode(cnode_json, func_graph, p);
}

AnfNodePtr AkgKernelJsonDecoder::DecodeOutput(const std::vector<nlohmann::json> &output_descs,
                                              const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> outputs{NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList output_abstract_list;
  for (const auto &output_desc : output_descs) {
    std::string name = output_desc[kJsonKeyTensorName];
    if (nodes_map_.count(name) == 0) {
      MS_LOG(ERROR) << "Output: " << name << " of graph not found.";
      return nullptr;
    }
    outputs.push_back(nodes_map_[name]);
    output_abstract_list.push_back(outputs.back()->abstract());
  }
  if (outputs.size() == 2) {
    func_graph->set_output(outputs[1]);
  } else {
    auto output = func_graph->NewCNode(outputs);
    output->set_abstract(std::make_shared<abstract::AbstractTuple>(output_abstract_list));
    func_graph->AddNode(output);
    func_graph->set_output(output);
  }
  return func_graph->output();
}

FuncGraphPtr AkgKernelJsonDecoder::DecodeFusedNodes(const nlohmann::json &kernel_json) {
  MS_LOG(DEBUG) << "Start decoding json: " << kernel_json;
  nodes_map_.clear();
  auto graph = std::make_shared<FuncGraph>();

  // decode parameters.
  std::vector<nlohmann::json> input_descs = kernel_json[kJsonKeyInputDesc];
  if (input_descs.empty()) {
    MS_LOG(ERROR) << "Error decoding parameter: no inputs for graph. Because " << kJsonKeyInputDesc
                  << " is empty in json: " << kernel_json;
    return nullptr;
  }
  for (size_t i = 0; i < input_descs.size(); ++i) {
    std::vector<nlohmann::json> input_desc = input_descs[i];
    auto parameter = DecodeParameter(input_desc[0], graph);
    MS_EXCEPTION_IF_NULL(parameter);
  }
  MS_LOG(DEBUG) << "Decode parameters successfully.";

  // decode cnodes in graph.
  std::vector<nlohmann::json> op_node_descs = kernel_json[kJsonKeyOpDesc];
  if (op_node_descs.empty()) {
    MS_LOG(ERROR) << "Error decoding cnodes: no cnodes for graph. Because " << kJsonKeyOpDesc
                  << " is empty in json: " << kernel_json;
    return nullptr;
  }
  for (const auto &op_desc : op_node_descs) {
    auto op_node = DecodeCNode(op_desc, graph, kernel_json[kJsonKeyProcess]);
    if (op_node == nullptr) {
      return nullptr;
    }
  }
  MS_LOG(DEBUG) << "Decode cnodes successfully.";

  // decode outputs of graph.
  std::vector<nlohmann::json> output_descs = kernel_json[kJsonKeyOutputDesc];
  if (output_descs.empty()) {
    MS_LOG(ERROR) << "Error decoding outputs: no outputs for graph. Because " << kJsonKeyOutputDesc
                  << " is empty in json: " << kernel_json;
    return nullptr;
  }
  auto output = DecodeOutput(output_descs, graph);
  if (output == nullptr) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "Decode json successfully, json: " << kernel_json;
  return graph;
}

FuncGraphPtr AkgKernelJsonDecoder::DecodeFusedNodes(const std::string &kernel_json_str) {
  auto kernel_json = nlohmann::json::parse(kernel_json_str);
  return DecodeFusedNodes(kernel_json);
}
}  // namespace kernel
}  // namespace mindspore
