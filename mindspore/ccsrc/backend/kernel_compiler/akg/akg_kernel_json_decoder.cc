/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/akg/akg_kernel_json_decoder.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"
#include "frontend/operator/ops.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "ir/meta_tensor.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "runtime/device/kernel_info.h"
#include "utils/convert_utils.h"
#include "utils/convert_utils_py.h"
#include "utils/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";

class AbstractShapeCreator {
 public:
  using AbstractShapeTransferFunc = std::function<ShapeVector(const ShapeVector &)>;
  /**
   * Get an abstract shape.
   * For a given device_shape and format, the available abstract_shape is not unique,
   * this interface only returns a legal abstract_shape without considering padding
   * so that the AnfAlgo's get device shape interface can get the right device_shape.
   */
  static ShapeVector GetFakeAbstractShape(const ShapeVector &device_shape, const std::string &format) {
    const std::map<std::string, AbstractShapeTransferFunc> fmap{
      {kOpFormat_NCHW, NchwAbstractShape},
      {kOpFormat_NHWC, NhwcAbstractShape},
      {kOpFormat_FRAC_NZ, FractalNzAbstractShape},
    };
    if (format == kOpFormat_ND || format == kOpFormat_DEFAULT) {
      return device_shape;
    }
    auto iter = fmap.find(format);
    if (iter == fmap.end()) {
      MS_LOG(WARNING) << "Unexpected format[" << format << "]";
      return device_shape;
    }
    return iter->second(device_shape);
  }

 private:
  static ShapeVector NchwAbstractShape(const ShapeVector &device_shape) { return device_shape; }
  static ShapeVector NhwcAbstractShape(const ShapeVector &device_shape) {
    if (device_shape.size() != 4) {
      MS_LOG(EXCEPTION) << "Shape size of NHWC should be 4, but got " << device_shape.size();
    }
    return {device_shape[0], device_shape[3], device_shape[1], device_shape[2]};
  }
  static ShapeVector FractalNzAbstractShape(const ShapeVector &device_shape) {
    if (device_shape.size() == 1 && (device_shape[0] == 1 || device_shape[0] % kCubeSize == 0)) {
      return device_shape;
    }
    if (device_shape.size() < 4) {
      MS_LOG(EXCEPTION) << "Shape size of FRACTAL_NZ should >= 4, but got " << device_shape.size();
    }
    ShapeVector shape;
    size_t dims = device_shape.size();
    size_t batch = dims - 4;
    for (size_t i = 0; i < batch; ++i) {
      shape.push_back(device_shape[i]);
    }
    int64_t m = device_shape[dims - 3] * device_shape[dims - 2];
    int64_t n = device_shape[dims - 4] * device_shape[dims - 1];
    shape.push_back(m);
    shape.push_back(n);

    return shape;
  }
};

class CNodeDecoder {
 public:
  explicit CNodeDecoder(std::map<std::string, AnfNodePtr> *nodes_map) : nodes_map_(*nodes_map) {}
  ~CNodeDecoder() = default;
  CNodePtr DecodeCNode(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph, kernel::Processor processor) {
    MS_LOG(DEBUG) << "start decode cnode, " << cnode_json;
    // decode attrs.
    if (!DecodeAttrs(cnode_json)) {
      MS_LOG(ERROR) << "Decode attrs failed.";
      return nullptr;
    }
    if (!DecodeInputDesc(cnode_json, func_graph) || cnode_ == nullptr) {
      MS_LOG(ERROR) << "Decode inputs failed.";
      return nullptr;
    }
    if (!DecodeOutputDesc(cnode_json, func_graph)) {
      MS_LOG(ERROR) << "Decode outputs failed.";
      return nullptr;
    }
    CreateKernelInfo(processor);
    CreateAbstract();
    return cnode_;
  }

 private:
  ValuePtr ParseValue(const nlohmann::json &attr_json, const std::string &type) {
    if (type == "str") {
      std::string value = attr_json[kJsonKeyValue];
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
      MS_LOG(ERROR) << "Unknown type of attr: " << type << ", json: \n" << attr_json;
      return nullptr;
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
    std::string op_name = cnode_json[kJsonKeyName];
    auto primitive = CreatePrimitiveWithAttrs(op_name);
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
        inputs.push_back(DecodeValueNode(input_desc, func_graph));
      } else if (nodes_map_.count(name) == 0) {
        MS_LOG(ERROR) << "Input: " << name << " of: " << op_name << " not found.";
        return false;
      } else {
        inputs.push_back(nodes_map_[name]);
      }
      input_formats_.push_back(input_desc[kJsonKeyFormat]);
      input_types_.push_back(DtypeToTypeId(input_desc[kJsonKeyDataType]));
      input_shapes_.push_back(input_desc[kJsonKeyShape]);
    }
    // new cnode.
    cnode_ = func_graph->NewCNode(inputs);
    func_graph->AddNode(cnode_);
    return true;
  }

  bool DecodeOutputDesc(const nlohmann::json &cnode_json, const FuncGraphPtr &func_graph) {
    std::vector<nlohmann::json> output_descs = cnode_json[kJsonKeyOutputDesc];
    AbstractBasePtr abstract(nullptr);
    if (output_descs.empty()) {
      MS_LOG(ERROR) << "No outputs found.";
      return false;
    } else if (output_descs.size() == 1) {
      // single output.
      nlohmann::json output_desc = output_descs[0];
      output_formats_.push_back(output_desc[kJsonKeyFormat]);
      output_types_.push_back(DtypeToTypeId(output_desc[kJsonKeyDataType]));
      output_shapes_.push_back(output_desc[kJsonKeyShape]);
      nodes_map_[output_desc[kJsonKeyTensorName]] = cnode_;
    } else {
      // multi outputs.
      for (size_t j = 0; j < output_descs.size(); ++j) {
        nlohmann::json output_desc = output_descs[j];
        output_formats_.push_back(output_desc[kJsonKeyFormat]);
        output_types_.push_back(DtypeToTypeId(output_desc[kJsonKeyDataType]));
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
      auto node = AnfAlgo::VisitKernel(inputs[index], 0);
      if ((node.first)->isa<Parameter>()) {
        auto parameter = (node.first)->cast<ParameterPtr>();
        bool is_weight = AnfAlgo::IsParameterWeight(parameter);
        kernel_info->set_feature_map_flag(!is_weight);
        if (!is_weight) {
          feature_map_input_indexs.push_back(index - 1);
        }
      }
      if (AnfAlgo::IsFeatureMapOutput(node.first)) {
        feature_map_input_indexs.push_back(index - 1);
      }
    }
    if (AnfAlgo::GetCNodeName(cnode_) == prim::kPrimCast->name()) {
      AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(false), cnode_);
    }
    if (inputs.size() == 1 || !feature_map_input_indexs.empty()) {
      kernel_info->set_feature_map_flag(true);
    }
    if (AnfAlgo::IsRealCNodeKernel(cnode_)) {
      AnfAlgo::SetNodeAttr(kIsFeatureMapOutput, MakeValue(kernel_info->is_feature_map()), cnode_);
      AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), cnode_);
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
    builder->SetFusionType(kernel::FusionType::OPAQUE);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode_.get());
  }

  void CreateAbstract() {
    auto shape = AbstractShapeCreator::GetFakeAbstractShape(output_shapes_[0], output_formats_[0]);
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(output_types_[0]), shape);
    cnode_->set_abstract(abstract);
  }

  PrimitivePtr CreatePrimitiveWithAttrs(const std::string &op_name) {
    auto primitive = std::make_shared<Primitive>(op_name);
    for (const auto &attr : cnode_attrs_) {
      primitive->AddAttr(attr.first, attr.second);
    }
    return primitive;
  }

  tensor::TensorPtr DecodeScalar(const nlohmann::json &scalar_json) {
    auto type_id = DtypeToTypeId(scalar_json[kJsonKeyDataType]);
    switch (type_id) {
      case kNumberTypeFloat16:
        return std::make_shared<tensor::Tensor>(static_cast<float>(scalar_json[kJsonKeyValue]), kFloat16);
      case kNumberTypeFloat32:
        return std::make_shared<tensor::Tensor>(static_cast<float>(scalar_json[kJsonKeyValue]), kFloat32);
      case kNumberTypeInt32:
        return std::make_shared<tensor::Tensor>(static_cast<int64_t>(scalar_json[kJsonKeyValue]), kInt32);
      default:
        MS_LOG(ERROR) << "Unknown type: " << scalar_json[kJsonKeyDataType];
        break;
    }
    return nullptr;
  }

  ValueNodePtr DecodeValueNode(const nlohmann::json &value_json, const FuncGraphPtr &func_graph) {
    MS_LOG(DEBUG) << "start decode value node, " << value_json;
    auto tensor = DecodeScalar(value_json);
    MS_EXCEPTION_IF_NULL(tensor);
    auto value_node = std::make_shared<ValueNode>(tensor);
    value_node->set_abstract(tensor->ToAbstract());
    // create kernel_info fo new value node.
    auto kernel_info = std::make_shared<device::KernelInfo>();
    value_node->set_kernel_info(kernel_info);
    // create kernel_build_info for new value node.
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    // layout info.
    builder->SetOutputsFormat(std::vector<std::string>{value_json[kJsonKeyFormat]});
    builder->SetOutputsDeviceType(std::vector<TypeId>{DtypeToTypeId(value_json[kJsonKeyDataType])});
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
};
}  // namespace

ParameterPtr AkgKernelJsonDecoder::DecodeParameter(const nlohmann::json &parameter_json,
                                                   const FuncGraphPtr &func_graph) {
  MS_LOG(DEBUG) << "start decode parameter, " << parameter_json;
  ParameterPtr new_parameter = func_graph->add_parameter();
  std::string name = parameter_json[kJsonKeyTensorName];
  new_parameter->set_name(name);
  std::string format = parameter_json[kJsonKeyFormat];
  TypeId dtype = DtypeToTypeId(parameter_json[kJsonKeyDataType]);
  ShapeVector shape = AbstractShapeCreator::GetFakeAbstractShape(parameter_json[kJsonKeyShape], format);
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
  MS_LOG(DEBUG) << "start decode, " << kernel_json;
  nodes_map_.clear();
  auto graph = std::make_shared<FuncGraph>();

  // decode parameters.
  std::vector<nlohmann::json> input_descs = kernel_json[kJsonKeyInputDesc];
  if (input_descs.empty()) {
    MS_LOG(ERROR) << "Error decode parameter, no inputs for graph.";
    return nullptr;
  }
  for (size_t i = 0; i < input_descs.size(); ++i) {
    std::vector<nlohmann::json> input_desc = input_descs[i];
    auto parameter = DecodeParameter(input_desc[0], graph);
    MS_EXCEPTION_IF_NULL(parameter);
  }
  MS_LOG(DEBUG) << "decode parameters success.";

  // decode cnodes in graph.
  std::vector<nlohmann::json> op_node_descs = kernel_json[kJsonKeyOpDesc];
  if (op_node_descs.empty()) {
    MS_LOG(ERROR) << "Error decode cnodes, no cnodes for graph.";
    return nullptr;
  }
  for (const auto &op_desc : op_node_descs) {
    auto op_node = DecodeCNode(op_desc, graph, kernel_json[kJsonKeyProcess]);
    MS_EXCEPTION_IF_NULL(op_node);
  }
  MS_LOG(DEBUG) << "decode cnodes success.";

  // decode outputs of graph.
  std::vector<nlohmann::json> output_descs = kernel_json[kJsonKeyOutputDesc];
  if (output_descs.empty()) {
    MS_LOG(ERROR) << "Error decode outputs, no outputs for graph.";
    return nullptr;
  }
  auto output = DecodeOutput(output_descs, graph);
  MS_EXCEPTION_IF_NULL(output);
  MS_LOG(DEBUG) << "decode success, " << kernel_json;
  return graph;
}

FuncGraphPtr AkgKernelJsonDecoder::DecodeFusedNodes(const std::string &kernel_json_str) {
  auto kernel_json = nlohmann::json::parse(kernel_json_str);
  return DecodeFusedNodes(kernel_json);
}

StitchInfo AkgKernelJsonDecoder::GetStitchInfo(const nlohmann::json &kernel_json) {
  StitchInfo info;
  if (kernel_json.find(kJsonKeyBufferStitch) != kernel_json.end()) {
    nlohmann::json buffer_stitch = kernel_json[kJsonKeyBufferStitch];
    if (buffer_stitch.find(kJsonKeyStitchOp) != buffer_stitch.end()) {
      std::vector<std::string> stitch_ops = buffer_stitch[kJsonKeyStitchOp];
      info.stitch_ops = stitch_ops;
    }
    if (buffer_stitch.find(kJsonKeyStitchAtomicOp) != buffer_stitch.end()) {
      std::vector<std::string> stitch_atomic_ops = buffer_stitch[kJsonKeyStitchAtomicOp];
      info.stitch_atomic_ops = stitch_atomic_ops;
    }
  }
  return info;
}

void AkgKernelJsonDecoder::SetStitchAttr(const nlohmann::json &op_desc, const StitchInfo &info, const CNodePtr &node) {
  std::vector<nlohmann::json> output_descs = op_desc[kJsonKeyOutputDesc];
  if (output_descs.empty() || output_descs[0].find(kJsonKeyTensorName) == output_descs[0].end()) return;
  std::string tensor_name = output_descs[0][kJsonKeyTensorName];
  if (std::find(info.stitch_ops.begin(), info.stitch_ops.end(), tensor_name) != info.stitch_ops.end()) {
    AnfAlgo::SetNodeAttr(kAttrStitch, MakeValue("common"), node);
  }
  if (std::find(info.stitch_atomic_ops.begin(), info.stitch_atomic_ops.end(), tensor_name) !=
      info.stitch_atomic_ops.end()) {
    AnfAlgo::SetNodeAttr(kAttrStitch, MakeValue("atomic"), node);
  }
}

bool AkgKernelJsonDecoder::DecodeSplitNodes(const nlohmann::json &kernel_json,
                                            const std::map<std::string, AnfNodePtr> &address_node_map,
                                            AnfNodePtrList *res_graphs) {
  MS_EXCEPTION_IF_NULL(res_graphs);
  MS_LOG(DEBUG) << "start decode, " << kernel_json;
  // decode cnodes in graph.
  std::vector<nlohmann::json> op_node_descs = kernel_json[kJsonKeyOpDesc];
  if (op_node_descs.empty()) {
    MS_LOG(ERROR) << "Error decode, no cnodes for graph." << kernel_json;
    return false;
  }
  StitchInfo info = GetStitchInfo(kernel_json);
  for (const auto &op_desc : op_node_descs) {
    if (op_desc.find(kJsonKeyPtrAddress) == op_desc.end() || op_desc[kJsonKeyPtrAddress].is_null()) {
      MS_LOG(ERROR) << "Decode failed, key: " << kJsonKeyPtrAddress << " not found in: " << op_desc;
      return false;
    }

    std::string ptr_address = op_desc[kJsonKeyPtrAddress];
    if (address_node_map.count(ptr_address) == 0) {
      MS_LOG(ERROR) << "Decode failed, ptr_address not found in map.";
      return false;
    }
    auto node = address_node_map.at(ptr_address)->cast<CNodePtr>();
    SetStitchAttr(op_desc, info, node);
    res_graphs->push_back(node);
  }
  MS_LOG(DEBUG) << "decode cnodes success, size: " << res_graphs->size();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
