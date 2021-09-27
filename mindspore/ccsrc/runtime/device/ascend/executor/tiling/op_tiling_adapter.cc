/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/executor/tiling/op_tiling_adapter.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_build.h"
#include "backend/kernel_compiler/tbe/tbe_dynaminc_shape_util.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/ascend/ge_types_convert.h"
#include "utils/utils.h"
#include "external/graph/tensor.h"
#include "external/register/op_tiling_registry.h"
#include "graph/utils/graph_utils.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"

namespace mindspore {
namespace device {
namespace tiling {
constexpr auto COMPILE_INFO_KEY = "compile_info_key";
constexpr auto COMPILE_INFO_JSON = "compile_info_json";
constexpr auto ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
constexpr auto ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
constexpr auto ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";
constexpr auto CONSTANTOP = "Constant";
constexpr auto ATTR_NAME_WEIGHTS = "value";
constexpr auto PARAM_DYNAMIC = "dynamic";

std::string OpTilingCalculateAdapter::GetRealOpType(const std::string &op_type) {
  static const std::map<std::string, std::string> kOpTypeMap = {
    {"SparseApplyFtrl", "SparseApplyFtrlD"},
    {"SparseApplyProximalAdagrad", "SparseApplyProximalAdagradD"},
    {"SparseGatherV2", "Gather"},
    {"Pad", "PadD"},
    {"Concat", "ConcatD"},
    {"Softmax", "SoftmaxV2"},
    {"DropoutDoMask", "DropOutDoMask"},
  };
  auto iter = kOpTypeMap.find(op_type);
  if (iter == kOpTypeMap.end()) {
    return op_type;
  }
  return iter->second;
}

std::string OpTilingCalculateAdapter::GetOutputName(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_names_.size() < index) {
    return "unknown_name";
  }
  return output_names_[index];
}

std::string OpTilingCalculateAdapter::GetInputName(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (input_names_.size() < index) {
    return "unknown_name";
  }
  return input_names_[index];
}

void OpTilingCalculateAdapter::ConvertInputShapeAndType(const CNodePtr &node, ge::OpDescPtr *op_desc) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(*op_desc);
  auto input_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; i++) {
    // ms info
    auto real_index = AnfAlgo::GetRealInputIndex(node, i);
    auto input_node_with_idx = AnfAlgo::GetPrevNodeOutput(node, real_index);
    auto input_node = input_node_with_idx.first;
    auto input_index = input_node_with_idx.second;
    auto ms_ori_shape = AnfAlgo::GetOutputInferShape(input_node, input_index);
    auto ms_shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    auto ms_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    auto ms_dtype = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);

    // ge info
    std::vector<int64_t> ge_shape;
    std::vector<int64_t> ge_ori_shape;
    ge::DataType ge_dtype = ascend::GeTypesConvert::TransTypeIdToGeDataType(ms_dtype);
    ge::Format ge_format = ascend::GeTypesConvert::GetGeFormat(ms_format, ms_shape.size());
    std::transform(ms_shape.begin(), ms_shape.end(), std::back_inserter(ge_shape),
                   [](size_t s) { return static_cast<int64_t>(s); });
    std::transform(ms_ori_shape.begin(), ms_ori_shape.end(), std::back_inserter(ge_ori_shape),
                   [](size_t s) { return static_cast<int64_t>(s); });

    auto input_name = GetInputName(node, real_index);
    ge::GeTensorDesc ge_tensor_desc;
    ge_tensor_desc.SetFormat(ge_format);
    ge_tensor_desc.SetDataType(ge_dtype);
    ge_tensor_desc.SetShape(ge::GeShape(ge_shape));
    ge_tensor_desc.SetOriginShape(ge::GeShape(ge_ori_shape));
    ge_tensor_desc.SetName(input_name);
    (void)(*op_desc)->AddInputDesc(input_name, ge_tensor_desc);
  }
}

void OpTilingCalculateAdapter::ConvertOutputShapeAndType(const CNodePtr &node, ge::OpDescPtr *op_desc) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(*op_desc);
  auto output_size = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_size; i++) {
    auto ms_shape = AnfAlgo::GetOutputDeviceShape(node, i);
    auto ms_ori_shape = AnfAlgo::GetOutputInferShape(node, i);
    auto ms_format = AnfAlgo::GetOutputFormat(node, i);
    auto ms_dtype = AnfAlgo::GetOutputDeviceDataType(node, i);

    std::vector<int64_t> ge_shape;
    std::vector<int64_t> ge_ori_shape;
    ge::DataType ge_dtype = ascend::GeTypesConvert::TransTypeIdToGeDataType(ms_dtype);
    ge::Format ge_format = ascend::GeTypesConvert::GetGeFormat(ms_format, ms_shape.size());
    std::transform(ms_shape.begin(), ms_shape.end(), std::back_inserter(ge_shape),
                   [](size_t s) { return static_cast<int64_t>(s); });
    std::transform(ms_ori_shape.begin(), ms_ori_shape.end(), std::back_inserter(ge_ori_shape),
                   [](size_t s) { return static_cast<int64_t>(s); });
    auto output_name = GetOutputName(node, i);
    ge::GeTensorDesc ge_tensor_desc;
    ge_tensor_desc.SetFormat(ge_format);
    ge_tensor_desc.SetDataType(ge_dtype);
    ge_tensor_desc.SetShape(ge::GeShape(ge_shape));
    ge_tensor_desc.SetOriginShape(ge::GeShape(ge_ori_shape));
    ge_tensor_desc.SetName(output_name);
    (void)(*op_desc)->AddOutputDesc(output_name, ge_tensor_desc);
  }
}

void OpTilingCalculateAdapter::ConvertCompileInfo(const CNodePtr &node, ge::OpDescPtr *op_desc) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(*op_desc);
  MS_LOG(INFO) << "For op " << op_name_ << ", get compile_info: " << op_compile_info_;
  std::string compile_info_key = std::to_string(std::hash<std::string>()(op_compile_info_));
  (void)ge::AttrUtils::SetStr(*(*op_desc), COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(*(*op_desc), COMPILE_INFO_JSON, op_compile_info_);
}

ge::NodePtr OpTilingCalculateAdapter::NewConstantOp(const CNodePtr &node, const std::string &name,
                                                    const tensor::TensorPtr &tensor_data, ge::ComputeGraphPtr *ge_graph,
                                                    size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(*ge_graph);
  MS_EXCEPTION_IF_NULL(tensor_data);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, CONSTANTOP);
  auto ms_format = AnfAlgo::GetPrevNodeOutputFormat(node, index);
  auto ms_ori_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, index);
  auto ms_dtype = AnfAlgo::GetPrevNodeOutputDeviceDataType(node, index);
  auto ms_shape = AnfAlgo::GetInputDeviceShape(node, index);

  std::vector<int64_t> ge_shape;
  std::vector<int64_t> ge_ori_shape;
  std::transform(ms_shape.begin(), ms_shape.end(), std::back_inserter(ge_shape),
                 [](size_t s) { return static_cast<int64_t>(s); });
  std::transform(ms_ori_shape.begin(), ms_ori_shape.end(), std::back_inserter(ge_ori_shape),
                 [](size_t s) { return static_cast<int64_t>(s); });
  ge::DataType ge_dtype = ascend::GeTypesConvert::TransTypeIdToGeDataType(ms_dtype);
  ge::Format ge_format = ascend::GeTypesConvert::GetGeFormat(ms_format, ms_shape.size());
  ge::GeTensorDesc ge_tensor_desc;
  ge_tensor_desc.SetFormat(ge_format);
  ge_tensor_desc.SetDataType(ge_dtype);
  ge_tensor_desc.SetShape(ge::GeShape(ge_shape));
  ge_tensor_desc.SetOriginShape(ge::GeShape(ge_ori_shape));
  ge_tensor_desc.SetName(name);
  ge::GeTensorPtr ge_tensor = std::make_shared<ge::GeTensor>(
    ge_tensor_desc, static_cast<uint8_t *>(tensor_data->data_c()), IntToSize(tensor_data->Size()));
  (void)op_desc->AddOutputDesc(name, ge_tensor_desc);
  ge::NodePtr constant_op = (*ge_graph)->AddNode(op_desc);
  ge::OpDescUtils::SetWeights(constant_op, {ge_tensor});
  (void)ge::AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor);
  return constant_op;
}

std::vector<std::tuple<std::size_t, ge::NodePtr>> OpTilingCalculateAdapter::ConvertDepends(
  const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map, ge::OpDescPtr *op_desc,
  ge::ComputeGraphPtr *ge_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(*op_desc);
  auto depends_list_me = abstract::GetDependsFormMap(node);
  if (depends_list_me.empty()) {
    MS_LOG(INFO) << "The node " << op_name_ << " has no infer depend ";
    return {};
  }
  auto has_input_name_attr = AnfAlgo::HasNodeAttr("input_names", node);
  if (!has_input_name_attr) {
    MS_LOG(EXCEPTION) << "Node should has attr: input_names. " << node->fullname_with_scope();
  }
  auto input_names_attr = AnfAlgo ::GetNodeAttr<std::vector<std::string>>(node, "input_names");
  std::vector<std::string> op_infer_depends;
  std::vector<std::tuple<std::size_t, ge::NodePtr>> constant_ops;
  for (auto index : depends_list_me) {
    if (LongToSize(index) > input_names_attr.size()) {
      MS_LOG(EXCEPTION) << "Input index " << index << " should less input_names' size " << input_names_attr.size();
    }
    auto iter = depend_tensor_map.find(LongToSize(index));
    if (iter == depend_tensor_map.end()) {
      MS_LOG(EXCEPTION) << "Input index " << index << " should less than depend_tensor_map' size "
                        << input_names_attr.size();
    }
    auto depend_name = input_names_attr[index];
    auto const_tensor = iter->second;
    ge::NodePtr ge_constant_op = NewConstantOp(node, depend_name, const_tensor, ge_graph, index);
    auto original_index = AnfAlgo::GetOriginalInputIndex(node, index);
    constant_ops.emplace_back(std::tuple<std::size_t, ge::NodePtr>(original_index, ge_constant_op));
    op_infer_depends.emplace_back(depend_name);
  }
  (void)(*op_desc)->SetOpInferDepends(op_infer_depends);
  return constant_ops;
}

void OpTilingCalculateAdapter::AddEdge(const ge::NodePtr &ge_node,
                                       const std::vector<std::tuple<std::size_t, ge::NodePtr>> &constant_ops) {
  MS_EXCEPTION_IF_NULL(ge_node);
  MS_LOG(INFO) << "Add edge for op " << op_name_;
  for (const auto &item : constant_ops) {
    auto index = std::get<0>(item);
    auto constant_op = std::get<1>(item);
    (void)ge_node->AddLinkFrom(index, constant_op);
  }
}

void OpTilingCalculateAdapter::InitOpIoName(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Get the every input name of " << op_name_;
  auto op_info_ptr = mindspore::kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name_, node);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  auto inputs_ptr = op_info_ptr->inputs_ptr();
  size_t dynamic_input_index = 0;
  std::vector<int64_t> dynamic_inputs_list;
  if (primitive->GetAttr(kAttrDynInputSizes) != nullptr) {
    dynamic_inputs_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrDynInputSizes));
  }
  for (const auto &item : inputs_ptr) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->param_type() == PARAM_DYNAMIC) {
      if (dynamic_input_index > dynamic_inputs_list.size()) {
        MS_LOG(EXCEPTION) << "Dynamic input index should less than the dynamic input's size.";
      }
      auto real_inputs_num = dynamic_inputs_list[dynamic_input_index];
      for (auto k = 0; k < real_inputs_num; k++) {
        std::string input_name = item->name() + "_dynamic_" + std::to_string(k);
        input_names_.emplace_back(input_name);
      }
      dynamic_input_index++;
    } else {
      input_names_.emplace_back(item->name());
    }
  }

  // output io names
  auto outputs_ptr = op_info_ptr->outputs_ptr();
  for (const auto &out_item : outputs_ptr) {
    MS_EXCEPTION_IF_NULL(out_item);
    output_names_.emplace_back(out_item->name());
  }
}

ge::NodePtr OpTilingCalculateAdapter::AnfNodeToGeNodeAdapter(
  const CNodePtr &node, ge::ComputeGraphPtr *ge_graph, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
  const std::string &op_compile_info) {
  MS_EXCEPTION_IF_NULL(node);
  op_name_ = AnfAlgo::GetCNodeName(node);
  MS_LOG(INFO) << "Convert anf node :" << op_name_ << " to ge node.";
  op_compile_info_ = op_compile_info;
  auto op_type = GetRealOpType(op_name_);
  (void)InitOpIoName(node);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(op_name_, op_type);
  MS_EXCEPTION_IF_NULL(op_desc);
  ConvertInputShapeAndType(node, &op_desc);
  ConvertOutputShapeAndType(node, &op_desc);
  ConvertCompileInfo(node, &op_desc);
  std::vector<std::tuple<std::size_t, ge::NodePtr>> constant_ops =
    ConvertDepends(node, depend_tensor_map, &op_desc, ge_graph);
  MS_EXCEPTION_IF_NULL(*ge_graph);
  auto ge_node = (*ge_graph)->AddNode(op_desc);
  MS_EXCEPTION_IF_NULL(ge_node);
  AddEdge(ge_node, constant_ops);
  return ge_node;
}
}  // namespace tiling
}  // namespace device
}  // namespace mindspore
