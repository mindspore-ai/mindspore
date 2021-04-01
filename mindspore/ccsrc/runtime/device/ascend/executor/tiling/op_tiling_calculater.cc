/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/executor/tiling/op_tiling_calculater.h"
#include <dlfcn.h>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/ascend/ge_types_convert.h"
#include "utils/utils.h"
#include "external/graph/tensor.h"
#include "external/register/op_tiling_registry.h"

namespace mindspore {
namespace device {
namespace ascend {
ge::Tensor MakeTempGeTensor(TypeId type_id) {
  auto ge_type = GeTypesConvert::TransTypeIdToGeDataType(type_id);
  ge::TensorDesc tensor_desc;
  tensor_desc.SetDataType(ge_type);
  ge::Tensor ge_tensor;
  ge_tensor.SetTensorDesc(tensor_desc);
  return ge_tensor;
}

void FeedTeOpTensorInputArg(const NotNull<CNodePtr> &cnode,
                            NotNull<std::vector<optiling::TeOpTensorArg> *> tensor_arg_list) {
  MS_LOG(INFO) << "FeedTeOpTensorInputArg start, node:" << cnode->fullname_with_scope();
  auto input_size = AnfAlgo::GetInputTensorNum(cnode.get());

  // Skip Dynamic Shape Depend Input

  for (size_t i = 0; i < input_size; ++i) {
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(cnode.get(), i);
    auto input_node = input_node_with_index.first;
    auto input_index = input_node_with_index.second;
    auto output_shape = AnfAlgo::GetOutputDeviceShape(input_node, input_index);
    auto output_format = AnfAlgo::GetOutputFormat(input_node, input_index);
    auto output_dtype = AnfAlgo::GetOutputDeviceDataType(input_node, input_index);
    auto iter = type_name_map.find(output_dtype);
    if (iter == type_name_map.end()) {
      MS_LOG(EXCEPTION) << "Cannot found typeId:" << output_dtype;
    }
    auto ge_output_dtype = iter->second;

    optiling::TeOpTensorArg tensor_arg;
    optiling::TeOpTensor tensor;
    tensor_arg.arg_type = optiling::TA_SINGLE;
    tensor.dtype = ge_output_dtype;
    tensor.shape.insert(tensor.shape.end(), output_shape.begin(), output_shape.end());

    tensor.format = GeTypesConvert::GetGeTilingFormat(GeTypesConvert::GetGeFormat(output_format, output_shape.size()));
    MS_LOG(INFO) << "Tiling Format:" << tensor.format;
    tensor_arg.tensor.emplace_back(tensor);
    tensor_arg_list->emplace_back(tensor_arg);
  }
}

void FeedTeOpTensorOutputArg(const NotNull<CNodePtr> &cnode,
                             NotNull<std::vector<optiling::TeOpTensorArg> *> tensor_arg_list) {
  MS_LOG(INFO) << "FeedTeOpTensorOutputArg start, node:" << cnode->fullname_with_scope();
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode.get());
  for (size_t i = 0; i < output_size; ++i) {
    auto output_shape = AnfAlgo::GetOutputDeviceShape(cnode.get(), i);
    auto output_format = AnfAlgo::GetOutputFormat(cnode.get(), i);
    auto data_type = AnfAlgo::GetOutputDeviceDataType(cnode.get(), i);
    auto iter = type_name_map.find(data_type);
    if (iter == type_name_map.end()) {
      MS_LOG(EXCEPTION) << "Cannot found typeId:" << data_type;
    }

    optiling::TeOpTensorArg tensor_arg;
    optiling::TeOpTensor tensor;
    tensor_arg.arg_type = optiling::TA_SINGLE;
    tensor.dtype = iter->second;
    tensor.shape.insert(tensor.shape.end(), output_shape.begin(), output_shape.end());
    tensor.format = GeTypesConvert::GetGeTilingFormat(GeTypesConvert::GetGeFormat(output_format, output_shape.size()));
    MS_LOG(INFO) << "Tiling Format:" << tensor.format;
    tensor_arg.tensor.emplace_back(tensor);
    tensor_arg_list->emplace_back(tensor_arg);
  }
}

void FeedTeOpConstTensor(const NotNull<CNodePtr> &cnode, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                         NotNull<std::map<std::string, optiling::TeConstTensorData> *> const_inputs) {
  MS_LOG(INFO) << "FeedTeOpConstTensor start, node:" << cnode->fullname_with_scope();
  auto depends_list_me = abstract::GetDependsFormMap(cnode);
  if (depends_list_me.empty()) {
    MS_LOG(INFO) << "No input depend found, " << cnode->fullname_with_scope();
    return;
  }

  std::vector<int> depends_list;
  (void)std::transform(depends_list_me.begin(), depends_list_me.end(), std::back_inserter(depends_list),
                       [](const int64_t &value) { return static_cast<int>(value); });
  for (auto index : depends_list) {
    auto iter = depend_tensor_map.find(IntToSize(index));
    if (iter == depend_tensor_map.end()) {
      MS_LOG(EXCEPTION) << "Index not found in depend_tensor_map";
    }

    auto const_tensor = iter->second;

    auto have_input_names_attr = AnfAlgo::HasNodeAttr("input_names", cnode);
    if (!have_input_names_attr) {
      MS_LOG(EXCEPTION) << "cnode:" << cnode->fullname_with_scope() << " no input_names attr";
    }
    auto input_names_attr = AnfAlgo::GetNodeAttr<std::vector<std::string>>(cnode.get(), "input_names");
    if (IntToSize(index) >= input_names_attr.size()) {
      MS_LOG(EXCEPTION) << "input index" << index << " >= input_name_attr.size:" << input_names_attr.size();
    }
    auto input_name = input_names_attr[index];
    MS_LOG(INFO) << "input_name is " << input_name;
    auto type_id = AnfAlgo::GetPrevNodeOutputDeviceDataType(cnode.get(), index);
    const_inputs->try_emplace(
      input_name, optiling::TeConstTensorData{static_cast<const uint8_t *>(const_tensor->data_c()),
                                              IntToSize(const_tensor->DataSize()), MakeTempGeTensor(type_id)});
  }
  MS_LOG(INFO) << "FeedTeOpConstTensor end";
}

void OpTilingCalculater::Init() {
  MS_LOG(INFO) << "Start init OpTilingCalculater";
  tiling_func_map_ = optiling::OpTilingRegistryInterf::RegisteredOpInterf();
  if (tiling_func_map_.empty()) {
    MS_LOG(EXCEPTION) << "Get register tiling func failed.";
  }
}

std::string GetRealOpType(const std::string &op_type) {
  static const std::map<std::string, std::string> kOpTypeMap = {
    {"SparseApplyFtrl", "SparseApplyFtrlD"},
    {"SparseApplyProximalAdagrad", "SparseApplyProximalAdagradD"},
    {"SparseGatherV2", "Gather"},
    {"Pad", "PadD"},
    {"Concat", "ConcatD"},
  };
  auto iter = kOpTypeMap.find(op_type);
  if (iter == kOpTypeMap.end()) {
    return op_type;
  }
  return iter->second;
}

void OpTilingCalculater::CalculateTiling(const NotNull<CNodePtr> &cnode, const optiling::OpCompileInfo &op_compile_info,
                                         const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                                         const NotNull<optiling::OpRunInfo *> op_run_info) {
  optiling::TeOpParas op_param;
  std::string op_type = AnfAlgo::GetCNodeName(cnode.get());
  MS_LOG(INFO) << "[DynamicShape] calculate tiling, op_type:" << op_type;

  FeedTeOpTensorInputArg(cnode, NOT_NULL(&op_param.inputs));
  FeedTeOpTensorOutputArg(cnode, NOT_NULL(&op_param.outputs));
  FeedTeOpConstTensor(cnode, depend_tensor_map, NOT_NULL(&op_param.const_inputs));

  op_type = GetRealOpType(op_type);
  auto iter = tiling_func_map_.find(op_type);
  if (iter == tiling_func_map_.end()) {
    iter = tiling_func_map_.find("AutoTiling");
    if (iter == tiling_func_map_.end()) {
      MS_LOG(EXCEPTION) << "AutoTiling Func Not Found";
    }
  }

  MS_LOG(INFO) << "Get tiling func:" << iter->first;

  if (iter != tiling_func_map_.end()) {
    bool ret = (iter->second)(op_param, op_compile_info, *op_run_info);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Calculate tiling failed";
    }
  } else {
    MS_LOG(EXCEPTION) << "Tiling func not found";
  }
  MS_LOG(INFO) << "CalculateTiling success";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
