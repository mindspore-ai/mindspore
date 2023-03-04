/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "c_api/src/utils.h"

void ConvertConstScalarInputToTensor(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNodeImpl>()) {
    return;
  }
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ScalarImpl>()) {
    return;
  }
  TensorPtr tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor of" << input_node->DebugString() << "failed";
    return;
  }
  value_node->set_value(tensor_ptr);
  value_node->set_abstract(tensor_ptr->ToAbstract());
}

std::vector<TensorPtr> ConvertOutputToTensor(const mindspore::BaseRef &output) {
  std::vector<TensorPtr> ref_outputs{};
  if (mindspore::utils::isa<mindspore::VectorRef>(output)) {
    auto vec_ref = mindspore::utils::cast<mindspore::VectorRef>(output);
    for (const auto &item : vec_ref) {
      // for multiple outputs, ascend will return a VectorRef of VectorRef.
      const std::vector<TensorPtr> &item_out = ConvertOutputToTensor(item);
      (void)ref_outputs.insert(ref_outputs.end(), item_out.begin(), item_out.end());
    }
  } else if (mindspore::utils::isa<TensorPtr>(output)) {
    auto tensor = std::dynamic_pointer_cast<TensorImpl>(output.copy());
    tensor->data_sync();
    ref_outputs.push_back(tensor);
  } else if (mindspore::utils::isa<ScalarPtr>(output)) {
    auto value = mindspore::utils::cast<ScalarPtr>(output);
    auto tensor = ScalarToTensor(value->cast<ScalarPtr>());
    ref_outputs.push_back(tensor);
  } else {
    MS_LOG(ERROR) << "Convert output to tensor failed, unrecognized output type: " << output.ToString();
  }
  return ref_outputs;
}

AbstractBasePtr GetAbstract(const TypePtr &type_ptr, const int64_t shape[], size_t shape_size, bool is_param) {
  if (shape == nullptr) {
    if (shape_size == 0) {
      if (is_param) {
        ShapeVector shape_vec{1};
        return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
      }
      return std::make_shared<AbstractScalarImpl>(type_ptr);
    } else {
      MS_LOG(ERROR) << "Input Handle [shape_size] should >= 0.";
      return nullptr;
    }
  }
  if (shape[0] == 0 && shape_size == 1) {
    ShapeVector shape_vec;
    return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
  }
  ShapeVector shape_vec(shape, shape + shape_size);
  return std::make_shared<AbstractTensorImpl>(type_ptr, shape_vec);
}

STATUS CheckCustomOpInfo(const CustomOpInfo &info) {
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.func_name != nullptr, RET_ERROR, "The func_name of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.func_type != nullptr, RET_ERROR, "The func_type of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.target != nullptr, RET_ERROR, "The target of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.input_name != nullptr, RET_ERROR,
                                "The input_name of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.output_name != nullptr, RET_ERROR,
                                "The output_name of custom op must be specified!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.input_num > 0, RET_ERROR, "The input_num of custom op must be a positive value!");
  MS_ERROR_IF_FALSE_W_RET_N_LOG(info.output_num > 0, RET_ERROR,
                                "The output_num of custom op must be a positive value!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_infer_func == nullptr && info.output_dtypes == nullptr, RET_ERROR,
                               "Either dtype infer function or output shape must be specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_infer_func != nullptr && info.output_dtypes != nullptr, RET_ERROR,
                               "Only one should be specified between dtype infer function and output shape!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.shape_infer_func == nullptr && info.output_shapes == nullptr, RET_ERROR,
                               "Either shape infer function or output shape must be specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.shape_infer_func != nullptr && info.output_shapes != nullptr, RET_ERROR,
                               "Only one should be specified between shape infer function and output shape!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.output_shapes != nullptr && info.output_dims == nullptr, RET_ERROR,
                               "Output dims must be specified if output_shapes are given!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.attr_name != nullptr && info.attr_num == 0, RET_ERROR,
                               "The attr_num of custom op must be none-zero if attr_name is specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.attr_name == nullptr && info.attr_num != 0, RET_ERROR,
                               "The attr_num of custom op must be zero if attr_name is not specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_formats != nullptr && info.dtype_formats_num == 0, RET_ERROR,
                               "The dtype_formats_num of custom op must be none-zero if dtype_formats is specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(info.dtype_formats == nullptr && info.dtype_formats_num != 0, RET_ERROR,
                               "The dtype_formats_num of custom op must be zero if dtype_formats is not specified!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(std::string(info.func_name).find(".so:") == std::string::npos, RET_ERROR,
                               "so file path and function name must be provided in func_name!");
  return RET_OK;
}

nlohmann::json ConvertOpInfoToJson(const CustomOpInfo &info) {
  nlohmann::json obj;
  obj["attr"] = {};
  std::string target = info.target;
  obj["target"] = target;
  obj["op_name"] = "Custom" + std::string(info.func_name);
  obj["fusion_tyoe"] = "OPAQUE";
  if (info.dtype_formats != nullptr) {
    std::vector<std::vector<std::string>> dtype_formats;
    for (size_t i = 0; i < info.dtype_formats_num; i++) {
      for (size_t j = 0; j < info.input_num + info.output_num; j++) {
        auto iter = kDTypeFmtEnumToStrMap.find(info.dtype_formats[i][j]);
        if (iter == kDTypeFmtEnumToStrMap.end()) {
          MS_LOG(ERROR) << "Unsupported DTypeFormat: " << info.dtype_formats[i][j];
          return {};
        }
        dtype_formats.push_back(iter->second);
      }
    }
    obj["dtype_format"] = {dtype_formats};
  }
  std::vector<nlohmann::json> js_inputs;
  for (size_t i = 0; i < info.input_num; i++) {
    nlohmann::json js_input;
    js_input["index"] = i;
    js_input["name"] = std::string(info.input_name[i]);
    js_input["paramType"] = "required";
    js_inputs.push_back(js_input);
  }
  obj["inputs"] = js_inputs;
  std::vector<nlohmann::json> js_outputs;
  for (size_t i = 0; i < info.output_num; i++) {
    nlohmann::json js_output;
    js_output["index"] = i;
    js_output["name"] = std::string(info.output_name[i]);
    js_output["paramType"] = "required";
    js_outputs.push_back(js_output);
  }
  obj["outputs"] = js_outputs;
  auto aot_imply_type = target == "Ascend" ? "BiSheng" : target;
  const std::map<std::string, std::string> func_type_to_imply_type = {
    {"hybrid", "AKG"},  {"akg", "AKG"},    {"tbe", "TBE"},         {"aicpu", "AICPU"},
    {"pyfunc", target}, {"julia", target}, {"aot", aot_imply_type}};
  auto iter = func_type_to_imply_type.find(std::string(info.func_type));
  if (iter == func_type_to_imply_type.end()) {
    MS_LOG(ERROR) << "Unsupported function type: " << std::string(info.func_type);
    return {};
  }
  auto imply_type = iter->second;
  obj["imply_type"] = imply_type;
  return obj;
}

size_t GetMaxMallocSize() {
  size_t max_malloc_size = 0;
#if defined(_MSC_VER) || defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  max_malloc_size = static_cast<size_t>(status.ullTotalPhys);
#else
  max_malloc_size = static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
  return max_malloc_size;
}
