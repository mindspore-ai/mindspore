/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_build.h"
#include <memory>
#include <map>
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "utils/ms_context.h"
#include "runtime/dev.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
namespace mindspore {
namespace kernel {
void GetRealInputSize(const nlohmann::json &input_json, std::vector<size_t> *input_size_list, size_t *size_i) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  MS_EXCEPTION_IF_NULL(size_i);
  size_t kMaxShapeIdx = 1;
  int64_t kDynShapeValue = -2;
  if (input_json[kJShape].size() == IntToSize(1) && input_json[kJShape][0] == kDynShapeValue) {
    auto input_max_shape = input_json[kJRange];
    for (auto &max_shape : input_max_shape) {
      if (max_shape[kMaxShapeIdx] < 0) {
        (*size_i) = SizetMulWithOverflowCheck((*size_i), 0);
      } else {
        (*size_i) = SizetMulWithOverflowCheck((*size_i), LongToSize(max_shape[kMaxShapeIdx]));
      }
    }
    MS_LOG(INFO) << "Dims is dynamic, change -2 Shape to Max Shape.";
  } else {
    for (size_t j = 0; j < input_json[kJShape].size(); ++j) {
      if (input_json[kJShape][j] == -1) {
        auto input_max_shape = input_json[kJRange];
        if (j >= input_max_shape.size()) {
          MS_LOG(EXCEPTION) << "Invalid Dynamic Shape Max Shape";
        }
        if (input_max_shape[j][kMaxShapeIdx] == -1) {
          MS_LOG(INFO) << "Change -1 Shape to 1";
          (*size_i) = SizetMulWithOverflowCheck((*size_i), 1);
        } else {
          MS_LOG(INFO) << "Change -1 Shape to input_max_shape[j][kMaxShapeIdx]:" << input_max_shape[j][kMaxShapeIdx];
          (*size_i) = SizetMulWithOverflowCheck((*size_i), LongToSize(input_max_shape[j][kMaxShapeIdx]));
        }
        continue;
      }
      (*size_i) = SizetMulWithOverflowCheck((*size_i), static_cast<size_t>(input_json[kJShape][j]));
    }
  }
  std::string dtype = input_json[kJDtype];
  size_t nbyte = tbe::GetDtypeNbyte(dtype);
  (*size_i) = SizetMulWithOverflowCheck(*size_i, nbyte);
  input_size_list->push_back((*size_i));
}

void GetInputSizeList(const nlohmann::json &input_json, std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  for (size_t i = 0; i < input_json.size(); i++) {
    if (input_json[i].is_array()) {
      for (size_t m = 0; m < input_json[i].size(); m++) {
        size_t size_i = 1;
        if (input_json[i][m][kJValid] == false) {
          continue;
        }
        GetRealInputSize(input_json[i][m], input_size_list, &size_i);
      }
    } else {
      size_t size_i = 1;
      if (input_json[i][kJValid] == false) {
        continue;
      }
      GetRealInputSize(input_json[i], input_size_list, &size_i);
    }
  }
}

void GetRealOutputSize(const nlohmann::json &output_json, std::vector<size_t> *output_size_list, size_t *size_i) {
  MS_EXCEPTION_IF_NULL(output_size_list);
  MS_EXCEPTION_IF_NULL(size_i);
  size_t kMaxShapeIdx = 1;
  int64_t kDynShapeValue = -2;
  if (output_json[kJShape].size() == IntToSize(1) && output_json[kJShape][0] == kDynShapeValue) {
    auto output_max_shape = output_json[kJRange];
    for (auto &max_shape : output_max_shape) {
      if (max_shape[kMaxShapeIdx] < 0) {
        (*size_i) = SizetMulWithOverflowCheck((*size_i), 0);
      } else {
        (*size_i) = SizetMulWithOverflowCheck(*size_i, LongToSize(max_shape[kMaxShapeIdx]));
      }
    }
    MS_LOG(INFO) << "Dims is dynamic, change -2 Shape to Max Shape.";
  } else {
    for (size_t j = 0; j < output_json[kJShape].size(); ++j) {
      if (output_json[kJShape][j] == -1) {
        auto output_max_shape = output_json[kJRange];
        if (j >= output_max_shape.size()) {
          MS_LOG(EXCEPTION) << "Invalid Dynamic Shape Max Shape";
        }
        if (output_max_shape[j][kMaxShapeIdx] == -1) {
          MS_LOG(INFO) << "Change -1 Shape to 1";
          (*size_i) = SizetMulWithOverflowCheck((*size_i), 1);
        } else {
          MS_LOG(INFO) << "Change -1 Shape to output_max_shape[j][kMaxShapeIdx]:" << output_max_shape[j][kMaxShapeIdx];
          (*size_i) = SizetMulWithOverflowCheck((*size_i), LongToSize(output_max_shape[j][kMaxShapeIdx]));
        }
        continue;
      }
      (*size_i) = SizetMulWithOverflowCheck(*size_i, static_cast<size_t>(output_json[kJShape][j]));
    }
  }
  std::string dtype = output_json[kJDtype];
  size_t nbyte = tbe::GetDtypeNbyte(dtype);
  (*size_i) = SizetMulWithOverflowCheck(*size_i, nbyte);
  output_size_list->push_back((*size_i));
}

void GetOutputSizeList(const nlohmann::json &output_json, std::vector<size_t> *output_size_list) {
  for (size_t i = 0; i < output_json.size(); i++) {
    if (output_json[i].is_array()) {
      for (size_t m = 0; m < output_json[i].size(); m++) {
        size_t size_i = 1;
        if (output_json[i][m][kJValid] == false) {
          std::string output_name = output_json[i][m][kJName];
          MS_LOG(INFO) << "Output name:" << output_name << " is optional, valid is false.";
          continue;
        }
        GetRealOutputSize(output_json[i][m], output_size_list, &size_i);
      }
    } else {
      size_t size_i = 1;
      if (output_json[i][kJValid] == false) {
        std::string output_name = output_json[i][kJName];
        MS_LOG(INFO) << "Output name:" << output_name << " is optional, valid is false.";
        continue;
      }
      GetRealOutputSize(output_json[i], output_size_list, &size_i);
    }
  }
}

bool TbeKernelBuild::GetIOSize(const nlohmann::json &kernel_json, std::vector<size_t> *input_size_list,
                               std::vector<size_t> *output_size_list) {
  if (input_size_list == nullptr || output_size_list == nullptr) {
    MS_LOG(ERROR) << "Input size or output size is nullptr";
    return false;
  }
  input_size_list->clear();
  output_size_list->clear();
  auto op_list = kernel_json["op_list"];
  for (size_t i = 0; i < op_list.size(); i++) {
    auto op_info = op_list[i];
    if (op_info["type"] != "Data") {
      GetInputSizeList(op_info["input_desc"], input_size_list);
      GetOutputSizeList(op_info["output_desc"], output_size_list);
    }
  }
  return true;
}

size_t TbeKernelBuild::GetIOSizeImpl(const nlohmann::json &desc) {
  size_t ret = 1;
  for (const auto &shape_item : desc[kJShape]) {
    ret = SizetMulWithOverflowCheck(ret, static_cast<size_t>(shape_item));
  }
  std::string data_type = desc[kJDataType];
  size_t nbyte = tbe::GetDtypeNbyte(data_type);
  ret = SizetMulWithOverflowCheck(ret, nbyte);
  return ret;
}

void TbeKernelBuild::CalInputSize(const nlohmann::json &fusion_op_list, std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  // cal input size for malloc
  for (const auto &op : fusion_op_list) {
    if (op[kJType] == "Data") {
      const auto &data_output_desc = op[kJOutputDesc];
      for (const auto &data_output : data_output_desc) {
        if (data_output[kJShape] == "NULL") {
          break;
        }
        input_size_list->push_back(GetIOSizeImpl(data_output));
      }
    }
  }
}

bool TbeKernelBuild::CalOutputSize(const nlohmann::json &fusion_op_list,
                                   const std::vector<mindspore::AnfNodePtr> &output_nodes,
                                   std::vector<size_t> *output_size_list) {
  MS_EXCEPTION_IF_NULL(output_size_list);
  // cal output size for malloc
  for (const auto &output_node : output_nodes) {
    auto kernel_idx = common::AnfAlgo::VisitKernel(output_node, 0);
    auto real_node = kernel_idx.first;
    size_t real_idx = kernel_idx.second;
    MS_EXCEPTION_IF_NULL(real_node);
    auto full_name = real_node->fullname_with_scope();
    for (const auto &op : fusion_op_list) {
      if (op[kJName] != full_name) {
        continue;
      }
      auto op_output_desces = op[kJOutputDesc];
      if (output_node != real_node) {
        // tuple_get item
        auto output_desc = op_output_desces[real_idx];
        if (output_desc[kJShape].empty()) {
          MS_LOG(INFO) << "Fusion error: output_desc's shape is empty. real_index " << real_idx;
          return false;
        }
        output_size_list->push_back(GetIOSizeImpl(output_desc));
      } else {
        for (const auto &output_desc : op_output_desces) {
          if (output_desc[kJShape].empty()) {
            continue;
          }
          output_size_list->push_back(GetIOSizeImpl(output_desc));
        }
      }
    }
  }
  return true;
}

bool TbeKernelBuild::GetIOSize(const nlohmann::json &fusion_op_list,
                               const std::vector<mindspore::AnfNodePtr> &output_nodes,
                               std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  MS_EXCEPTION_IF_NULL(output_size_list);
  input_size_list->clear();
  output_size_list->clear();
  // cal input size for malloc
  CalInputSize(fusion_op_list, input_size_list);
  // cal output size for malloc
  return CalOutputSize(fusion_op_list, output_nodes, output_size_list);
}
}  // namespace kernel
}  // namespace mindspore
