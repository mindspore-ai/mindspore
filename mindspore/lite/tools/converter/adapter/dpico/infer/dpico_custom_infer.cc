/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "infer/dpico_custom_infer.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "common/string_util.h"
#include "common/op_attr.h"
#include "common/infer_util.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::CustomInterface;
using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace kernel {
namespace {
constexpr int kOmParameterNum = 1;
Status FetchAttrs(const schema::Primitive &primitive, std::map<std::string, std::string> *attrs) {
  if (attrs == nullptr) {
    MS_LOG(ERROR) << "function input parameter is nullptr.";
    return kLiteError;
  }
  auto param = primitive.value_as_Custom();
  if (dpico::CheckCustomParam(param, "DPICO") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param attr is nullptr.";
    return kLiteError;
  }
  if (param->attr()->size() < 1) {
    MS_LOG(ERROR) << "There are at least 1 attribute of Custom";
    return kLiteError;
  }
  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    auto output_info = param->attr()->Get(i)->data();
    if (output_info == nullptr) {
      return kLiteError;
    }
    int buf_size = static_cast<int>(output_info->size());
    std::string attr;
    for (int j = 0; j < buf_size; j++) {
      attr.push_back(static_cast<char>(output_info->Get(j)));
    }
    auto attr_name = param->attr()->Get(i)->name()->str();
    (void)attrs->emplace(attr_name, attr);
  }
  return kSuccess;
}

Status GetCustomShape(const std::map<std::string, std::string> &attrs, const std::string &attr_name, size_t tensor_num,
                      std::vector<std::vector<int64_t>> *shapes) {
  if (shapes == nullptr) {
    MS_LOG(ERROR) << "the function input parameter is nullptr.";
    return kLiteError;
  }
  if (attrs.find(attr_name) == attrs.end()) {
    MS_LOG(ERROR) << "custom node should have " << attr_name << " val.";
    return kLiteError;
  }
  auto attr = attrs.at(attr_name);
  if (attr.empty()) {
    MS_LOG(ERROR) << "custom node should have " << attr_name << " val.";
    return kLiteError;
  }
  auto split_shape_str = dpico::SplitString(attr, ',');
  size_t index = 0;
  for (size_t i = 0; i < split_shape_str.size(); i++) {
    auto dim_size = std::stoul(split_shape_str.at(i));
    std::vector<int64_t> shape;
    for (size_t j = i + 1; j < i + 1 + dim_size; j++) {
      if (j >= split_shape_str.size()) {
        MS_LOG(ERROR) << "split_shape_str val is invalid. ";
        return kLiteError;
      }
      shape.push_back(std::stoul(split_shape_str.at(j)));
    }
    i += dim_size;
    if (tensor_num < index) {
      MS_LOG(ERROR) << "shape index " << index << " is greater than custom tensor_num " << tensor_num;
      return kLiteError;
    }
    shapes->push_back(shape);
    index++;
  }
  return kSuccess;
}

Status SetOutputFormat(const std::map<std::string, std::string> &attrs, std::vector<mindspore::MSTensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "the function input parameter is nullptr.";
    return kLiteError;
  }
  if (attrs.find(dpico::kOutputsFormat) == attrs.end()) {
    MS_LOG(ERROR) << "custom node should have " << dpico::kOutputsFormat << " val.";
    return kLiteError;
  }
  auto output_format_str = attrs.at(dpico::kOutputsFormat);
  auto output_format = dpico::SplitString(output_format_str, ',');
  if (output_format.size() > outputs->size()) {
    MS_LOG(ERROR) << "output format attr is invalid, the number of which is out of range.";
    return kLiteError;
  }
  for (size_t i = 0; i < output_format.size(); ++i) {
    if (!dpico::IsValidUnsignedNum(output_format[i])) {
      MS_LOG(ERROR) << "output format must be an unsigned int";
      return kLiteError;
    }
    auto format = std::stoi(output_format[i]);
    if (format != static_cast<int64_t>(NHWC) && format != static_cast<int64_t>(NCHW)) {
      MS_LOG(ERROR) << "output format is invalid, which should be NHWC or NCHW.";
      return kLiteError;
    }
    outputs->at(i).SetFormat(static_cast<Format>(format));
  }
  return kSuccess;
}

bool InferDone(const std::vector<mindspore::MSTensor> &tensors) {
  for (auto &tensor : tensors) {
    auto shape = tensor.Shape();
    if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
      return false;
    }
  }
  return true;
}
}  // namespace

Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  for (auto &output : *outputs) {
    output.SetDataType(DataType::kNumberTypeFloat32);
    output.SetFormat(inputs->front().format());
  }
  std::map<std::string, std::string> attrs;
  if (FetchAttrs(*primitive, &attrs) != kSuccess) {
    MS_LOG(ERROR) << "fetch attrs from primitive failed.";
    return kLiteError;
  }
  if (SetOutputFormat(attrs, outputs) != kSuccess) {
    MS_LOG(ERROR) << "set output format failed.";
    return kLiteError;
  }
  if (!InferDone(*inputs)) {
    return kLiteInferInvalid;
  }
  std::vector<std::vector<int64_t>> inputs_shape;
  if (GetCustomShape(attrs, dpico::kInputsShape, inputs->size(), &inputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "parser inputs_shape attribute failed.";
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> outputs_shape;
  if (GetCustomShape(attrs, dpico::kOutputsShape, outputs->size(), &outputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "parser outputs_shape attribute failed.";
    return kLiteError;
  }
  if (inputs_shape.size() != inputs->size() - kOmParameterNum) {
    MS_LOG(ERROR) << "inputs num:" << (inputs->size() - kOmParameterNum)
                  << "should be equal to inputs_shape num:" << inputs_shape.size();
    return kLiteError;
  }
  if (inputs_shape[0].size() != (*inputs)[0].Shape().size()) {
    MS_LOG(ERROR) << "input[0] shape dim size is invalid. " << inputs_shape[0].size()
                  << "!=" << (*inputs)[0].Shape().size();
    return kLiteError;
  }

  dpico::OmNetType om_net_type;
  if (dpico::GetOmNetType(primitive, &om_net_type) != RET_OK) {
    MS_LOG(ERROR) << "get om net type failed.";
    return kLiteError;
  }

  bool resize_flag = false;
  int resize_num = 1;
  for (size_t i = 0; i < inputs_shape[0].size(); i++) {
    if (inputs_shape[0][i] != (*inputs)[0].Shape()[i]) {
      if (i == 0) {
        resize_flag = true;
        resize_num = static_cast<int>((*inputs)[0].Shape()[i]);
      } else {
        MS_LOG(ERROR) << "Custom of DPICO only support batch_num resize.";
        return kLiteError;
      }
    }
  }
  if (resize_flag) {
    for (auto &output_shape : outputs_shape) {
      output_shape[0] = resize_num;
      if (om_net_type == dpico::OmNetType::kRecurrent) {
        MS_LOG(INFO) << "only output_0 has the information about time steps.";
        break;
      }
    }
  }

  for (size_t i = 0; i < outputs->size(); i++) {
    (*outputs)[i].SetShape(outputs_shape[i]);
  }
  return kSuccess;
}

std::shared_ptr<KernelInterface> CustomInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<CustomInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, DPICO, CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore
