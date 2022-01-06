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

#include "src/custom_infer.h"
#include <map>
#include <string>
#include "include/api/format.h"
#include "include/registry/register_kernel_interface.h"
#include "src/common_utils.h"

using mindspore::kernel::KernelInterface;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace dpico {
namespace {
constexpr int kDecimal = 10;
constexpr auto kInputShape = "inputs_shape";
constexpr auto kOutputShape = "outputs_shape";
constexpr auto kOutputsFormat = "outputs_format";
std::vector<std::string> SplitString(const std::string &raw_str, char delimiter) {
  if (raw_str.empty()) {
    return {};
  }
  std::vector<std::string> res;
  std::string::size_type last_pos = 0;
  auto cur_pos = raw_str.find(delimiter);
  while (cur_pos != std::string::npos) {
    res.push_back(raw_str.substr(last_pos, cur_pos - last_pos));
    cur_pos++;
    last_pos = cur_pos;
    cur_pos = raw_str.find(delimiter, cur_pos);
  }
  if (last_pos < raw_str.size()) {
    res.push_back(raw_str.substr(last_pos, raw_str.size() - last_pos + 1));
  }
  return res;
}

Status GetCustomShape(const std::map<std::string, std::string> &attrs, const std::string &attr_name,
                      std::vector<std::vector<int64_t>> *shapes) {
  if (shapes == nullptr) {
    MS_LOG(ERROR) << "the function input parameter is nullptr.";
    return kLiteError;
  }
  auto attr = attrs.at(attr_name);
  if (attr.empty()) {
    MS_LOG(ERROR) << attr_name.c_str() << " data is empty.";
    return kLiteError;
  }
  char delims[] = ",";
  char *res = nullptr;
  char *save_ptr = nullptr;
  res = strtok_r(attr.data(), delims, &save_ptr);
  while (res != nullptr) {
    int64_t ndims = strtol(res, &res, kDecimal);
    int j = 0;
    std::vector<int64_t> shape;
    shape.resize(ndims);
    for (; j < ndims; j++) {
      res = strtok_r(NULL, delims, &save_ptr);
      shape[j] = static_cast<int64_t>(strtol(res, &res, kDecimal));
    }
    shapes->push_back(shape);

    res = strtok_r(NULL, delims, &save_ptr);
  }
  return kSuccess;
}

Status DetermineBatchSize(const std::vector<int64_t> &input_shape_lite, const std::vector<int64_t> &input_shape_dpico,
                          int *resize_num, bool *is_resize) {
  if (resize_num == nullptr || is_resize == nullptr) {
    MS_LOG(ERROR) << "the function input parameter is nullptr.";
    return kLiteError;
  }
  if (input_shape_lite.size() != input_shape_dpico.size()) {
    MS_LOG(ERROR) << "both input shape from lite and dpico cannot match.";
    return kLiteError;
  }
  for (size_t i = 0; i < input_shape_dpico.size(); i++) {
    if (input_shape_dpico[i] != input_shape_lite[i]) {
      if (i == 0) {
        *is_resize = true;
        *resize_num = input_shape_lite[i];
      } else {
        MS_LOG(ERROR) << "Custom of DPICO only support batch_num resize.";
        return kLiteError;
      }
    }
  }
  return kSuccess;
}

Status SetOutputFormat(const std::map<std::string, std::string> &attrs, std::vector<mindspore::MSTensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "the function input parameter is nullptr.";
    return kLiteError;
  }
  if (attrs.find(kOutputsFormat) == attrs.end()) {
    MS_LOG(ERROR) << "custom node should have " << kOutputsFormat << " attr.";
    return kLiteError;
  }
  auto output_format_str = attrs.at(kOutputsFormat);
  auto output_format = SplitString(output_format_str, ',');
  if (output_format.size() > outputs->size()) {
    MS_LOG(ERROR) << "output format attr is invalid, the number of which is out of range.";
    return kLiteError;
  }
  for (size_t i = 0; i < output_format.size(); ++i) {
    if (!lite::IsValidUnsignedNum(output_format[i])) {
      MS_LOG(ERROR) << "output format must be an unsigned int.";
      return kLiteError;
    }
    auto format = std::stoi(output_format[i]);
    if (format != NHWC && format != NCHW) {
      MS_LOG(ERROR) << "output format is invalid, which should be NHWC or NCHW.";
      return kLiteError;
    }
    outputs->at(i).SetFormat(static_cast<Format>(format));
  }
  return kSuccess;
}
}  // namespace

std::shared_ptr<KernelInterface> CustomInferCreater() {
  auto infer = new (std::nothrow) CustomInterface();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return std::shared_ptr<KernelInterface>(infer);
}

Status CustomInterface::InferShapeJudge(std::vector<mindspore::MSTensor> *inputs,
                                        const std::vector<std::vector<int64_t>> &inputs_shape) const {
  size_t inputs_size_without_om_model = inputs->size() - 1;
  if (inputs_shape.size() != inputs_size_without_om_model) {
    MS_LOG(ERROR) << "inputs num diff inputs_shape num.";
    return kLiteError;
  }
  if (inputs_shape[0].size() != (*inputs)[0].Shape().size()) {
    MS_LOG(ERROR) << "shape size err. " << inputs_shape[0].size() << ", " << (*inputs)[0].Shape().size();
    return kLiteError;
  }
  return kSuccess;
}

Status CustomInterface::InferRecurrentTwoOutputProcess(const mindspore::schema::Primitive *primitive,
                                                       const kernel::Kernel *kernel,
                                                       std::vector<std::vector<int64_t>> *outputs_shape) const {
  if (primitive == nullptr || outputs_shape == nullptr) {
    return kLiteError;
  }
  lite::OmNetType net_type{lite::OmNetType_CNN};
  if (kernel != nullptr) {
    auto net_type_str = kernel->GetAttr(lite::kNetType);
    if (!net_type_str.empty()) {
      if (!lite::IsValidUnsignedNum(net_type_str)) {
        MS_LOG(ERROR) << "net_type must be an unsigned int.";
        return kLiteError;
      }
      auto net_type_int = std::stoi(net_type_str);
      if (net_type_int < lite::OmNetType_CNN || net_type_int > lite::OmNetType_RECURRENT) {
        MS_LOG(ERROR) << "net_type attr is invalid, value is " << net_type_int;
        return kLiteError;
      }
      net_type = static_cast<lite::OmNetType>(net_type_int);
    }
  } else {
    auto ret = JudgeOmNetType(*primitive, &net_type);
    if (ret != lite::SUCCESS) {
      MS_LOG(ERROR) << "get model attr failed";
      return kLiteError;
    }
  }
  if (net_type == lite::OmNetType_RECURRENT && outputs_shape->size() > 1) {
    if ((*outputs_shape)[1].empty()) {
      return kLiteError;
    }
    (*outputs_shape)[1][0] = 1;
  }
  return kSuccess;
}

Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const mindspore::schema::Primitive *primitive, const kernel::Kernel *kernel) {
  if (inputs->size() < lite::kMinInputSize) {
    MS_LOG(ERROR) << "Inputs size is less than 2";
    return kLiteError;
  }
  if (outputs->empty()) {
    MS_LOG(ERROR) << "Outputs size 0";
    return kLiteError;
  }
  std::map<std::string, std::string> attrs;
  schema::PrimitiveType type;
  if (kernel != nullptr) {
    attrs.emplace(kInputShape, kernel->GetAttr(kInputShape));
    attrs.emplace(kOutputShape, kernel->GetAttr(kOutputShape));
    attrs.emplace(kOutputsFormat, kernel->GetAttr(kOutputsFormat));
    type = kernel->type();
  } else {
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr.";
      return kLiteError;
    }
    lite::ExtractAttrsFromPrimitive(primitive, &attrs);
    type = primitive->value_type();
  }
  if (type != mindspore::schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return kLiteError;
  }
  for (size_t i = 0; i < outputs->size(); i++) {
    (*outputs)[i].SetDataType(DataType::kNumberTypeFloat32);
    (*outputs)[i].SetFormat(Format::NCHW);
  }
  if (SetOutputFormat(attrs, outputs) != kSuccess) {
    MS_LOG(ERROR) << "set output format failed.";
    return kLiteError;
  }
  if (!lite::InferDone(*inputs)) {
    return kLiteInferInvalid;
  }
  std::vector<std::vector<int64_t>> inputs_shape;
  if (GetCustomShape(attrs, "inputs_shape", &inputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "parser inputs_shape attribute err.";
    return kLiteError;
  }
  std::vector<std::vector<int64_t>> outputs_shape;
  if (GetCustomShape(attrs, "outputs_shape", &outputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "parser outputs_shape attribute err.";
    return kLiteError;
  }
  if (InferShapeJudge(inputs, inputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "input shape err.";
    return kLiteError;
  }
  bool resize_flag = false;
  int resize_num = 1;
  if (DetermineBatchSize((*inputs)[0].Shape(), inputs_shape[0], &resize_num, &resize_flag) != kSuccess) {
    MS_LOG(ERROR) << "determine batch size failed.";
    return kLiteError;
  }
  if (resize_flag) {
    for (auto &output_shape : outputs_shape) {
      output_shape[0] = resize_num;
    }
  }
  if (InferRecurrentTwoOutputProcess(primitive, kernel, &outputs_shape) != kSuccess) {
    MS_LOG(ERROR) << "Infer Recurrent Two Output Process err.";
    return kLiteError;
  }

  for (size_t i = 0; i < outputs->size(); i++) {
    (*outputs)[i].SetShape(outputs_shape[i]);
  }
  return kSuccess;
}
}  // namespace dpico
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, DPICO, dpico::CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore
