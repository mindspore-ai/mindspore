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

#include "ops/grad/pool_grad.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
static std::map<std::string, int64_t> pad_map = {
  {"CALCULATED", PadMode::PAD},
  {"PAD", PadMode::PAD},
  {"SAME", PadMode::SAME},
  {"VALID", PadMode::VALID},
};

static std::map<std::string, int64_t> dataformat_map = {
  {"NCHW", Format::NCHW},
  {"NHWC", Format::NHWC},
  {"NCDHW", Format::NCDHW},
  {"NDHWC", Format::NDHWC},
};

MIND_API_OPERATOR_IMPL(PoolGrad, BaseOperator);
std::vector<int64_t> PoolGrad::_grad_check_vector(const std::string &arg_name, std::vector<int64_t> arg_val,
                                                  const std::string &op_name) {
  std::vector<int64_t> ret;
  std::string error_msg = "For '" + op_name + "'," + " the '" + arg_name +
                          "' must be a vector of one, two or four "
                          "positive int number(s), but got error arg_val size.";
  switch (SizeToLong(arg_val.size())) {
    case 1:
      ret = {1, 1, arg_val[0], arg_val[0]};
      break;
    case 2:
      ret = {1, 1, arg_val[0], arg_val[1]};
      break;
    case 4:
      ret = arg_val;
      break;
    default:
      MS_LOG(EXCEPTION) << error_msg;
  }
  for (auto it : arg_val) {
    if (it <= 0) {
      MS_LOG(EXCEPTION) << error_msg;
    }
  }
  return ret;
}

void PoolGrad::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                    const PadMode &pad_mode, const Format &format) {
  this->set_kernel_size(kernel_size);
  this->set_strides(strides);
  this->set_pad_mode(pad_mode);
  this->set_format(format);
}

void PoolGrad::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  std::vector<int64_t> k_size = _grad_check_vector(kKernelSize, kernel_size, this->name());
  (void)this->AddAttr(kKernelSize, api::MakeValue(k_size));
}

void PoolGrad::set_strides(const std::vector<int64_t> &strides) {
  std::vector<int64_t> strides_ = _grad_check_vector(kStrides, strides, this->name());
  (void)this->AddAttr(kStrides, api::MakeValue(strides_));
}

void PoolGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

void PoolGrad::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

std::vector<int64_t> PoolGrad::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> PoolGrad::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode PoolGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return PadMode(GetValue<int64_t>(value_ptr));
  }
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = pad_map.find(attr_value_str);
  if (iter == pad_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid pad mode " << attr_value_str << " use CALCULATED, PAD, VALID or SAME";
  }
  return PadMode(iter->second);
}

Format PoolGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return Format(GetValue<int64_t>(value_ptr));
  }
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = dataformat_map.find(attr_value_str);
  if (iter == dataformat_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid format " << attr_value_str << " use NCHW, NHWC NCDHW or NDHWC";
  }
  return Format(iter->second);
}

REGISTER_PRIMITIVE_C(kNamePoolGrad, PoolGrad);
}  // namespace ops
}  // namespace mindspore
