/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file error_util.cpp
 * \brief
 */
#include <map>
#include "error_util.h"
#include "error_code.h"
#include "op_log.h"

using namespace std;
using namespace ge;

namespace ge {
std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode) { return "E" + std::to_string(errCode); }

std::string GetShapeErrMsg(uint32_t index, const std::string &wrong_shape, const std::string &correct_shape) {
  std::string msg = ConcatString(index, "th input has wrong shape", wrong_shape, ", it should be ", correct_shape);
  return msg;
}

std::string GetAttrValueErrMsg(const std::string &attr_name, const std::string &wrong_val,
                               const std::string &correct_val) {
  std::string msg = ConcatString("attr[", attr_name, "], has wrong value[", wrong_val, "], it should be ", correct_val);
  return msg;
}

std::string GetAttrSizeErrMsg(const std::string &attr_name, const std::string &wrong_size,
                              const std::string &correct_size) {
  std::string msg =
    ConcatString("attr[", attr_name, "], has wrong size[", wrong_size, "], it should be ", correct_size);
  return msg;
}

std::string GetInputInvalidErrMsg(const std::string &param_name) {
  std::string msg = ConcatString("get ", param_name, " failed");
  return msg;
}

std::string GetShapeSizeErrMsg(uint32_t index, const std::string &wrong_shape_size,
                               const std::string &correct_shape_size) {
  std::string msg =
    ConcatString(index, "th input has wrong shape size ", wrong_shape_size, ", it should be ", correct_shape_size);
  return msg;
}

std::string GetInputFormatNotSupportErrMsg(const std::string &param_name, const std::string &expected_format_list,
                                           const std::string &data_format) {
  std::string msg =
    ConcatString("[", param_name, "], has wrong format [", data_format, "], it should be in ", expected_format_list);
  return msg;
}

std::string GetInputDtypeNotSupportErrMsg(const std::string &param_name, const std::string &expected_dtype_list,
                                          const std::string &data_dtype) {
  std::string msg =
    ConcatString("[", param_name, "], has wrong dtype [", data_dtype, "], it should be in ", expected_dtype_list);
  return msg;
}

std::string GetInputDTypeErrMsg(const std::string &param_name, const std::string &expected_dtype,
                                const std::string &data_dtype) {
  std::string msg =
    ConcatString("[", param_name, "], has wrong dtype [", data_dtype, "], it should be ", expected_dtype);
  return msg;
}

std::string GetInputFormatErrMsg(const std::string &param_name, const std::string &expected_format,
                                 const std::string &data_format) {
  std::string msg =
    ConcatString("[", param_name, "], has wrong format [", data_format, "], it should be in ", expected_format);
  return msg;
}

std::string SetAttrErrMsg(const std::string &param_name) {
  std::string msg = ConcatString("set param [", param_name, "] failed");
  return msg;
}

std::string UpdateParamErrMsg(const std::string &param_name) {
  std::string msg = ConcatString("update [", param_name, "] failed");
  return msg;
}

std::string GetParamOutRangeErrMsg(const std::string &param_name, const std::string &range,
                                   const std::string &real_value) {
  std::string msg = ConcatString("the parameter [", param_name, "] should be in the range of [", range,
                                 "], but actually is ", real_value);
  return msg;
}

std::string OtherErrMsg(const std::string &error_detail) {
  std::string msg = error_detail;
  return msg;
}
}  // namespace ge
