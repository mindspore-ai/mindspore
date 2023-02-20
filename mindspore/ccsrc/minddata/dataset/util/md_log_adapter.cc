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

#include "minddata/dataset/util/md_log_adapter.h"

#include <sstream>

namespace mindspore {
namespace dataset {
Status MDLogAdapter::Apply(Status *rc) {
  std::string status_msg = ConstructMsg(rc->StatusCode(), rc->CodeAsString(rc->StatusCode()), "", rc->GetLineOfCode(),
                                        rc->GetFileName(), rc->GetErrDescription());
  rc->SetStatusMsg(status_msg);
  return *rc;
}

std::string MDLogAdapter::ConstructMsg(const enum StatusCode &status_code, const std::string &code_as_string,
                                       const std::string &status_msg, const int line_of_code,
                                       const std::string &file_name, const std::string &err_description) {
  std::ostringstream ss;
  std::string kSplitLine = std::string(66, '-') + "\n";
  std::string err_ori = err_description;

  /// Python Runtime Error
  ss << code_as_string << ". \n\n";

  /// Python Stack
  std::string user_err;
  std::string user_stack;
  if (status_code == StatusCode::kMDPyFuncException) {
    std::string at_stack = "\n\nAt:\n";
    if (err_ori.find(at_stack) != std::string::npos) {
      user_stack = err_ori.substr(0, err_ori.find(at_stack));
      user_err = "Execute user Python code failed, check 'Python Call Stack' above.";
      ss << kSplitLine << "- Python Call Stack: \n" << kSplitLine;
      ss << user_stack << "\n\n";
    } else {
      user_err = err_ori;
    }
  }

  /// Summary Message
  ss << kSplitLine << "- Dataset Pipeline Error Message: \n" << kSplitLine;
  if (!user_err.empty()) {
    ss << "[ERROR] " + user_err + "\n\n";
  } else {
    user_err = err_description;
    if (*user_err.rbegin() != '.') {
      user_err += '.';
    }
    ss << "[ERROR] " + user_err + "\n\n";
  }

  /// C++ Stack
  if (!file_name.empty()) {
    ss << kSplitLine << "- C++ Call Stack: (For framework developers) \n" << kSplitLine;
    std::string cpp_trace = std::string(file_name) + "(" + std::to_string(line_of_code) + ").\n";
    ss << cpp_trace << "\n\n";
  }

  return ss.str();
}
}  //  namespace dataset
}  //  namespace mindspore
