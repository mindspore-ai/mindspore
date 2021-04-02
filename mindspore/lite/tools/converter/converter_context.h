/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_RETURN_CODE_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_RETURN_CODE_H

#include <string>
#include <set>
#include <map>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace lite {
class ReturnCode {
 public:
  ~ReturnCode() = default;
  static ReturnCode *GetSingleReturnCode() {
    static ReturnCode returnCode;
    return &returnCode;
  }
  void UpdateReturnCode(STATUS status) {
    if (statusCode == RET_OK) {
      statusCode = status;
    }
  }
  STATUS GetReturnCode() const { return statusCode; }

 private:
  ReturnCode() { statusCode = RET_OK; }
  int statusCode;
};

class NoSupportOp {
 public:
  ~NoSupportOp() = default;
  static NoSupportOp *GetInstance() {
    static NoSupportOp noSupportOp;
    return &noSupportOp;
  }
  void SetFmkType(const std::string &fmk_type) { fmkType = fmk_type; }
  void InsertOp(const std::string &op_name) { noSupportOps.insert(op_name); }
  void PrintOps() const {
    if (!noSupportOps.empty()) {
      MS_LOG(ERROR) << "===========================================";
      MS_LOG(ERROR) << "UNSUPPORTED OP LIST:";
      for (auto &op_name : noSupportOps) {
        MS_LOG(ERROR) << "FMKTYPE: " << fmkType << ", OP TYPE: " << op_name;
      }
      MS_LOG(ERROR) << "===========================================";
    }
  }

 private:
  NoSupportOp() { noSupportOps.clear(); }
  std::set<std::string> noSupportOps;
  std::string fmkType;
};

class TensorDataType {
 public:
  ~TensorDataType() = default;
  static TensorDataType *GetInstance() {
    static TensorDataType tensorDataType;
    return &tensorDataType;
  }
  void UpdateTensorType(int32_t index, int32_t type) { tensorDataTypeMap[index] = type; }
  int32_t GetTensorType(int32_t index) const {
    if (tensorDataTypeMap.find(index) == tensorDataTypeMap.end()) {
      return TypeId::kTypeUnknown;
    }
    return tensorDataTypeMap.at(index);
  }

 private:
  TensorDataType() {}
  std::map<int32_t, int32_t> tensorDataTypeMap;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_RETURN_CODE_H
