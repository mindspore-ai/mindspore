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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPLIB_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPLIB_H_
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <nlohmann/json.hpp>
#include "utils/ms_utils.h"
#include "kernel/oplib/opinfo.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
class BACKEND_EXPORT OpLib {
 public:
  OpLib() = default;
  virtual ~OpLib() = default;
  static bool RegOp(const std::string &json_string, const std::string &impl_path);
  static bool LoadSuperBarConfig(const std::string &suber_bar_config);
  static std::shared_ptr<OpInfo> FindOp(const std::string &op_name, OpImplyType imply_type,
                                        bool is_dynamic_shape = false);
  static std::map<mindspore::kernel::OpImplyType, std::map<std::string, std::shared_ptr<OpInfo>>> &GetOpInfoMap();

 private:
  static bool RegOpFromLocalInfo();
  static std::shared_ptr<OpInfo> DecodeOpInfo(const nlohmann::json &obj, const OpImplyType &imply_type,
                                              const std::string &impl_path);
  static bool DecodeAttr(const nlohmann::json &obj, const OpImplyType &imply_type,
                         const std::shared_ptr<OpInfo> &op_info);
  static bool DecodeDtypeFormat(const nlohmann::json &dtype_format, const std::shared_ptr<OpIOInfo> &op_io,
                                size_t index);
  static void DecodeTBESpecificInfo(const nlohmann::json &obj, const std::shared_ptr<OpInfo> &op_info);
  static bool DecodeInputOutput(const nlohmann::json &obj, OpImplyType imply_type, bool is_input,
                                const std::shared_ptr<OpInfo> &op_info, const nlohmann::json &dtype_format);
  static bool GetRefInfo(const std::shared_ptr<OpInfo> &op_info);

  friend class OpInfoUtils;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPLIB_H_
