/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ADAPTER_INFO_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ADAPTER_INFO_H_

#include <vector>
#include <string>
#include <map>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/hash_map.h"
#include "include/transform/graph_ir/types.h"

namespace mindspore {
namespace transform {
typedef enum { ALLOW_FP32_TO_FP16, FORCE_FP32 } AclPrecisionMode;

struct AclSpecialInfo {
  std::vector<std::string> ori_format{};
  std::vector<std::string> dev_format{};
  std::string reshape_type{};
};

class AclAdapterInfo {
 public:
  explicit AclAdapterInfo(const std::string &op_type) : op_type_(op_type) {}
  ~AclAdapterInfo() = default;

  AclAdapterInfo &Input(size_t index, const std::vector<std::string> &ori_format = {},
                        const std::vector<std::string> &dev_format = {}, const std::string &reshape_type = {}) {
    AclSpecialInfo info;
    info.ori_format = ori_format;
    info.dev_format = dev_format;
    info.reshape_type = reshape_type;
    (void)input_info_.emplace(index, info);
    return *this;
  }

  AclAdapterInfo &set_is_3d_ops() {
    is_3d_ops_ = true;
    return *this;
  }

  AclAdapterInfo &set_is_need_retrieve_output_shape() {
    is_need_retrieve_output_shape_ = true;
    return *this;
  }

  AclAdapterInfo &set_run_mode(bool is_dynamic) {
    is_dynamic_ = is_dynamic;
    return *this;
  }

  AclAdapterInfo &set_precision_mode(AclPrecisionMode precision_mode) {
    precision_mode_ = precision_mode;
    return *this;
  }

  AclAdapterInfo &set_extra_supported_datatype(const std::vector<ge::DataType> &supported_datatype) {
    extra_supported_datatype_ = supported_datatype;
    return *this;
  }

  const std::string &op_type() const { return op_type_; }
  const bool &is_3d() const { return is_3d_ops_; }
  const bool &is_need_retrieve_output_shape() const { return is_need_retrieve_output_shape_; }
  const bool &is_dynamic() const { return is_dynamic_; }
  const AclPrecisionMode &precision_mode() const { return precision_mode_; }
  const std::map<size_t, AclSpecialInfo> &inputs() const { return input_info_; }
  const std::vector<ge::DataType> &extra_supported_datatype() const { return extra_supported_datatype_; }

 private:
  std::string op_type_;
  bool is_3d_ops_{false};
  bool is_need_retrieve_output_shape_{false};
  bool is_dynamic_{true};
  AclPrecisionMode precision_mode_{ALLOW_FP32_TO_FP16};  // 910 default mix precision.
  std::map<size_t, AclSpecialInfo> input_info_{};
  std::vector<ge::DataType> extra_supported_datatype_{};
};

class AclAdapterManager {
 public:
  static AclAdapterManager &GetInstance();
  AclAdapterInfo &Register(const std::string &op_type);

  bool CheckAclAdapter(const std::string &op_type);
  const AclAdapterInfo &GetOpInfo(const std::string &op_type) const;

 private:
  AclAdapterManager() = default;
  ~AclAdapterManager() = default;
  mindspore::HashMap<std::string, AclAdapterInfo> op_cache_;
};

#define REGISTER_ACL_IMPL(ctr, name) \
  static transform::AclAdapterInfo &register_acl##name##ctr = AclAdapterManager::GetInstance().Register(#name)

#define REGISTER_ACL_OP(name) REGISTER_ACL_IMPL(__COUNTER__, name)
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_ADAPTER_INFO_H_
