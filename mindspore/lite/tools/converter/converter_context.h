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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_CONTEXT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_CONTEXT_H_

#include <string>
#include <set>
#include <map>
#include <vector>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "ir/dtype/type_id.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace lite {
class ReturnCode {
 public:
  static ReturnCode *GetSingleReturnCode() {
    static ReturnCode return_code;
    return &return_code;
  }
  void UpdateReturnCode(STATUS status) {
    if (status_code_ == RET_OK) {
      status_code_ = status;
    }
  }
  STATUS status_code() const { return status_code_; }

 private:
  ReturnCode() = default;
  virtual ~ReturnCode() = default;
  int status_code_ = RET_OK;
};

class NotSupportOp {
 public:
  static NotSupportOp *GetInstance() {
    static NotSupportOp not_support_op;
    return &not_support_op;
  }
  void set_fmk_type(const std::string &fmk_type) { fmk_type_ = fmk_type; }
  void InsertOp(const std::string &op_name) { (void)not_support_ops_.insert(op_name); }
  void PrintOps() const {
    if (!not_support_ops_.empty()) {
      MS_LOG(ERROR) << "===========================================";
      MS_LOG(ERROR) << "UNSUPPORTED OP LIST:";
      for (auto &op_name : not_support_ops_) {
        MS_LOG(ERROR) << "FMKTYPE: " << fmk_type_ << ", OP TYPE: " << op_name;
      }
      MS_LOG(ERROR) << "===========================================";
    }
  }

 private:
  NotSupportOp() = default;
  virtual ~NotSupportOp() = default;
  std::set<std::string> not_support_ops_;
  std::string fmk_type_;
};

class ConverterInnerContext {
 public:
  static ConverterInnerContext *GetInstance() {
    static ConverterInnerContext converter_context;
    return &converter_context;
  }

  void UpdateGraphInputDType(int32_t index, int32_t dtype) { graph_input_data_type_map_[index] = dtype; }
  int32_t GetGraphInputDType(int32_t index) const {
    if (graph_input_data_type_map_.find(index) == graph_input_data_type_map_.end()) {
      return TypeId::kTypeUnknown;
    }
    return graph_input_data_type_map_.at(index);
  }

  void UpdateGraphOutputDType(int32_t index, int32_t dtype) { graph_output_data_type_map_[index] = dtype; }
  int32_t GetGraphOutputDType(int32_t index) const {
    if (graph_output_data_type_map_.find(index) == graph_output_data_type_map_.end()) {
      return TypeId::kTypeUnknown;
    }
    return graph_output_data_type_map_.at(index);
  }

  void UpdateGraphInputTensorShape(const std::string &tensor_name, const std::vector<int64_t> &shape) {
    graph_input_tensor_shape_map_[tensor_name] = shape;
    MS_LOG(INFO) << "Update shape of input " << tensor_name << " to " << shape;
  }
  std::vector<int64_t> GetGraphInputTensorShape(const std::string &tensor_name) const {
    if (graph_input_tensor_shape_map_.find(tensor_name) == graph_input_tensor_shape_map_.end()) {
      return {};
    }
    return graph_input_tensor_shape_map_.at(tensor_name);
  }
  size_t GetGraphInputTensorShapeMapSize() const { return graph_input_tensor_shape_map_.size(); }

  void SetGraphOutputTensorNames(const std::vector<std::string> &output_names) {
    graph_output_tensor_names_ = output_names;
  }

  const std::vector<std::string> GetGraphOutputTensorNames() const { return graph_output_tensor_names_; }

  void SetExternalUsedConfigInfos(const std::string &section,
                                  const std::map<std::string, std::string> &external_infos) {
    for (auto const &external_info : external_infos) {
      if (external_used_config_infos_[section].find(external_info.first) !=
          external_used_config_infos_[section].end()) {
        MS_LOG(WARNING) << "This content " << external_info.first
                        << " has been saved. Now the value will be overwrite.";
      }
      external_used_config_infos_[section][external_info.first] = external_info.second;
    }
  }

  const std::map<std::string, std::map<std::string, std::string>> &GetExternalUsedConfigInfos() const {
    return external_used_config_infos_;
  }

  void SetTargetDevice(const std::string &target_device) { target_device_ = target_device; }
  std::string GetTargetDevice() const { return target_device_; }

 private:
  ConverterInnerContext() {
    (void)external_used_config_infos_.emplace(mindspore::converter::KCommonQuantParam,
                                              std::map<std::string, std::string>{});
    (void)external_used_config_infos_.emplace(mindspore::converter::KFullQuantParam,
                                              std::map<std::string, std::string>{});
    (void)external_used_config_infos_.emplace(mindspore::converter::KDataPreProcess,
                                              std::map<std::string, std::string>{});
    (void)external_used_config_infos_.emplace(mindspore::converter::KMixBitWeightQuantParam,
                                              std::map<std::string, std::string>{});
  }
  virtual ~ConverterInnerContext() = default;
  std::map<int32_t, int32_t> graph_input_data_type_map_;
  std::map<int32_t, int32_t> graph_output_data_type_map_;
  std::map<std::string, std::vector<int64_t>> graph_input_tensor_shape_map_;
  std::vector<std::string> graph_output_tensor_names_;
  std::map<std::string, std::map<std::string, std::string>> external_used_config_infos_;
  std::string target_device_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_CONTEXT_H_
