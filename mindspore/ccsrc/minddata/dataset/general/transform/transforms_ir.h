/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_DATA_TRANSFORMS_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_DATA_TRANSFORMS_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/kernels/tensor_operation.h"

namespace mindspore {
namespace dataset {
// Transform operations for performing data transformation.
namespace transforms {
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kComposeOperation[] = "Compose";
constexpr char kParseExampleOperation[] = "ParseExample";
constexpr char kPluginOperation[] = "Plugin";
constexpr char kPreBuiltOperation[] = "PreBuilt";
constexpr char kTypeCastOperation[] = "TypeCast";
/* ####################################### Derived TensorOperation classes ################################# */

class ComposeOperation : public TensorOperation {
 public:
  explicit ComposeOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

  ~ComposeOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kComposeOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation);

  // Get the compose type: kInvalid / kAscend910B / kCpu.
  virtual MapTargetDevice Type() {
    bool have_dvpp = false;
    bool have_cpu = false;
    for (auto &item : transforms_) {
      if (item->Type() == MapTargetDevice::kAscend910B) {
        have_dvpp = true;
      } else if (item->Type() == MapTargetDevice::kCpu) {
        have_cpu = true;
      } else {
        MS_LOG(ERROR) << "The transform: " << item->Name() << " is not Ascend or Cpu.";
        return MapTargetDevice::kInvalid;
      }
    }

    if (have_dvpp && have_cpu) {
      MS_LOG(ERROR) << "Currently, it is not supported to mix DVPP transforms with CPU transforms in Compose.";
      return MapTargetDevice::kInvalid;
    } else if (have_dvpp) {
      return MapTargetDevice::kAscend910B;
    } else {
      return MapTargetDevice::kCpu;
    }
  }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class ParseExampleOperation : public TensorOperation {
 public:
  ParseExampleOperation(DataSchema schema, std::vector<std::string> column_list, bool parallel_parse);

  ~ParseExampleOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  std::string Name() const override { return kParseExampleOperation; }

 private:
  DataSchema schema_;
  std::vector<std::string> column_list_;
  bool parallel_parse_;
};

class PreBuiltOperation : public TensorOperation {
 public:
  explicit PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op);

  ~PreBuiltOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::shared_ptr<TensorOp> op_;
};

class TypeCastOperation : public TensorOperation {
 public:
  explicit TypeCastOperation(const DataType &data_type);  // Used for C++ API

  explicit TypeCastOperation(const std::string &data_type);  // Used for Pybind

  ~TypeCastOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kTypeCastOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  DataType data_type_;
};

class PluginOperation : public TensorOperation {
 public:
  explicit PluginOperation(const std::string &lib_path, const std::string &func_name, const std::string &user_args)
      : lib_path_(lib_path), func_name_(func_name), user_args_(user_args) {}

  ~PluginOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPluginOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::string lib_path_;
  std::string func_name_;
  std::string user_args_;
};
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_DATA_TRANSFORMS_IR_H_
