/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
// Transform operations for performing data transformation.
namespace transforms {
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kComposeOperation[] = "Compose";
constexpr char kConcatenateOperation[] = "Concatenate";
constexpr char kDuplicateOperation[] = "Duplicate";
constexpr char kFillOperation[] = "Fill";
constexpr char kMaskOperation[] = "Mask";
constexpr char kOneHotOperation[] = "OneHot";
constexpr char kPadEndOperation[] = "PadEnd";
constexpr char kPreBuiltOperation[] = "PreBuilt";
constexpr char kSliceOperation[] = "Slice";
constexpr char kRandomApplyOperation[] = "RandomApply";
constexpr char kRandomChoiceOperation[] = "RandomChoice";
constexpr char kTypeCastOperation[] = "TypeCast";
constexpr char kUniqueOperation[] = "Unique";
constexpr char kPluginOperation[] = "Plugin";
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

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class ConcatenateOperation : public TensorOperation {
 public:
  ConcatenateOperation(int8_t axis, const std::shared_ptr<Tensor> &prepend, const std::shared_ptr<Tensor> &append);

  ~ConcatenateOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kConcatenateOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int8_t axis_;
  std::shared_ptr<Tensor> prepend_;
  std::shared_ptr<Tensor> append_;
};

class DuplicateOperation : public TensorOperation {
 public:
  DuplicateOperation() = default;

  ~DuplicateOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDuplicateOperation; }

  static Status from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation);
};

class FillOperation : public TensorOperation {
 public:
  explicit FillOperation(const std::shared_ptr<Tensor> &fill_value);

  ~FillOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kFillOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::shared_ptr<Tensor> fill_value_;
};

class MaskOperation : public TensorOperation {
 public:
  MaskOperation(RelationalOp op, const std::shared_ptr<Tensor> &constant, const DataType &dtype);

  ~MaskOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kMaskOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(const nlohmann::json &op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  RelationalOp op_;
  std::shared_ptr<Tensor> constant_;
  DataType dtype_;
};

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes, double smoothing_rate = 0.0);

  ~OneHotOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kOneHotOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_classes_;
  double smoothing_rate_;
};

class PadEndOperation : public TensorOperation {
 public:
  PadEndOperation(const TensorShape &pad_shape, const std::shared_ptr<Tensor> &pad_value);

  ~PadEndOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPadEndOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  TensorShape pad_shape_;
  std::shared_ptr<Tensor> pad_value_;
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

class RandomApplyOperation : public TensorOperation {
 public:
  RandomApplyOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms, double prob);

  ~RandomApplyOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomApplyOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  double prob_;
};

class RandomChoiceOperation : public TensorOperation {
 public:
  explicit RandomChoiceOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

  ~RandomChoiceOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomChoiceOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class SliceOperation : public TensorOperation {
 public:
  explicit SliceOperation(const std::vector<SliceOption> &slice_input);

  ~SliceOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSliceOperation; }

 private:
  std::vector<SliceOption> slice_input_;
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

#ifndef ENABLE_ANDROID
class UniqueOperation : public TensorOperation {
 public:
  UniqueOperation() = default;

  ~UniqueOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUniqueOperation; }

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);
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
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_DATA_TRANSFORMS_IR_H_
