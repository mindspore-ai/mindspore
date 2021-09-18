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

  ~ComposeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kComposeOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class ConcatenateOperation : public TensorOperation {
 public:
  explicit ConcatenateOperation(int8_t axis, const std::shared_ptr<Tensor> &prepend,
                                const std::shared_ptr<Tensor> &append);

  ~ConcatenateOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kConcatenateOperation; }

 private:
  int8_t axis_;
  std::shared_ptr<Tensor> prepend_;
  std::shared_ptr<Tensor> append_;
};

class DuplicateOperation : public TensorOperation {
 public:
  DuplicateOperation() = default;

  ~DuplicateOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDuplicateOperation; }
};

class FillOperation : public TensorOperation {
 public:
  explicit FillOperation(const std::shared_ptr<Tensor> &fill_value);

  ~FillOperation() = default;

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
  explicit MaskOperation(RelationalOp op, const std::shared_ptr<Tensor> &constant, const DataType &dtype);

  ~MaskOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kMaskOperation; }

 private:
  RelationalOp op_;
  std::shared_ptr<Tensor> constant_;
  DataType dtype_;
};

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes);

  ~OneHotOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kOneHotOperation; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_classes_;
};

class PadEndOperation : public TensorOperation {
 public:
  explicit PadEndOperation(const TensorShape &pad_shape, const std::shared_ptr<Tensor> &pad_value);

  ~PadEndOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPadEndOperation; }

 private:
  TensorShape pad_shape_;
  std::shared_ptr<Tensor> pad_value_;
};

class PreBuiltOperation : public TensorOperation {
 public:
  explicit PreBuiltOperation(std::shared_ptr<TensorOp> tensor_op);

  ~PreBuiltOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::shared_ptr<TensorOp> op_;
};

class RandomApplyOperation : public TensorOperation {
 public:
  explicit RandomApplyOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms, double prob);

  ~RandomApplyOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomApplyOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  double prob_;
};

class RandomChoiceOperation : public TensorOperation {
 public:
  explicit RandomChoiceOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms);

  ~RandomChoiceOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRandomChoiceOperation; }

 private:
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

class SliceOperation : public TensorOperation {
 public:
  explicit SliceOperation(const std::vector<SliceOption> &slice_input);

  ~SliceOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSliceOperation; }

 private:
  std::vector<SliceOption> slice_input_;
};

class TypeCastOperation : public TensorOperation {
 public:
  explicit TypeCastOperation(const DataType &data_type);     // Used for C++ API
  explicit TypeCastOperation(const std::string &data_type);  // Used for Pybind

  ~TypeCastOperation() = default;

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

  ~UniqueOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUniqueOperation; }
};

class PluginOperation : public TensorOperation {
 public:
  explicit PluginOperation(const std::string &lib_path, const std::string &func_name, const std::string &user_args)
      : lib_path_(lib_path), func_name_(func_name), user_args_(user_args) {}

  ~PluginOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kPluginOperation; }

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
