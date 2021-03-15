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

#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kComposeOperation[] = "Compose";
constexpr char kDuplicateOperation[] = "Duplicate";
constexpr char kOneHotOperation[] = "OneHot";
constexpr char kPreBuiltOperation[] = "PreBuilt";
constexpr char kRandomApplyOperation[] = "RandomApply";
constexpr char kRandomChoiceOperation[] = "RandomChoice";
constexpr char kTypeCastOperation[] = "TypeCast";
constexpr char kUniqueOperation[] = "Unique";

// Transform operations for performing data transformation.
namespace transforms {
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

class DuplicateOperation : public TensorOperation {
 public:
  DuplicateOperation() = default;

  ~DuplicateOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDuplicateOperation; }
};

class OneHotOperation : public TensorOperation {
 public:
  explicit OneHotOperation(int32_t num_classes);

  ~OneHotOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kOneHotOperation; }

 private:
  float num_classes_;
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
class TypeCastOperation : public TensorOperation {
 public:
  explicit TypeCastOperation(std::string data_type);

  ~TypeCastOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kTypeCastOperation; }

 private:
  std::string data_type_;
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
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_DATA_TRANSFORMS_IR_H_
