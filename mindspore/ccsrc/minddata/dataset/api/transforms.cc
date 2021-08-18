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

#include "minddata/dataset/include/dataset/transforms.h"

#include <algorithm>

#include "mindspore/ccsrc/minddata/dataset/core/type_id.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

namespace mindspore {
namespace dataset {

// Transform operations for data.
namespace transforms {

// API CLASS FOR DATA TRANSFORM OPERATIONS
// (In alphabetical order)

// Constructor to Compose.
struct Compose::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

Compose::Compose(const std::vector<TensorTransform *> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::shared_ptr<TensorTransform>> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

Compose::Compose(const std::vector<std::reference_wrapper<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> Compose::Parse() { return std::make_shared<ComposeOperation>(data_->transforms_); }

// Constructor to Concatenate
struct Concatenate::Data {
  explicit Data(int8_t axis, const MSTensor &prepend, const MSTensor &append)
      : axis_(axis), prepend_(prepend), append_(append) {}
  int8_t axis_;
  MSTensor prepend_;
  MSTensor append_;
};

Concatenate::Concatenate(int8_t axis, const MSTensor &prepend, const MSTensor &append)
    : data_(std::make_shared<Data>(axis, prepend, append)) {}

std::shared_ptr<TensorOperation> Concatenate::Parse() {
#ifndef ENABLE_ANDROID
  std::shared_ptr<Tensor> out_prepend, out_append;
  Status rc = Tensor::CreateFromMSTensor(data_->prepend_, &out_prepend);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating prepend constant tensor. " << rc;
    return nullptr;
  }
  rc = Tensor::CreateFromMSTensor(data_->append_, &out_append);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating append constant tensor. " << rc;
    return nullptr;
  }
  return std::make_shared<ConcatenateOperation>(data_->axis_, out_prepend, out_append);
#else
  MS_LOG(ERROR) << "Concatenate op is not supported for Android.";
  return nullptr;
#endif  // not ENABLE_ANDROID
}

// Constructor to Duplicate
Duplicate::Duplicate() {}

std::shared_ptr<TensorOperation> Duplicate::Parse() { return std::make_shared<DuplicateOperation>(); }

// Constructor to Fill
struct Fill::Data {
  explicit Data(const MSTensor &fill_value) : fill_value_(fill_value) {}
  MSTensor fill_value_;
};

Fill::Fill(const MSTensor &fill_value) : data_(std::make_shared<Data>(fill_value)) {}

std::shared_ptr<TensorOperation> Fill::Parse() {
#ifndef ENABLE_ANDROID
  std::shared_ptr<Tensor> out_fill_value;
  Status rc = Tensor::CreateFromMSTensor(data_->fill_value_, &out_fill_value);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating fill value tensor. " << rc;
    return nullptr;
  }
  return std::make_shared<FillOperation>(out_fill_value);
#else
  MS_LOG(ERROR) << "Fill op is not supported for Android.";
  return nullptr;
#endif  // not ENABLE_ANDROID
}

// Constructor to Mask
struct Mask::Data {
  explicit Data(RelationalOp op, const MSTensor &constant, mindspore::DataType ms_type)
      : op_(op), constant_(constant), ms_type_(ms_type) {}
  RelationalOp op_;
  MSTensor constant_;
  mindspore::DataType ms_type_;
};

Mask::Mask(RelationalOp op, const MSTensor &constant, mindspore::DataType ms_type)
    : data_(std::make_shared<Data>(op, constant, ms_type)) {}

std::shared_ptr<TensorOperation> Mask::Parse() {
#ifndef ENABLE_ANDROID
  std::shared_ptr<Tensor> out_constant;
  Status rc = Tensor::CreateFromMSTensor(data_->constant_, &out_constant);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating constant tensor. " << rc;
    return nullptr;
  }

  DataType de_type = dataset::MSTypeToDEType(static_cast<TypeId>(data_->ms_type_));
  return std::make_shared<MaskOperation>(data_->op_, out_constant, de_type);
#else
  MS_LOG(ERROR) << "Mask op is not supported for Android.";
  return nullptr;
#endif  // not ENABLE_ANDROID
}

// Constructor to OneHot
struct OneHot::Data {
  explicit Data(int32_t num_classes) : num_classes_(num_classes) {}
  int32_t num_classes_;
};

OneHot::OneHot(int32_t num_classes) : data_(std::make_shared<Data>(num_classes)) {}

std::shared_ptr<TensorOperation> OneHot::Parse() { return std::make_shared<OneHotOperation>(data_->num_classes_); }

// Constructor to PadEnd
struct PadEnd::Data {
  explicit Data(const std::vector<dsize_t> &pad_shape, const MSTensor &pad_value)
      : pad_shape_(pad_shape), pad_value_(pad_value) {}
  std::vector<dsize_t> pad_shape_;
  MSTensor pad_value_;
};

PadEnd::PadEnd(const std::vector<dsize_t> &pad_shape, const MSTensor &pad_value)
    : data_(std::make_shared<Data>(pad_shape, pad_value)) {}

std::shared_ptr<TensorOperation> PadEnd::Parse() {
#ifndef ENABLE_ANDROID
  std::shared_ptr<Tensor> pad_value;
  Status rc = Tensor::CreateFromMSTensor(data_->pad_value_, &pad_value);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error creating value constant tensor. " << rc;
    return nullptr;
  }
  return std::make_shared<PadEndOperation>(TensorShape(data_->pad_shape_), pad_value);
#else
  MS_LOG(ERROR) << "PadEnd op is not supported for Android.";
  return nullptr;
#endif  // not ENABLE_ANDROID
}

// Constructor to RandomApply.
struct RandomApply::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
  double prob_;
};

RandomApply::RandomApply(const std::vector<TensorTransform *> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
  data_->prob_ = prob;
}

RandomApply::RandomApply(const std::vector<std::shared_ptr<TensorTransform>> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
  data_->prob_ = prob;
}

RandomApply::RandomApply(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, double prob)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
  data_->prob_ = prob;
}

std::shared_ptr<TensorOperation> RandomApply::Parse() {
  return std::make_shared<RandomApplyOperation>(data_->transforms_, data_->prob_);
}

// Constructor to RandomChoice.
struct RandomChoice::Data {
  std::vector<std::shared_ptr<TensorOperation>> transforms_;
};

RandomChoice::RandomChoice(const std::vector<TensorTransform *> &transforms) : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform *const op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomChoice::RandomChoice(const std::vector<std::shared_ptr<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](const std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                         return op != nullptr ? op->Parse() : nullptr;
                       });
}

RandomChoice::RandomChoice(const std::vector<std::reference_wrapper<TensorTransform>> &transforms)
    : data_(std::make_shared<Data>()) {
  (void)std::transform(transforms.begin(), transforms.end(), std::back_inserter(data_->transforms_),
                       [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
}

std::shared_ptr<TensorOperation> RandomChoice::Parse() {
  return std::make_shared<RandomChoiceOperation>(data_->transforms_);
}

// Constructor to Slice
struct Slice::Data {
  explicit Data(const std::vector<SliceOption> &slice_input) : slice_input_(slice_input) {}
  std::vector<SliceOption> slice_input_;
};

Slice::Slice(const std::vector<SliceOption> &slice_input) : data_(std::make_shared<Data>(slice_input)) {}

std::shared_ptr<TensorOperation> Slice::Parse() {
#ifndef ENABLE_ANDROID
  return std::make_shared<SliceOperation>(data_->slice_input_);
#else
  MS_LOG(ERROR) << "Slice op is not supported for Android.";
  return nullptr;
#endif  // not ENABLE_ANDROID
}

// Constructor to TypeCast
struct TypeCast::Data {
  dataset::DataType data_type_;
};

TypeCast::TypeCast(mindspore::DataType data_type) : data_(std::make_shared<Data>()) {
  data_->data_type_ = dataset::MSTypeToDEType(static_cast<TypeId>(data_type));
}

std::shared_ptr<TensorOperation> TypeCast::Parse() { return std::make_shared<TypeCastOperation>(data_->data_type_); }

// Constructor to Unique
Unique::Unique() {}

#ifndef ENABLE_ANDROID
std::shared_ptr<TensorOperation> Unique::Parse() { return std::make_shared<UniqueOperation>(); }
#else
std::shared_ptr<TensorOperation> Unique::Parse() {
  MS_LOG(ERROR) << "Unique op is not supported for Android.";
  return nullptr;
}
#endif
}  // namespace transforms
}  // namespace dataset
}  // namespace mindspore
