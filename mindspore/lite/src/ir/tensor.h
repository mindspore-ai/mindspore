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

#ifndef MINDSPORE_LITE_SRC_IR_TENSOR_H_
#define MINDSPORE_LITE_SRC_IR_TENSOR_H_

#include <memory>
#include <vector>
#include <string>
#include "ir/meta_tensor.h"
#include "include/ms_tensor.h"
#include "ir/dtype/type_id.h"
#include "src/runtime/allocator.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
namespace tensor {

struct QuantArg {
  double scale;
  int32_t zeroPoint;
};

class Tensor : public mindspore::tensor::MetaTensor {
 public:
  Tensor() : MetaTensor() {}

  Tensor(const TypeId data_type, const std::vector<int> &shape, const schema::Format &format = schema::Format_NHWC,
         schema::NodeType tensorType = schema::NodeType_Parameter);

  Tensor(const Tensor &tensor);

  ~Tensor() override;

  int CopyTensorData(const Tensor &srcTensor);

  int CopyTensor(const Tensor &srcTensor, bool copyData = false);

  MS_DECLARE_PARENT(Tensor, MetaTensor)

  virtual Tensor &operator=(const Tensor &tensor);

  virtual bool operator==(const Tensor &tensor);

  bool operator==(const Value &other) const override;

  int32_t Batch() const;

  int32_t Channel() const;

  int32_t Height() const;

  int32_t Width() const;

  int32_t ElementsC4Num() const;

  int DataSize() const { return this->ElementsNum(); }

  size_t Size() const {
    size_t size = 0;
    switch (this->data_type_) {
      case kNumberTypeFloat64:
        size = sizeof(double);
        break;
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        size = sizeof(float);
        break;
      case kNumberTypeInt8:
        size = sizeof(int8_t);
        break;
      case kNumberTypeUInt8:
        size = sizeof(uint8_t);
        break;
      case kNumberTypeFloat16:
        size = sizeof(int16_t);
        break;
      case kNumberTypeInt16:
        size = sizeof(int16_t);
        break;
      case kNumberTypeInt32:
        size = sizeof(int32_t);
        break;
      case kNumberTypeInt64:
        size = sizeof(int64_t);
        break;
      case kNumberTypeUInt16:
        size = sizeof(uint16_t);
        break;
      case kNumberTypeUInt32:
        size = sizeof(uint32_t);
        break;
      case kNumberTypeUInt64:
        size = sizeof(uint64_t);
        break;
      case kNumberTypeBool:
        size = sizeof(bool);
        break;
      default:
        MS_LOG(ERROR) << "Not support the type: " << this->data_type_;
        return 0;
    }
    size *= (format_ == schema::Format_NC4HW4 || format_ == schema::Format_NHWC4) ? ElementsC4Num()
                                                                                  : MetaTensor::ElementsNum();

    return size;
  }

  void set_allocator(mindspore::lite::Allocator *allocator) { allocator_ = allocator; }

  int MallocData(mindspore::lite::Allocator *allocator = nullptr) {
    if (nullptr != this->data_) {
      return 0;
    }
    if (allocator != nullptr) {
      allocator_ = allocator;
    }
    if (allocator_ == nullptr) {
      this->data_ = malloc(this->Size());
    } else {
      this->data_ = allocator_->Malloc(this->Size());
    }
    if (nullptr == this->data_) {
      MS_LOG(ERROR) << "Malloc tensor data failed, size=" << this->Size();
      return -1;
    }

    return 0;
  }

  int FreeData() {
    if (nullptr == this->data_) {
      return 0;
    }
    if (nullptr == allocator_) {
      free(this->data_);
    } else {
      allocator_->Free(this->data_);
      this->data_ = nullptr;
    }

    return 0;
  }

  void *Data() { return data_; }

  void SetData(void *data) { this->data_ = data; }

  schema::NodeType TensorType() { return this->tensorType; }

  void SetFormat(schema::Format format) { this->format_ = format; }

  schema::Format GetFormat() { return this->format_; }

  size_t RefCount() { return this->refCount; }

  void SetRefCount(size_t refCount) { this->refCount = refCount; }

  void decRefCount() { this->refCount--; }

  std::string ToString() const override;

  void AddQuantParam(const tensor::QuantArg &quant_arg);

  std::vector<tensor::QuantArg> GetQuantParams() const;

 protected:
  void *data_ = nullptr;
  void *device_data_ = nullptr;
  schema::NodeType tensorType;
  schema::Format format_;
  size_t refCount = 0;
  std::vector<tensor::QuantArg> quant_params_;
  mindspore::lite::Allocator *allocator_ = nullptr;
};

class LiteTensor : public mindspore::tensor::MSTensor {
 public:
  LiteTensor();

  LiteTensor(TypeId data_type, const std::vector<int> &shape);

  explicit LiteTensor(tensor::Tensor *tensor_ptr);

  ~LiteTensor() override;

  TypeId data_type() const override;

  TypeId set_data_type(TypeId data_type) override;

  std::vector<int> shape() const override;

  size_t set_shape(const std::vector<int> &shape) override;

  int DimensionSize(size_t index) const override;

  int ElementsNum() const override;

  std::size_t hash() const override;

  tensor::Tensor *tensor() const;

  size_t Size() const override;

  void *MutableData() const override;

  void SetTensorImpl(tensor::Tensor *tensor);

 protected:
  tensor::Tensor *tensor_impl_;
};

using TensorPtr = std::shared_ptr<tensor::Tensor>;
}  // namespace tensor
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_IR_TENSOR_H_
