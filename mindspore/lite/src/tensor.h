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
#include <numeric>
#include <functional>
#include "include/ms_tensor.h"
#include "src/runtime/allocator.h"

#include "src/common/log_adapter.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
struct QuantArg {
  double scale;
  int32_t zeroPoint;
  double var_corr{1};
  double mean_corr{0};
  std::vector<float> clusters{};
};

class Tensor : public mindspore::tensor::MSTensor {
 public:
  enum Category {
    CONST,  // weight tensor
    VAR     // activation tensor
  };
  Tensor() = default;

  Tensor(const TypeId data_type, const std::vector<int> &shape,
         const schema::Format &format = schema::Format::Format_NHWC, Category category = VAR);

  Tensor(const Tensor &tensor);

  virtual ~Tensor();

  int CopyTensorData(const Tensor &srcTensor);

  int CopyTensor(const Tensor &srcTensor, bool copyData = false);

  virtual Tensor &operator=(const Tensor &tensor);

  virtual bool operator==(const Tensor &tensor);

  TypeId data_type() const override { return data_type_; }

  void set_data_type(TypeId data_type) { data_type_ = data_type; }

  std::vector<int> shape() const override { return shape_; }

  void set_shape(const std::vector<int> &shape) { shape_ = shape; }

  int DimensionSize(size_t index) const override {
    int dim_size = -1;
    if (index < shape_.size()) {
      dim_size = shape_[index];
    } else {
      MS_LOG(ERROR) << "Dimension index is wrong: " << index;
    }
    return dim_size;
  }

  int ElementsNum() const override {
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int>());
  }

  int32_t Batch() const;

  int32_t Channel() const;

  int32_t Height() const;

  int32_t Width() const;

  int32_t ElementsC4Num() const;

  size_t Size() const override {
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
      case kObjectTypeString:
        size = sizeof(char);
        break;
      default:
        MS_LOG(ERROR) << "Not support the type: " << this->data_type_;
        return 0;
    }
    size *= (format_ == schema::Format::Format_NC4HW4 || format_ == schema::Format::Format_NHWC4) ? ElementsC4Num()
                                                                                                  : ElementsNum();

    return size;
  }

  void set_allocator(mindspore::lite::Allocator *allocator) { allocator_ = allocator; }

  int MallocData(mindspore::lite::Allocator *allocator = nullptr);

  int FreeData();

  void *MutableData() override;

  void *data_c() const { return data_; }

  void SetData(void *data) { this->data_ = data; }

  Category category() { return this->category_; }

  void SetFormat(schema::Format format) { this->format_ = format; }

  schema::Format GetFormat() { return this->format_; }

  size_t RefCount() { return this->refCount; }

  void SetRefCount(size_t refCount) { this->refCount = refCount; }

  void decRefCount() { this->refCount--; }

  std::string ToString() const;

  void AddQuantParam(const QuantArg &quant_arg);

  std::vector<QuantArg> GetQuantParams() const;

  void Prepare() {
    if (allocator_ != nullptr) {
      data_ = allocator_->Prepare(data_);
    }
  }

 protected:
  void *data_ = nullptr;
  void *device_data_ = nullptr;
  TypeId data_type_;
  std::vector<int> shape_;
  schema::Format format_;
  Category category_;
  size_t refCount = 0;
  std::vector<QuantArg> quant_params_;
  mindspore::lite::Allocator *allocator_ = nullptr;
};

inline Tensor::Category TensorCategory(const schema::Tensor *tensor) {
  return (tensor->nodeType() == schema::NodeType::NodeType_ValueNode) ? Tensor::Category::CONST : Tensor::Category::VAR;
}
inline Tensor::Category TensorCategory(const schema::NodeType type) {
  return (type == schema::NodeType::NodeType_ValueNode) ? Tensor::Category::CONST : Tensor::Category::VAR;
}
std::vector<tensor::MSTensor *> TensorVectorCast(const std::vector<Tensor *> &src);
}  // namespace lite
}  // namespace mindspore

using TensorPtr = std::shared_ptr<mindspore::lite::Tensor>;
#endif  // MINDSPORE_LITE_SRC_IR_TENSOR_H_
