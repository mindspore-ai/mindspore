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

#ifndef MINDSPORE_LITE_SRC_TENSOR_H_
#define MINDSPORE_LITE_SRC_TENSOR_H_

#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include "include/ms_tensor.h"
#include "src/runtime/allocator.h"

#include "src/common/log_adapter.h"
#include "schema/model_generated.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
struct QuantArg {
  double scale;
  int32_t zeroPoint;
  float var_corr{1};
  float mean_corr{0};
  bool inited;
  std::vector<float> clusters{};
  int bitNum;
  int roundType;
  int multiplier;
  int dstDtype;
};

class Tensor : public mindspore::tensor::MSTensor {
 public:
  enum Category {
    CONST_TENSOR,  // weight tensor
    CONST_SCALAR,  // weight scalar
    VAR,           // activation tensor
    GRAPH_INPUT,
    GRAPH_OUTPUT,
  };
  Tensor() = default;

  Tensor(TypeId data_type, std::vector<int> shape, const schema::Format &format = schema::Format::Format_NHWC,
         Category category = VAR);

  Tensor(const Tensor &tensor) = delete;

  Tensor(Tensor &&other) = delete;

  Tensor &operator=(const Tensor &tensor) = delete;

  Tensor &operator=(Tensor &&src) = delete;

  ~Tensor() override;

  static int CopyTensorData(const Tensor &src_tensor, Tensor *dst_tensor);

  static Tensor *CopyTensor(const Tensor &src_tensor, bool copy_data = false);

  virtual bool operator==(const Tensor &tensor);

  void set_tensor_name(const std::string &name) override { tensor_name_ = name; }

  std::string tensor_name() const override { return tensor_name_; }

  TypeId data_type() const override { return data_type_; }

  void set_data_type(TypeId data_type) { data_type_ = data_type; }

  std::vector<int> shape() const override { return shape_; }

  void set_shape(const std::vector<int> &shape) override { shape_ = shape; }

  int DimensionSize(size_t index) const;

  int ElementsNum() const override;

  int32_t Batch() const;

  int32_t Channel() const;

  int32_t Height() const;

  int32_t Width() const;

  int32_t ElementsC4Num() const;

  size_t Size() const override;

  void set_allocator(mindspore::Allocator *allocator) { allocator_ = allocator; }

  mindspore::Allocator *allocator() const { return this->allocator_; }

  virtual int MallocData(const mindspore::Allocator *allocator = nullptr);

  virtual void FreeData();

  void *MutableData() override;

  void *data() override { return this->data_; }

  virtual void *data_c() const {
    if (this->root_tensor_ != nullptr) {
      return this->root_tensor_->data_;
    }
    return data_;
  }

  void set_data(void *data) override { this->data_ = data; }

  Category category() const { return this->category_; }

  void set_category(Category category) { this->category_ = category; }

  void set_format(schema::Format format) { this->format_ = format; }

  schema::Format format() const { return this->format_; }

  size_t ref_count() const { return this->ref_count_; }

  size_t init_ref_count() const { return this->init_ref_count_; }

  void set_ref_count(size_t ref_count) { this->ref_count_ = ref_count; }

  void set_init_ref_count(size_t ref_count) { this->init_ref_count_ = ref_count; }

  void ResetRefCount() { this->ref_count_ = this->init_ref_count_; }

  void DecRefCount();

  std::string ToString() const;

  void AddQuantParam(const QuantArg &quant_arg);

  std::vector<QuantArg> quant_params() const;

  std::vector<float> quant_clusters() const;

  void set_quant_clusters(const std::vector<float> &clusters);

  bool enable_huffman_code() const;

  void set_enable_huffman_code(bool enable_huffman_code);

  virtual bool IsConst() const {
    return (this->category_ == CONST_TENSOR || this->category_ == CONST_SCALAR) && this->data_ != nullptr;
  }

  bool IsScalar() const { return this->category_ == CONST_SCALAR && this->data_ != nullptr; }

  bool IsGraphInput() const { return this->category_ == GRAPH_INPUT; }

  bool IsGraphOutput() const { return this->category_ == GRAPH_OUTPUT; }

  void Prepare() {
    if (allocator_ != nullptr) {
      data_ = allocator_->Prepare(data_);
    }
  }

  virtual int set_root_tensor(Tensor *tensor);

  Tensor *root_tensor() const { return this->root_tensor_; }

  bool IsReady() const {
    return this->IsConst() || (this->IsGraphInput() && this->data_ != nullptr) || this->ref_count_ >= 1;
  }

 private:
  template <typename T>
  std::string DataToString(void *data, size_t data_number) const {
    if (data == nullptr) {
      return "Data of tensor is nullptr";
    }
    std::ostringstream oss;
    auto casted_data = static_cast<T *>(data);
    for (size_t i = 0; i < 40 && i < data_number; i++) {
      oss << " " << casted_data[i];
    }
    return oss.str();
  }

 protected:
  std::string tensor_name_;
  void *data_ = nullptr;
  TypeId data_type_;
  std::vector<int> shape_;
  schema::Format format_;
  Category category_;
  size_t ref_count_ = 0;
  size_t init_ref_count_ = 0;
  std::vector<QuantArg> quant_params_;
  std::vector<float> quant_clusters_;
  mindspore::Allocator *allocator_ = nullptr;
  Tensor *root_tensor_ = nullptr;
  bool enable_huffman_code_ = false;
};

inline size_t DataTypeSize(const TypeId type) {
  switch (type) {
    case kNumberTypeFloat64:
      return sizeof(double);
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return sizeof(float);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeFloat16:
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeInt64:
      return sizeof(int64_t);
    case kNumberTypeUInt16:
      return sizeof(uint16_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    case kNumberTypeUInt64:
      return sizeof(uint64_t);
    case kNumberTypeBool:
      return sizeof(bool);
    case kObjectTypeString:
      return sizeof(char);
    case kObjectTypeTensorType:
      return 0;
    default:
      MS_LOG(ERROR) << "Not support the type: " << type;
      return 0;
  }
}

inline Tensor::Category TensorCategory(const int node_type, const size_t shape_num, const TypeId data_type,
                                       const size_t data_size) {
  return (node_type == NodeType_ValueNode)
           ? (shape_num == 0 && data_size == DataTypeSize(data_type) ? Tensor::Category::CONST_SCALAR
                                                                     : Tensor::Category::CONST_TENSOR)
           : Tensor::Category::VAR;
}

inline Tensor::Category TensorCategory(const schema::Tensor *tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr";
    return Tensor::VAR;
  }
  auto shape_num = tensor->dims() == nullptr ? 0 : tensor->dims()->size();
  auto data_size = tensor->data() == nullptr ? 0 : tensor->data()->size();
  return TensorCategory(tensor->nodeType(), shape_num, TypeId(tensor->dataType()), data_size);
}

std::vector<tensor::MSTensor *> TensorVectorCast(const std::vector<Tensor *> &src);
}  // namespace lite
}  // namespace mindspore

using TensorPtr = std::shared_ptr<mindspore::lite::Tensor>;
#endif  // MINDSPORE_LITE_SRC_TENSOR_H_
