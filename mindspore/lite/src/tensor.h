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

#ifndef MINDSPORE_LITE_SRC_TENSOR_H_
#define MINDSPORE_LITE_SRC_TENSOR_H_

#include <math.h>
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <atomic>
#include "include/api/format.h"
#include "include/lite_types.h"
#include "nnacl/tensor_c.h"
#include "src/litert/inner_allocator.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "src/litert/tensor_category.h"

namespace mindspore {
namespace lite {
#define STATIC_ALLOCATION -271964
#define RUNTIME_REFCOUNT 0x9999
#define IS_STATIC_ALLOCATOR(allocator) ((allocator != nullptr) && (allocator->RefCount(nullptr) == STATIC_ALLOCATION))
#define IS_RUNTIME_ALLOCATOR(allocator) ((allocator != nullptr) && (allocator->RefCount(nullptr) == RUNTIME_REFCOUNT))
struct LiteQuantParam {
  double scale;
  int32_t zeroPoint;
  float var_corr{1};
  float mean_corr{0};
  bool inited{false};
  std::vector<float> clusters{};
  int bitNum{8};
  int roundType{1};
  int multiplier{1};
  int dstDtype{32};
  // dynamic range
  double min{-255.0};
  double max{255.0};
};

enum CompressType {
  kNoCompression = 0,
  kIndexing = 1,
  kSparse = 2,
  kFSE = 3,
  kBitPacking = 4,
  kFSEInt = 5,
  kFSEInfer = 6
};

class Tensor {
 public:
  Tensor() { tensor_c_ = {kTypeUnknown, NHWC, nullptr, 0}; }

  Tensor(TypeId data_type, std::vector<int> shape, const mindspore::Format &format = mindspore::NHWC,
         Category category = VAR);

  Tensor(const Tensor &tensor) = delete;

  Tensor(Tensor &&other) = delete;

  Tensor &operator=(const Tensor &tensor) = delete;

  Tensor &operator=(Tensor &&src) = delete;

  virtual ~Tensor();

  static Tensor *CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape, const void *data,
                              size_t data_len);
  static Tensor *CreateTensorByDeepCopy(const std::string &name, TypeId type, const std::vector<int> &shape,
                                        const void *data, size_t data_len);

  static int CopyTensorData(const Tensor &src_tensor, Tensor *dst_tensor);

  static Tensor *CopyTensor(const Tensor &src_tensor, bool copy_data = false, AllocatorPtr allocator = nullptr);

  virtual bool operator==(const Tensor &tensor);

  void set_tensor_name(const std::string &name) { tensor_name_ = name; }

  std::string tensor_name() const { return tensor_name_; }

  TypeId data_type() const { return static_cast<TypeId>(tensor_c_.data_type_); }

  void set_data_type(TypeId data_type) { tensor_c_.data_type_ = data_type; }

  std::vector<int> shape() const {
    return std::vector<int>(tensor_c_.shape_, tensor_c_.shape_ + tensor_c_.shape_size_);
  }

  void set_shape(const std::vector<int> &shape) {
    if (shape.size() > MAX_SHAPE_SIZE) {
      FreeData();
      tensor_c_.shape_size_ = 0;
      MS_LOG(WARNING) << "The shape-size has exceeded the limit 8, now is " << shape.size();
      return;
    }
    tensor_c_.shape_size_ = shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      tensor_c_.shape_[i] = shape[i];
    }
  }

  int DimensionSize(size_t index) const;

  int64_t ElementsNum() const;

  int32_t Batch() const;

  int32_t Channel() const;

  int32_t Height() const;

  int32_t Width() const;

  int64_t ElementsC4Num() const;

  int64_t ElementsC8Num() const;
  virtual size_t Size() const;

  virtual void set_allocator(AllocatorPtr allocator) { allocator_ = allocator; }

  AllocatorPtr allocator() const { return allocator_; }

  virtual int MallocData(const AllocatorPtr allocator = nullptr);

  virtual void FreeData();

  virtual void *MutableData();

  void *ReallocData();

  virtual void *data() { return tensor_c_.data_; }

  virtual void *data() const { return tensor_c_.data_; }

  // note: in the case of that old_data is valid, set_data just releases the ownership of it but not frees it. Of
  //       course, you can call FreeData before calling set_data to ensure the data can be freed by current tensor.
  void set_data(void *data, bool own_data = true) {
    if (allocator_ != nullptr && this->tensor_c_.data_ != data) {
      (void)allocator_->IncRefCount(data, 1);
      (void)allocator_->DecRefCount(this->tensor_c_.data_, 1);
    }
    this->tensor_c_.data_ = data;
    this->own_data_ = own_data;
  }

  void set_device_data(void *data) { device_data_ = data; }

  void *device_data() const { return device_data_; }

  Category category() const { return this->category_; }

  void set_category(Category category) { this->category_ = category; }

  void set_format(mindspore::Format format) { this->tensor_c_.format_ = format; }

  mindspore::Format format() const { return static_cast<mindspore::Format>(this->tensor_c_.format_); }
  virtual int ref_count() const { return ref_count_; }

  virtual int init_ref_count() const { return static_cast<int>(this->init_ref_count_); }

  virtual void set_ref_count(int ref_count) { ref_count_ = ref_count; }

  virtual void set_init_ref_count(int ref_count) { this->init_ref_count_ = ref_count; }

  virtual void ResetRefCount() { set_ref_count(static_cast<int>(this->init_ref_count_)); }

  virtual void IncRefCount() { ++ref_count_; }

  virtual void DecRefCount();

  std::string ToString() const;

  void AddQuantParam(const LiteQuantParam &quant_param);

  void ClearQuantParam();

  std::vector<LiteQuantParam> quant_params() const;

  void set_quant_params(std::vector<LiteQuantParam>);

  std::vector<float> quant_clusters() const;

  void set_quant_clusters(const std::vector<float> &clusters);

  virtual bool IsConst() const {
    return (this->category_ == CONST_TENSOR || this->category_ == CONST_SCALAR) && this->tensor_c_.data_ != nullptr;
  }

  bool IsScalar() const { return this->category_ == CONST_SCALAR && this->tensor_c_.data_ != nullptr; }

  bool IsGraphInput() const { return this->category_ == GRAPH_INPUT; }

  bool IsGraphOutput() const { return this->category_ == GRAPH_OUTPUT; }

  void Prepare() {
    if (allocator_ != nullptr) {
      tensor_c_.data_ = allocator_->Prepare(tensor_c_.data_);
    }
  }

  bool IsReady() const {
    return this->IsConst() || (this->IsGraphInput() && this->tensor_c_.data_ != nullptr) || ref_count() >= 1;
  }

  bool own_data() const { return this->own_data_; }

  virtual void set_own_data(bool own_data) { this->own_data_ = own_data; }

  template <typename T>
  int Scale(float scale) {
    T cast_scale = static_cast<T>(scale);
    auto data = reinterpret_cast<T *>(tensor_c_.data_);
    if (data == nullptr) {
      return RET_ERROR;
    }
    int length = ElementsNum();
    for (int i = 0; i < length; i++) {
      data[i] *= cast_scale;
    }
    scale_ *= scale;
    return RET_OK;
  }

  float get_scale() const { return this->scale_; }

  void set_scale(float scale) { this->scale_ = scale; }

  CompressType get_compress_type() const { return this->compress_type_; }

  void set_compress_type(CompressType compression_type) { this->compress_type_ = compression_type; }

  void set_compressed_size(size_t compressed_size) { this->compressed_size_ = compressed_size; }

  bool IsScale() const { return (std::fabs(this->scale_ - 1.0f) > 1.0e-05); }

  TensorC *ConvertToTensorC() { return &tensor_c_; }

 private:
  template <typename T>
  std::string DataToString(void *data, size_t data_number, size_t print_len = 40) const {
    if (data == nullptr) {
      return "Data of tensor is nullptr";
    }
    std::ostringstream oss;
    auto casted_data = static_cast<T *>(data);
    for (size_t i = 0; i < print_len && i < data_number; i++) {
      oss << " " << casted_data[i];
    }
    return oss.str();
  }

 protected:
  TensorC tensor_c_;
  std::string tensor_name_;
  Category category_ = VAR;
  std::atomic_int ref_count_ = {0};
  int init_ref_count_ = 0;
  std::vector<LiteQuantParam> quant_params_;
  std::vector<float> quant_clusters_;
  AllocatorPtr allocator_ = nullptr;
  bool own_data_{false};
  float scale_ = 1.0f;
  void *device_data_ = nullptr;
  CompressType compress_type_ = kNoCompression;
  size_t compressed_size_ = 0;
};
}  // namespace lite
}  // namespace mindspore

using TensorPtr = std::shared_ptr<mindspore::lite::Tensor>;
#endif  // MINDSPORE_LITE_SRC_TENSOR_H_
