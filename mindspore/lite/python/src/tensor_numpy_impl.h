/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_NUMPY_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_NUMPY_IMPL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <set>

#include "src/common/log_adapter.h"
#include "common/mutable_tensor_impl.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;
namespace mindspore {
class TensorNumpyImpl : public MutableTensorImpl {
 public:
  TensorNumpyImpl(const std::string &name, py::buffer_info &&buffer, const std::vector<int64_t> &ms_shape)
      : name_(name), buffer_(std::move(buffer)), ms_shape_(ms_shape) {}
  ~TensorNumpyImpl() {
    {
      py::gil_scoped_acquire acquire;
      { buffer_ = py::buffer_info(); }
    }
  }
  const std::vector<int64_t> &Shape() const override { return ms_shape_; }
  void SetShape(const std::vector<int64_t> &shape) override {
    MS_LOG(ERROR) << "Cannot call SetShape for numpy tensor";
  }

  enum DataType DataType() const override { return GetDataType(buffer_); }
  void SetDataType(mindspore::DataType data_type) override {
    MS_LOG(ERROR) << "Cannot call SetDataType for numpy tensor";
  }

  void SetName(const std::string &name) override { name_ = name; }
  const std::string &Name() const override { return name_; }

  mindspore::Format Format() const override { return format_; }
  void SetFormat(mindspore::Format format) override { MS_LOG(ERROR) << "Cannot call SetFormat for numpy tensor"; }

  void SetAllocator(const std::shared_ptr<Allocator> &allocator) override {
    MS_LOG(ERROR) << "Cannot call SetAllocator for numpy tensor";
  }
  std::shared_ptr<Allocator> GetAllocator() const override { return nullptr; }

  std::vector<QuantParam> GetQuantParams() const override { return {}; }
  void SetQuantParams(const std::vector<QuantParam> &quant_param) override {
    MS_LOG(ERROR) << "Cannot call SetQuantParams for numpy tensor";
  }

  int64_t ElementNum() const override { return buffer_.size; }
  size_t DataSize() const override { return buffer_.size * buffer_.itemsize; }

  void SetDeviceData(void *data) override { MS_LOG(ERROR) << "Cannot call SetDeviceData for numpy tensor"; }
  void *GetDeviceData() override { return nullptr; }
  bool IsConst() const override { return false; }
  void SetIsConst(bool is_const) { MS_LOG(ERROR) << "Cannot call SetIsConst for numpy tensor"; }

  bool IsDevice() const override { return false; }

  std::shared_ptr<const void> Data() const override {
    auto data = static_cast<const uint8_t *>(buffer_.ptr);
    return std::shared_ptr<const void>(data, [](const void *) {});
  }

  void SetData(void *data, bool own_data) override { MS_LOG(ERROR) << "Cannot call SetData for numpy tensor"; }

  void *MutableData() override { return buffer_.ptr; }

  std::shared_ptr<Impl> Clone() const override {
    MS_LOG(ERROR) << "Cannot call Clone for numpy tensor";
    return nullptr;
  }

  static enum DataType GetDataType(const py::buffer_info &buf) {
    std::set<char> fp_format = {'e', 'f', 'd'};
    std::set<char> int_format = {'b', 'h', 'i', 'l', 'q'};
    std::set<char> uint_format = {'B', 'H', 'I', 'L', 'Q'};
    if (buf.format.size() == 1) {
      char format = buf.format.front();
      if (fp_format.find(format) != fp_format.end()) {
        switch (buf.itemsize) {
          case 2:
            return DataType::kNumberTypeFloat16;
          case 4:
            return DataType::kNumberTypeFloat32;
          case 8:
            return DataType::kNumberTypeFloat64;
        }
      } else if (int_format.find(format) != int_format.end()) {
        switch (buf.itemsize) {
          case 1:
            return DataType::kNumberTypeInt8;
          case 2:
            return DataType::kNumberTypeInt16;
          case 4:
            return DataType::kNumberTypeInt32;
          case 8:
            return DataType::kNumberTypeInt64;
        }
      } else if (uint_format.find(format) != uint_format.end()) {
        switch (buf.itemsize) {
          case 1:
            return DataType::kNumberTypeUInt8;
          case 2:
            return DataType::kNumberTypeUInt16;
          case 4:
            return DataType::kNumberTypeUInt32;
          case 8:
            return DataType::kNumberTypeUInt64;
        }
      } else if (format == '?') {
        return DataType::kNumberTypeBool;
      }
    }
    MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
    return DataType::kTypeUnknown;
  }

 protected:
  std::string name_;
  enum Format format_ = mindspore::NCHW;

  py::buffer_info buffer_;
  std::vector<int64_t> ms_shape_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_NUMPY_IMPL_H_
