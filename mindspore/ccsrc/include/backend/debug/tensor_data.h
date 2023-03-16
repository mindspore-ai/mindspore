/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_
#define MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "base/float16.h"
#include "utils/log_adapter.h"
#include "mindapi/base/type_id.h"
#ifndef OFFLINE_DBG_MODE
#include "ir/tensor.h"
#endif

constexpr size_t kFloat32Size = 4;
constexpr size_t kFloat64Size = 8;

namespace mindspore {
typedef enum DbgDataType : unsigned int {
  DT_UNDEFINED = 0,
  // Basic types.
  DT_BOOL = 1,  // bool

  DT_INT8 = 2,   // int8_t
  DT_INT16 = 3,  // int16_t
  DT_INT32 = 4,  // int32_t
  DT_INT64 = 5,  // int64_t

  DT_UINT8 = 6,   // uint8_t
  DT_UINT16 = 7,  // uint16_t
  DT_UINT32 = 8,  // uint32_t
  DT_UINT64 = 9,  // uint64_t

  DT_FLOAT16 = 10,  // float 16
  DT_FLOAT32 = 11,  // float 32
  DT_FLOAT64 = 12,  // float 64

  DT_STRING = 13,  // string
  DT_TENSOR = 14,  // tensor
  DT_GRAPH = 15,   // graph

  // list type
  DT_BOOLS = 16,  // list of bool

  DT_INTS8 = 17,   // list of int8_t
  DT_INTS16 = 18,  // list of int16_t
  DT_INTS32 = 19,  // list of int32_t
  DT_INTS64 = 20,  // list of int64_t

  DT_UINTS8 = 21,   // list of uint8_t
  DT_UINTS16 = 22,  // list of uint16_t
  DT_UINTS32 = 23,  // list of uint32_t
  DT_UINTS64 = 24,  // list of uint64_t

  DT_FLOATS16 = 25,  // list of float16
  DT_FLOATS32 = 26,  // list of float32
  DT_FLOATS64 = 27,  // list of float64

  DT_STRINGS = 28,  // list of string
  DT_TENSORS = 29,  // list of tensor
  DT_GRAPHS = 30,   // list of graph

  DT_TUPLE = 31,  // tuple
  DT_LIST = 32,   // list
  DT_DICT = 33,   // dictionary

  // other types
  DT_NONE = 34,      // None
  DT_SYM_INST = 35,  // Symbolic Key Instance

  // type related type
  DT_BASE_INT = 36,    // type generic int
  DT_BASE_UINT = 37,   // type generate unsigned int
  DT_BASE_FLOAT = 38,  // type generate float
  DT_TYPE = 39,        // type type
  DT_ANY = 40,         // type any
  DT_REFKEY = 41,      // type refkey
  DT_REF = 42          // type ref
} DbgDataType;

class TensorData {
 public:
  TensorData() : slot_(0), execution_order_(-1) {}

  TensorData(const TensorData &obj) {
    MS_LOG(INFO) << "Copy Constructor";
    this->name_ = obj.name_;
    this->execution_order_ = obj.execution_order_;
    this->slot_ = obj.slot_;
    this->size_ = obj.size_;
    this->data_type_ = obj.data_type_;
    this->data_type_size_ = obj.data_type_size_;
    this->shape_ = obj.shape_;
    this->iteration_ = obj.iteration_;
    this->prev_iteration_ = obj.prev_iteration_;
    this->device_id_ = obj.device_id_;
    this->data_ptr_ = obj.data_ptr_;
    this->root_graph_id_ = obj.root_graph_id_;
    this->is_output_ = obj.is_output_;
    this->time_stamp_ = obj.time_stamp_;
#ifndef OFFLINE_DBG_MODE
    this->format_ = obj.format_;
    this->tensor_ptr_ = obj.tensor_ptr_;
#endif
  }

  TensorData &operator=(const TensorData &other) {
    if (this != &other) {
      MS_LOG(INFO) << "Copy Constructor";
      this->name_ = other.name_;
      this->execution_order_ = other.execution_order_;
      this->slot_ = other.slot_;
      this->size_ = other.size_;
      this->data_type_ = other.data_type_;
      this->data_type_size_ = other.data_type_size_;
      this->shape_ = other.shape_;
      this->iteration_ = other.iteration_;
      this->prev_iteration_ = other.prev_iteration_;
      this->device_id_ = other.device_id_;
      this->data_ptr_ = other.data_ptr_;
      this->root_graph_id_ = other.root_graph_id_;
      this->is_output_ = other.is_output_;
      this->time_stamp_ = other.time_stamp_;
#ifndef OFFLINE_DBG_MODE
      this->format_ = other.format_;
      this->tensor_ptr_ = other.tensor_ptr_;
#endif
    }
    return *this;
  }

  ~TensorData() { DeleteDataPtr(); }

  void DeleteDataPtr() noexcept {
#ifndef OFFLINE_DBG_MODE
    this->tensor_ptr_ = nullptr;
    this->data_ptr_ = nullptr;
#else
    if (this->data_ptr_ != nullptr) {
      delete[] this->data_ptr_;
      this->data_ptr_ = nullptr;
      this->size_ = 0;
    }
#endif
  }

  std::string GetName() const { return this->name_; }

  std::string GetTimeStamp() const { return this->time_stamp_; }

  size_t GetSlot() const { return this->slot_; }

  int GetExecutionOrder() const { return this->execution_order_; }

  void SetExecutionOrder(int execution_order) { this->execution_order_ = execution_order; }

  void SetName(const std::string &name) { this->name_ = name; }

  void SetTimeStamp(const std::string &time_stamp) { this->time_stamp_ = time_stamp; }

#ifndef OFFLINE_DBG_MODE
  void SetTensor(const mindspore::tensor::TensorPtr &out_tensor) { this->tensor_ptr_ = out_tensor; }

  void SetFormat(const std::string &format) { this->format_ = format; }

  std::string GetFormat() { return this->format_; }
#endif

  void SetSlot(size_t slot) { this->slot_ = slot; }

  const char *GetDataPtr() const { return this->data_ptr_; }

  void SetDataPtr(char *data_ptr) { this->data_ptr_ = data_ptr; }

  uint64_t GetNumElements() const {
    if (data_type_size_ == 0) {
      return 0;
    }
    return size_ / data_type_size_;
  }

  uint64_t GetByteSize() const { return this->size_; }

  void SetByteSize(uint64_t size) { this->size_ = size; }

  std::vector<int64_t> GetShape() const { return this->shape_; }

  void SetShape(const std::vector<int64_t> &shape) { this->shape_ = shape; }

  unsigned int GetIteration() const { return this->iteration_; }

  void SetIteration(unsigned int iteration) { this->iteration_ = iteration; }

  unsigned int GetPrevIteration() const { return this->prev_iteration_; }

  void SetPrevIteration(unsigned int prev_iteration) { this->prev_iteration_ = prev_iteration; }

  unsigned int GetDeviceId() const { return this->device_id_; }

  void SetDeviceId(unsigned int device_id) { this->device_id_ = device_id; }

  unsigned int GetRootGraphId() const { return this->root_graph_id_; }

  void SetRootGraphId(unsigned int root_graph_id) { this->root_graph_id_ = root_graph_id; }

  DbgDataType GetType() const { return this->data_type_; }

  std::string GetTypeString() const {
    const std::map<DbgDataType, std::string> kDbgDataTypeToStringMap = {
      {DT_BOOL, "bool"},     {DT_INT8, "int8"},       {DT_INT16, "int16"},     {DT_INT32, "int32"},
      {DT_INT64, "int64"},   {DT_UINT8, "uint8"},     {DT_UINT16, "uint16"},   {DT_UINT32, "uint32"},
      {DT_UINT64, "uint64"}, {DT_FLOAT16, "float16"}, {DT_FLOAT32, "float32"}, {DT_FLOAT64, "float64"}};
    auto iter_type = kDbgDataTypeToStringMap.find(data_type_);
    if (iter_type == kDbgDataTypeToStringMap.end()) {
      return std::string();
    } else {
      return iter_type->second;
    }
  }

  void SetType(TypeId type) { ConvertMsToDbgType(type); }

  void SetType(const std::string &type_name) { ConvertStringToDbgType(type_name); }

  bool GetIsOutput() const { return this->is_output_; }

  void SetIsOutput(bool is_output) { this->is_output_ = is_output; }

  void ConvertMsToDbgType(TypeId type) {
    switch (type) {
      case TypeId::kNumberTypeBool:
        this->data_type_ = DbgDataType::DT_BOOL;
        this->data_type_size_ = sizeof(bool);
        break;
      case TypeId::kNumberTypeInt8:
        this->data_type_ = DbgDataType::DT_INT8;
        this->data_type_size_ = sizeof(int8_t);
        break;
      case TypeId::kNumberTypeInt16:
        this->data_type_ = DbgDataType::DT_INT16;
        this->data_type_size_ = sizeof(int16_t);
        break;
      case TypeId::kNumberTypeInt32:
        this->data_type_ = DbgDataType::DT_INT32;
        this->data_type_size_ = sizeof(int32_t);
        break;
      case TypeId::kNumberTypeInt64:
        this->data_type_ = DbgDataType::DT_INT64;
        this->data_type_size_ = sizeof(int64_t);
        break;
      case TypeId::kNumberTypeUInt8:
        this->data_type_ = DbgDataType::DT_UINT8;
        this->data_type_size_ = sizeof(uint8_t);
        break;
      case TypeId::kNumberTypeUInt16:
        this->data_type_ = DbgDataType::DT_UINT16;
        this->data_type_size_ = sizeof(uint16_t);
        break;
      case TypeId::kNumberTypeUInt32:
        this->data_type_ = DbgDataType::DT_UINT32;
        this->data_type_size_ = sizeof(uint32_t);
        break;
      case TypeId::kNumberTypeUInt64:
        this->data_type_ = DbgDataType::DT_UINT64;
        this->data_type_size_ = sizeof(uint64_t);
        break;
      case TypeId::kNumberTypeFloat16:
        this->data_type_ = DbgDataType::DT_FLOAT16;
        this->data_type_size_ = sizeof(float16);
        break;
      case TypeId::kNumberTypeFloat32:
        this->data_type_ = DbgDataType::DT_FLOAT32;
        this->data_type_size_ = kFloat32Size;
        break;
      case TypeId::kNumberTypeFloat64:
        this->data_type_ = DbgDataType::DT_FLOAT64;
        this->data_type_size_ = kFloat64Size;
        break;
      case TypeId::kNumberTypeInt:
        this->data_type_ = DbgDataType::DT_BASE_INT;
        this->data_type_size_ = sizeof(int);
        break;
      case TypeId::kNumberTypeUInt:
        this->data_type_ = DbgDataType::DT_BASE_UINT;
        this->data_type_size_ = sizeof(uint);
        break;
      case TypeId::kNumberTypeFloat:
        this->data_type_ = DbgDataType::DT_BASE_FLOAT;
        this->data_type_size_ = sizeof(uint);
        break;
      default:
        MS_LOG(EXCEPTION) << "Unexpected type id: " << type;
    }
  }

  bool ConvertNpyStringToDbgType(const std::string &type_name) {
    if (type_name == "b1") {
      this->data_type_ = DbgDataType::DT_BOOL;
      this->data_type_size_ = sizeof(bool);
      return true;
    } else if (type_name == "i1") {
      this->data_type_ = DbgDataType::DT_INT8;
      this->data_type_size_ = sizeof(int8_t);
      return true;
    } else if (type_name == "i2") {
      this->data_type_ = DbgDataType::DT_INT16;
      this->data_type_size_ = sizeof(int16_t);
      return true;
    } else if (type_name == "i4") {
      this->data_type_ = DbgDataType::DT_INT32;
      this->data_type_size_ = sizeof(int32_t);
      return true;
    } else if (type_name == "i8") {
      this->data_type_ = DbgDataType::DT_INT64;
      this->data_type_size_ = sizeof(int64_t);
      return true;
    } else if (type_name == "u1") {
      this->data_type_ = DbgDataType::DT_UINT8;
      this->data_type_size_ = sizeof(uint8_t);
      return true;
    } else if (type_name == "u2") {
      this->data_type_ = DbgDataType::DT_UINT16;
      this->data_type_size_ = sizeof(uint16_t);
      return true;
    } else if (type_name == "u4") {
      this->data_type_ = DbgDataType::DT_UINT32;
      this->data_type_size_ = sizeof(uint32_t);
      return true;
    } else if (type_name == "u8") {
      this->data_type_ = DbgDataType::DT_UINT64;
      this->data_type_size_ = sizeof(uint64_t);
      return true;
    } else if (type_name == "f2") {
      this->data_type_ = DbgDataType::DT_FLOAT16;
      this->data_type_size_ = sizeof(float16);
      return true;
    } else if (type_name == "f4") {
      this->data_type_ = DbgDataType::DT_FLOAT32;
      this->data_type_size_ = kFloat32Size;
      return true;
    } else if (type_name == "f8") {
      this->data_type_ = DbgDataType::DT_FLOAT64;
      this->data_type_size_ = kFloat64Size;
      return true;
    } else {
      return false;
    }
  }

  void ConvertStringToDbgType(const std::string &type_name) {
    std::string type_name_lower = type_name;
    std::string trans_true_prefix = "kNumberType";
    if (type_name.find(trans_true_prefix) == 0) {
      type_name_lower = type_name.substr(trans_true_prefix.length());
    }
    (void)std::transform(type_name_lower.begin(), type_name_lower.end(), type_name_lower.begin(), ::tolower);
    if (type_name_lower == "bool") {
      this->data_type_ = DbgDataType::DT_BOOL;
      this->data_type_size_ = sizeof(bool);
    } else if (type_name_lower == "int8") {
      this->data_type_ = DbgDataType::DT_INT8;
      this->data_type_size_ = sizeof(int8_t);
    } else if (type_name_lower == "int16") {
      this->data_type_ = DbgDataType::DT_INT16;
      this->data_type_size_ = sizeof(int16_t);
    } else if (type_name_lower == "int32") {
      this->data_type_ = DbgDataType::DT_INT32;
      this->data_type_size_ = sizeof(int32_t);
    } else if (type_name_lower == "int64") {
      this->data_type_ = DbgDataType::DT_INT64;
      this->data_type_size_ = sizeof(int64_t);
    } else if (type_name_lower == "uint8") {
      this->data_type_ = DbgDataType::DT_UINT8;
      this->data_type_size_ = sizeof(uint8_t);
    } else if (type_name_lower == "uint16") {
      this->data_type_ = DbgDataType::DT_UINT16;
      this->data_type_size_ = sizeof(uint16_t);
    } else if (type_name_lower == "uint32") {
      this->data_type_ = DbgDataType::DT_UINT32;
      this->data_type_size_ = sizeof(uint32_t);
    } else if (type_name_lower == "uint64") {
      this->data_type_ = DbgDataType::DT_UINT64;
      this->data_type_size_ = sizeof(uint64_t);
    } else if (type_name_lower == "float16") {
      this->data_type_ = DbgDataType::DT_FLOAT16;
      this->data_type_size_ = sizeof(float16);
    } else if (type_name_lower == "float32") {
      this->data_type_ = DbgDataType::DT_FLOAT32;
      this->data_type_size_ = kFloat32Size;
    } else if (type_name_lower == "float64") {
      this->data_type_ = DbgDataType::DT_FLOAT64;
      this->data_type_size_ = kFloat64Size;
    } else if (type_name_lower == "") {
      this->data_type_ = DbgDataType::DT_UNDEFINED;
      this->data_type_size_ = 0;
    } else {
      if (!ConvertNpyStringToDbgType(type_name_lower)) {
        MS_LOG(EXCEPTION) << "Unexpected type name: " << type_name;
      }
    }
  }

 private:
  char *data_ptr_{nullptr};                           // pointer to the pre-allocated memory
  uint64_t size_{0};                                  // size_ in bytes
  DbgDataType data_type_{DbgDataType::DT_UNDEFINED};  // internal debugger type
  size_t data_type_size_{0};
  std::vector<int64_t> shape_;
  std::string name_;
  uint64_t slot_;
  unsigned int iteration_{0};
  unsigned int prev_iteration_{0};
  unsigned int device_id_{0};
  unsigned int root_graph_id_{0};
  bool is_output_{true};
  int execution_order_{-1};
  std::string time_stamp_;

#ifndef OFFLINE_DBG_MODE
  std::string format_{""};
  mindspore::tensor::TensorPtr tensor_ptr_{nullptr};
#endif
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_
