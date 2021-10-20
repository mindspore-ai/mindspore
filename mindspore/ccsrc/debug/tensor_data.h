/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include "mindspore/core/utils/log_adapter.h"
#ifdef ONLINE_DBG_MODE
#include "ir/tensor.h"
#endif

#ifdef ONLINE_DBG_MODE
namespace mindspore {
#endif

namespace MsTypeId {
typedef enum MsTypeId : unsigned int {
  kTypeUnknown = 0,
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAnything,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeRowTensorType,
  kObjectTypeSparseTensorType,
  kObjectTypeUndeterminedType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeComplex64,
  kNumberTypeEnd
} MsTypeId;
}  // namespace MsTypeId

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
  DT_ANYTHING = 40,    // type anything
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
    this->data_ptr_ = obj.data_ptr_;
    this->size_ = obj.size_;
    this->data_type_ = obj.data_type_;
    this->data_type_size_ = obj.data_type_size_;
    this->shape_ = obj.shape_;
    this->iteration_ = obj.iteration_;
    this->device_id_ = obj.device_id_;
    this->data_ptr_ = obj.data_ptr_;
    this->root_graph_id_ = obj.root_graph_id_;
    this->is_output_ = obj.is_output_;
#ifdef ONLINE_DBG_MODE
    this->tensor_ptr_ = obj.tensor_ptr_;
#endif
  }

#ifdef OFFLINE_DBG_MODE
  ~TensorData() { DeleteDataPtr(); }
#else
  ~TensorData() {}
#endif

  void DeleteDataPtr() {
    if (this->data_ptr_ != NULL) {
      delete this->data_ptr_;
      this->data_ptr_ = NULL;
      this->size_ = 0;
    }
  }

  std::string GetName() const { return this->name_; }

  std::string GetTimeStamp() const { return this->time_stamp_; }

  size_t GetSlot() const { return this->slot_; }

  int GetExecutionOrder() const { return this->execution_order_; }

  void SetExecutionOrder(int execution_order) { this->execution_order_ = execution_order; }

  void SetName(const std::string &name) { this->name_ = name; }

  void SetTimeStamp(const std::string &time_stamp) { this->time_stamp_ = time_stamp; }

#ifdef ONLINE_DBG_MODE
  void SetTensor(mindspore::tensor::TensorPtr out_tensor) { this->tensor_ptr_ = out_tensor; }
#endif

  void SetSlot(size_t slot) { this->slot_ = slot; }

  const char *GetDataPtr() const { return this->data_ptr_; }

  void SetDataPtr(char *data_ptr) { this->data_ptr_ = data_ptr; }

  uint32_t GetNumElements() { return size_ / data_type_size_; }

  uint64_t GetByteSize() const { return this->size_; }

  void SetByteSize(uint64_t size) { this->size_ = size; }

  std::vector<int64_t> GetShape() const { return this->shape_; }

  void SetShape(std::vector<int64_t> shape) { this->shape_ = shape; }

  unsigned int GetIteration() const { return this->iteration_; }

  void SetIteration(unsigned int iteration) { this->iteration_ = iteration; }

  unsigned int GetDeviceId() const { return this->device_id_; }

  void SetDeviceId(unsigned int device_id) { this->device_id_ = device_id; }

  unsigned int GetRootGraphId() const { return this->root_graph_id_; }

  void SetRootGraphId(unsigned int root_graph_id) { this->root_graph_id_ = root_graph_id; }

  DbgDataType GetType() const { return this->data_type_; }

  void SetType(unsigned int type) { ConvertMsToDbgType(type); }

  void SetType(std::string type_name) { ConvertStringToDbgType(type_name); }

  bool GetIsOutput() const { return this->is_output_; }

  void SetIsOutput(bool is_output) { this->is_output_ = is_output; }

  void ConvertMsToDbgType(uint32_t type) {
    switch (type) {
      case MsTypeId::kNumberTypeBool:
        this->data_type_ = DbgDataType::DT_BOOL;
        this->data_type_size_ = 1;
        break;
      case MsTypeId::kNumberTypeInt8:
        this->data_type_ = DbgDataType::DT_INT8;
        this->data_type_size_ = 1;
        break;
      case MsTypeId::kNumberTypeInt16:
        this->data_type_ = DbgDataType::DT_INT16;
        this->data_type_size_ = 2;
        break;
      case MsTypeId::kNumberTypeInt32:
        this->data_type_ = DbgDataType::DT_INT32;
        this->data_type_size_ = 4;
        break;
      case MsTypeId::kNumberTypeInt64:
        this->data_type_ = DbgDataType::DT_INT64;
        this->data_type_size_ = 8;
        break;
      case MsTypeId::kNumberTypeUInt8:
        this->data_type_ = DbgDataType::DT_UINT8;
        this->data_type_size_ = 1;
        break;
      case MsTypeId::kNumberTypeUInt16:
        this->data_type_ = DbgDataType::DT_UINT16;
        this->data_type_size_ = 2;
        break;
      case MsTypeId::kNumberTypeUInt32:
        this->data_type_ = DbgDataType::DT_UINT32;
        this->data_type_size_ = 4;
        break;
      case MsTypeId::kNumberTypeUInt64:
        this->data_type_ = DbgDataType::DT_UINT64;
        this->data_type_size_ = 8;
        break;
      case MsTypeId::kNumberTypeFloat16:
        this->data_type_ = DbgDataType::DT_FLOAT16;
        this->data_type_size_ = 2;
        break;
      case MsTypeId::kNumberTypeFloat32:
        this->data_type_ = DbgDataType::DT_FLOAT32;
        this->data_type_size_ = 4;
        break;
      case MsTypeId::kNumberTypeFloat64:
        this->data_type_ = DbgDataType::DT_FLOAT64;
        this->data_type_size_ = 8;
        break;
      case MsTypeId::kNumberTypeInt:
        this->data_type_ = DbgDataType::DT_BASE_INT;
        this->data_type_size_ = 4;
        break;
      case MsTypeId::kNumberTypeUInt:
        this->data_type_ = DbgDataType::DT_BASE_UINT;
        this->data_type_size_ = 4;
        break;
      case MsTypeId::kNumberTypeFloat:
        this->data_type_ = DbgDataType::DT_BASE_FLOAT;
        this->data_type_size_ = 4;
        break;
      default:
        MS_LOG(EXCEPTION) << "Unexpected type id: " << type;
    }
  }

  bool ConvertNpyStringToDbgType(const std::string &type_name) {
    if (type_name == "b1") {
      this->data_type_ = DbgDataType::DT_BOOL;
      this->data_type_size_ = 1;
      return true;
    } else if (type_name == "i1") {
      this->data_type_ = DbgDataType::DT_INT8;
      this->data_type_size_ = 1;
      return true;
    } else if (type_name == "i2") {
      this->data_type_ = DbgDataType::DT_INT16;
      this->data_type_size_ = 2;
      return true;
    } else if (type_name == "i4") {
      this->data_type_ = DbgDataType::DT_INT32;
      this->data_type_size_ = 4;
      return true;
    } else if (type_name == "i8") {
      this->data_type_ = DbgDataType::DT_INT64;
      this->data_type_size_ = 8;
      return true;
    } else if (type_name == "u1") {
      this->data_type_ = DbgDataType::DT_UINT8;
      this->data_type_size_ = 1;
      return true;
    } else if (type_name == "u2") {
      this->data_type_ = DbgDataType::DT_UINT16;
      this->data_type_size_ = 2;
      return true;
    } else if (type_name == "u4") {
      this->data_type_ = DbgDataType::DT_UINT32;
      this->data_type_size_ = 4;
      return true;
    } else if (type_name == "u8") {
      this->data_type_ = DbgDataType::DT_UINT64;
      this->data_type_size_ = 8;
      return true;
    } else if (type_name == "f2") {
      this->data_type_ = DbgDataType::DT_FLOAT16;
      this->data_type_size_ = 2;
      return true;
    } else if (type_name == "f4") {
      this->data_type_ = DbgDataType::DT_FLOAT32;
      this->data_type_size_ = 4;
      return true;
    } else if (type_name == "f8") {
      this->data_type_ = DbgDataType::DT_FLOAT64;
      this->data_type_size_ = 8;
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
      this->data_type_size_ = 1;
    } else if (type_name_lower == "int8") {
      this->data_type_ = DbgDataType::DT_INT8;
      this->data_type_size_ = 1;
    } else if (type_name_lower == "int16") {
      this->data_type_ = DbgDataType::DT_INT16;
      this->data_type_size_ = 2;
    } else if (type_name_lower == "int32") {
      this->data_type_ = DbgDataType::DT_INT32;
      this->data_type_size_ = 4;
    } else if (type_name_lower == "int64") {
      this->data_type_ = DbgDataType::DT_INT64;
      this->data_type_size_ = 8;
    } else if (type_name_lower == "uint8") {
      this->data_type_ = DbgDataType::DT_UINT8;
      this->data_type_size_ = 1;
    } else if (type_name_lower == "uint16") {
      this->data_type_ = DbgDataType::DT_UINT16;
      this->data_type_size_ = 2;
    } else if (type_name_lower == "uint32") {
      this->data_type_ = DbgDataType::DT_UINT32;
      this->data_type_size_ = 4;
    } else if (type_name_lower == "uint64") {
      this->data_type_ = DbgDataType::DT_UINT64;
      this->data_type_size_ = 8;
    } else if (type_name_lower == "float16") {
      this->data_type_ = DbgDataType::DT_FLOAT16;
      this->data_type_size_ = 2;
    } else if (type_name_lower == "float32") {
      this->data_type_ = DbgDataType::DT_FLOAT32;
      this->data_type_size_ = 4;
    } else if (type_name_lower == "float64") {
      this->data_type_ = DbgDataType::DT_FLOAT64;
      this->data_type_size_ = 8;
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
  unsigned int data_type_size_{0};
  std::vector<int64_t> shape_;
  std::string name_;
  uint64_t slot_;
  unsigned int iteration_{0};
  unsigned int device_id_{0};
  unsigned int root_graph_id_{0};
  bool is_output_{true};
  int execution_order_{-1};
  std::string time_stamp_;

#ifdef ONLINE_DBG_MODE
  mindspore::tensor::TensorPtr tensor_ptr_{nullptr};
#endif
};
#ifdef ONLINE_DBG_MODE
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_DEBUG_TENSOR_DATA_H_
