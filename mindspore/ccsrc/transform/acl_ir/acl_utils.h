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
#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_UTILS_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_UTILS_H_

#include <string>
#include <vector>
#include <utility>
#include <map>
#include "acl/acl_op_compiler.h"
#include "acl/acl_base.h"
#include "include/transform/graph_ir/types.h"
#include "transform/acl_ir/ge_adapter_info.h"

#define MAX_INPUT_TO_HOST 25

namespace mindspore {
namespace transform {
struct AclExecParam {
  std::vector<const aclTensorDesc *> input_desc;
  std::vector<const aclDataBuffer *> input_buffer;
  std::vector<const aclTensorDesc *> output_desc;
  std::vector<aclDataBuffer *> output_buffer;
  aclopAttr *attr = nullptr;
};

class AclInputToHost {
 public:
  AclInputToHost() { clear(); }

  void clear() {
    for (auto &item : input_to_host_) {
      item = nullptr;
    }
    size_ = 0;
  }

  tensor::TensorPtr get(size_t index) const {
    if (index >= MAX_INPUT_TO_HOST) {
      MS_LOG(EXCEPTION) << "Index is bigger than max input to host size, index: " << index
                        << ", max_input_to_host: " << MAX_INPUT_TO_HOST;
    }
    return input_to_host_[index];
  }

  void emplace(size_t index, const tensor::TensorPtr &tensor_ptr) {
    auto origin_tensor = get(index);
    if (origin_tensor == nullptr && tensor_ptr != nullptr) {
      size_++;
    }
    input_to_host_[index] = tensor_ptr;
  }

  void build(const std::map<uint32_t, tensor::TensorPtr> &inputs_on_host) {
    clear();
    for (auto &kv : inputs_on_host) {
      emplace(kv.first, kv.second);
    }
  }

  bool empty() const { return size_ == 0; }

 private:
  tensor::TensorPtr input_to_host_[MAX_INPUT_TO_HOST];
  size_t size_{};
};

class AclAttrMaker {
 public:
  static void SetAttr(const string &attr_name, const bool value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const int64_t value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const float value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::string &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::vector<uint8_t> &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::vector<int64_t> &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::vector<float> &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::vector<std::string> &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const std::vector<std::vector<int64_t>> &value, aclopAttr *attr);
  static void SetAttr(const string &attr_name, const ::ge::DataType value, aclopAttr *attr);
};  // class AclAttrMaker

class AclTensorDescMaker {
 public:
  AclTensorDescMaker() = default;
  ~AclTensorDescMaker() = default;

  AclTensorDescMaker &Create(aclDataType data_type, const ShapeVector &shape, aclFormat format) {
    acl_desc_ = aclCreateTensorDesc(data_type, shape.size(), shape.data(), format);
    MS_EXCEPTION_IF_NULL(acl_desc_);
    return *this;
  }

  AclTensorDescMaker &Create(aclDataType data_type, aclFormat format) {
    acl_desc_ = aclCreateTensorDesc(data_type, 0, nullptr, format);
    MS_EXCEPTION_IF_NULL(acl_desc_);
    return *this;
  }

  AclTensorDescMaker &SetFormat(aclFormat format) {
    auto ret = aclSetTensorFormat(acl_desc_, format);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl set tensor format failed!";
    }
    return *this;
  }

  AclTensorDescMaker &SetShape(const ShapeVector &shape) {
    if (!shape.empty()) {
      auto ret = aclSetTensorShape(acl_desc_, shape.size(), shape.data());
      if (ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Acl set tensor shape failed!";
      }
    }
    return *this;
  }

  AclTensorDescMaker &SetName(const std::string &name) {
    if (!name.empty()) {
      aclSetTensorDescName(acl_desc_, name.c_str());
    }
    return *this;
  }

  AclTensorDescMaker &SetTensorPlaceMent(const aclMemType &mem_type) {
    auto ret = aclSetTensorPlaceMent(acl_desc_, mem_type);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "ACL set host tensor failed!";
    }
    return *this;
  }

  aclTensorDesc *Get() const { return acl_desc_; }

 private:
  aclTensorDesc *acl_desc_ = nullptr;
};  // class AclTensorDescMaker

class AclTensorBufferMaker {
 public:
  AclTensorBufferMaker(void *addr, size_t size, TypeId type = kTypeUnknown) {
    size_t type_size = 1;
    if (type != kTypeUnknown) {
      type_size = GetTypeByte(TypeIdToType(type));
    }
    auto real_size = type_size * size;
    if (addr == nullptr || real_size == 0) {
      data_buffer_ = aclCreateDataBuffer(nullptr, 0);
    } else {
      data_buffer_ = aclCreateDataBuffer(addr, size);
    }
  }

  explicit AclTensorBufferMaker(const TensorPtr &tensor) {
    if (tensor->Size() == 0) {
      data_buffer_ = aclCreateDataBuffer(nullptr, 0);
    } else {
      data_buffer_ = aclCreateDataBuffer(tensor->data_c(), tensor->Size());
    }
  }

  ~AclTensorBufferMaker() = default;

  inline aclDataBuffer *Get() const { return data_buffer_; }

 private:
  aclDataBuffer *data_buffer_ = nullptr;
};  // class AclTensorBufferMaker

class AclRunner {
 public:
  AclRunner() = default;
  ~AclRunner();

  void SetName(const std::string &op_type) { op_type_ = op_type; }

  const string &GetName() const { return op_type_; }

  void SetStaticMode();

  void SetDynamicMode();

  void SetPrecisionMode(const AclPrecisionMode mode);

  void ResizeOpInputs(size_t size) {
    (void)std::for_each(acl_param_.input_desc.begin(), acl_param_.input_desc.end(), [](const aclTensorDesc *desc) {
      if (desc != nullptr) {
        aclDestroyTensorDesc(desc);
      }
    });
    (void)std::for_each(acl_param_.input_buffer.begin(), acl_param_.input_buffer.end(),
                        [](const aclDataBuffer *buffer) {
                          if (buffer != nullptr) {
                            aclDestroyDataBuffer(buffer);
                          }
                        });
    acl_param_.input_desc.clear();
    acl_param_.input_desc.resize(size, nullptr);
    acl_param_.input_buffer.clear();
    acl_param_.input_buffer.resize(size, nullptr);
  }

  void SetInput(size_t i, const aclTensorDesc *desc, const aclDataBuffer *buffer) {
    if (i >= acl_param_.input_desc.size() || i >= acl_param_.input_buffer.size()) {
      MS_LOG(EXCEPTION) << "Index " << i << " is out of bounds " << acl_param_.input_desc.size();
    }

    if (acl_param_.input_desc[i] != nullptr) {
      aclDestroyTensorDesc(acl_param_.input_desc[i]);
    }
    if (acl_param_.input_buffer[i] != nullptr) {
      aclDestroyDataBuffer(acl_param_.input_buffer[i]);
    }

    acl_param_.input_desc[i] = desc;
    acl_param_.input_buffer[i] = buffer;
  }

  size_t GetNumRealInputs() const {
    if (acl_param_.input_desc.empty()) {
      return 0;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(acl_param_.input_desc.size() == acl_param_.input_buffer.size(),
                               "Acl param input_desc size is not equal to acl param input_buffer size");
    for (size_t i = acl_param_.input_desc.size() - 1; i >= 0; --i) {
      if (acl_param_.input_desc[i] != nullptr && acl_param_.input_buffer[i] != nullptr) {
        return i + 1;
      }
    }
    return 0;
  }

  void ResizeOpOutputs(size_t size) {
    (void)std::for_each(acl_param_.output_desc.begin(), acl_param_.output_desc.end(), [](const aclTensorDesc *desc) {
      if (desc != nullptr) {
        aclDestroyTensorDesc(desc);
      }
    });
    (void)std::for_each(acl_param_.output_buffer.begin(), acl_param_.output_buffer.end(),
                        [](const aclDataBuffer *buffer) {
                          if (buffer != nullptr) {
                            aclDestroyDataBuffer(buffer);
                          }
                        });
    acl_param_.output_desc.clear();
    acl_param_.output_desc.resize(size, nullptr);
    acl_param_.output_buffer.clear();
    acl_param_.output_buffer.resize(size, nullptr);
  }

  void SetOutput(size_t i, const aclTensorDesc *desc, aclDataBuffer *buffer) {
    if (i >= acl_param_.output_desc.size() || i >= acl_param_.output_buffer.size()) {
      MS_LOG(EXCEPTION) << "Index " << i << " is out of bounds " << acl_param_.output_desc.size();
    }

    if (acl_param_.output_desc[i] != nullptr) {
      aclDestroyTensorDesc(acl_param_.output_desc[i]);
    }
    if (acl_param_.output_buffer[i] != nullptr) {
      aclDestroyDataBuffer(acl_param_.output_buffer[i]);
    }

    acl_param_.output_desc[i] = desc;
    acl_param_.output_buffer[i] = buffer;
  }

  size_t GetNumRealOutputs() const {
    if (acl_param_.output_desc.empty()) {
      return 0;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(acl_param_.output_desc.size() == acl_param_.output_buffer.size(),
                               "Acl param output_desc size is not equal to acl param output_buffer size");
    for (size_t i = acl_param_.output_desc.size() - 1; i >= 0; --i) {
      if (acl_param_.output_desc[i] != nullptr && acl_param_.output_buffer[i] != nullptr) {
        return i + 1;
      }
    }
    return 0;
  }

  template <typename data_type>
  void AddAttr(const std::string &attrName, data_type value) {
    InitAttr();
    MS_LOG(DEBUG) << "set acl attr:" << attrName << " value:" << value;
    AclAttrMaker::SetAttr(attrName, value, acl_param_.attr);
  }

  void AoeDump();

  void Run(void *stream_ptr, bool is_sync);

  std::vector<std::vector<int64_t>> SyncData();

  void Reset();

 private:
  void InitAttr() {
    if (acl_param_.attr == nullptr) {
      acl_param_.attr = aclopCreateAttr();
    }
  }

  std::string op_type_;
  bool is_dynamic_{true};
  AclExecParam acl_param_;
};  // class AclRunner
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_ACL_UTILS_H_
