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

#ifndef MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H
#define MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H

#include <string>
#include <vector>
#include "ir/anf.h"
#include "include/api/dual_abi_helper.h"
#include "mindspore/ccsrc/include/common/utils/anfalgo.h"

namespace mindspore {
class AotKernelDataDualABI {
 public:
  AotKernelDataDualABI() = default;
  virtual ~AotKernelDataDualABI() = default;
};

class AotExtraDualABI {
 public:
  AotExtraDualABI() = default;
  virtual ~AotExtraDualABI() = default;
  virtual bool HasAttr(std::string name) { return HasAttr(StringToChar(name)); }

  template <typename T>
  inline T Attr(std::string name) const {
    MS_EXCEPTION_IF_CHECK_FAIL(name.length() > 0, "The input name is an empty string");
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  void SetKernelData(AotKernelDataDualABI *kernel_data) { kernel_data_ = kernel_data; }
  AotKernelDataDualABI *KernelData() const { return kernel_data_; }

  void DestructKernelData() {
    delete kernel_data_;
    kernel_data_ = nullptr;
  }

 private:
  virtual bool HasAttr(std::vector<char> name) = 0;
  virtual bool GetAttrBool(std::vector<char> name) = 0;
  virtual int64_t GetAttrInt(std::vector<char> name) = 0;
  virtual float GetAttrFloat(std::vector<char> name) = 0;
  virtual std::vector<char> GetAttrStr(std::vector<char> name) = 0;

  virtual std::vector<int64_t> GetAttrIntVec(std::vector<char> name) = 0;
  virtual std::vector<float> GetAttrFloatVec(std::vector<char> name) = 0;
  virtual std::vector<std::vector<int64_t>> GetAttrInt2DVec(std::vector<char> name) = 0;
  virtual std::vector<std::vector<float>> GetAttrFloat2DVec(std::vector<char> name) = 0;
  std::vector<size_t> workspace_;

  AotKernelDataDualABI *kernel_data_{nullptr};
};

class AotExtraDualABIImpl : public AotExtraDualABI {
 public:
  AotExtraDualABIImpl() : prim_(nullptr) {}
  virtual ~AotExtraDualABIImpl() = default;
  void SetKernelPrim(const PrimitivePtr &prim) { prim_ = prim; }

 private:
  bool HasAttr(std::vector<char> name) final { return prim_ != nullptr && prim_->HasAttr(CharToString(name)); }

  bool GetAttrBool(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<bool>(value);
  }
  int64_t GetAttrInt(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<int64_t>(value);
  }
  float GetAttrFloat(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<float>(value);
  }
  std::vector<char> GetAttrStr(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return StringToChar(GetValue<std::string>(value));
  }

  std::vector<int64_t> GetAttrIntVec(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<std::vector<int64_t>>(value);
  }
  std::vector<float> GetAttrFloatVec(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<std::vector<float>>(value);
  }
  std::vector<std::vector<int64_t>> GetAttrInt2DVec(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<std::vector<std::vector<int64_t>>>(value);
  }
  std::vector<std::vector<float>> GetAttrFloat2DVec(std::vector<char> name) {
    MS_EXCEPTION_IF_NULL(prim_);
    auto value = prim_->GetAttr(CharToString(name));
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << prim_->ToString() << ", there is no attribute called " << name << "! ";
    }
    return GetValue<std::vector<std::vector<float>>>(value);
  }
  PrimitivePtr prim_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H
