/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H
#define MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H

#include <string>
#include <vector>
#include "ir/anf.h"
#include "mindspore/ccsrc/include/common/utils/anfalgo.h"

namespace mindspore {
class AotKernelData {
 public:
  AotKernelData() = default;
  virtual ~AotKernelData() = default;
};

class AotExtra {
 public:
  AotExtra() = default;
  virtual ~AotExtra() = default;
  virtual bool HasAttr(std::string name) = 0;

  template <typename T>
  inline T Attr(std::string name) const {
    MS_EXCEPTION_IF_CHECK_FAIL(name.length() > 0, "The input name is an empty string");
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  void SetKernelData(AotKernelData *kernel_data) { kernel_data_ = kernel_data; }
  AotKernelData *KernelData() const { return kernel_data_; }

  void DestructKernelData() {
    delete kernel_data_;
    kernel_data_ = nullptr;
  }

 private:
  virtual bool GetAttrBool(std::string name) = 0;
  virtual int64_t GetAttrInt(std::string name) = 0;
  virtual float GetAttrFloat(std::string name) = 0;
  virtual std::string GetAttrStr(std::string name) = 0;

  virtual std::vector<int64_t> GetAttrIntVec(std::string name) = 0;
  virtual std::vector<float> GetAttrFloatVec(std::string name) = 0;
  virtual std::vector<std::vector<int64_t>> GetAttrInt2DVec(std::string name) = 0;
  virtual std::vector<std::vector<float>> GetAttrFloat2DVec(std::string name) = 0;
  std::vector<size_t> workspace_;

  AotKernelData *kernel_data_{nullptr};
};

class AotExtraImpl : public AotExtra {
 public:
  AotExtraImpl() : prim_(nullptr) {}
  virtual ~AotExtraImpl() = default;
  void SetKernelPrim(const PrimitivePtr &prim) { prim_ = prim; }
  bool HasAttr(std::string name) final { return prim_ != nullptr && prim_->HasAttr(name); }

 private:
  bool GetAttrBool(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<bool>(prim_->GetAttr(name));
  }
  int64_t GetAttrInt(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<int64_t>(prim_->GetAttr(name));
  }
  float GetAttrFloat(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<float>(prim_->GetAttr(name));
  }
  std::string GetAttrStr(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<std::string>(prim_->GetAttr(name));
  }

  std::vector<int64_t> GetAttrIntVec(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<std::vector<int64_t>>(prim_->GetAttr(name));
  }
  std::vector<float> GetAttrFloatVec(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<std::vector<float>>(prim_->GetAttr(name));
  }
  std::vector<std::vector<int64_t>> GetAttrInt2DVec(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<std::vector<std::vector<int64_t>>>(prim_->GetAttr(name));
  }
  std::vector<std::vector<float>> GetAttrFloat2DVec(std::string name) {
    MS_EXCEPTION_IF_NULL(prim_);
    return GetValue<std::vector<std::vector<float>>>(prim_->GetAttr(name));
  }
  PrimitivePtr prim_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H
