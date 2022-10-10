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
  virtual ~AotKernelData() = 0;
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
    if (kernel_data_ != nullptr) {
      delete kernel_data_;
      kernel_data_ = nullptr;
    }
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
  AotExtraImpl() : cnode_(nullptr) {}
  virtual ~AotExtraImpl() = default;
  void SetKernelNode(const CNodePtr &cnode) { cnode_ = cnode; }
  bool HasAttr(std::string name) final { return common::AnfAlgo::HasNodeAttr(name, cnode_); }

 private:
  bool GetAttrBool(std::string name) { return common::AnfAlgo::GetNodeAttr<bool>(this->cnode_, name); }
  int64_t GetAttrInt(std::string name) { return common::AnfAlgo::GetNodeAttr<int64_t>(this->cnode_, name); }
  float GetAttrFloat(std::string name) { return common::AnfAlgo::GetNodeAttr<float>(this->cnode_, name); }
  std::string GetAttrStr(std::string name) { return common::AnfAlgo::GetNodeAttr<std::string>(this->cnode_, name); }

  std::vector<int64_t> GetAttrIntVec(std::string name) {
    return common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(this->cnode_, name);
  }
  std::vector<float> GetAttrFloatVec(std::string name) {
    return common::AnfAlgo::GetNodeAttr<std::vector<float>>(this->cnode_, name);
  }
  std::vector<std::vector<int64_t>> GetAttrInt2DVec(std::string name) {
    return common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(this->cnode_, name);
  }
  std::vector<std::vector<float>> GetAttrFloat2DVec(std::string name) {
    return common::AnfAlgo::GetNodeAttr<std::vector<std::vector<float>>>(this->cnode_, name);
  }
  CNodePtr cnode_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H
