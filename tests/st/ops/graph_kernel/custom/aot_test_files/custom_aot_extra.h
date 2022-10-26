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

#ifndef ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H
#define ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H

#include <string>
#include <vector>
#include <iostream>

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
  inline T Attr(std::string name) {
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

template <>
inline bool AotExtra::Attr(std::string name) {
  return GetAttrBool(name);
}

template <>
inline int64_t AotExtra::Attr(std::string name) {
  return GetAttrInt(name);
}

template <>
inline float AotExtra::Attr(std::string name) {
  return GetAttrFloat(name);
}

template <>
inline std::string AotExtra::Attr(std::string name) {
  return GetAttrStr(name);
}

template <>
inline std::vector<int64_t> AotExtra::Attr(std::string name) {
  return GetAttrIntVec(name);
}

template <>
inline std::vector<float> AotExtra::Attr(std::string name) {
  return GetAttrFloatVec(name);
}

template <>
inline std::vector<std::vector<int64_t>> AotExtra::Attr(std::string name) {
  return GetAttrInt2DVec(name);
}

template <>
inline std::vector<std::vector<float>> AotExtra::Attr(std::string name) {
  return GetAttrFloat2DVec(name);
}
#endif  // ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_H
