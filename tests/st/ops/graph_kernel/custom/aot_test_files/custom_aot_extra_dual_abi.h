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

#ifndef ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H
#define ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H

#include <string>
#include <vector>
#include <iostream>

inline std::vector<char> StringToChar(const std::string &s) { return std::vector<char>(s.begin(), s.end()); }

inline std::string CharToString(const std::vector<char> &c) { return std::string(c.begin(), c.end()); }

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
  inline T Attr(std::string name) {
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

template <>
inline bool AotExtraDualABI::Attr(std::string name) {
  return GetAttrBool(StringToChar(name));
}

template <>
inline int64_t AotExtraDualABI::Attr(std::string name) {
  return GetAttrInt(StringToChar(name));
}

template <>
inline float AotExtraDualABI::Attr(std::string name) {
  return GetAttrFloat(StringToChar(name));
}

template <>
inline std::string AotExtraDualABI::Attr(std::string name) {
  return CharToString(GetAttrStr(StringToChar(name)));
}

template <>
inline std::vector<int64_t> AotExtraDualABI::Attr(std::string name) {
  return GetAttrIntVec(StringToChar(name));
}

template <>
inline std::vector<float> AotExtraDualABI::Attr(std::string name) {
  return GetAttrFloatVec(StringToChar(name));
}

template <>
inline std::vector<std::vector<int64_t>> AotExtraDualABI::Attr(std::string name) {
  return GetAttrInt2DVec(StringToChar(name));
}

template <>
inline std::vector<std::vector<float>> AotExtraDualABI::Attr(std::string name) {
  return GetAttrFloat2DVec(StringToChar(name));
}
#endif  // ST_MINDSPORE_CCSRC_UTILS_CUSTOM_AOT_EXTRA_DUAL_ABI_H
