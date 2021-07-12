/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_TYPES_H
#define MINDSPORE_INCLUDE_API_TYPES_H

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "include/api/data_type.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/format.h"

#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif

namespace mindspore {
enum ModelType : uint32_t {
  kMindIR = 0,
  kAIR = 1,
  kOM = 2,
  kONNX = 3,
  kFlatBuffer = 4,
  // insert new data type here
  kUnknownType = 0xFFFFFFFF
};

enum QuantizationType : uint32_t {
  kNoQuant = 0,
  kWeightQuant = 1,
  kFullQuant = 2,
  kUnknownQuantType = 0xFFFFFFFF
};

enum OptimizationLevel : uint32_t {
  kO0 = 0,  // Do not change
  kO2 = 2,  // Cast network to float16, keep batchnorm and loss in float32,
  kO3 = 3,  // Cast network to float16, including bacthnorm
  kAuto = 4,  // Choose optimization based on device
  kOptimizationType = 0xFFFFFFFF
};

class Allocator;
class MS_API MSTensor {
 public:
  class Impl;

  static inline MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                       const void *data, size_t data_len) noexcept;
  static inline MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          const void *data, size_t data_len) noexcept;
  static inline MSTensor *CreateDevTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          const void *data, size_t data_len) noexcept;
  static inline MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str);
  static inline std::vector<std::string> TensorToStrings(const MSTensor &tensor);
  static void DestroyTensorPtr(MSTensor *tensor) noexcept;

  MSTensor();
  explicit MSTensor(const std::shared_ptr<Impl> &impl);
  inline MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data,
                  size_t data_len);
  explicit MSTensor(std::nullptr_t);
  ~MSTensor();

  inline std::string Name() const;
  enum DataType DataType() const;
  const std::vector<int64_t> &Shape() const;
  int64_t ElementNum() const;

  std::shared_ptr<const void> Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool IsDevice() const;

  MSTensor *Clone() const;
  bool operator==(std::nullptr_t) const;
  bool operator!=(std::nullptr_t) const;
  bool operator==(const MSTensor &tensor) const;

  void SetShape(const std::vector<int64_t> &shape);
  void SetDataType(enum DataType data_type);
  void SetTensorName(const std::string &name);
  void SetAllocator(std::shared_ptr<Allocator> allocator);
  std::shared_ptr<Allocator> allocator() const;
  void SetFormat(mindspore::Format format);
  mindspore::Format format() const;
  void SetData(void *data);
  const std::shared_ptr<Impl> impl() const { return impl_; }

 private:
  // api without std::string
  static MSTensor *CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                const void *data, size_t data_len) noexcept;
  static MSTensor *CreateRefTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept;
  static MSTensor *CreateDevTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept;
  static MSTensor *CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &str);
  static std::vector<std::vector<char>> TensorToStringChars(const MSTensor &tensor);

  MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
           size_t data_len);
  std::vector<char> CharName() const;

  friend class ModelImpl;
  std::shared_ptr<Impl> impl_;
};

class MS_API Buffer {
 public:
  Buffer();
  Buffer(const void *data, size_t data_len);
  ~Buffer();

  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool ResizeData(size_t data_len);
  bool SetData(const void *data, size_t data_len);

  Buffer Clone() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

MSTensor *MSTensor::CreateTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                 const void *data, size_t data_len) noexcept {
  return CreateTensor(StringToChar(name), type, shape, data, data_len);
}

MSTensor *MSTensor::CreateRefTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                    const void *data, size_t data_len) noexcept {
  return CreateRefTensor(StringToChar(name), type, shape, data, data_len);
}

MSTensor *MSTensor::CreateDevTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                    const void *data, size_t data_len) noexcept {
  return CreateDevTensor(StringToChar(name), type, shape, data, data_len);
}

MSTensor *MSTensor::StringsToTensor(const std::string &name, const std::vector<std::string> &str) {
  return CharStringsToTensor(StringToChar(name), VectorStringToChar(str));
}

std::vector<std::string> MSTensor::TensorToStrings(const MSTensor &tensor) {
  return VectorCharToString(TensorToStringChars(tensor));
}

MSTensor::MSTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                   size_t data_len)
    : MSTensor(StringToChar(name), type, shape, data, data_len) {}

std::string MSTensor::Name() const { return CharToString(CharName()); }


struct MS_API Key {
  const size_t max_key_len = 32;
  size_t len;
  unsigned char key[32];
  Key() : len(0) {}
  explicit Key(const char *dec_key, size_t key_len);
};
constexpr char kDecModeAesGcm[] = "AES-GCM";

/// \brief CallBackParam defined input arguments for callBack function.
struct MSCallBackParam {
  std::string node_name_; /**< node name argument */
  std::string node_type_; /**< node type argument */
};

/// \brief KernelCallBack defined the function pointer for callBack.
using MSKernelCallBack = std::function<bool(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                            const MSCallBackParam &opInfo)>;

std::vector<char> CharVersion();
inline std::string Version() { return CharToString(CharVersion()); }

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_TYPES_H
