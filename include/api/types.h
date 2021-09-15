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

#ifndef MS_API
#ifdef _WIN32
#define MS_API __declspec(dllexport)
#else
#define MS_API __attribute__((visibility("default")))
#endif
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

enum QuantizationType : uint32_t { kNoQuant = 0, kWeightQuant = 1, kFullQuant = 2, kUnknownQuantType = 0xFFFFFFFF };

enum OptimizationLevel : uint32_t {
  kO0 = 0,    // Do not change
  kO2 = 2,    // Cast network to float16, keep batchnorm and loss in float32,
  kO3 = 3,    // Cast network to float16, including bacthnorm
  kAuto = 4,  // Choose optimization based on device
  kOptimizationType = 0xFFFFFFFF
};

struct QuantParam {
  int bit_num;
  double scale;
  int32_t zero_point;
};

class Allocator;
/// \brief The MSTensor class defines a tensor in MindSpore.
class MS_API MSTensor {
 public:
  class Impl;
  /// \brief Creates a MSTensor object, whose data need to be copied before accessed by Model, must be used in pairs
  /// with DestroyTensorPtr.
  ///
  /// \param[in] name The name of the MSTensor.
  /// \param[in] type The data type of the MSTensor.
  /// \param[in] shape The shape of the MSTensor.
  /// \param[in] data The data pointer that points to allocated memory.
  /// \param[in] data_len The length of the memory, in bytes.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                       const void *data, size_t data_len) noexcept;

  /// \brief Creates a MSTensor object, whose data can be directly accessed by Model, must be used in pairs with
  /// DestroyTensorPtr.
  ///
  /// \param[in] name The name of the MSTensor.
  /// \param[in] type The data type of the MSTensor.
  /// \param[in] shape The shape of the MSTensor.
  /// \param[in] data The data pointer that points to allocated memory.
  /// \param[in] data_len The length of the memory, in bytes.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          const void *data, size_t data_len) noexcept;

  /// \brief Creates a MSTensor object, whose device data can be directly accessed by Model, must be used in pairs with
  /// DestroyTensorPtr.
  ///
  /// \param[in] name The name of the MSTensor.
  /// \param[in] type The data type of the MSTensor.
  /// \param[in] shape The shape of the MSTensor.
  /// \param[in] data The data pointer that points to device memory.
  /// \param[in] data_len The length of the memory, in bytes.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateDevTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          const void *data, size_t data_len) noexcept;

  /// \brief Creates a MSTensor object from local image file, must be used in pairs with DestroyTensorPtr.
  ///
  /// \param[in] image_file Path of image file.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateImageTensor(const std::string &image_file) noexcept;

  /// \brief Create a string type MSTensor object whose data can be accessed by Model only after being copied, must be
  /// used in pair with DestroyTensorPtr.
  ///
  /// \param[in] name The name of the MSTensor.
  /// \param[in] str A vector container containing several strings.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *StringsToTensor(const std::string &name, const std::vector<std::string> &str);

  /// \brief Parse the string type MSTensor object into strings.
  ///
  /// \param[in] tensor A MSTensor object.
  ///
  /// \return A vector container containing several strings.
  static inline std::vector<std::string> TensorToStrings(const MSTensor &tensor);

  /// \brief Destroy an object created by Clone, StringsToTensor, CreateRefTensor, CreateDevTensor or CreateTensor. Do
  /// not use it to destroy MSTensor from other sources.
  ///
  /// \param[in] tensor A MSTensor object.
  static void DestroyTensorPtr(MSTensor *tensor) noexcept;

  MSTensor();
  explicit MSTensor(const std::shared_ptr<Impl> &impl);
  inline MSTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data,
                  size_t data_len);
  explicit MSTensor(std::nullptr_t);
  ~MSTensor();

  /// \brief Obtains the name of the MSTensor.
  ///
  /// \return The name of the MSTensor.
  inline std::string Name() const;

  /// \brief Obtains the data type of the MSTensor.
  ///
  /// \return The data type of the MSTensor.
  enum DataType DataType() const;

  /// \brief Obtains the shape of the MSTensor.
  ///
  /// \return The shape of the MSTensor.
  const std::vector<int64_t> &Shape() const;

  /// \brief Obtains the number of elements of the MSTensor.
  ///
  /// \return The number of elements of the MSTensor.
  int64_t ElementNum() const;

  /// \brief Obtains a shared pointer to the copy of data of the MSTensor. The data can be read on host.
  ///
  /// \return A shared pointer to the copy of data of the MSTensor.
  std::shared_ptr<const void> Data() const;

  /// \brief Obtains the pointer to the data of the MSTensor. If the MSTensor is a device tensor, the data cannot be
  /// accessed directly on host.
  ///
  /// \return A pointer to the data of the MSTensor.
  void *MutableData();

  /// \brief Obtains the length of the data of the MSTensor, in bytes.
  ///
  /// \return The length of the data of the MSTensor, in bytes.
  size_t DataSize() const;

  /// \brief Get whether the MSTensor data is const data
  ///
  /// \return Const flag of MSTensor
  bool IsConst() const;

  /// \brief Gets the boolean value that indicates whether the memory of MSTensor is on device.
  ///
  /// \return The boolean value that indicates whether the memory of MSTensor is on device.
  bool IsDevice() const;

  /// \brief Gets a deep copy of the MSTensor, must be used in pair with DestroyTensorPtr.
  ///
  /// \return A pointer points to a deep copy of the MSTensor.
  MSTensor *Clone() const;

  /// \brief Gets the boolean value that indicates whether the MSTensor is valid.
  ///
  /// \return The boolean value that indicates whether the MSTensor is valid.
  bool operator==(std::nullptr_t) const;

  /// \brief Gets the boolean value that indicates whether the MSTensor is valid.
  ///
  /// \return The boolean value that indicates whether the MSTensor is valid.
  bool operator!=(std::nullptr_t) const;

  /// \brief Get the boolean value that indicates whether the MSTensor equals tensor.
  ///
  /// \param[in] another MSTensor.
  ///
  /// \return The boolean value that indicates whether the MSTensor equals tensor.
  bool operator==(const MSTensor &tensor) const;

  /// \brief Set the shape of for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] Shape of the MSTensor, a vector of int64_t.
  void SetShape(const std::vector<int64_t> &shape);

  /// \brief Set the data type for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] The data type of the MSTensor.
  void SetDataType(enum DataType data_type);

  /// \brief Set the name for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] The name of the MSTensor.
  void SetTensorName(const std::string &name);

  /// \brief Set the Allocator for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] A pointer to Allocator.
  void SetAllocator(std::shared_ptr<Allocator> allocator);

  /// \brief Obtain the Allocator of the MSTensor. Only valid for Lite.
  ///
  /// \return A pointer to Allocator.
  std::shared_ptr<Allocator> allocator() const;

  /// \brief Set the format for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] The format of the MSTensor.
  void SetFormat(mindspore::Format format);

  /// \brief Obtain the format of the MSTensor. Only valid for Lite.
  ///
  /// \return The format of the MSTensor.
  mindspore::Format format() const;

  /// \brief Set the data for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] A pointer to the data of the MSTensor.
  void SetData(void *data);

  /// \brief Get the quantization parameters of the MSTensor. Only valid for Lite.
  ///
  /// \return The quantization parameters of the MSTensor.
  std::vector<QuantParam> QuantParams() const;

  /// \brief Set the quantization parameters for the MSTensor. Only valid for Lite.
  ///
  /// \param[in] The quantization parameters of the MSTensor.
  void SetQuantParams(std::vector<QuantParam> quant_params);

  const std::shared_ptr<Impl> impl() const { return impl_; }

 private:
  // api without std::string
  static MSTensor *CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                const void *data, size_t data_len) noexcept;
  static MSTensor *CreateRefTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept;
  static MSTensor *CreateDevTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len) noexcept;
  static MSTensor *CreateImageTensor(const std::vector<char> &image_file) noexcept;
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

MSTensor *MSTensor::CreateImageTensor(const std::string &image_file) noexcept {
  return CreateImageTensor(StringToChar(image_file));
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

using Key = struct Key {
  const size_t max_key_len = 32;
  size_t len;
  unsigned char key[32];
  Key() : len(0) {}
  explicit Key(const char *dec_key, size_t key_len);
};

constexpr char kDecModeAesGcm[] = "AES-GCM";

/// \brief CallBackParam defined input arguments for callBack function.
struct MSCallBackParam {
  std::string node_name; /**< node name argument */
  std::string node_type; /**< node type argument */
};

/// \brief KernelCallBack defined the function pointer for callBack.
using MSKernelCallBack = std::function<bool(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                            const MSCallBackParam &opInfo)>;

std::vector<char> CharVersion();
inline std::string Version() { return CharToString(CharVersion()); }

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_TYPES_H
