/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "include/api/visible.h"

namespace mindspore {
enum ModelType : uint32_t {
  kMindIR = 0,               ///< Model type is MindIR
  kAIR = 1,                  ///< Model type is AIR
  kOM = 2,                   ///< Model type is OM
  kONNX = 3,                 ///< Model type is ONNX
  kMindIR_Lite = 4,          ///< Model type is MindIR_LITE
  kUnknownType = 0xFFFFFFFF  ///< Unknown model type
};

enum QuantizationType : uint32_t {
  kNoQuant = 0,                   ///< Do not quantize
  kWeightQuant = 1,               ///< Only Quantize weight
  kFullQuant = 2,                 ///< Quantize whole network
  kUnknownQuantType = 0xFFFFFFFF  ///< Quantization type unknown
};

enum OptimizationLevel : uint32_t {
  kO0 = 0,                        ///< Do not optimize
  kO2 = 2,                        ///< Cast network to float16, keep batchnorm and loss in float32,
  kO3 = 3,                        ///< Cast network to float16, including bacthnorm
  kAuto = 4,                      ///< Choose optimization based on device
  kOptimizationType = 0xFFFFFFFF  ///< Unknown optimization type
};

struct QuantParam {
  int bit_num;         ///< Quantization bit num
  double scale;        ///< Quantization scale
  int32_t zero_point;  ///< Quantization zero point
  double min;          ///< Quantization min value
  double max;          ///< Quantization max value
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
  /// \param[in] own_data Whether the data memory should be freed in MSTensor destruction.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateRefTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                          const void *data, size_t data_len, bool own_data = true) noexcept;

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
  static inline MSTensor CreateDeviceTensor(const std::string &name, DataType type, const std::vector<int64_t> &shape,
                                            void *data, size_t data_len) noexcept;

  /// \brief Creates a MSTensor object from local file, must be used in pairs with DestroyTensorPtr.
  ///
  /// \param[in] file Path of file to be read.
  /// \param[in] type The data type of the MSTensor.
  /// \param[in] shape The shape of the MSTensor.
  ///
  /// \return A pointer of MSTensor.
  static inline MSTensor *CreateTensorFromFile(const std::string &file, DataType type = DataType::kNumberTypeUInt8,
                                               const std::vector<int64_t> &shape = {}) noexcept;

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

  /// \brief Destroy an object created by Clone, StringsToTensor, CreateRefTensor or CreateTensor. Do
  /// not use it to destroy MSTensor from other sources.
  ///
  /// \param[in] tensor A MSTensor object.
  static void DestroyTensorPtr(MSTensor *tensor) noexcept;

  MSTensor();
  explicit MSTensor(const std::shared_ptr<Impl> &impl);
  // if malloc data, user need to free after constructing MSTensor, else memory leak.
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

  /// \brief Get the boolean value that indicates whether the MSTensor not equals tensor.
  ///
  /// \param[in] another MSTensor.
  ///
  /// \return The boolean value that indicates whether the MSTensor not equals tensor.
  bool operator!=(const MSTensor &tensor) const;

  /// \brief Set the shape of for the MSTensor.
  ///
  /// \param[in] shape Shape of the MSTensor, a vector of int64_t.
  void SetShape(const std::vector<int64_t> &shape);

  /// \brief Set the data type for the MSTensor.
  ///
  /// \param[in] data_type The data type of the MSTensor.
  void SetDataType(enum DataType data_type);

  /// \brief Set the name for the MSTensor.
  ///
  /// \param[in] name The name of the MSTensor.
  inline void SetTensorName(const std::string &name);

  /// \brief Set the Allocator for the MSTensor.
  ///
  /// \param[in] allocator A pointer to Allocator.
  void SetAllocator(std::shared_ptr<Allocator> allocator);

  /// \brief Obtain the Allocator of the MSTensor.
  ///
  /// \return A pointer to Allocator.
  std::shared_ptr<Allocator> allocator() const;

  /// \brief Set the format for the MSTensor.
  ///
  /// \param[in] format The format of the MSTensor.
  void SetFormat(mindspore::Format format);

  /// \brief Obtain the format of the MSTensor.
  ///
  /// \return The format of the MSTensor.
  mindspore::Format format() const;

  /// \brief Set the data for the MSTensor.
  ///
  /// \note Deprecated, this interface will be removed in the next iteration
  ///
  /// \note A pointer to the data should be created by malloc interface
  ///
  /// \note The memory pointed to origin data pointer of MSTensor needs to be managed by the user
  ///
  /// \param[in] data A pointer to the data of the MSTensor.
  /// \param[in] own_data Whether the data memory should be freed in MSTensor destruction.
  void SetData(void *data, bool own_data = true);

  /// \brief Set the device data address for the MSTensor. Only valid for Lite.
  ///
  /// \note The memory pointed to origin data pointer of MSTensor needs to be managed by the user
  ///
  /// \param[in] data A pointer to the device data of the MSTensor.
  void SetDeviceData(void *data);

  /// \brief Get the device data address of the MSTensor set by SetDeviceData. Only valid for Lite.
  ///
  /// \return A pointer to the device data of the MSTensor.
  void *GetDeviceData();

  /// \brief Get the quantization parameters of the MSTensor.
  ///
  /// \return The quantization parameters of the MSTensor.
  std::vector<QuantParam> QuantParams() const;

  /// \brief Set the quantization parameters for the MSTensor.
  ///
  /// \param[in] quant_params The quantization parameters of the MSTensor.
  void SetQuantParams(std::vector<QuantParam> quant_params);

  const std::shared_ptr<Impl> impl() const { return impl_; }

 private:
  // api without std::string
  static MSTensor *CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                const void *data, size_t data_len) noexcept;
  static MSTensor *CreateRefTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                   const void *data, size_t data_len, bool own_data) noexcept;
  static MSTensor CreateDeviceTensor(const std::vector<char> &name, enum DataType type,
                                     const std::vector<int64_t> &shape, void *data, size_t data_len) noexcept;
  static MSTensor *CreateTensorFromFile(const std::vector<char> &file, enum DataType type,
                                        const std::vector<int64_t> &shape) noexcept;
  static MSTensor *CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &str);
  static std::vector<std::vector<char>> TensorToStringChars(const MSTensor &tensor);

  MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
           size_t data_len);
  std::vector<char> CharName() const;
  void SetTensorName(const std::vector<char> &name);

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
                                    const void *data, size_t data_len, bool own_data) noexcept {
  return CreateRefTensor(StringToChar(name), type, shape, data, data_len, own_data);
}

MSTensor MSTensor::CreateDeviceTensor(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                                      void *data, size_t data_len) noexcept {
  return CreateDeviceTensor(StringToChar(name), type, shape, data, data_len);
}

MSTensor *MSTensor::CreateTensorFromFile(const std::string &file, enum DataType type,
                                         const std::vector<int64_t> &shape) noexcept {
  return CreateTensorFromFile(StringToChar(file), type, shape);
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

void MSTensor::SetTensorName(const std::string &name) { SetTensorName(StringToChar(name)); }

using Key = struct MS_API Key {
  const size_t max_key_len = 32;
  size_t len = 0;
  unsigned char key[32] = {0};
  Key() : len(0) {}
  explicit Key(const char *dec_key, size_t key_len);
};

constexpr char kDecModeAesGcm[] = "AES-GCM";

/// \brief CallBackParam defined input arguments for callBack function.
struct MSCallBackParam {
  std::string node_name; /**< node name argument */
  std::string node_type; /**< node type argument */
  double execute_time;   /**< gpu execute time */
};

/// \brief KernelCallBack defined the function pointer for callBack.
using MSKernelCallBack =
  std::function<bool(const std::vector<MSTensor> & /* inputs */, const std::vector<MSTensor> & /* outputs */,
                     const MSCallBackParam &opInfo)>;

MS_API std::vector<char> CharVersion();
inline std::string Version() { return CharToString(CharVersion()); }

}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_TYPES_H
