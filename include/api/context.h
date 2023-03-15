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
#ifndef MINDSPORE_INCLUDE_API_CONTEXT_H
#define MINDSPORE_INCLUDE_API_CONTEXT_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "include/api/types.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
enum DelegateMode {
  kNoDelegate = 0,
  kCoreML = 1,
  kNNAPI = 2,
};

enum DeviceType {
  kCPU = 0,
  kGPU,
  kKirinNPU,
  kAscend,
  kAscend910,
  kAscend310,
  kCustomDevice,
  kAllDevice,
  // add new type here
  kInvalidDeviceType = 100,
};

class Allocator;
class AbstractDelegate;
class Delegate;
class DeviceInfoContext;

/// \brief Context is used to store environment variables during execution.
class MS_API Context {
 public:
  struct Data;
  Context();
  ~Context() = default;
  Context(const Context &rhs) : data_(rhs.data_) {}

  /// \brief Set the number of threads at runtime.
  ///
  /// \param[in] thread_num the number of threads at runtime.
  void SetThreadNum(int32_t thread_num);

  /// \brief Get the current thread number setting.
  ///
  /// \return The current thread number setting.
  int32_t GetThreadNum() const;

  /// \brief Set the parallel number of operators at runtime.
  ///
  /// \param[in] parallel_num the parallel number of operators at runtime.
  void SetInterOpParallelNum(int32_t parallel_num);

  /// \brief Get the current operators parallel number setting.
  ///
  /// \return The current operators parallel number setting.
  int32_t GetInterOpParallelNum() const;

  /// \brief Set the thread affinity to CPU cores.
  ///
  /// \param[in] mode: 0: no affinities, 1: big cores first, 2: little cores first
  void SetThreadAffinity(int mode);

  /// \brief Get the thread affinity of CPU cores.
  ///
  /// \return Thread affinity to CPU cores. 0: no affinities, 1: big cores first, 2: little cores first
  int GetThreadAffinityMode() const;

  /// \brief Set the thread lists to CPU cores.
  ///
  /// \note If core_list and mode are set by SetThreadAffinity at the same time, the core_list is effective, but the
  /// mode is not effective.
  ///
  /// \param[in] core_list: a vector of thread core lists.
  void SetThreadAffinity(const std::vector<int> &core_list);

  /// \brief Get the thread lists of CPU cores.
  ///
  /// \return core_list: a vector of thread core lists.
  std::vector<int32_t> GetThreadAffinityCoreList() const;

  /// \brief Set the status whether to perform model inference or training in parallel.
  ///
  /// \param[in] is_parallel: true, parallel; false, not in parallel.
  void SetEnableParallel(bool is_parallel);

  /// \brief Get the status whether to perform model inference or training in parallel.
  ///
  /// \return Bool value that indicates whether in parallel.
  bool GetEnableParallel() const;

  /// \brief Set built-in delegate mode to access third-party AI framework.
  ///
  /// \param[in] mode the built-in delegate mode.
  void SetBuiltInDelegate(DelegateMode mode);

  /// \brief Get the built-in delegate mode of the third-party AI framework.
  ///
  /// \return the built-in delegate mode.
  DelegateMode GetBuiltInDelegate() const;

  /// \brief Set Delegate to access third-party AI framework.
  ///
  /// \param[in] delegate the custom delegate.
  void set_delegate(const std::shared_ptr<AbstractDelegate> &delegate);

  // deprecated
  void SetDelegate(const std::shared_ptr<Delegate> &delegate);

  /// \brief Get the delegate of the third-party AI framework.
  ///
  /// \return Pointer to the custom delegate.
  std::shared_ptr<AbstractDelegate> get_delegate() const;

  // deprecated
  std::shared_ptr<Delegate> GetDelegate() const;

  /// \brief Set quant model to run as float model in multi device.
  ///
  /// \param[in] float_mode: true, run as float model; false, not run as float model.
  void SetMultiModalHW(bool float_mode);

  /// \brief Get the mode of the model run.
  ///
  /// \return Bool value that indicates whether run as float model
  bool GetMultiModalHW() const;

  /// \brief Get a mutable reference of DeviceInfoContext vector in this context. Only MindSpore Lite supports
  /// heterogeneous scenarios with multiple members in the vector.
  ///
  /// \return Mutable reference of DeviceInfoContext vector in this context.
  std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo();

 private:
  std::shared_ptr<Data> data_;
};

/// \brief DeviceInfoContext defines different device contexts.
class MS_API DeviceInfoContext : public std::enable_shared_from_this<DeviceInfoContext> {
 public:
  struct Data;

  DeviceInfoContext();
  virtual ~DeviceInfoContext() = default;

  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  virtual enum DeviceType GetDeviceType() const = 0;

  /// \brief A similar function to RTTI is provided when the -fno-rtti compilation option is turned on, which converts
  /// DeviceInfoContext to a shared pointer of type T, and returns nullptr if the conversion fails.
  ///
  /// \return A pointer of type T after conversion. If the conversion fails, it will be nullptr.
  template <class T>
  std::shared_ptr<T> Cast() {
    static_assert(std::is_base_of<DeviceInfoContext, T>::value, "Wrong cast type.");
    if (GetDeviceType() != T().GetDeviceType()) {
      return nullptr;
    }

    return std::static_pointer_cast<T>(shared_from_this());
  }
  /// \brief obtain provider's name
  ///
  /// \return provider's name.
  inline std::string GetProvider() const;

  /// \brief set provider's name.
  ///
  /// \param[in] provider define the provider's name.
  inline void SetProvider(const std::string &provider);

  /// \brief obtain provider's device type.
  ///
  /// \return provider's device type.
  inline std::string GetProviderDevice() const;

  /// \brief set provider's device type.
  ///
  /// \param[in] device define the provider's device type.EG: CPU.
  inline void SetProviderDevice(const std::string &device);

  /// \brief set memory allocator.
  ///
  /// \param[in] allocator define the memory allocator which can be defined by user.
  void SetAllocator(const std::shared_ptr<Allocator> &allocator);

  /// \brief obtain memory allocator.
  ///
  /// \return memory allocator.
  std::shared_ptr<Allocator> GetAllocator() const;

 protected:
  std::vector<char> GetProviderChar() const;
  void SetProvider(const std::vector<char> &provider);
  std::vector<char> GetProviderDeviceChar() const;
  void SetProviderDevice(const std::vector<char> &device);

  std::shared_ptr<Data> data_;
};

std::string DeviceInfoContext::GetProvider() const { return CharToString(GetProviderChar()); }
void DeviceInfoContext::SetProvider(const std::string &provider) { SetProvider(StringToChar(provider)); }
std::string DeviceInfoContext::GetProviderDevice() const { return CharToString(GetProviderDeviceChar()); }
void DeviceInfoContext::SetProviderDevice(const std::string &device) { SetProviderDevice(StringToChar(device)); }

/// \brief Derived from DeviceInfoContext, The configuration of the model running auto on the Host Devices, include
/// CPU/GPU/NPU/Ascend310/Ascend910. This option is only valid for MindSpore Lite.
class MS_API AutoDeviceInfo : public DeviceInfoContext {
 public:
  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  enum DeviceType GetDeviceType() const override { return DeviceType::kAllDevice; };
};

/// \brief Derived from DeviceInfoContext, The configuration of the model running on the CPU. This option is only valid
/// for MindSpore Lite.
class MS_API CPUDeviceInfo : public DeviceInfoContext {
 public:
  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  enum DeviceType GetDeviceType() const override { return DeviceType::kCPU; };

  /// \brief Set enables to perform the float16 inference
  ///
  /// \param[in] is_fp16 Enable float16 inference or not.
  void SetEnableFP16(bool is_fp16);

  /// \brief Get enables to perform the float16 inference
  ///
  /// \return Whether enable float16 inference.
  bool GetEnableFP16() const;
};

/// \brief Derived from DeviceInfoContext, The configuration of the model running on the NPU. This option is only valid
/// for MindSpore Lite.
class MS_API KirinNPUDeviceInfo : public DeviceInfoContext {
 public:
  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  enum DeviceType GetDeviceType() const override { return DeviceType::kKirinNPU; };

  /// \brief Set enables to perform the float16 inference
  ///
  /// \param[in] is_fp16 Enable float16 inference or not.
  void SetEnableFP16(bool is_fp16);

  /// \brief Get enables to perform the float16 inference
  ///
  /// \return Whether enable float16 inference.
  bool GetEnableFP16() const;

  /// \brief Set the NPU frequency.
  ///
  /// \param[in] frequency Can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme
  /// performance), default as 3.
  void SetFrequency(int frequency);

  /// \brief Get the NPU frequency.
  ///
  /// \return NPU frequency
  int GetFrequency() const;
};

/// \brief Derived from DeviceInfoContext, The configuration of the model running on the GPU.
class MS_API GPUDeviceInfo : public DeviceInfoContext {
 public:
  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  enum DeviceType GetDeviceType() const override { return DeviceType::kGPU; };

  /// \brief Set device id.
  ///
  /// \param[in] device_id The device id.
  void SetDeviceID(uint32_t device_id);

  /// \brief Get the device id.
  ///
  /// \return The device id.
  uint32_t GetDeviceID() const;

  /// \brief Get the distribution rank id.
  ///
  /// \return The device id.
  int GetRankID() const;

  /// \brief Get the distribution group size.
  ///
  /// \return The device id.
  int GetGroupSize() const;

  /// \brief Set the precision mode.
  ///
  /// \param[in] precision_mode Optional "origin", "fp16". "origin" is set as default.
  inline void SetPrecisionMode(const std::string &precision_mode);

  /// \brief Get the precision mode.
  ///
  /// \return The precision mode.
  inline std::string GetPrecisionMode() const;

  /// \brief Set enables to perform the float16 inference
  ///
  /// \param[in] is_fp16 Enable float16 inference or not.
  void SetEnableFP16(bool is_fp16);

  /// \brief Get enables to perform the float16 inference
  ///
  /// \return Whether enable float16 inference.
  bool GetEnableFP16() const;

  /// \brief Set enables to sharing mem with OpenGL
  ///
  /// \param[in] is_enable_gl_texture Enable sharing OpenCL Memory with OpenGL or not.
  void SetEnableGLTexture(bool is_enable_gl_texture);

  /// \brief Get enables to sharing mem with OpenGL
  ///
  /// \return Whether enable sharing mem with OpenGL.
  bool GetEnableGLTexture() const;

  /// \brief Set current OpenGL context
  ///
  /// \param[in] gl_context Current OpenGL context.
  void SetGLContext(void *gl_context);

  /// \brief Get current OpenGL context
  ///
  /// \return the OpenCL context by OpenGL used.
  void *GetGLContext() const;

  /// \brief Set current OpenGL display
  ///
  /// \param[in] gl_display Current OpenGL display.
  void SetGLDisplay(void *gl_display);

  /// \brief Get current OpenGL display
  ///
  /// \return the OpenCL display by OpenGL used.
  void *GetGLDisplay() const;

 private:
  void SetPrecisionMode(const std::vector<char> &precision_mode);
  std::vector<char> GetPrecisionModeChar() const;
};

void GPUDeviceInfo::SetPrecisionMode(const std::string &precision_mode) {
  SetPrecisionMode(StringToChar(precision_mode));
}
std::string GPUDeviceInfo::GetPrecisionMode() const { return CharToString(GetPrecisionModeChar()); }

/// \brief Derived from DeviceInfoContext, The configuration of the model running on the Ascend. This option is
/// invalid for MindSpore Lite.
class MS_API AscendDeviceInfo : public DeviceInfoContext {
 public:
  /// \brief Get the type of this DeviceInfoContext.
  ///
  /// \return Type of this DeviceInfoContext.
  enum DeviceType GetDeviceType() const override { return DeviceType::kAscend; };

  /// \brief Set device id.
  ///
  /// \param[in] device_id The device id.
  void SetDeviceID(uint32_t device_id);

  /// \brief Get the device id.
  ///
  /// \return The device id.
  uint32_t GetDeviceID() const;

  /// \brief Set AIPP configuration file path.
  ///
  /// \param[in] cfg_path AIPP configuration file path.
  inline void SetInsertOpConfigPath(const std::string &cfg_path);

  /// \brief Get AIPP configuration file path.
  ///
  /// \return AIPP configuration file path.
  inline std::string GetInsertOpConfigPath() const;

  /// \brief Set format of model inputs.
  ///
  /// \param[in] format Optional "NCHW", "NHWC", etc.
  inline void SetInputFormat(const std::string &format);

  /// \brief Get format of model inputs.
  ///
  /// \return The format of model inputs.
  inline std::string GetInputFormat() const;

  /// \brief Set shape of model inputs.
  ///
  /// \param[in] shape e.g. "input_op_name1: 1,2,3,4;input_op_name2: 4,3,2,1".
  inline void SetInputShape(const std::string &shape);

  /// \brief Get shape of model inputs.
  ///
  /// \return The shape of model inputs.
  inline std::string GetInputShape() const;

  /// \brief Set shape of model inputs.
  ///
  /// \param[in] shape e.g. {{1, {1,2,3,4}}, {2, {4,3,2,1}}} means the first input shape 1,2,3,4 and the second input
  /// shape 4,3,2,1.
  void SetInputShapeMap(const std::map<int, std::vector<int>> &shape);

  /// \brief Get shape of model inputs.
  ///
  /// \return The shape of model inputs.
  std::map<int, std::vector<int>> GetInputShapeMap() const;

  void SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size);
  inline std::string GetDynamicBatchSize() const;

  /// \brief Set the dynamic image size of model inputs.
  ///
  /// \param[in] dynamic_image_size size hw e.g. "66,88;32,64" means h1:66,w1:88; h2:32,w2:64.
  inline void SetDynamicImageSize(const std::string &dynamic_image_size);

  /// \brief Get dynamic image size of model inputs.
  ///
  /// \return The image size of model inputs.
  inline std::string GetDynamicImageSize() const;

  /// \brief Set type of model outputs.
  ///
  /// \param[in] output_type FP32, UINT8 or FP16, default as FP32.
  void SetOutputType(enum DataType output_type);

  /// \brief Get type of model outputs.
  ///
  /// \return The set type of model outputs.
  enum DataType GetOutputType() const;

  /// \brief Set precision mode of model.
  ///
  /// \param[in] precision_mode Optional "force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype" and
  /// "allow_mix_precision", "force_fp16" is set as default
  inline void SetPrecisionMode(const std::string &precision_mode);

  /// \brief Get precision mode of model.
  ///
  /// \return The set type of model outputs
  inline std::string GetPrecisionMode() const;

  /// \brief Set op select implementation mode.
  ///
  /// \param[in] op_select_impl_mode Optional "high_performance" and "high_precision", "high_performance" is set as
  /// default.
  inline void SetOpSelectImplMode(const std::string &op_select_impl_mode);

  /// \brief Get op select implementation mode.
  ///
  /// \return The set op select implementation mode.
  inline std::string GetOpSelectImplMode() const;

  inline void SetFusionSwitchConfigPath(const std::string &cfg_path);
  inline std::string GetFusionSwitchConfigPath() const;

  // Optional "l1_optimize", "l2_optimize", "off_optimize" or "l1_and_l2_optimize", default as "l2_optimize"
  inline void SetBufferOptimizeMode(const std::string &buffer_optimize_mode);
  inline std::string GetBufferOptimizeMode() const;

 private:
  void SetInsertOpConfigPath(const std::vector<char> &cfg_path);
  std::vector<char> GetInsertOpConfigPathChar() const;

  void SetInputFormat(const std::vector<char> &format);
  std::vector<char> GetInputFormatChar() const;

  void SetInputShape(const std::vector<char> &shape);
  std::vector<char> GetInputShapeChar() const;

  std::vector<char> GetDynamicBatchSizeChar() const;

  void SetDynamicImageSize(const std::vector<char> &dynamic_image_size);
  std::vector<char> GetDynamicImageSizeChar() const;

  void SetPrecisionMode(const std::vector<char> &precision_mode);
  std::vector<char> GetPrecisionModeChar() const;

  void SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode);
  std::vector<char> GetOpSelectImplModeChar() const;

  void SetFusionSwitchConfigPath(const std::vector<char> &cfg_path);
  std::vector<char> GetFusionSwitchConfigPathChar() const;

  void SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode);
  std::vector<char> GetBufferOptimizeModeChar() const;
};

using Ascend310DeviceInfo = AscendDeviceInfo;
using Ascend910DeviceInfo = AscendDeviceInfo;

void AscendDeviceInfo::SetInsertOpConfigPath(const std::string &cfg_path) {
  SetInsertOpConfigPath(StringToChar(cfg_path));
}
std::string AscendDeviceInfo::GetInsertOpConfigPath() const { return CharToString(GetInsertOpConfigPathChar()); }

void AscendDeviceInfo::SetInputFormat(const std::string &format) { SetInputFormat(StringToChar(format)); }
std::string AscendDeviceInfo::GetInputFormat() const { return CharToString(GetInputFormatChar()); }

void AscendDeviceInfo::SetInputShape(const std::string &shape) { SetInputShape(StringToChar(shape)); }
std::string AscendDeviceInfo::GetInputShape() const { return CharToString(GetInputShapeChar()); }

std::string AscendDeviceInfo::GetDynamicBatchSize() const { return CharToString(GetDynamicBatchSizeChar()); }

void AscendDeviceInfo::SetDynamicImageSize(const std::string &dynamic_image_size) {
  SetDynamicImageSize(StringToChar(dynamic_image_size));
}

std::string AscendDeviceInfo::GetDynamicImageSize() const { return CharToString(GetDynamicImageSizeChar()); }

void AscendDeviceInfo::SetPrecisionMode(const std::string &precision_mode) {
  SetPrecisionMode(StringToChar(precision_mode));
}
std::string AscendDeviceInfo::GetPrecisionMode() const { return CharToString(GetPrecisionModeChar()); }

void AscendDeviceInfo::SetOpSelectImplMode(const std::string &op_select_impl_mode) {
  SetOpSelectImplMode(StringToChar(op_select_impl_mode));
}
std::string AscendDeviceInfo::GetOpSelectImplMode() const { return CharToString(GetOpSelectImplModeChar()); }

void AscendDeviceInfo::SetFusionSwitchConfigPath(const std::string &cfg_path) {
  SetFusionSwitchConfigPath(StringToChar(cfg_path));
}
std::string AscendDeviceInfo::GetFusionSwitchConfigPath() const {
  return CharToString(GetFusionSwitchConfigPathChar());
}

void AscendDeviceInfo::SetBufferOptimizeMode(const std::string &buffer_optimize_mode) {
  SetBufferOptimizeMode(StringToChar(buffer_optimize_mode));
}
std::string AscendDeviceInfo::GetBufferOptimizeMode() const { return CharToString(GetBufferOptimizeModeChar()); }
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CONTEXT_H
