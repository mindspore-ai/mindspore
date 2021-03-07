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
enum DeviceType {
  kCPU = 0,
  kMaliGPU,
  kNvidiaGPU,
  kKirinNPU,
  kAscend910,
  kAscend310,
  // add new type here
  kInvalidDeviceType = 100,
};

class Allocator;
class DeviceInfoContext;

class MS_API Context {
 public:
  Context();
  ~Context() = default;

  void SetThreadNum(int32_t thread_num);
  int32_t GetThreadNum() const;

  void SetAllocator(const std::shared_ptr<Allocator> &allocator);
  std::shared_ptr<Allocator> GetAllocator() const;

  std::vector<std::shared_ptr<DeviceInfoContext>> &MutableDeviceInfo();

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

class MS_API DeviceInfoContext : public std::enable_shared_from_this<DeviceInfoContext> {
 public:
  struct Data;

  DeviceInfoContext();
  virtual ~DeviceInfoContext() = default;
  virtual enum DeviceType GetDeviceType() const = 0;

  template <class T>
  std::shared_ptr<T> Cast() {
    static_assert(std::is_base_of<DeviceInfoContext, T>::value, "Wrong cast type.");
    if (GetDeviceType() != T().GetDeviceType()) {
      return nullptr;
    }

    return std::static_pointer_cast<T>(shared_from_this());
  }

 protected:
  std::shared_ptr<Data> data_;
};

class MS_API CPUDeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kCPU; };

  /// \brief Set the thread affinity to CPU cores.
  ///
  /// \param mode: 0: no affinities, 1: big cores first, 2: little cores first
  void SetThreadAffinity(int mode);
  int GetThreadAffinity() const;
  void SetEnableFP16(bool is_fp16);
  bool GetEnableFP16() const;
};

class MS_API MaliGPUDeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kMaliGPU; };

  void SetEnableFP16(bool is_fp16);
  bool GetEnableFP16() const;
};

class MS_API KirinNPUDeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kKirinNPU; };

  void SetFrequency(int frequency);
  int GetFrequency() const;
};

class MS_API NvidiaGPUDeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kNvidiaGPU; };

  void SetDeviceID(uint32_t device_id);
  uint32_t GetDeviceID() const;

  void SetGpuTrtInferMode(bool gpu_trt_infer_mode);
  bool GetGpuTrtInferMode() const;
};

class MS_API Ascend910DeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kAscend910; };

  void SetDeviceID(uint32_t device_id);
  uint32_t GetDeviceID() const;
};

class MS_API Ascend310DeviceInfo : public DeviceInfoContext {
 public:
  enum DeviceType GetDeviceType() const override { return DeviceType::kAscend310; };

  void SetDeviceID(uint32_t device_id);
  uint32_t GetDeviceID() const;

  inline void SetDumpConfigPath(const std::string &cfg_path);
  inline std::string GetDumpConfigPath() const;

  // aipp config file
  inline void SetInsertOpConfigPath(const std::string &cfg_path);
  inline std::string GetInsertOpConfigPath() const;

  // nchw or nhwc
  inline void SetInputFormat(const std::string &format);
  inline std::string GetInputFormat() const;

  // Mandatory while dynamic batch: e.g. "input_op_name1: 1,2,3,4;input_op_name2: 4,3,2,1"
  inline void SetInputShape(const std::string &shape);
  inline std::string GetInputShape() const;

  void SetInputShapeMap(const std::map<int, std::vector<int>> &shape);
  std::map<int, std::vector<int>> GetInputShapeMap() const;

  void SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size);
  inline std::string GetDynamicBatchSize() const;

  // FP32, UINT8 or FP16, default as FP32
  void SetOutputType(enum DataType output_type);
  enum DataType GetOutputType() const;

  // "force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype" or "allow_mix_precision", default as "force_fp16"
  inline void SetPrecisionMode(const std::string &precision_mode);
  inline std::string GetPrecisionMode() const;

  // Optional "high_performance" and "high_precision", "high_performance" is set as default
  inline void SetOpSelectImplMode(const std::string &op_select_impl_mode);
  inline std::string GetOpSelectImplMode() const;

  inline void SetFusionSwitchConfigPath(const std::string &cfg_path);
  inline std::string GetFusionSwitchConfigPath() const;

  // Optional "l1_optimize", "l2_optimize", "off_optimize" or "l1_and_l2_optimize", default as "l2_optimize"
  inline void SetBufferOptimizeMode(const std::string &buffer_optimize_mode);
  inline std::string GetBufferOptimizeMode() const;

 private:
  void SetDumpConfigPath(const std::vector<char> &cfg_path);
  std::vector<char> GetDumpConfigPathChar() const;

  void SetInsertOpConfigPath(const std::vector<char> &cfg_path);
  std::vector<char> GetInsertOpConfigPathChar() const;

  void SetInputFormat(const std::vector<char> &format);
  std::vector<char> GetInputFormatChar() const;

  void SetInputShape(const std::vector<char> &shape);
  std::vector<char> GetInputShapeChar() const;

  std::vector<char> GetDynamicBatchSizeChar() const;

  void SetPrecisionMode(const std::vector<char> &precision_mode);
  std::vector<char> GetPrecisionModeChar() const;

  void SetOpSelectImplMode(const std::vector<char> &op_select_impl_mode);
  std::vector<char> GetOpSelectImplModeChar() const;

  void SetFusionSwitchConfigPath(const std::vector<char> &cfg_path);
  std::vector<char> GetFusionSwitchConfigPathChar() const;

  void SetBufferOptimizeMode(const std::vector<char> &buffer_optimize_mode);
  std::vector<char> GetBufferOptimizeModeChar() const;
};

void Ascend310DeviceInfo::SetDumpConfigPath(const std::string &cfg_path) { SetDumpConfigPath(StringToChar(cfg_path)); }
std::string Ascend310DeviceInfo::GetDumpConfigPath() const { return CharToString(GetDumpConfigPathChar()); }

void Ascend310DeviceInfo::SetInsertOpConfigPath(const std::string &cfg_path) {
  SetInsertOpConfigPath(StringToChar(cfg_path));
}
std::string Ascend310DeviceInfo::GetInsertOpConfigPath() const { return CharToString(GetInsertOpConfigPathChar()); }

void Ascend310DeviceInfo::SetInputFormat(const std::string &format) { SetInputFormat(StringToChar(format)); }
std::string Ascend310DeviceInfo::GetInputFormat() const { return CharToString(GetInputFormatChar()); }

void Ascend310DeviceInfo::SetInputShape(const std::string &shape) { SetInputShape(StringToChar(shape)); }
std::string Ascend310DeviceInfo::GetInputShape() const { return CharToString(GetInputShapeChar()); }

std::string Ascend310DeviceInfo::GetDynamicBatchSize() const { return CharToString(GetDynamicBatchSizeChar()); }

void Ascend310DeviceInfo::SetPrecisionMode(const std::string &precision_mode) {
  SetPrecisionMode(StringToChar(precision_mode));
}
std::string Ascend310DeviceInfo::GetPrecisionMode() const { return CharToString(GetPrecisionModeChar()); }

void Ascend310DeviceInfo::SetOpSelectImplMode(const std::string &op_select_impl_mode) {
  SetOpSelectImplMode(StringToChar(op_select_impl_mode));
}
std::string Ascend310DeviceInfo::GetOpSelectImplMode() const { return CharToString(GetOpSelectImplModeChar()); }

void Ascend310DeviceInfo::SetFusionSwitchConfigPath(const std::string &cfg_path) {
  SetFusionSwitchConfigPath(StringToChar(cfg_path));
}
std::string Ascend310DeviceInfo::GetFusionSwitchConfigPath() const {
  return CharToString(GetFusionSwitchConfigPathChar());
}

void Ascend310DeviceInfo::SetBufferOptimizeMode(const std::string &buffer_optimize_mode) {
  SetBufferOptimizeMode(StringToChar(buffer_optimize_mode));
}
std::string Ascend310DeviceInfo::GetBufferOptimizeMode() const { return CharToString(GetBufferOptimizeModeChar()); }
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_CONTEXT_H
