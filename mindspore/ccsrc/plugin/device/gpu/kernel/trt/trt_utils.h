/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRT_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRT_UTILS_H_

#include <utility>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <NvInfer.h>
#include "utils/log_adapter.h"
#include "utils/singleton.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "ir/dtype/type.h"

namespace mindspore {
class TrtUtils {
 public:
  static TypeId TrtDtypeToMsDtype(const nvinfer1::DataType &trt_dtype) {
    static std::map<nvinfer1::DataType, TypeId> type_list = {{nvinfer1::DataType::kFLOAT, TypeId::kNumberTypeFloat32},
                                                             {nvinfer1::DataType::kHALF, TypeId::kNumberTypeFloat16},
                                                             {nvinfer1::DataType::kINT8, TypeId::kNumberTypeInt8},
                                                             {nvinfer1::DataType::kINT32, TypeId::kNumberTypeInt32}};

    auto iter = type_list.find(trt_dtype);
    if (iter == type_list.end()) {
      MS_LOG(EXCEPTION) << "Invalid Tensor-RT dtype: " << trt_dtype;
    }
    return iter->second;
  }

  static std::variant<bool, nvinfer1::DataType> MsDtypeToTrtDtype(const TypeId &ms_dtype) {
    static std::map<TypeId, nvinfer1::DataType> type_list = {{TypeId::kNumberTypeFloat32, nvinfer1::DataType::kFLOAT},
                                                             {TypeId::kNumberTypeFloat16, nvinfer1::DataType::kHALF},
                                                             {TypeId::kNumberTypeInt8, nvinfer1::DataType::kINT8},
                                                             {TypeId::kNumberTypeInt, nvinfer1::DataType::kINT32},
                                                             {TypeId::kNumberTypeInt32, nvinfer1::DataType::kINT32}};
    auto iter = type_list.find(ms_dtype);
    if (iter == type_list.end()) {
      MS_LOG(WARNING) << "data type not support: " << ms_dtype;
      return false;
    }
    return iter->second;
  }

  static nvinfer1::Dims MsDimsToTrtDims(const std::vector<size_t> &ms_shape, bool ignore_batch_dim = false) {
    nvinfer1::Dims trt_dims;
    size_t offset = ignore_batch_dim ? 1 : 0;
    for (size_t i = offset; i < ms_shape.size(); ++i) {
      trt_dims.d[i - offset] = SizeToInt(ms_shape[i]);
    }
    trt_dims.nbDims = ms_shape.size() - offset;
    return trt_dims;
  }

  static nvinfer1::Dims MsDimsToTrtDims(const ShapeVector &ms_shape, bool ignore_batch_dim = false) {
    nvinfer1::Dims trt_dims;
    size_t offset = ignore_batch_dim ? 1 : 0;
    for (size_t i = offset; i < ms_shape.size(); ++i) {
      trt_dims.d[i - offset] = LongToInt(ms_shape[i]);
    }
    trt_dims.nbDims = ms_shape.size() - offset;
    return trt_dims;
  }

  static ShapeVector TrtDimsToMsDims(const nvinfer1::Dims &trt_dims) {
    ShapeVector shape;
    std::transform(trt_dims.d, trt_dims.d + trt_dims.nbDims, std::back_inserter(shape),
                   [](const uint32_t &value) { return static_cast<int64_t>(value); });
    return shape;
  }

  static bool IsSameShape(const nvinfer1::Dims &lhs, const nvinfer1::Dims &rhs) {
    if (lhs.nbDims != rhs.nbDims) {
      return false;
    }

    for (int32_t i = 0; i < lhs.nbDims; i++) {
      if (lhs.d[i] != rhs.d[i]) {
        return false;
      }
    }

    return true;
  }
};

class TrtLogger : public nvinfer1::ILogger {
 public:
  TrtLogger() {
    log_level_ = MsLogLevel::kWarning;  // set default log level to WARNING
    const char *glog_config = std::getenv("GLOG_v");
    if (glog_config == nullptr) {
      return;
    }

    std::string str_level{glog_config};
    if (str_level.size() == 1) {
      int ch = str_level.c_str()[0];
      ch = ch - '0';  // subtract ASCII code of '0', which is 48
      if (ch >= MsLogLevel::kDebug && ch <= MsLogLevel::kException) {
        log_level_ = static_cast<MsLogLevel>(ch);
      }
    }
  }
  // Redirect Tensor-RT inner log to GLOG
  void log(Severity severity, const char *msg) noexcept override {
#ifdef USE_GLOG
#define google mindspore_private
    static std::map<Severity, std::tuple<MsLogLevel, int, std::string>> logger_map = {
      {Severity::kVERBOSE, {MsLogLevel::kDebug, google::GLOG_INFO, "VERBOSE"}},
      {Severity::kINFO, {MsLogLevel::kInfo, google::GLOG_INFO, "INFO"}},
      {Severity::kWARNING, {MsLogLevel::kWarning, google::GLOG_WARNING, "WARNING"}},
      {Severity::kERROR, {MsLogLevel::kError, google::GLOG_ERROR, "ERROR"}},
      {Severity::kINTERNAL_ERROR, {MsLogLevel::kError, google::GLOG_ERROR, "INTERNAL ERROR"}}};

    static const size_t kMsLogLevelIndex = 0;
    static const size_t kGoogleLogLevelIndex = 1;
    static const size_t kLogLevelDescriptionIndex = 2;

    auto iter = logger_map.find(severity);
    if (iter == logger_map.end()) {
      google::LogMessage("", 0, google::GLOG_WARNING).stream() << "Unrecognized severity type: " << msg << std::endl;
      return;
    }

    auto level = iter->second;
    // discard log
    if (std::get<kMsLogLevelIndex>(level) < log_level_) {
      return;
    }

    google::LogMessage("", 0, std::get<kGoogleLogLevelIndex>(level)).stream()
      << "[TensorRT " << std::get<kLogLevelDescriptionIndex>(level) << "] " << msg << std::endl;
#undef google
#endif  // USE_GLOG
  }

 private:
  MsLogLevel log_level_;
};

// Using RAII to avoid tensor-rt object leakage
template <typename T>
inline std::shared_ptr<T> TrtPtr(T *obj) {
  return std::shared_ptr<T>(obj, [](T *obj) {
    if (obj) obj->destroy();
  });
}

#define TRT_VARIANT_CHECK(input, expect, ret) \
  do {                                        \
    if ((input.index()) != (expect)) {        \
      return ret;                             \
    }                                         \
  } while (0)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRT_UTILS_H_
