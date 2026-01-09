/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_UTILS_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_UTILS_H_

#include <functional>
#include <vector>
#include <string>
#include <map>
#include "acl/acl.h"
#include "include/pynative/utils/pyboost/custom/pyboost_extension.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace ms::pynative {
using mindspore::device::ascend::aclTensor;

enum asdFftType {
  ASCEND_FFT_C2C = 0x10,
  ASCEND_FFT_C2R = 0x11,
  ASCEND_FFT_R2C = 0x12,
  ASCEND_STFT_C2C = 0x20,
  ASCEND_STFT_C2R = 0x21,
  ASCEND_STFT_R2C = 0x22,
};

enum asdFftDirection {
  ASCEND_FFT_FORWARD = 0x10,
  ASCEND_FFT_BACKWARD = 0x11,
};

enum asdFft1dDimType {
  ASCEND_FFT_HORIZONTAL = 0x10,
  ASCEND_FFT_VERTICAL = 0x11,
};

struct FFTParam {
  int64_t fftXSize = 0;
  int64_t fftYSize = 0;
  asdFftType fftType = asdFftType::ASCEND_FFT_C2C;
  asdFftDirection direction = asdFftDirection::ASCEND_FFT_FORWARD;
  int64_t batchSize = 0;
  asdFft1dDimType dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
};

typedef void *asdFftHandle;
typedef int (*_asdFftCreate)(asdFftHandle &handle);
typedef int (*_asdFftDestroy)(asdFftHandle handle);
typedef int (*_asdFftSetStream)(asdFftHandle handle, void *stream);
typedef int (*_asdFftSynchronize)(asdFftHandle handle);
typedef int (*_asdFftGetWorkspaceSize)(asdFftHandle handle, size_t &size);
typedef int (*_asdFftSetWorkspace)(asdFftHandle handle, void *work_space);
typedef int (*_asdFftMakePlan1D)(asdFftHandle handle, int64_t fftXSize, asdFftType fftType, asdFftDirection direction,
                                 int64_t batch_size, asdFft1dDimType dim_type);
typedef int (*_asdFftMakePlan2D)(asdFftHandle handle, int64_t fftXSize, int64_t fftYSize, asdFftType fftType,
                                 asdFftDirection direction, int64_t batch_size);

using AsdFftExecFunc = int (*)(asdFftHandle handle, aclTensor *input, aclTensor *output);

#define GET_API_FUNC(func_name) reinterpret_cast<_##func_name>(GetAsdSipApiFuncAddr(#func_name))

inline std::string GetAsdSipLibPath() {
  auto ascend_path = mindspore::device::ascend::GetAscendPath();
  const std::string kLatest = "latest";
  auto posLatest = ascend_path.rfind(kLatest);
  if (posLatest != std::string::npos) {
    return ascend_path + "/../../nnal/asdsip/latest/lib/libasdsip.so";
  }
  return ascend_path + "/../nnal/asdsip/latest/lib/libasdsip.so";
}

inline uint64_t HashFFTParam(const FFTParam &param) {
  static std::hash<std::string> hash_func;
  std::string param_str = std::to_string(param.fftXSize) + std::to_string(param.fftYSize) +
                          std::to_string(static_cast<int>(param.fftType)) +
                          std::to_string(static_cast<int>(param.direction)) + std::to_string(param.batchSize) +
                          std::to_string(static_cast<int>(param.dimType));
  return hash_func(param_str);
}

inline void *GetAsdSipApiFuncAddr(const char *api_name) {
  static auto handle = dlopen(GetAsdSipLibPath().c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(EXCEPTION)
      << "Load libasdsip.so from " << GetAsdSipLibPath() << " failed, error is: " << dlerror()
      << ", you should check the nnal package whether installed correctly and source the set_env.sh in Asdsip.";
  }
  auto func_addr = dlsym(handle, api_name);
  if (func_addr == nullptr) {
    MS_LOG(EXCEPTION) << "Get api " << api_name << " from " << GetAsdSipLibPath() << " failed.";
  }
  return func_addr;
}

inline int AsdFftCreate(asdFftHandle &handle) {
  static auto asd_fft_create = GET_API_FUNC(asdFftCreate);
  if (asd_fft_create == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftCreate is nullptr.";
  }
  return asd_fft_create(handle);
}

inline int AsdFftMakePlan1D(asdFftHandle handle, int64_t fftXSize, asdFftType fftType, asdFftDirection direction,
                            int64_t batch_size, asdFft1dDimType dim_type) {
  static auto asd_sip_make_plan_1d = GET_API_FUNC(asdFftMakePlan1D);
  if (asd_sip_make_plan_1d == nullptr) {
    MS_LOG(EXCEPTION) << "asdSipFftMakePlan1D is nullptr.";
  }
  return asd_sip_make_plan_1d(handle, fftXSize, fftType, direction, batch_size, dim_type);
}

inline int AsdFftMakePlan2D(asdFftHandle handle, int64_t fftXSize, int64_t fftYSize, asdFftType fftType,
                            asdFftDirection direction, int64_t batch_size) {
  static auto asd_sip_make_plan_2d = GET_API_FUNC(asdFftMakePlan2D);
  if (asd_sip_make_plan_2d == nullptr) {
    MS_LOG(EXCEPTION) << "asdSipFftMakePlan2D is nullptr.";
  }
  return asd_sip_make_plan_2d(handle, fftXSize, fftYSize, fftType, direction, batch_size);
}

inline asdFftHandle CreateHandle(const FFTParam &param) {
  asdFftHandle handle;
  auto ret = AsdFftCreate(handle);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "AsdFftCreate failed.";
  }
  if (param.fftYSize == 0) {
    AsdFftMakePlan1D(handle, param.fftXSize, param.fftType, param.direction, param.batchSize, param.dimType);
  } else {
    AsdFftMakePlan2D(handle, param.fftXSize, param.fftYSize, param.fftType, param.direction, param.batchSize);
  }
  return handle;
}

inline int AsdFftDestroy(asdFftHandle handle) {
  static auto asd_fft_destrory = GET_API_FUNC(asdFftDestroy);
  if (asd_fft_destrory == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftDestroy is nullptr.";
  }
  return asd_fft_destrory(handle);
}

inline int AsdFftSetStream(asdFftHandle handle, void *stream) {
  static auto asd_fft_set_stream = GET_API_FUNC(asdFftSetStream);
  if (asd_fft_set_stream == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftSetStream is nullptr.";
  }
  return asd_fft_set_stream(handle, stream);
}

inline int AsdFftSynchronize(asdFftHandle handle) {
  static auto asd_fft_synchronize = GET_API_FUNC(asdFftSynchronize);
  if (asd_fft_synchronize == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftSynchronize is nullptr.";
  }
  return asd_fft_synchronize(handle);
}

inline int AsdFftGetWorkSpaceSize(asdFftHandle handle, size_t &size) {
  static auto asd_fft_get_workspace = GET_API_FUNC(asdFftGetWorkspaceSize);
  if (asd_fft_get_workspace == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftGetWorkSpaceSize is nullptr.";
  }
  return asd_fft_get_workspace(handle, size);
}

inline int AsdFftSetWorkSpace(asdFftHandle handle, void *work_space) {
  static auto asd_fft_set_workspace = GET_API_FUNC(asdFftSetWorkspace);
  if (asd_fft_set_workspace == nullptr) {
    MS_LOG(EXCEPTION) << "asdFftSetWorkSpace is nullptr.";
  }
  return asd_fft_set_workspace(handle, work_space);
}

inline void DestrotyHandle(asdFftHandle handle) {
  AsdFftSynchronize(handle);
  AsdFftDestroy(handle);
}

}  // namespace ms::pynative
#endif  // MINDSPORE_CCSRC_MS_EXTENSION_ASCEND_ASDSIP_UTILS_H_
