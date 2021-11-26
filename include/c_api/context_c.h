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
#ifndef MINDSPORE_INCLUDE_C_API_CONTEXT_C_H
#define MINDSPORE_INCLUDE_C_API_CONTEXT_C_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "include/c_api/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *MSContextHandle;
typedef void *MSDeviceInfoHandle;

/// \brief Create a context object.
///
/// \return Context object handle.
MS_API MSContextHandle MSContextCreate();

/// \brief Destroy the context object.
///
/// \param[in] context Context object handle address.
MS_API void MSContextDestroy(MSContextHandle *context);

/// \brief Set the number of threads at runtime.
///
/// \param[in] context Context object handle.
/// \param[in] thread_num the number of threads at runtime.
MS_API void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num);

/// \brief Obtain the current thread number setting.
///
/// \param[in] context Context object handle.
///
/// \return The current thread number setting.
MS_API int32_t MSContextGetThreadNum(const MSContextHandle context);

/// \brief Set the thread affinity to CPU cores.
///
/// \param[in] context Context object handle.
/// \param[in] mode: 0: no affinities, 1: big cores first, 2: little cores first
MS_API void MSContextSetThreadAffinityMode(MSContextHandle context, int mode);

/// \brief Obtain the thread affinity of CPU cores.
///
/// \param[in] context Context object handle.
///
/// \return Thread affinity to CPU cores. 0: no affinities, 1: big cores first, 2: little cores first
MS_API int MSContextGetThreadAffinityMode(const MSContextHandle context);

/// \brief Set the thread lists to CPU cores.
///
/// \note If core_list and mode are set by MSContextSetThreadAffinityMode at the same time,
/// the core_list is effective, but the mode is not effective.
///
/// \param[in] context Context object handle.
/// \param[in] core_list: a array of thread core lists.
/// \param[in] core_num The number of core.
MS_API void MSContextSetThreadAffinityCoreList(MSContextHandle context, const int32_t *core_list, size_t core_num);

/// \brief Obtain the thread lists of CPU cores.
///
/// \param[in] context Context object handle.
/// \param[out] core_num The number of core.
///
/// \return a array of thread core lists.
MS_API const int32_t *MSContextGetThreadAffinityCoreList(const MSContextHandle context, size_t *core_num);

/// \brief Set the status whether to perform model inference or training in parallel.
///
/// \param[in] context Context object handle.
/// \param[in] is_parallel: true, parallel; false, not in parallel.
MS_API void MSContextSetEnableParallel(MSContextHandle context, bool is_parallel);

/// \brief Obtain the status whether to perform model inference or training in parallel.
///
/// \param[in] context Context object handle.
///
/// \return Bool value that indicates whether in parallel.
MS_API bool MSContextGetEnableParallel(const MSContextHandle context);

/// \brief Add device info to context object.
///
/// \param[in] context Context object handle.
/// \param[in] device_info Device info object handle.
MS_API void MSContextAddDeviceInfo(MSContextHandle context, MSDeviceInfoHandle device_info);

/// \brief Create a device info object.
///
/// \param[in] device_info Device info object handle.
///
/// \return Device info object handle.
MS_API MSDeviceInfoHandle MSDeviceInfoCreate(MSDeviceType device_type);

/// \brief Destroy the device info object.
///
/// \param[in] device_info Device info object handle address.
MS_API void MSDeviceInfoDestroy(MSDeviceInfoHandle *device_info);

/// \brief Set provider's name.
///
/// \param[in] device_info Device info object handle.
/// \param[in] provider define the provider's name.
MS_API void MSDeviceInfoSetProvider(MSDeviceInfoHandle device_info, const char *provider);

/// \brief Obtain provider's name
///
/// \param[in] device_info Device info object handle.
///
/// \return provider's name.
MS_API const char *MSDeviceInfoGetProvider(const MSDeviceInfoHandle device_info);

/// \brief Set provider's device type.
///
/// \param[in] device_info Device info object handle.
/// \param[in] device define the provider's device type. EG: CPU.
MS_API void MSDeviceInfoSetProviderDevice(MSDeviceInfoHandle device_info, const char *device);

/// \brief Obtain provider's device type.
///
/// \param[in] device_info Device info object handle.
///
/// \return provider's device type.
MS_API const char *MSDeviceInfoGetProviderDevice(const MSDeviceInfoHandle device_info);

/// \brief Obtain the device type of the device info.
///
/// \param[in] device_info Device info object handle.
///
/// \return Device Type of the device info.
MS_API MSDeviceType MSDeviceInfoGetDeviceType(const MSDeviceInfoHandle device_info);

/// \brief Set enables to perform the float16 inference, Only valid for CPU/GPU.
///
/// \param[in] device_info Device info object handle.
/// \param[in] is_fp16 Enable float16 inference or not.
MS_API void MSDeviceInfoSetEnableFP16(MSDeviceInfoHandle device_info, bool is_fp16);

/// \brief Obtain enables to perform the float16 inference, Only valid for CPU/GPU.
///
/// \param[in] device_info Device info object handle.
///
/// \return Whether enable float16 inference.
MS_API bool MSDeviceInfoGetEnableFP16(const MSDeviceInfoHandle device_info);

/// \brief Set the NPU frequency, Only valid for NPU.
///
/// \param[in] device_info Device info object handle.
/// \param[in] frequency Can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme
/// performance), default as 3.
MS_API void MSDeviceInfoSetFrequency(MSDeviceInfoHandle device_info, int frequency);

/// \brief Obtain the NPU frequency, Only valid for NPU.
///
/// \param[in] device_info Device info object handle.
///
/// \return NPU frequency
MS_API int MSDeviceInfoGetFrequency(const MSDeviceInfoHandle device_info);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_CONTEXT_C_H
