/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "acl/acl_rt.h"

/**
 * @ingroup AscendCL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count,
                                         aclrtMemcpyKind kind) {
  return ACL_SUCCESS;
}

/**
 * @ingroup AscendCL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                                              aclrtMemcpyKind kind, aclrtStream stream) {
  return ACL_SUCCESS;
}

/**
 * @ingroup AscendCL
 * @brief Specify the device used for computing in the current process, and implicitly create a default context
 *
 * @param deviceId [IN]    the device id of the resource used by this process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId) { return ACL_SUCCESS; }

/**
 * @ingroup AscendCL
 * @brief Specify the device used for computing in the current process, and implicitly create a default context
 *
 * @param deviceId [IN]    the device id of the resource used by this process
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtResetDevice(int32_t deviceId) { return ACL_SUCCESS; }

/**
 * @ingroup AscendCL
 * @brief asynchronously initialize memory and set contents to specified value
 *
 * @par Function
 * The memory to be initialized is on the Host or Device side
 * and the system uses address to recognize that
 *
 * @param dst [IN]      destination address pointer
 * @param destMax [IN]      max length of the destination address memory
 * @param value [IN]      set value
 * @param count [IN]      the number of byte to set
 * @param stream [IN]      asynchronized task stream
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclrtSynchronizeStream
 */
ACL_FUNC_VISIBILITY aclError aclrtMemsetAsync(void *dst, size_t destMax, int32_t value, size_t count,
                                              aclrtStream stream) {
  return ACL_SUCCESS;
}

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for waiting of op
 *
 * @param timeout [IN]    op wait timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout) { return ACL_SUCCESS; }

/**
 * @ingroup AscendCL
 * @brief Set the timeout interval for executing of op
 *
 * @param timeout [IN]    op execute timeout
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtSetOpExecuteTimeOut(uint32_t timeout) { return ACL_SUCCESS; }

/**
 * @ingroup AscendCL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtDestroyStream(aclrtStream stream) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtDestroyStreamForce(aclrtStream stream) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtSetCurrentContext(aclrtContext context) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtGetDeviceCount(uint32_t *count) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtGetCurrentContext(aclrtContext *context) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtFree(void *devPtr) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStream(aclrtStream stream) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtDestroyEvent(aclrtEvent event) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeEvent(aclrtEvent event) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) {
  return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError aclrtCreateStream(aclrtStream *stream) { return ACL_SUCCESS; }

ACL_FUNC_VISIBILITY aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtDestroyContext(aclrtContext context) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtUnmapMem(void *virPtr) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr,
                                                    uint64_t flags) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size,
                                                 const aclrtPhysicalMemProp *prop, uint64_t flags) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle,
                                         uint64_t flags) {
  return ACL_SUCCESS;
}
ACL_FUNC_VISIBILITY aclError aclrtFreePhysical(aclrtDrvMemHandle handle) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY aclError aclrtReleaseMemAddress(void *virPtr) { return ACL_SUCCESS; }
ACL_FUNC_VISIBILITY const char *aclrtGetSocName() { return "910"; }
