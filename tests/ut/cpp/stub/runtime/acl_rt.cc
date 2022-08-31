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
  return ACL_ERROR_NONE;
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
  return ACL_ERROR_NONE;
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
ACL_FUNC_VISIBILITY aclError aclrtSetDevice(int32_t deviceId) {
  return ACL_ERROR_NONE;
}

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
  return ACL_ERROR_NONE;
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
ACL_FUNC_VISIBILITY aclError aclrtSetOpWaitTimeout(uint32_t timeout) {
  return ACL_SUCCESS;
}
