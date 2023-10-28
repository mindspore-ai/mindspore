//
// Created by jojo on 2023/10/28.
//

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_

#include "transform/acl_ir/op_api_exec.h"

// TODO: Get kCubeMathType by SOC version
#define LAUNCH_ACLNN_CUBE(aclnn_api, stream_ptr, ...)                                                        \
  int8_t cube_math_type = 0;                                                                                 \
  auto [workspace_size, executor, after_launch_func] = GEN_EXECUTOR(aclnn_api, __VA_ARGS__, cube_math_type); \
  if (workspace_size == 0) {                                                                                 \
    RUN_OP_API(aclnn_api, stream_ptr, nullptr, 0, executor, after_launch_func);                              \
  } else {                                                                                                   \
    auto workspace_device_address =                                                                          \
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, workspace_size);                  \
    RUN_OP_API(aclnn_api, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor,   \
               after_launch_func);                                                                           \
  }

#define LAUNCH_ACLNN(aclnn_api, stream_ptr, ...)                                                           \
  auto [workspace_size, executor, after_launch_func] = GEN_EXECUTOR(aclnn_api, __VA_ARGS__);               \
  if (workspace_size == 0) {                                                                               \
    RUN_OP_API(aclnn_api, stream_ptr, nullptr, 0, executor, after_launch_func);                            \
  } else {                                                                                                 \
    auto workspace_device_address =                                                                        \
      runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, workspace_size);                \
    RUN_OP_API(aclnn_api, stream_ptr, workspace_device_address->GetMutablePtr(), workspace_size, executor, \
               after_launch_func);                                                                         \
  }

#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
