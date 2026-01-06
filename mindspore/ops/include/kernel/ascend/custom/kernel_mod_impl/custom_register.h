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

#ifndef MINDSPORE_CCSRC_EXTENSION_COMMON_UTILS_H_
#define MINDSPORE_CCSRC_EXTENSION_COMMON_UTILS_H_

#define DECLARE_CUSTOM_OP_IMPL(OpName, OpFuncImplClass) \
  static OpFuncImplClass g##OpName##FuncImplReal;       \
  OpFuncImpl &gCustom_##OpName##FuncImpl = g##OpName##FuncImplReal

#define MS_CUSTOM_OPS_REGISTER(OpName, OpFuncImplClass, KernelClass) \
  namespace mindspore::ops {                                         \
  DECLARE_CUSTOM_OP_IMPL(OpName, OpFuncImplClass);                   \
  } /* namespace mindspore::ops */                                   \
                                                                     \
  namespace mindspore::kernel {                                      \
  MS_CUSTOM_KERNEL_FACTORY_REG("Custom_" #OpName, KernelClass);      \
  } /* namespace mindspore::kernel */

#endif  // MINDSPORE_CCSRC_EXTENSION_COMMON_UTILS_H_
