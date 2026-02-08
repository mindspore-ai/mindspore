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
#ifndef MINDSPORE_INCLUDE_CUSTOM_OP_API_H_
#define MINDSPORE_INCLUDE_CUSTOM_OP_API_H_
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ir/tensor.h"

// pyboost headfiles
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/backend/common/device_address_utils.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "mindspore/ccsrc/tools/profiler/profiler.h"

// ascend files
#ifdef CUSTOM_ASCEND_OP
#include "include/pynative/backward/function.h"
#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/custom_launch_aclnn.h"
#endif  // CUSTOM_ASCEND_OP

// custom api
#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "include/pynative/utils/pyboost/custom/tensor_utils.h"
#include "include/pynative/utils/pyboost/custom/pyboost_extension.h"
#include "include/kernel/ascend/custom/kernel_mod_impl/custom_register.h"
#include "include/kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"

// ascend files
#ifdef CUSTOM_ASCEND_OP
#include "include/kernel/ascend/custom/pyboost_impl/aclnn_op_runner.h"
// MindSpore core includes
#include "mindspore/core/include/utils/ms_utils.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/utils/convert_utils_base.h"
#include "mindspore/core/include/utils/check_convert_utils.h"
#include "mindspore/core/include/utils/ms_context.h"

// MindSpore ops includes
#include "mindspore/ops/ops_utils/op_utils.h"

// MindSpore runtime includes
#include "mindspore/ccsrc/include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "mindspore/ccsrc/plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "include/utils/utils.h"
#include "mindspore/ccsrc/include/runtime/hardware_abstract/device_context/device_context.h"
#include "mindspore/ccsrc/include/backend/common/ms_device_shape_transfer.h"

// Internal kernel includes
#include "mindspore/ops/kernel/ascend/internal/internal_ascend_adapter.h"

#ifdef CUSTOM_ENABLE_ATB
#include "include/kernel/ascend/custom/pyboost_impl/atb_common.h"
#endif  // CUSTOM_ENABLE_ATB
#ifdef CUSTOM_ENABLE_ASDSIP
#include "include/kernel/ascend/custom/pyboost_impl/asdsip_common.h"
#endif  // CUSTOM_ENABLE_ASDSIP
#endif  // CUSTOM_ASCEND_OP

// The BaseTensor is deprecated
namespace mindspore {
namespace tensor {
using BaseTensor = Tensor;
using BaseTensorPtr = TensorPtr;
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_CUSTOM_OP_API_H_
