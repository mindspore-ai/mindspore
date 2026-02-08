/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
#define TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H

#include <memory>

#include "common/common_test.h"
#define private public
#define protected public
#include "abstract/abstract_function.h"
#include "backend/ms_backend/runtime/graph_scheduler/control_flow/control_node_parser.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "device_address/device_address.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "common/test_device_address.h"
#define private public
#define protected public

namespace mindspore {
namespace runtime {
namespace test {
using abstract::AbstractFuncUnion;
using abstract::AbstractTensor;
using abstract::AbstractTensorPtr;
using abstract::AnalysisContext;
using abstract::FuncGraphAbstractClosure;
using device::DeviceAddress;
using device::DeviceAddressPtr;
using device::DeviceContextKey;
using device::DeviceContextRegister;
using device::DeviceType;
using kernel::AddressPtr;
using kernel::KernelTensorPtr;
using session::KernelGraph;

}  // namespace test
}  // namespace runtime
}  // namespace mindspore
#endif  // TESTS_UT_CPP_COMMON_DEVICE_COMMON_TEST_H
