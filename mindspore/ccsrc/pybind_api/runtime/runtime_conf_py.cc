/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "pybind_api/runtime/runtime_api.h"

namespace py = pybind11;
namespace mindspore {
namespace runtime {
void RegRuntimeConf(py::module *m) {
  (void)py::class_<RuntimeConf, std::shared_ptr<RuntimeConf>>(*m, "RuntimeConf")
    .def_static("get_instance", &RuntimeConf::GetInstance, "Get Runtime Instance")
    .def("set_launch_blocking", &RuntimeConf::set_launch_blocking, "Set launch blocking")
    .def("set_dispatch_threads_num", &RuntimeConf::set_dispatch_threads_num, "Set dispatch threads num")
    .def("is_dispatch_threads_num_configured", &RuntimeConf::IsDispatchThreadsNumConfigured,
         "Is dispatch threads num configured")
    .def("set_kernel_launch_group", &RuntimeConf::set_kernel_launch_group, "Set kernel threads num")
    .def("is_kernel_launch_group_configured", &RuntimeConf::IsKernelLaunchGroupConfigured,
         "Is kernel launch group configured")
    .def("set_op_threads_num", &RuntimeConf::set_op_threads_num, "Set op threads num")
    .def("is_op_threads_num_configured", &RuntimeConf::IsOpThreadsNumConfigured, "Is op threads num configured")
    .def("set_memory", &RuntimeConf::set_memory, "Set memory")
    .def("is_memory_configured", &RuntimeConf::IsMemoryConfigured, "Is memory configured")
    .def("is_launch_blocking", &RuntimeConf::IsSetLaunchBlocking, "Check whether launch blocking configured.")
    .def("is_thread_bind_core_configured", &RuntimeConf::IsThreadBindCoreConfigured,
         "Check whether thread_bind_core configured.")
    .def("set_thread_bind_core_configured", &RuntimeConf::SetThreadBindCoreConfigured,
         "Set thread_bind_core configured.")
    .def("thread_bind_core", &RuntimeConf::BindThreadCore, "Bind thread to specific cpus with policy generated")
    .def("set_kernel_launch_capture", &RuntimeConf::SetEnableKernelLaunchCapture,
         "Enable capture graph when launch some kernels")
    .def("get_enable_kernel_launch_capture", &RuntimeConf::GetEnableKernelLaunchCapture,
         "Is kernel launch capture configured");
}
}  // namespace runtime
}  // namespace mindspore
