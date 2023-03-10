/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/api/context.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mindspore::lite {
namespace py = pybind11;

void ContextPyBind(const py::module &m) {
  (void)py::enum_<DeviceType>(m, "DeviceType", py::arithmetic())
    .value("kCPU", DeviceType::kCPU)
    .value("kGPU", DeviceType::kGPU)
    .value("kKirinNPU", DeviceType::kKirinNPU)
    .value("kAscend", DeviceType::kAscend);

  (void)py::class_<DeviceInfoContext, std::shared_ptr<DeviceInfoContext>>(m, "DeviceInfoContextBind");

  (void)py::class_<CPUDeviceInfo, DeviceInfoContext, std::shared_ptr<CPUDeviceInfo>>(m, "CPUDeviceInfoBind")
    .def(py::init<>())
    .def("get_device_type", &CPUDeviceInfo::GetDeviceType)
    .def("set_enable_fp16", &CPUDeviceInfo::SetEnableFP16)
    .def("get_enable_fp16", &CPUDeviceInfo::GetEnableFP16);

  (void)py::class_<GPUDeviceInfo, DeviceInfoContext, std::shared_ptr<GPUDeviceInfo>>(m, "GPUDeviceInfoBind")
    .def(py::init<>())
    .def("get_device_type", &GPUDeviceInfo::GetDeviceType)
    .def("set_device_id", &GPUDeviceInfo::SetDeviceID)
    .def("get_device_id", &GPUDeviceInfo::GetDeviceID)
    .def("set_enable_fp16", &GPUDeviceInfo::SetEnableFP16)
    .def("get_enable_fp16", &GPUDeviceInfo::GetEnableFP16)
    .def("get_rank_id", &GPUDeviceInfo::GetRankID)
    .def("get_group_size", &GPUDeviceInfo::GetGroupSize);

  (void)py::class_<AscendDeviceInfo, DeviceInfoContext, std::shared_ptr<AscendDeviceInfo>>(m, "AscendDeviceInfoBind")
    .def(py::init<>())
    .def("get_device_type", &AscendDeviceInfo::GetDeviceType)
    .def("set_device_id", &AscendDeviceInfo::SetDeviceID)
    .def("get_device_id", &AscendDeviceInfo::GetDeviceID)
    .def("set_input_format",
         [](AscendDeviceInfo &device_info, const std::string &format) { device_info.SetInputFormat(format); })
    .def("get_input_format", &AscendDeviceInfo::GetInputFormat)
    .def("set_input_shape", &AscendDeviceInfo::SetInputShapeMap)
    .def("get_input_shape", &AscendDeviceInfo::GetInputShapeMap)
    .def("set_precision_mode", [](AscendDeviceInfo &device_info,
                                  const std::string &precision_mode) { device_info.SetPrecisionMode(precision_mode); })
    .def("get_precision_mode", &AscendDeviceInfo::GetPrecisionMode)
    .def("set_op_select_impl_mode",
         [](AscendDeviceInfo &device_info, const std::string &op_select_impl_mode) {
           device_info.SetOpSelectImplMode(op_select_impl_mode);
         })
    .def("get_op_select_impl_mode", &AscendDeviceInfo::GetOpSelectImplMode)
    .def("set_dynamic_batch_size", &AscendDeviceInfo::SetDynamicBatchSize)
    .def("get_dynamic_batch_size", &AscendDeviceInfo::GetDynamicBatchSize)
    .def("set_dynamic_image_size",
         [](AscendDeviceInfo &device_info, const std::string &dynamic_image_size) {
           device_info.SetDynamicImageSize(dynamic_image_size);
         })
    .def("get_dynamic_image_size", &AscendDeviceInfo::GetDynamicImageSize)
    .def("set_fusion_switch_config_path",
         [](AscendDeviceInfo &device_info, const std::string &cfg_path) {
           device_info.SetFusionSwitchConfigPath(cfg_path);
         })
    .def("get_fusion_switch_config_path", &AscendDeviceInfo::GetFusionSwitchConfigPath)
    .def("set_insert_op_cfg_path", [](AscendDeviceInfo &device_info,
                                      const std::string &cfg_path) { device_info.SetInsertOpConfigPath(cfg_path); })
    .def("get_insert_op_cfg_path", &AscendDeviceInfo::GetInsertOpConfigPath);

  (void)py::class_<Context, std::shared_ptr<Context>>(m, "ContextBind")
    .def(py::init<>())
    .def("append_device_info",
         [](Context &context, const std::shared_ptr<DeviceInfoContext> &device_info) {
           context.MutableDeviceInfo().push_back(device_info);
         })
    .def("clear_device_info", [](Context &context) { context.MutableDeviceInfo().clear(); })
    .def("set_thread_num", &Context::SetThreadNum)
    .def("get_thread_num", &Context::GetThreadNum)
    .def("set_inter_op_parallel_num", &Context::SetInterOpParallelNum)
    .def("get_inter_op_parallel_num", &Context::GetInterOpParallelNum)
    .def("set_thread_affinity_mode", py::overload_cast<int>(&Context::SetThreadAffinity))
    .def("get_thread_affinity_mode", &Context::GetThreadAffinityMode)
    .def("set_thread_affinity_core_list", py::overload_cast<const std::vector<int> &>(&Context::SetThreadAffinity))
    .def("get_thread_affinity_core_list", &Context::GetThreadAffinityCoreList)
    .def("set_enable_parallel", &Context::SetEnableParallel)
    .def("get_enable_parallel", &Context::GetEnableParallel)
    .def("get_device_list", [](Context &context) {
      std::string result;
      auto &device_list = context.MutableDeviceInfo();
      for (auto &device : device_list) {
        result += std::to_string(device->GetDeviceType());
        result += ", ";
      }
      return result;
    });
}
}  // namespace mindspore::lite
