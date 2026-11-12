/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"

#include "minddata/dataset/core/client.h"  // DE client
#if defined(ENABLE_D)
#include "minddata/dataset/core/device_buffer.h"
#endif
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/message_queue.h"
#include "minddata/dataset/core/shared_memory_queue.h"

#include "minddata/dataset/include/dataset/constants.h"
#if defined(ENABLE_D)
#include "minddata/dataset/vision/kernels/dvpp/acl_adapter.h"
#endif

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(GlobalContext, 0, ([](const py::module *m) {
                  (void)py::class_<GlobalContext>(*m, "GlobalContext")
                    .def_static("config_manager", &GlobalContext::config_manager, py::return_value_policy::reference)
                    .def_static("profiling_manager", &GlobalContext::profiling_manager,
                                py::return_value_policy::reference);
                }));

PYBIND_REGISTER(ConfigManager, 0, ([](const py::module *m) {
                  (void)py::class_<ConfigManager, std::shared_ptr<ConfigManager>>(*m, "ConfigManager")
                    .def("__str__", &ConfigManager::ToString)
                    .def("get_auto_num_workers", &ConfigManager::auto_num_workers)
                    .def("get_callback_timeout", &ConfigManager::callback_timeout)
                    .def("get_monitor_sampling_interval", &ConfigManager::monitor_sampling_interval)
                    .def("get_num_parallel_workers", &ConfigManager::num_parallel_workers)
                    .def("get_numa_enable", &ConfigManager::numa_enable)
                    .def("set_numa_enable", &ConfigManager::set_numa_enable)
                    .def("get_op_connector_size", &ConfigManager::op_connector_size)
                    .def("get_seed", &ConfigManager::seed)
                    .def("set_rank_id", &ConfigManager::set_rank_id)
                    .def("get_rank_id", &ConfigManager::rank_id)
                    .def("get_worker_connector_size", &ConfigManager::worker_connector_size)
                    .def("set_auto_num_workers", &ConfigManager::set_auto_num_workers)
                    .def("set_auto_worker_config", &ConfigManager::set_auto_worker_config_)
                    .def("set_callback_timeout", &ConfigManager::set_callback_timeout)
                    .def("set_monitor_sampling_interval", &ConfigManager::set_monitor_sampling_interval)
                    .def("set_num_parallel_workers",
                         [](ConfigManager &c, int32_t num) { THROW_IF_ERROR(c.set_num_parallel_workers(num)); })
                    .def("set_op_connector_size", &ConfigManager::set_op_connector_size)
                    .def("set_sending_batches", &ConfigManager::set_sending_batches)
                    .def("set_seed", &ConfigManager::set_seed)
                    .def("set_worker_connector_size", &ConfigManager::set_worker_connector_size)
                    .def("set_enable_shared_mem", &ConfigManager::set_enable_shared_mem)
                    .def("get_enable_shared_mem", &ConfigManager::enable_shared_mem)
                    .def("set_auto_offload", &ConfigManager::set_auto_offload)
                    .def("get_auto_offload", &ConfigManager::get_auto_offload)
                    .def("set_enable_watchdog", &ConfigManager::set_enable_watchdog)
                    .def("get_enable_watchdog", &ConfigManager::enable_watchdog)
                    .def("set_multiprocessing_timeout_interval", &ConfigManager::set_multiprocessing_timeout_interval)
                    .def("get_multiprocessing_timeout_interval", &ConfigManager::multiprocessing_timeout_interval)
                    .def("set_dynamic_shape", &ConfigManager::set_dynamic_shape)
                    .def("get_dynamic_shape", &ConfigManager::dynamic_shape)
                    .def("set_fast_recovery", &ConfigManager::set_fast_recovery)
                    .def("get_fast_recovery", &ConfigManager::fast_recovery)
                    .def("get_debug_mode", &ConfigManager::get_debug_mode)
                    .def("set_error_samples_mode", &ConfigManager::set_error_samples_mode)
                    .def("get_error_samples_mode", &ConfigManager::get_error_samples_mode)
                    .def("set_iterator_mode", &ConfigManager::set_iterator_mode)
                    .def("get_iterator_mode", &ConfigManager::get_iterator_mode)
                    .def("set_multiprocessing_start_method", &ConfigManager::set_multiprocessing_start_method)
                    .def("get_multiprocessing_start_method", &ConfigManager::get_multiprocessing_start_method)
                    .def("set_video_backend", &ConfigManager::set_video_backend)
                    .def("get_video_backend", &ConfigManager::get_video_backend)
                    .def("load", [](ConfigManager &c, const std::string &s) { THROW_IF_ERROR(c.LoadFile(s)); });
                }));

PYBIND_REGISTER(Tensor, 0, ([](const py::module *m) {
                  (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor", py::buffer_protocol())
                    .def(py::init([](py::array arr) {
                      std::shared_ptr<Tensor> out;
                      THROW_IF_ERROR(Tensor::CreateFromNpArray(arr, &out));
                      return out;
                    }))
                    .def_buffer([](Tensor &tensor) {
                      py::buffer_info info;
                      THROW_IF_ERROR(Tensor::GetBufferInfo(&tensor, &info));
                      return info;
                    })
                    .def("__str__", &Tensor::ToString)
                    .def("shape", &Tensor::shape)
                    .def("type", &Tensor::type)
                    .def("as_python",
                         [](py::object &t) {
                           auto &tensor = py::cast<Tensor &>(t);
                           py::dict res;
                           THROW_IF_ERROR(tensor.GetDataAsPythonObject(&res));
                           return res;
                         })
                    .def("as_array", [](py::object &t) {
                      auto &tensor = py::cast<Tensor &>(t);
                      if (tensor.type().IsString()) {
                        py::array res;
                        THROW_IF_ERROR(tensor.GetDataAsNumpyStrings(&res));
                        return res;
                      }
                      py::buffer_info info;
                      THROW_IF_ERROR(Tensor::GetBufferInfo(&tensor, &info));
                      return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr, t);
                    });
                }));

#if defined(ENABLE_D)
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;

PYBIND_REGISTER(
  DeviceBuffer, 0, ([](const py::module *m) {
    (void)py::class_<DeviceBuffer, std::shared_ptr<DeviceBuffer>>(*m, "DeviceBuffer", py::buffer_protocol())
      .def(py::init([](const std::vector<size_t> &shape) { return std::make_shared<DeviceBuffer>(shape); }),
           py::call_guard<py::gil_scoped_release>())
      .def("size", &DeviceBuffer::GetBufferSize)
      .def_property_readonly("shape", ([](DeviceBuffer &device_buffer) {
                               auto shape = device_buffer.GetShape();
                               return py::tuple(py::cast(shape));
                             }))
      .def(
        "__getitem__",
        [](const std::shared_ptr<DeviceBuffer> &device_buffer, ptrdiff_t offset) {
          return std::make_shared<DeviceBuffer>(device_buffer, offset);
        },
        py::call_guard<py::gil_scoped_release>())
      .def_static(
        "from_numpy",
        [](py::array array) {
          if (!array.dtype().is(py::dtype::of<uint8_t>())) {
            throw std::runtime_error("Only support converting numpy array of uint8 to DeviceBuffer, but got: " +
                                     std::string(array.dtype().str()));
          }
          if (array.size() == 0) {
            throw std::runtime_error("Cannot convert empty numpy array to DeviceBuffer.");
          }
          std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
          auto device_buffer = std::make_shared<DeviceBuffer>(shape);
          THROW_IF_ERROR(AclAdapter::GetInstance().DvppMemcpy(device_buffer->GetBuffer(),
                                                              device_buffer->GetBufferSize(), array.mutable_data(0),
                                                              array.nbytes(), ACL_MEMCPY_HOST_TO_DEVICE));
          return device_buffer;
        })
      .def("numpy", [](const std::shared_ptr<DeviceBuffer> &device_buffer) {
        auto array = py::array_t<uint8_t>(device_buffer->GetShape(), device_buffer->GetStrides());
        THROW_IF_ERROR(AclAdapter::GetInstance().DvppMemcpy(array.mutable_data(0), device_buffer->GetBufferSize(),
                                                            device_buffer->GetBuffer(), device_buffer->GetBufferSize(),
                                                            ACL_MEMCPY_DEVICE_TO_HOST));
        return array;
      });
  }));
#endif

PYBIND_REGISTER(TensorShape, 0, ([](const py::module *m) {
                  (void)py::class_<TensorShape>(*m, "TensorShape")
                    .def(py::init<py::list>())
                    .def("__str__", &TensorShape::ToString)
                    .def("as_list", &TensorShape::AsPyList)
                    .def("is_known", &TensorShape::known);
                }));

PYBIND_REGISTER(DataType, 0, ([](const py::module *m) {
                  (void)py::class_<DataType>(*m, "DataType")
                    .def(py::init<std::string>())
                    .def(py::self == py::self)
                    .def("is_python", &DataType::IsPython)
                    .def("__str__", &DataType::ToString)
                    .def("__deepcopy__", [](py::object &t, const py::dict &memo) { return t; });
                }));

PYBIND_REGISTER(AutoAugmentPolicy, 0, ([](const py::module *m) {
                  (void)py::enum_<AutoAugmentPolicy>(*m, "AutoAugmentPolicy", py::arithmetic())
                    .value("DE_AUTO_AUGMENT_POLICY_IMAGENET", AutoAugmentPolicy::kImageNet)
                    .value("DE_AUTO_AUGMENT_POLICY_CIFAR10", AutoAugmentPolicy::kCifar10)
                    .value("DE_AUTO_AUGMENT_POLICY_SVHN", AutoAugmentPolicy::kSVHN)
                    .export_values();
                }));

PYBIND_REGISTER(BorderType, 0, ([](const py::module *m) {
                  (void)py::enum_<BorderType>(*m, "BorderType", py::arithmetic())
                    .value("DE_BORDER_CONSTANT", BorderType::kConstant)
                    .value("DE_BORDER_EDGE", BorderType::kEdge)
                    .value("DE_BORDER_REFLECT", BorderType::kReflect)
                    .value("DE_BORDER_SYMMETRIC", BorderType::kSymmetric)
                    .export_values();
                }));

PYBIND_REGISTER(InterpolationMode, 0, ([](const py::module *m) {
                  (void)py::enum_<InterpolationMode>(*m, "InterpolationMode", py::arithmetic())
                    .value("DE_INTER_LINEAR", InterpolationMode::kLinear)
                    .value("DE_INTER_CUBIC", InterpolationMode::kCubic)
                    .value("DE_INTER_AREA", InterpolationMode::kArea)
                    .value("DE_INTER_NEAREST_NEIGHBOUR", InterpolationMode::kNearestNeighbour)
                    .value("DE_INTER_PILCUBIC", InterpolationMode::kCubicPil)
                    .export_values();
                }));

PYBIND_REGISTER(ImageBatchFormat, 0, ([](const py::module *m) {
                  (void)py::enum_<ImageBatchFormat>(*m, "ImageBatchFormat", py::arithmetic())
                    .value("DE_IMAGE_BATCH_FORMAT_NHWC", ImageBatchFormat::kNHWC)
                    .value("DE_IMAGE_BATCH_FORMAT_NCHW", ImageBatchFormat::kNCHW)
                    .export_values();
                }));

PYBIND_REGISTER(SliceMode, 0, ([](const py::module *m) {
                  (void)py::enum_<SliceMode>(*m, "SliceMode", py::arithmetic())
                    .value("DE_SLICE_PAD", SliceMode::kPad)
                    .value("DE_SLICE_DROP", SliceMode::kDrop)
                    .export_values();
                }));

PYBIND_REGISTER(ConvertMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ConvertMode>(*m, "ConvertMode", py::arithmetic())
                    .value("DE_COLOR_BGR2BGRA", ConvertMode::COLOR_BGR2BGRA)
                    .value("DE_COLOR_RGB2RGBA", ConvertMode::COLOR_RGB2RGBA)
                    .value("DE_COLOR_BGRA2BGR", ConvertMode::COLOR_BGRA2BGR)
                    .value("DE_COLOR_RGBA2RGB", ConvertMode::COLOR_RGBA2RGB)
                    .value("DE_COLOR_BGR2RGBA", ConvertMode::COLOR_BGR2RGBA)
                    .value("DE_COLOR_RGB2BGRA", ConvertMode::COLOR_RGB2BGRA)
                    .value("DE_COLOR_RGBA2BGR", ConvertMode::COLOR_RGBA2BGR)
                    .value("DE_COLOR_BGRA2RGB", ConvertMode::COLOR_BGRA2RGB)
                    .value("DE_COLOR_BGR2RGB", ConvertMode::COLOR_BGR2RGB)
                    .value("DE_COLOR_RGB2BGR", ConvertMode::COLOR_RGB2BGR)
                    .value("DE_COLOR_BGRA2RGBA", ConvertMode::COLOR_BGRA2RGBA)
                    .value("DE_COLOR_RGBA2BGRA", ConvertMode::COLOR_RGBA2BGRA)
                    .value("DE_COLOR_BGR2GRAY", ConvertMode::COLOR_BGR2GRAY)
                    .value("DE_COLOR_RGB2GRAY", ConvertMode::COLOR_RGB2GRAY)
                    .value("DE_COLOR_GRAY2BGR", ConvertMode::COLOR_GRAY2BGR)
                    .value("DE_COLOR_GRAY2RGB", ConvertMode::COLOR_GRAY2RGB)
                    .value("DE_COLOR_GRAY2BGRA", ConvertMode::COLOR_GRAY2BGRA)
                    .value("DE_COLOR_GRAY2RGBA", ConvertMode::COLOR_GRAY2RGBA)
                    .value("DE_COLOR_BGRA2GRAY", ConvertMode::COLOR_BGRA2GRAY)
                    .value("DE_COLOR_RGBA2GRAY", ConvertMode::COLOR_RGBA2GRAY)
                    .export_values();
                }));

PYBIND_REGISTER(ImageReadMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ImageReadMode>(*m, "ImageReadMode", py::arithmetic())
                    .value("DE_IMAGE_READ_MODE_UNCHANGED", ImageReadMode::kUNCHANGED)
                    .value("DE_IMAGE_READ_MODE_GRAYSCALE", ImageReadMode::kGRAYSCALE)
                    .value("DE_IMAGE_READ_MODE_COLOR", ImageReadMode::kCOLOR)
                    .export_values();
                }));

PYBIND_REGISTER(ErrorSamplesMode, 0, ([](const py::module *m) {
                  (void)py::enum_<ErrorSamplesMode>(*m, "ErrorSamplesMode", py::arithmetic())
                    .value("DE_ERROR_SAMPLES_MODE_RETURN", ErrorSamplesMode::kReturn)
                    .value("DE_ERROR_SAMPLES_MODE_REPLACE", ErrorSamplesMode::kReplace)
                    .value("DE_ERROR_SAMPLES_MODE_SKIP", ErrorSamplesMode::kSkip)
                    .export_values();
                }));

#if !defined(_WIN32) && !defined(_WIN64)
PYBIND_REGISTER(MsgType, 0, ([](const py::module *m) {
                  (*m).attr("WORKER_SEND_DATA_MSG") = kWorkerSendDataMsg;
                  (*m).attr("MASTER_SEND_DATA_MSG") = kMasterSendDataMsg;
                }));

PYBIND_REGISTER(SharedMemoryQueue, 0, ([](const py::module *m) {
                  (void)py::class_<SharedMemoryQueue, std::shared_ptr<SharedMemoryQueue>>(*m, "SharedMemoryQueue")
                    .def(py::init([](key_t key) {  // the key_t is int32_t
                      return std::make_shared<SharedMemoryQueue>(key);
                    }))
                    .def("from_tensor_row",
                         [](SharedMemoryQueue &shm_queue, const TensorRow &in_row) {
                           THROW_IF_ERROR(shm_queue.FromTensorRow(in_row));
                           return shm_queue;
                         })
                    .def("from_tensor_table",
                         [](SharedMemoryQueue &shm_queue, const TensorTable &in_table, const CBatchInfo &batch_info,
                            bool concat_batch) {
                           THROW_IF_ERROR(shm_queue.FromTensorTable(in_table, &batch_info, &concat_batch));
                           return shm_queue;
                         })
                    .def("to_tensor_row",
                         [](SharedMemoryQueue &shm_queue, int shm_id, uint64_t shm_size) {
                           std::shared_ptr<TensorRow> tensor_row = std::make_shared<TensorRow>();
                           THROW_IF_ERROR(shm_queue.ToTensorRow(tensor_row.get(), shm_id, shm_size));
                           return tensor_row;
                         })
                    .def("to_tensor_table",
                         [](SharedMemoryQueue &shm_queue, int shm_id, uint64_t shm_size) {
                           TensorTable tensor_table;
                           CBatchInfo batch_info;
                           bool concat_batch;
                           THROW_IF_ERROR(
                             shm_queue.ToTensorTable(&tensor_table, &batch_info, &concat_batch, shm_id, shm_size));
                           return std::make_tuple(tensor_table, batch_info, concat_batch);
                         })
                    .def("release",
                         [](SharedMemoryQueue &shm_queue) {
                           THROW_IF_ERROR(shm_queue.ReleaseCurrentShm());
                           return shm_queue;
                         })
                    .def("set_release_flag", &SharedMemoryQueue::SetReleaseFlag)
                    .def("get_shm_id", &SharedMemoryQueue::GetShmID)
                    .def("get_shm_size", &SharedMemoryQueue::GetShmSize);
                }));

PYBIND_REGISTER(MessageQueue, 0, ([](const py::module *m) {
                  (void)py::class_<MessageQueue, std::shared_ptr<MessageQueue>>(*m, "MessageQueue")
                    .def(py::init([](key_t key) {  // the key_t is int32_t
                      return std::make_shared<MessageQueue>(key);
                    }))
                    .def("set_release_flag", &MessageQueue::SetReleaseFlag)
                    .def("msg_snd",
                         [](MessageQueue &msg_queue, int64_t mtype, int shm_id, uint64_t shm_size) {
                           THROW_IF_ERROR(msg_queue.MsgSnd(mtype, shm_id, shm_size));
                           return msg_queue;
                         })
                    .def("msg_rcv",
                         [](MessageQueue &msg_queue, int mtype) {
                           THROW_IF_ERROR(msg_queue.MsgRcv(mtype));
                           return msg_queue;
                         })
                    .def("serialize_status",
                         [](MessageQueue &msg_queue, int32_t status_code, int32_t line_of_code,
                            const std::string &filename, const std::string &err_desc) {
                           THROW_IF_ERROR(msg_queue.SerializeStatus(status_code, line_of_code, filename, err_desc));
                           return msg_queue;
                         })
                    .def("release", &MessageQueue::ReleaseQueue)
                    .def("message_queue_state", &MessageQueue::MessageQueueState)
                    .def_readonly("shm_id", &MessageQueue::shm_id_)
                    .def_readonly("shm_size", &MessageQueue::shm_size_)
                    .def_readonly("msg_queue_id", &MessageQueue::msg_queue_id_);
                }));

PYBIND_REGISTER(MessageState, 0, ([](const py::module *m) {
                  (void)py::enum_<MessageState>(*m, "MessageState", py::arithmetic())
                    .value("INIT", mindspore::dataset::MessageState::kInit)
                    .value("RUNNING", mindspore::dataset::MessageState::kRunning)
                    .value("RELEASED", mindspore::dataset::MessageState::kReleased)
                    .export_values();
                }));

PYBIND_REGISTER(TensorRow, 0, ([](const py::module *m) {
                  (void)py::class_<TensorRow, std::shared_ptr<TensorRow>>(*m, "TensorRow").def(py::init([]() {
                    return std::make_shared<TensorRow>();
                  }));
                }));

PYBIND_REGISTER(ConvertTensorRowToPyTuple, 0, ([](py::module *m) {
                  (void)m->def("convert_tensor_row_to_py_tuple", ([](const TensorRow &input) {
                                 py::tuple output(input.size());
                                 THROW_IF_ERROR(ConvertTensorRowToPyTuple(input, &output));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(ConvertPyTupleToTensorRow, 0, ([](py::module *m) {
                  (void)m->def("convert_py_tuple_to_tensor_row", ([](const py::tuple &input) {
                                 TensorRow output;
                                 THROW_IF_ERROR(ConvertPyTupleToTensorRow(input, &output));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(ConvertTensorTableToPyTupleList, 0, ([](py::module *m) {
                  (void)m->def("convert_tensor_table_to_py_tuple_list", ([](const TensorTable &input) {
                                 py::tuple output(input.size());
                                 THROW_IF_ERROR(ConvertTensorTableToPyTupleList(input, &output));
                                 return output;
                               }));
                }));

PYBIND_REGISTER(ConvertPyTupleListToTensorTable, 0, ([](py::module *m) {
                  (void)m->def("convert_py_tuple_list_to_tensor_table", ([](const py::tuple &input) {
                                 TensorTable output;
                                 bool concat_batch;
                                 THROW_IF_ERROR(ConvertPyTupleListToTensorTable(input, &output, &concat_batch));
                                 return std::make_tuple(output, concat_batch);
                               }));
                }));

PYBIND_REGISTER(StatusCode, 0, ([](const py::module *m) {
                  (void)py::enum_<StatusCode>(*m, "StatusCode", py::module_local())
                    .value("MD_PY_FUNC_EXCEPTION", mindspore::StatusCode::kMDPyFuncException)
                    .export_values();
                }));
#endif
}  // namespace dataset
}  // namespace mindspore
