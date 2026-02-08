/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/util/shared_mem.h"
#include "minddata/dataset/util/sig_handler.h"

namespace mindspore {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
PYBIND_REGISTER(SharedMemory, 0, ([](const py::module *m) {
                  (void)py::class_<SharedMem, std::shared_ptr<SharedMem>>(*m, "SharedMemory")
                    .def(py::init([](const py::object &name, bool create, int fd, size_t size) {
                      std::string shm_name;
                      if (py::isinstance<py::none>(name)) {
                        shm_name = GenerateShmName();
                      } else {
                        shm_name = py::cast<std::string>(name);
                      }
                      return std::make_shared<SharedMem>(shm_name, create, fd, size);
                    }))
                    .def("buf",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return py::array_t<uint8_t>({shared_memory.Size()}, {sizeof(uint8_t)},
                                                       reinterpret_cast<uint8_t *>(shared_memory.Buf()),
                                                       py::capsule(shared_memory.Buf(), [](void *v) {}));
                         })
                    .def("name",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return shared_memory.Name();
                         })
                    .def("fd",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return shared_memory.Fd();
                         })
                    .def("size", [](py::object &obj) {
                      auto &shared_memory = py::cast<SharedMem &>(obj);
                      return shared_memory.Size();
                    });
                }));

PYBIND_REGISTER(ReleaseShmAndMsgByWorkerPIDs, 0, ([](py::module *m) {
                  (void)m->def("release_shm_and_msg_by_worker_pids",
                               ([](const std::vector<int> &pids) { ReleaseShmAndMsgByWorkerPIDs(pids); }));
                }));
#endif

PYBIND_REGISTER(RegisterWorkerHandlers, 0, ([](py::module *m) {
                  (void)m->def("register_worker_handlers", ([]() { RegisterWorkerHandlers(); }));
                }));

PYBIND_REGISTER(RegisterWorkerPIDs, 0, ([](py::module *m) {
                  (void)m->def("register_worker_pids",
                               ([](int64_t id, const std::vector<int> &pids) { RegisterWorkerPIDs(id, pids); }));
                }));

PYBIND_REGISTER(DeregisterWorkerPIDs, 0, ([](py::module *m) {
                  (void)m->def("deregister_worker_pids", ([](int64_t id) { DeregisterWorkerPIDs(id); }));
                }));

PYBIND_REGISTER(RegisterShmIDAndMsgID, 0, ([](py::module *m) {
                  (void)m->def("register_shm_id_and_msg_id", ([](std::string pid, int32_t shm_id, int32_t msg_id) {
                                 RegisterShmIDAndMsgID(pid, shm_id, msg_id);
                               }));
                }));

PYBIND_REGISTER(UnlockShmIDAndMsgIDMutex, 0, ([](py::module *m) {
                  (void)m->def("unlock_shm_id_and_msg_id_mutex", ([]() { UnlockShmIDAndMsgIDMutex(); }));
                }));

PYBIND_REGISTER(CheckIfWorkerExit, 0,
                ([](py::module *m) { (void)m->def("check_if_worker_exit", ([]() { return CheckIfWorkerExit(); })); }));
}  // namespace dataset
}  // namespace mindspore
