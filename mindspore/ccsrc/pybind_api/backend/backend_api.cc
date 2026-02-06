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

#include "pybind_api/backend/backend_api.h"
#include <memory>
#include <string>
#include "pybind11/pybind11.h"
#include "include/cluster/topology/ps_context.h"
#include "include/cluster/init.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/cluster/rpc/tcp_store.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "pybind_api/backend/graph/pipeline_py.h"
#include "pybind_api/backend/graph/custom_pass_py.h"
#include "pybind_api/backend/graph/custom_backend_py.h"

using TCPStoreClient = mindspore::distributed::cluster::TCPStoreClient;
namespace py = pybind11;

namespace mindspore {
void RegGraphFragment(py::module *m) {
  (void)py::class_<mindspore::backend::GraphFragment, mindspore::backend::GraphFragmentPtr>(*m, "_GraphFragment_")
    .def(py::init([](const py::object &graph_fragment) {
           auto graph_fragment_ = graph_fragment.cast<mindspore::backend::GraphFragmentPtr>();
           return std::make_shared<mindspore::backend::GraphFragment>(graph_fragment_);
         }),
         py::arg("input"))
    .def("__call__", &mindspore::backend::GraphFragment::Run, "Executor run function.")
    .def("__str__", &mindspore::backend::GraphFragment::ToString, "Executor run function.")
    .def("id_", &mindspore::backend::GraphFragment::id, "Executor run function.")
    .def("is_graph_", &mindspore::backend::GraphFragment::is_graph, "Executor run function.")
    .def("py_key_", &mindspore::backend::GraphFragment::py_key, "Executor run function.")
    .def("args_list_", &mindspore::backend::GraphFragment::args_list, "Executor run function.");
}

void RegBackendModule(py::module *m) {
  mindspore::graph::RegCustomPass(m);
  mindspore::graph::RegCustomBackend(m);
  mindspore::backend::RegBackendGraphMock(m);
  RegGraphFragment(m);
  (void)m->def("reset_op_id", &mindspore::pipeline::ResetOpId, "Reset Operator Id");
  (void)m->def("reset_op_id_with_offset", &mindspore::pipeline::ResetOpIdWithOffset, "Reset Operator Id With Offset");
  (void)m->def("init_hccl", (void (*)()) & mindspore::pipeline::InitHccl, "Init Hccl");
  (void)m->def(
    "_init_hccl_with_store",
    [](std::optional<std::string> init_method, int64_t timeout, uint32_t world_size, uint32_t node_id,
       const py::object &store) {
      std::shared_ptr<TCPStoreClient> store_client = nullptr;
      if (!store.is_none()) {
        store_client = store.attr("instance").cast<std::shared_ptr<TCPStoreClient>>();
      }
      mindspore::pipeline::InitHccl(init_method, timeout, world_size, node_id, store_client);
    },
    py::arg("init_method"), py::arg("timeout"), py::arg("world_size"), py::arg("node_id"), py::arg("store"),
    "Init Hccl without scheduler process");
  (void)m->def("finalize_hccl", &mindspore::pipeline::FinalizeHccl, "Finalize Hccl");
  (void)m->def("_finalize_collective", &mindspore::distributed::FinalizeCollective, "Finalize Collective");
  (void)m->def("get_hccl_rank_id", &mindspore::pipeline::GetHcclRankId, "Get Hccl Rank Id");
  (void)m->def("get_hccl_rank_size", &mindspore::pipeline::GetHcclRankSize, "Get Hccl Rank Size");
  (void)m->def("init_exec_dataset", &mindspore::pipeline::InitExecDataset, py::arg("queue_name"), py::arg("size"),
               py::arg("batch_size"), py::arg("types"), py::arg("shapes"), py::arg("input_indexs"),
               py::arg("phase") = py::str("dataset"), py::arg("need_run") = py::bool_(true), "Init and exec dataset.");
  (void)m->def("init_cluster", (bool (*)()) & mindspore::distributed::Initialize, "Init Cluster");
  (void)m->def(
    "_init_cluster_with_store",
    [](std::optional<std::string> init_method, int64_t timeout, uint32_t world_size, uint32_t node_id,
       const py::object &store) {
      auto store_client = store.attr("instance").cast<std::shared_ptr<TCPStoreClient>>();
      mindspore::distributed::Initialize(init_method, timeout, world_size, node_id, store_client);
    },
    py::arg("init_method"), py::arg("timeout"), py::arg("world_size"), py::arg("node_id"), py::arg("store"),
    "Init Cluster without scheduler process");
  (void)m->def("set_cluster_exit_with_exception", &mindspore::distributed::set_cluster_exit_with_exception,
               "Set this process exits with exception.");
  (void)m->def("_encrypt", &mindspore::pipeline::PyEncrypt, "Encrypt the data.");
  (void)m->def("_decrypt", &mindspore::pipeline::PyDecrypt, "Decrypt the data.");
  (void)m->def("_decrypt_data", &mindspore::pipeline::PyDecryptData, "Decrypt the bytes data.");
  (void)m->def("_is_cipher_file", &mindspore::pipeline::PyIsCipherFile, "Determine whether the file is encrypted");
  (void)m->def("_bind_device_ctx", &mindspore::pipeline::BindDeviceCtx, "Bind device context to current thread");
  (void)m->def("swap_cache", &mindspore::pipeline::SwapCache, py::arg("host"), py::arg("device"),
               py::arg("block_mapping"), py::arg("is_device_to_host"), "Swap Cache for PageAttention.");
}

}  // namespace mindspore
