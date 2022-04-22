/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "frontend/operator/composite/composite.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/operator/composite/list_append_operation.h"
#include "frontend/operator/composite/list_insert_operation.h"
#include "frontend/operator/composite/map.h"
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/operator/composite/vmap.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "frontend/operator/composite/zip_operation.h"
namespace mindspore {
namespace prim {
REGISTER_PYBIND_WITH_PARENT_NAME(
  CompositeOpsGroup_, MetaFuncGraph, ([](const py::module *m) {
    //  Reg HyperMap
    (void)py::class_<HyperMapPy, MetaFuncGraph, std::shared_ptr<HyperMapPy>>(*m, "HyperMap_")
      .def(py::init<bool, std::shared_ptr<MultitypeFuncGraph>>(), py::arg("reverse"), py::arg("ops"))
      .def(py::init<bool>(), py::arg("reverse"));

    // Reg Tail
    (void)py::class_<Tail, MetaFuncGraph, std::shared_ptr<Tail>>(*m, "Tail_").def(py::init<std::string &>());

    // Reg GradOperation
    (void)py::class_<GradOperation, MetaFuncGraph, std::shared_ptr<GradOperation>>(*m, "GradOperation_")
      .def(py::init<std::string &>(), py::arg("fn"))
      .def(py::init<std::string &, bool, bool, bool, bool>(), py::arg("fn"), py::arg("get_all"), py::arg("get_by_list"),
           py::arg("sens_param"), py::arg("get_by_position"));

    // Reg VmapOperation
    (void)py::class_<VmapOperation, MetaFuncGraph, std::shared_ptr<VmapOperation>>(*m, "VmapOperation_")
      .def(py::init<const std::string &>(), py::arg("fn"));

    // Reg TaylorOperation
    (void)py::class_<TaylorOperation, MetaFuncGraph, std::shared_ptr<TaylorOperation>>(*m, "TaylorOperation_")
      .def(py::init<const std::string &>(), py::arg("fn"));

    // Reg TupleAdd
    (void)py::class_<TupleAdd, MetaFuncGraph, std::shared_ptr<TupleAdd>>(*m, "TupleAdd_")
      .def(py::init<std::string &>());

    // Reg TupleGetItemTensor
    (void)py::class_<TupleGetItemTensor, MetaFuncGraph, std::shared_ptr<TupleGetItemTensor>>(*m, "TupleGetItemTensor_")
      .def(py::init<std::string &>());

    // Reg ListSliceSetItem
    (void)py::class_<ListSliceSetItem, MetaFuncGraph, std::shared_ptr<ListSliceSetItem>>(*m, "ListSliceSetItem_")
      .def(py::init<const std::string &>());

    // Reg SequenceSliceGetItem
    (void)py::class_<SequenceSliceGetItem, MetaFuncGraph, std::shared_ptr<SequenceSliceGetItem>>(
      *m, "SequenceSliceGetItem_")
      .def(py::init<std::string &, std::string &, std::string &>());

    // Reg Shard
    (void)py::class_<Shard, MetaFuncGraph, std::shared_ptr<Shard>>(*m, "Shard_")
      .def(py::init<const std::string &>(), py::arg("fn"));

    // Reg ListAppend
    (void)py::class_<ListAppend, MetaFuncGraph, std::shared_ptr<ListAppend>>(*m, "ListAppend_")
      .def(py::init<std::string &>());

    // Reg ListInsert
    (void)py::class_<ListInsert, MetaFuncGraph, std::shared_ptr<ListInsert>>(*m, "ListInsert_")
      .def(py::init<const std::string &>());

    // Reg MapPy
    (void)py::class_<MapPy, MetaFuncGraph, std::shared_ptr<MapPy>>(*m, "Map_")
      .def(py::init<bool, std::shared_ptr<MultitypeFuncGraph>>(), py::arg("reverse"), py::arg("ops"))
      .def(py::init<bool>(), py::arg("reverse"));

    // Reg MultitypeFuncGraph
    (void)py::class_<MultitypeFuncGraph, MetaFuncGraph, std::shared_ptr<MultitypeFuncGraph>>(*m, "MultitypeFuncGraph_")
      .def(py::init<std::string &>())
      .def("register_fn", &MultitypeFuncGraph::PyRegister);

    // Reg UnpackCall
    (void)py::class_<UnpackCall, MetaFuncGraph, std::shared_ptr<UnpackCall>>(*m, "UnpackCall_")
      .def(py::init<std::string &>());

    // Reg ZipOperation
    (void)py::class_<ZipOperation, MetaFuncGraph, std::shared_ptr<ZipOperation>>(*m, "ZipOperation_")
      .def(py::init<std::string &>());

    // Reg VmapGeneralPreprocess
    (void)py::class_<VmapGeneralPreprocess, MetaFuncGraph, std::shared_ptr<VmapGeneralPreprocess>>(
      *m, "VmapGeneralPreprocess_")
      .def(py::init<std::string &>(), py::arg("fn"));
  }));
}  // namespace prim
}  // namespace mindspore
