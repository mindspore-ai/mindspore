/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/kernels/data/concatenate_op.h"
#include "minddata/dataset/kernels/data/duplicate_op.h"
#include "minddata/dataset/kernels/data/fill_op.h"
#include "minddata/dataset/kernels/data/mask_op.h"
#include "minddata/dataset/kernels/data/one_hot_op.h"
#include "minddata/dataset/kernels/data/pad_end_op.h"
#include "minddata/dataset/kernels/data/slice_op.h"
#include "minddata/dataset/kernels/data/to_float16_op.h"
#include "minddata/dataset/kernels/data/type_cast_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(ConcatenateOp, 1, ([](const py::module *m) {
                  (void)py::class_<ConcatenateOp, TensorOp, std::shared_ptr<ConcatenateOp>>(
                    *m, "ConcatenateOp", "Tensor operation concatenate tensors.")
                    .def(py::init<int8_t, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(), py::arg("axis"),
                         py::arg("prepend").none(true), py::arg("append").none(true));
                }));

PYBIND_REGISTER(DuplicateOp, 1, ([](const py::module *m) {
                  (void)py::class_<DuplicateOp, TensorOp, std::shared_ptr<DuplicateOp>>(*m, "DuplicateOp",
                                                                                        "Duplicate tensor.")
                    .def(py::init<>());
                }));

PYBIND_REGISTER(FillOp, 1, ([](const py::module *m) {
                  (void)py::class_<FillOp, TensorOp, std::shared_ptr<FillOp>>(
                    *m, "FillOp", "Tensor operation to return tensor filled with same value as input fill value.")
                    .def(py::init<std::shared_ptr<Tensor>>());
                }));

PYBIND_REGISTER(MaskOp, 1, ([](const py::module *m) {
                  (void)py::class_<MaskOp, TensorOp, std::shared_ptr<MaskOp>>(
                    *m, "MaskOp", "Tensor mask operation using relational comparator")
                    .def(py::init<RelationalOp, std::shared_ptr<Tensor>, DataType>());
                }));

PYBIND_REGISTER(OneHotOp, 1, ([](const py::module *m) {
                  (void)py::class_<OneHotOp, TensorOp, std::shared_ptr<OneHotOp>>(
                    *m, "OneHotOp", "Tensor operation to apply one hot encoding. Takes number of classes.")
                    .def(py::init<int32_t>());
                }));

PYBIND_REGISTER(PadEndOp, 1, ([](const py::module *m) {
                  (void)py::class_<PadEndOp, TensorOp, std::shared_ptr<PadEndOp>>(
                    *m, "PadEndOp", "Tensor operation to pad end of tensor with a pad value.")
                    .def(py::init<TensorShape, std::shared_ptr<Tensor>>());
                }));

PYBIND_REGISTER(SliceOp, 1, ([](const py::module *m) {
                  (void)py::class_<SliceOp, TensorOp, std::shared_ptr<SliceOp>>(*m, "SliceOp",
                                                                                "Tensor slice operation.")
                    .def(py::init<bool>())
                    .def(py::init([](const py::list &py_list) {
                      std::vector<dsize_t> c_list;
                      for (auto l : py_list) {
                        if (!l.is_none()) {
                          c_list.push_back(py::reinterpret_borrow<py::int_>(l));
                        }
                      }
                      return std::make_shared<SliceOp>(c_list);
                    }))
                    .def(py::init([](const py::tuple &py_slice) {
                      if (py_slice.size() != 3) {
                        THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Wrong slice object"));
                      }
                      Slice c_slice;
                      if (!py_slice[0].is_none() && !py_slice[1].is_none() && !py_slice[2].is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice[0]),
                                        py::reinterpret_borrow<py::int_>(py_slice[1]),
                                        py::reinterpret_borrow<py::int_>(py_slice[2]));
                      } else if (py_slice[0].is_none() && py_slice[2].is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice[1]));
                      } else if (!py_slice[0].is_none() && !py_slice[1].is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice[0]),
                                        py::reinterpret_borrow<py::int_>(py_slice[1]));
                      }

                      if (!c_slice.valid()) {
                        THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Wrong slice object"));
                      }
                      return std::make_shared<SliceOp>(c_slice);
                    }));
                }));

PYBIND_REGISTER(ToFloat16Op, 1, ([](const py::module *m) {
                  (void)py::class_<ToFloat16Op, TensorOp, std::shared_ptr<ToFloat16Op>>(
                    *m, "ToFloat16Op", py::dynamic_attr(),
                    "Tensor operator to type cast float32 data to a float16 type.")
                    .def(py::init<>());
                }));

PYBIND_REGISTER(TypeCastOp, 1, ([](const py::module *m) {
                  (void)py::class_<TypeCastOp, TensorOp, std::shared_ptr<TypeCastOp>>(
                    *m, "TypeCastOp", "Tensor operator to type cast data to a specified type.")
                    .def(py::init<DataType>(), py::arg("data_type"))
                    .def(py::init<std::string>(), py::arg("data_type"));
                }));

PYBIND_REGISTER(RelationalOp, 0, ([](const py::module *m) {
                  (void)py::enum_<RelationalOp>(*m, "RelationalOp", py::arithmetic())
                    .value("EQ", RelationalOp::kEqual)
                    .value("NE", RelationalOp::kNotEqual)
                    .value("LT", RelationalOp::kLess)
                    .value("LE", RelationalOp::kLessEqual)
                    .value("GT", RelationalOp::kGreater)
                    .value("GE", RelationalOp::kGreaterEqual)
                    .export_values();
                }));

}  // namespace dataset
}  // namespace mindspore
