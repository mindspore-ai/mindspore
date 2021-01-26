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
#include "minddata/dataset/core/tensor_helpers.h"
#include "minddata/dataset/kernels/data/concatenate_op.h"
#include "minddata/dataset/kernels/data/duplicate_op.h"
#include "minddata/dataset/kernels/data/fill_op.h"
#include "minddata/dataset/kernels/data/mask_op.h"
#include "minddata/dataset/kernels/data/one_hot_op.h"
#include "minddata/dataset/kernels/data/pad_end_op.h"
#include "minddata/dataset/kernels/data/slice_op.h"
#include "minddata/dataset/kernels/data/to_float16_op.h"
#include "minddata/dataset/kernels/data/type_cast_op.h"
#include "minddata/dataset/kernels/data/unique_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(ConcatenateOp, 1, ([](const py::module *m) {
                  (void)py::class_<ConcatenateOp, TensorOp, std::shared_ptr<ConcatenateOp>>(*m, "ConcatenateOp")
                    .def(py::init<int8_t, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>());
                }));

PYBIND_REGISTER(
  DuplicateOp, 1, ([](const py::module *m) {
    (void)py::class_<DuplicateOp, TensorOp, std::shared_ptr<DuplicateOp>>(*m, "DuplicateOp").def(py::init<>());
  }));

PYBIND_REGISTER(UniqueOp, 1, ([](const py::module *m) {
                  (void)py::class_<UniqueOp, TensorOp, std::shared_ptr<UniqueOp>>(*m, "UniqueOp").def(py::init<>());
                }));

PYBIND_REGISTER(
  FillOp, 1, ([](const py::module *m) {
    (void)py::class_<FillOp, TensorOp, std::shared_ptr<FillOp>>(*m, "FillOp").def(py::init<std::shared_ptr<Tensor>>());
  }));

PYBIND_REGISTER(MaskOp, 1, ([](const py::module *m) {
                  (void)py::class_<MaskOp, TensorOp, std::shared_ptr<MaskOp>>(*m, "MaskOp")
                    .def(py::init<RelationalOp, std::shared_ptr<Tensor>, DataType>());
                }));

PYBIND_REGISTER(
  OneHotOp, 1, ([](const py::module *m) {
    (void)py::class_<OneHotOp, TensorOp, std::shared_ptr<OneHotOp>>(*m, "OneHotOp").def(py::init<int32_t>());
  }));

PYBIND_REGISTER(PadEndOp, 1, ([](const py::module *m) {
                  (void)py::class_<PadEndOp, TensorOp, std::shared_ptr<PadEndOp>>(*m, "PadEndOp")
                    .def(py::init<TensorShape, std::shared_ptr<Tensor>>());
                }));

PYBIND_REGISTER(SliceOption, 0, ([](const py::module *m) {
                  (void)py::class_<SliceOption>(*m, "SliceOption")
                    .def(py::init([](const py::slice &py_slice) {
                      Slice c_slice;
                      if (!py_slice.attr("start").is_none() && !py_slice.attr("stop").is_none() &&
                          !py_slice.attr("step").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("start")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("stop")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("step")));
                      } else if (py_slice.attr("start").is_none() && py_slice.attr("step").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("stop")));
                      } else if (!py_slice.attr("start").is_none() && !py_slice.attr("stop").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("start")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("stop")));
                      }

                      if (!c_slice.valid()) {
                        THROW_IF_ERROR(
                          Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Wrong slice object"));
                      }
                      return SliceOption(c_slice);
                    }))
                    .def(py::init([](const py::list &py_list) {
                      std::vector<dsize_t> indices;
                      for (auto l : py_list) {
                        indices.push_back(py::reinterpret_borrow<py::int_>(l));
                      }
                      return SliceOption(indices);
                    }))
                    .def(py::init<bool>())
                    .def(py::init<SliceOption>());
                }));

PYBIND_REGISTER(SliceOp, 1, ([](const py::module *m) {
                  (void)py::class_<SliceOp, TensorOp, std::shared_ptr<SliceOp>>(*m, "SliceOp")
                    .def(py::init<std::vector<SliceOption>>());
                }));

PYBIND_REGISTER(ToFloat16Op, 1, ([](const py::module *m) {
                  (void)py::class_<ToFloat16Op, TensorOp, std::shared_ptr<ToFloat16Op>>(*m, "ToFloat16Op",
                                                                                        py::dynamic_attr())
                    .def(py::init<>());
                }));

PYBIND_REGISTER(TypeCastOp, 1, ([](const py::module *m) {
                  (void)py::class_<TypeCastOp, TensorOp, std::shared_ptr<TypeCastOp>>(*m, "TypeCastOp")
                    .def(py::init<DataType>())
                    .def(py::init<std::string>());
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
