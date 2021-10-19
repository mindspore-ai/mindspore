/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(CreateDct, 1, ([](py::module *m) {
                  (void)m->def("CreateDct", ([](int32_t n_mfcc, int32_t n_mels, NormMode norm) {
                                 std::shared_ptr<Tensor> out;
                                 THROW_IF_ERROR(Dct(&out, n_mfcc, n_mels, norm));
                                 return out;
                               }));
                }));

PYBIND_REGISTER(NormMode, 0, ([](const py::module *m) {
                  (void)py::enum_<NormMode>(*m, "NormMode", py::arithmetic())
                    .value("DE_NORMMODE_NONE", NormMode::kNone)
                    .value("DE_NORMMODE_ORTHO", NormMode::kOrtho)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
