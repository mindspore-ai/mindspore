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
                  (void)m->def("create_dct", ([](int32_t n_mfcc, int32_t n_mels, NormMode norm) {
                                 std::shared_ptr<Tensor> out;
                                 THROW_IF_ERROR(Dct(&out, n_mfcc, n_mels, norm));
                                 return out;
                               }));
                }));

PYBIND_REGISTER(MelscaleFbanks, 1, ([](py::module *m) {
                  (void)m->def("melscale_fbanks", ([](int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                                                      int32_t sample_rate, NormType norm, MelType mel_type) {
                                 std::shared_ptr<Tensor> fb;
                                 THROW_IF_ERROR(CreateFbanks<float>(&fb, n_freqs, f_min, f_max, n_mels, sample_rate,
                                                                    norm, mel_type));
                                 return fb;
                               }));
                }));

PYBIND_REGISTER(MelType, 0, ([](const py::module *m) {
                  (void)py::enum_<MelType>(*m, "MelType", py::arithmetic())
                    .value("DE_MEL_TYPE_HTK", MelType::kHtk)
                    .value("DE_MEL_TYPE_SLANEY", MelType::kSlaney)
                    .export_values();
                }));

PYBIND_REGISTER(NormType, 0, ([](const py::module *m) {
                  (void)py::enum_<NormType>(*m, "NormType", py::arithmetic())
                    .value("DE_NORM_TYPE_NONE", NormType::kNone)
                    .value("DE_NORM_TYPE_SLANEY", NormType::kSlaney)
                    .export_values();
                }));

PYBIND_REGISTER(LinearFbanks, 1, ([](py::module *m) {
                  (void)m->def("linear_fbanks",
                               ([](int32_t n_freqs, float f_min, float f_max, int32_t n_filter, int32_t sample_rate) {
                                 std::shared_ptr<Tensor> fb;
                                 THROW_IF_ERROR(CreateLinearFbanks(&fb, n_freqs, f_min, f_max, n_filter, sample_rate));
                                 return fb;
                               }));
                }));

PYBIND_REGISTER(NormMode, 0, ([](const py::module *m) {
                  (void)py::enum_<NormMode>(*m, "NormMode", py::arithmetic())
                    .value("DE_NORM_MODE_NONE", NormMode::kNone)
                    .value("DE_NORM_MODE_ORTHO", NormMode::kOrtho)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
