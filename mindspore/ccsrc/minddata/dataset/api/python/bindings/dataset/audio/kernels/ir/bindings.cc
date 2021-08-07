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
#include "minddata/dataset/audio/ir/kernels/allpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/amplitude_to_db_ir.h"
#include "minddata/dataset/audio/ir/kernels/angle_ir.h"
#include "minddata/dataset/audio/ir/kernels/band_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandreject_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_stretch_ir.h"
#include "minddata/dataset/include/dataset/transforms.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  AllpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::AllpassBiquadOperation, TensorOperation, std::shared_ptr<audio::AllpassBiquadOperation>>(
      *m, "AllpassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q) {
        auto allpass_biquad = std::make_shared<audio::AllpassBiquadOperation>(sample_rate, central_freq, Q);
        THROW_IF_ERROR(allpass_biquad->ValidateParams());
        return allpass_biquad;
      }));
  }));

PYBIND_REGISTER(
  AmplitudeToDBOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::AmplitudeToDBOperation, TensorOperation, std::shared_ptr<audio::AmplitudeToDBOperation>>(
      *m, "AmplitudeToDBOperation")
      .def(py::init([](ScaleType stype, float ref_value, float amin, float top_db) {
        auto amplitude_to_db = std::make_shared<audio::AmplitudeToDBOperation>(stype, ref_value, amin, top_db);
        THROW_IF_ERROR(amplitude_to_db->ValidateParams());
        return amplitude_to_db;
      }));
  }));

PYBIND_REGISTER(ScaleType, 0, ([](const py::module *m) {
                  (void)py::enum_<ScaleType>(*m, "ScaleType", py::arithmetic())
                    .value("DE_SCALETYPE_MAGNITUDE", ScaleType::kMagnitude)
                    .value("DE_SCALETYPE_POWER", ScaleType::kPower)
                    .export_values();
                }));

PYBIND_REGISTER(AngleOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::AngleOperation, TensorOperation, std::shared_ptr<audio::AngleOperation>>(
                    *m, "AngleOperation")
                    .def(py::init([]() {
                      auto angle = std::make_shared<audio::AngleOperation>();
                      THROW_IF_ERROR(angle->ValidateParams());
                      return angle;
                    }));
                }));

PYBIND_REGISTER(
  BandBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BandBiquadOperation, TensorOperation, std::shared_ptr<audio::BandBiquadOperation>>(
      *m, "BandBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q, bool noise) {
        auto band_biquad = std::make_shared<audio::BandBiquadOperation>(sample_rate, central_freq, Q, noise);
        THROW_IF_ERROR(band_biquad->ValidateParams());
        return band_biquad;
      }));
  }));

PYBIND_REGISTER(
  BandpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BandpassBiquadOperation, TensorOperation, std::shared_ptr<audio::BandpassBiquadOperation>>(
      *m, "BandpassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain) {
        auto bandpass_biquad =
          std::make_shared<audio::BandpassBiquadOperation>(sample_rate, central_freq, Q, const_skirt_gain);
        THROW_IF_ERROR(bandpass_biquad->ValidateParams());
        return bandpass_biquad;
      }));
  }));

PYBIND_REGISTER(BandrejectBiquadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::BandrejectBiquadOperation, TensorOperation,
                                   std::shared_ptr<audio::BandrejectBiquadOperation>>(*m, "BandrejectBiquadOperation")
                    .def(py::init([](int32_t sample_rate, float central_freq, float Q) {
                      auto bandreject_biquad =
                        std::make_shared<audio::BandrejectBiquadOperation>(sample_rate, central_freq, Q);
                      THROW_IF_ERROR(bandreject_biquad->ValidateParams());
                      return bandreject_biquad;
                    }));
                }));

PYBIND_REGISTER(
  BassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BassBiquadOperation, TensorOperation, std::shared_ptr<audio::BassBiquadOperation>>(
      *m, "BassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float gain, float central_freq, float Q) {
        auto bass_biquad = std::make_shared<audio::BassBiquadOperation>(sample_rate, gain, central_freq, Q);
        THROW_IF_ERROR(bass_biquad->ValidateParams());
        return bass_biquad;
      }));
  }));

PYBIND_REGISTER(
  TimeStretchOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::TimeStretchOperation, TensorOperation, std::shared_ptr<audio::TimeStretchOperation>>(
      *m, "TimeStretchOperation")
      .def(py::init([](float hop_length, int n_freq, float fixed_rate) {
        auto timestretch = std::make_shared<audio::TimeStretchOperation>(hop_length, n_freq, fixed_rate);
        THROW_IF_ERROR(timestretch->ValidateParams());
        return timestretch;
      }));
  }));
}  // namespace dataset
}  // namespace mindspore
