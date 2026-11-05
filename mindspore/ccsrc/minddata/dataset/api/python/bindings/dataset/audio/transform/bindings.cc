/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(ScaleType, 0, ([](const py::module *m) {
                  (void)py::enum_<ScaleType>(*m, "ScaleType", py::arithmetic())
                    .value("DE_SCALE_TYPE_MAGNITUDE", ScaleType::kMagnitude)
                    .value("DE_SCALE_TYPE_POWER", ScaleType::kPower)
                    .export_values();
                }));

PYBIND_REGISTER(DensityFunction, 0, ([](const py::module *m) {
                  (void)py::enum_<DensityFunction>(*m, "DensityFunction", py::arithmetic())
                    .value("DE_DENSITY_FUNCTION_TPDF", DensityFunction::kTPDF)
                    .value("DE_DENSITY_FUNCTION_RPDF", DensityFunction::kRPDF)
                    .value("DE_DENSITY_FUNCTION_GPDF", DensityFunction::kGPDF)
                    .export_values();
                }));

PYBIND_REGISTER(FadeShape, 0, ([](const py::module *m) {
                  (void)py::enum_<FadeShape>(*m, "FadeShape", py::arithmetic())
                    .value("DE_FADE_SHAPE_LINEAR", FadeShape::kLinear)
                    .value("DE_FADE_SHAPE_EXPONENTIAL", FadeShape::kExponential)
                    .value("DE_FADE_SHAPE_LOGARITHMIC", FadeShape::kLogarithmic)
                    .value("DE_FADE_SHAPE_QUARTER_SINE", FadeShape::kQuarterSine)
                    .value("DE_FADE_SHAPE_HALF_SINE", FadeShape::kHalfSine)
                    .export_values();
                }));

PYBIND_REGISTER(Modulation, 0, ([](const py::module *m) {
                  (void)py::enum_<Modulation>(*m, "Modulation", py::arithmetic())
                    .value("DE_MODULATION_SINUSOIDAL", Modulation::kSinusoidal)
                    .value("DE_MODULATION_TRIANGULAR", Modulation::kTriangular)
                    .export_values();
                }));

PYBIND_REGISTER(Interpolation, 0, ([](const py::module *m) {
                  (void)py::enum_<Interpolation>(*m, "Interpolation", py::arithmetic())
                    .value("DE_INTERPOLATION_LINEAR", Interpolation::kLinear)
                    .value("DE_INTERPOLATION_QUADRATIC", Interpolation::kQuadratic)
                    .export_values();
                }));

PYBIND_REGISTER(ResampleMethod, 0, ([](const py::module *m) {
                  (void)py::enum_<ResampleMethod>(*m, "ResampleMethod", py::arithmetic())
                    .value("DE_RESAMPLE_SINC_INTERPOLATION", ResampleMethod::kSincInterpolation)
                    .value("DE_RESAMPLE_KAISER_WINDOW", ResampleMethod::kKaiserWindow)
                    .export_values();
                }));

PYBIND_REGISTER(WindowType, 0, ([](const py::module *m) {
                  (void)py::enum_<WindowType>(*m, "WindowType", py::arithmetic())
                    .value("DE_WINDOW_TYPE_BARTLETT", WindowType::kBartlett)
                    .value("DE_WINDOW_TYPE_BLACKMAN", WindowType::kBlackman)
                    .value("DE_WINDOW_TYPE_HAMMING", WindowType::kHamming)
                    .value("DE_WINDOW_TYPE_HANN", WindowType::kHann)
                    .value("DE_WINDOW_TYPE_KAISER", WindowType::kKaiser)
                    .export_values();
                }));

PYBIND_REGISTER(GainType, 0, ([](const py::module *m) {
                  (void)py::enum_<GainType>(*m, "GainType", py::arithmetic())
                    .value("DE_GAIN_TYPE_AMPLITUDE", GainType::kAmplitude)
                    .value("DE_GAIN_TYPE_POWER", GainType::kPower)
                    .value("DE_GAIN_TYPE_DB", GainType::kDb)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
