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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MATH_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MATH_UTILS_H_

#include <memory>
#include <random>
#include <vector>
#include "minddata/dataset/util/status.h"

#define CV_PI 3.1415926535897932384626433832795

namespace mindspore {
namespace dataset {

/// \brief Returns lower and upper pth percentiles of the input histogram.
/// \param[in] hist: Input histogram (mutates the histogram for computation purposes)
/// \param[in] hi_p: Right side percentile
/// \param[in] low_p: Left side percentile
/// \param[out] hi: Value at high end percentile
/// \param[out] lo: Value at low end percentile
Status ComputeUpperAndLowerPercentiles(std::vector<int32_t> *hist, int32_t hi_p, int32_t low_p, int32_t *hi,
                                       int32_t *lo);

/// \brief Converts degrees input to radians.
/// \param[in] degrees: Input degrees
/// \param[out] radians_target: Radians output
Status DegreesToRadians(float_t degrees, float_t *radians_target);

/// \brief Generates a random real number in [a,b).
/// \param[in] a: Start of range
/// \param[in] b: End of range
/// \param[in] rnd: Random device
/// \param[out] result: Random number in range [a,b)
Status GenerateRealNumber(float_t a, float_t b, std::mt19937 *rnd, float_t *result);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MATH_UTILS_H_
