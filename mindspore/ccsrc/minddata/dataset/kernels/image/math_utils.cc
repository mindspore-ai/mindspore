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

#include "minddata/dataset/kernels/image/math_utils.h"

#include <algorithm>
#include <string>

namespace mindspore {
namespace dataset {
Status ComputeUpperAndLowerPercentiles(std::vector<int32_t> *hist, int32_t hi_p, int32_t low_p, int32_t *hi,
                                       int32_t *lo) {
  CHECK_FAIL_RETURN_UNEXPECTED(hist != nullptr, "hist is nullptr");
  CHECK_FAIL_RETURN_UNEXPECTED(hi != nullptr, "hi is nullptr");
  CHECK_FAIL_RETURN_UNEXPECTED(lo != nullptr, "lo is nullptr");
  try {
    int32_t n = std::accumulate(hist->begin(), hist->end(), 0);
    constexpr float kMaxPerc = 100.0;
    int32_t cut = static_cast<int32_t>((low_p / kMaxPerc) * n);
    for (int32_t lb = 0; lb < hist->size() && cut > 0; lb++) {
      if (cut > (*hist)[lb]) {
        cut -= (*hist)[lb];
        (*hist)[lb] = 0;
      } else {
        (*hist)[lb] -= cut;
        cut = 0;
      }
    }
    cut = static_cast<int32_t>((hi_p / kMaxPerc) * n);
    for (auto ub_iter = hist->end() - 1; ub_iter >= hist->begin() && cut > 0; ub_iter--) {
      if (cut > *ub_iter) {
        cut -= *ub_iter;
        *ub_iter = 0;
      } else {
        *ub_iter -= cut;
        cut = 0;
      }
    }
    *lo = 0;
    *hi = hist->size() - 1;
    for (; (*lo) < (*hi) && !(*hist)[*lo]; (*lo)++) {
    }
    for (; (*hi) >= 0 && !(*hist)[*hi]; (*hi)--) {
    }
  } catch (const std::exception &e) {
    std::string err_message = "AutoContrast: ComputeUpperAndLowerPercentiles failed: ";
    err_message += e.what();
    RETURN_STATUS_UNEXPECTED(err_message);
  }
  return Status::OK();
}

Status DegreesToRadians(float_t degrees, float_t *radians_target) {
  CHECK_FAIL_RETURN_UNEXPECTED(radians_target != nullptr, "radians_target is nullptr");
  *radians_target = CV_PI * degrees / 180.0;
  return Status::OK();
}

Status GenerateRealNumber(float_t a, float_t b, std::mt19937 *rnd, float_t *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(rnd != nullptr, "rnd is nullptr");
  CHECK_FAIL_RETURN_UNEXPECTED(result != nullptr, "result is nullptr");
  try {
    std::uniform_real_distribution<float_t> distribution{a, b};
    *result = distribution(*rnd);
  } catch (const std::exception &e) {
    std::string err_message = "RandomAffine: GenerateRealNumber failed: ";
    err_message += e.what();
    RETURN_STATUS_UNEXPECTED(err_message);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
