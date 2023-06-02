/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
 * limitations under the License
 */

#ifndef MINDSPORE_CORE_OPS_FFT_WITH_SIZE_H_
#define MINDSPORE_CORE_OPS_FFT_WITH_SIZE_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFFTWithSize = "FFTWithSize";
/// \brief Fast Fourier Transform
/// Refer to Python API @ref mindspore.ops.FFTWithSize for more details.
class MIND_API FFTWithSize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FFTWithSize);
  /// \brief Constructor.
  FFTWithSize() : BaseOperator(kNameFFTWithSize) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FFTWithSize for the inputs.
  void Init(const int64_t signal_ndim = 1, const bool inverse = false, const bool real = false,
            const std::string &norm = "backward", const bool onesided = false,
            const std::vector<int64_t> &signal_sizes = {});
  /// \brief Set signal_ndim.
  void set_signal_ndim(const int64_t signal_ndim);
  /// \brief Set inverse.
  void set_inverse(const bool inverse);
  /// \brief Set real.
  void set_real(const bool real);
  /// \brief Set norm.
  void set_norm(const std::string &norm);
  /// \brief Set onesided.
  void set_onesided(const bool onesided);
  /// \brief Set signal_sizes.
  void set_signal_sizes(const std::vector<int64_t> &signal_sizes);

  /// \brief Get signal_ndim.
  int64_t get_signal_ndim() const;
  /// \brief Get inverse.
  bool get_inverse() const;
  /// \brief Get real.
  bool get_real() const;
  /// \brief Get norm.
  std::string get_norm() const;
  /// \brief Get onesided.
  bool get_onesided() const;
  /// \brief Get signal_sizes.
  std::vector<int64_t> get_signal_sizes() const;
};
MIND_API abstract::AbstractBasePtr FFTWithSizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimFFTWithSizePtr = std::shared_ptr<FFTWithSize>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FFT_WITH_SIZE_H_
