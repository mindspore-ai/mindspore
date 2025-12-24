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
 * limitations under the License.
 */

#ifndef MINDSPORE_CORE_OPS_STFT_H_
#define MINDSPORE_CORE_OPS_STFT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSTFT = "STFT";
constexpr int64_t kSTFT2DInputDims = 2;
constexpr int64_t kSTFT1DWindowDims = 1;
constexpr int64_t kSTFT1DSignalInput = 1;
constexpr int64_t kSTFT2DSignalInput = 2;

/// \brief 3D Average pooling operation. Refer to Python API @ref mindspore.ops.STFT for more details.
class OPS_API STFT : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(STFT);
  /// \brief Constructor.
  STFT() : BaseOperator(kNameSTFT) { InitIOName({"input"}, {"output"}); }
  void Init(int64_t n_fft, int64_t hop_length, int64_t win_length, bool normalized, bool onesided, bool return_complex);
  void set_n_fft(int64_t n_fft);
  void set_hop_length(int64_t hop_length);
  void set_win_length(int64_t win_length);
  void set_normalized(bool normalized);
  void set_onesided(bool onesided);
  void set_return_complex(bool return_complex);
  int64_t get_n_fft() const;
  int64_t get_hop_length() const;
  int64_t get_win_length() const;
  bool get_normalized() const;
  bool get_onesided() const;
  bool get_return_complex() const;
};

OPS_API abstract::AbstractBasePtr STFTInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_STFT_H_
