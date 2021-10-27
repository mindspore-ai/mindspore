/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/audio_spectrogram.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
int64_t Log2Ceil(int64_t length) {
  if (length == 0) {
    return -1;
  }
  int64_t floor = 0;
  for (int64_t i = 4; i >= 0; --i) {
    const int64_t shift = static_cast<int64_t>(1UL << static_cast<unsigned>(i));
    int64_t tmp = SizeToLong(static_cast<uint64_t>(length) >> static_cast<uint64_t>(shift));
    if (tmp != 0) {
      length = tmp;
      floor += shift;
    }
  }
  auto unsigned_length = LongToUlong(length);
  return unsigned_length == (unsigned_length & ~(unsigned_length - 1)) ? floor : floor + 1;
}

int64_t GetFftLength(int64_t length) {
  int64_t shift = Log2Ceil(length);
  return SizeToLong(1UL << LongToSize(shift));
}

abstract::ShapePtr AudioSpectrogramInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (input_shape.size() != 2) {
    MS_LOG(ERROR) << "input shape is error, which need to be 2 dimensions";
  }
  auto window_size = GetValue<int64_t>(primitive->GetAttr(kWindowSize));
  if (window_size < 2) {
    MS_LOG(ERROR) << "window size is too short, now is " << window_size;
  }
  auto stride_size = GetValue<int64_t>(primitive->GetAttr(kStride));
  if (stride_size < 1) {
    MS_LOG(ERROR) << "stride must be positive, now is " << stride_size;
  }
  std::vector<int64_t> infer_shape;
  infer_shape.push_back(input_shape[1]);
  int64_t sample_sub_window = input_shape[0] - window_size;
  infer_shape.push_back(sample_sub_window < 0 ? 0 : 1 + sample_sub_window / stride_size);
  int64_t fft_length = GetFftLength(window_size);
  infer_shape.push_back(fft_length / 2 + 1);
  MS_LOG(ERROR) << infer_shape;
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr AudioSpectrogramInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t x_index = 0;
  return CheckAndConvertUtils::GetInputTensorType(input_args, x_index, prim->name());
}
}  // namespace

void AudioSpectrogram::set_window_size(const int64_t window_size) {
  (void)this->AddAttr(kWindowSize, MakeValue(window_size));
}
int64_t AudioSpectrogram::get_window_size() const {
  auto value_ptr = GetAttr(kWindowSize);
  return GetValue<int64_t>(value_ptr);
}

void AudioSpectrogram::set_stride(const int64_t stride) { (void)this->AddAttr(kStride, MakeValue(stride)); }
int64_t AudioSpectrogram::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<int64_t>(value_ptr);
}

void AudioSpectrogram::set_mag_square(const bool mag_square) { (void)this->AddAttr(kMagSquare, MakeValue(mag_square)); }
bool AudioSpectrogram::get_mag_square() const {
  auto value_ptr = GetAttr(kMagSquare);
  return GetValue<bool>(value_ptr);
}
void AudioSpectrogram::Init(const int64_t window_size, const int64_t stride, const bool mag_square) {
  this->set_window_size(window_size);
  this->set_stride(stride);
  this->set_mag_square(mag_square);
}

AbstractBasePtr AudioSpectrogramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  return std::make_shared<abstract::AbstractTensor>(AudioSpectrogramInferType(primitive, input_args),
                                                    AudioSpectrogramInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameAudioSpectrogram, AudioSpectrogram);
}  // namespace ops
}  // namespace mindspore
