/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef LITE_MINDSPORE_LITE_C_OPS_AUDIO_SPECTROGRAM_H_
#define LITE_MINDSPORE_LITE_C_OPS_AUDIO_SPECTROGRAM_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class AudioSpectrogram : public PrimitiveC {
 public:
  AudioSpectrogram() = default;
  ~AudioSpectrogram() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(AudioSpectrogram, PrimitiveC);
  explicit AudioSpectrogram(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetWindowSize(int window_size) { this->primitive_->value.AsAudioSpectrogram()->windowSize = window_size; }
  void SetStride(int stride) { this->primitive_->value.AsAudioSpectrogram()->stride = stride; }
  void SetMagSquare(bool mag_square) { this->primitive_->value.AsAudioSpectrogram()->magSquare = mag_square; }
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int GetWindowSize() const;
  int GetStride() const;
  bool GetMagSquare() const;
  int Log2Ceil(uint32_t length);
  uint32_t GetFftLength(uint32_t length);
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_AUDIO_SPECTROGRAM_H_
