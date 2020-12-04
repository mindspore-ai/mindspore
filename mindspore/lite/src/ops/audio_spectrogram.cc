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

#include "src/ops/audio_spectrogram.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int AudioSpectrogram::GetWindowSize() const { return this->primitive_->value.AsAudioSpectrogram()->windowSize; }
int AudioSpectrogram::GetStride() const { return this->primitive_->value.AsAudioSpectrogram()->stride; }
bool AudioSpectrogram::GetMagSquare() const { return this->primitive_->value.AsAudioSpectrogram()->magSquare; }

#else
int AudioSpectrogram::GetWindowSize() const { return this->primitive_->value_as_AudioSpectrogram()->windowSize(); }
int AudioSpectrogram::GetStride() const { return this->primitive_->value_as_AudioSpectrogram()->stride(); }
bool AudioSpectrogram::GetMagSquare() const { return this->primitive_->value_as_AudioSpectrogram()->magSquare(); }
int AudioSpectrogram::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_AudioSpectrogram();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Add return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateAudioSpectrogram(*fbb, attr->windowSize(), attr->stride(), attr->magSquare());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_AudioSpectrogram, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *AudioSpectrogramCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<AudioSpectrogram>(primitive);
}
Registry AudioSpectrogramRegistry(schema::PrimitiveType_AudioSpectrogram, AudioSpectrogramCreator);
#endif
int AudioSpectrogram::Log2Ceil(uint32_t length) {
  if (length == 0) {
    return -1;
  }
  int floor = 0;
  for (int i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    uint32_t tmp = length >> shift;
    if (tmp != 0) {
      length = tmp;
      floor += shift;
    }
  }
  return length == (length & ~(length - 1)) ? floor : floor + 1;
}
uint32_t AudioSpectrogram::GetFftLength(uint32_t length) {
  int shift = Log2Ceil(length);
  return 1 << shift;
}
int AudioSpectrogram::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != 2) {
    MS_LOG(ERROR) << "input shape is error, which need to be 2 dimensions";
    return RET_ERROR;
  }
  if (GetWindowSize() < 2) {
    MS_LOG(ERROR) << "window size is too short, now is " << GetWindowSize();
    return RET_ERROR;
  }
  if (GetStride() < 1) {
    MS_LOG(ERROR) << "stride must be positive, now is " << GetStride();
    return RET_ERROR;
  }
  std::vector<int> output_shape(3);
  output_shape[0] = input_shape[1];
  // output height
  int sample_sub_window = input_shape[0] - GetWindowSize();
  output_shape[1] = sample_sub_window < 0 ? 0 : 1 + sample_sub_window / GetStride();
  // compute fft length
  int fft_length = GetFftLength(GetWindowSize());
  output_shape[2] = fft_length / 2 + 1;
  outputs_.front()->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
