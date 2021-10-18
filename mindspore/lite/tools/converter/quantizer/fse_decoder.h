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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H

#include <cstdint>
#include "tools/converter/quantizer/fse_bit_stream.h"
#include "src/tensor.h"

namespace mindspore::lite::quant {
class FSEDecoder {
 public:
  FSEDecoder() = default;
  ~FSEDecoder() = default;

  static int DeCompress(const schema::Tensor &src_tensor, Tensor *dst_tensor);

 private:
  static int FSEDecode(BitStream *bs, float *buff, int buff_count, uint32_t *frequency, int frequency_count,
                       const float *centroids, int table_log);

  static int FSECreateStatesForDecoding(const uint32_t *symbol_frequency, int symbol_frequency_count, int table_log,
                                        uint16_t *new_state, uint8_t *bit_count, uint16_t *symbol_table);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_FSE_DECODER_H
