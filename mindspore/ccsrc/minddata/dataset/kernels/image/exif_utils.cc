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

#include "minddata/dataset/kernels/image/exif_utils.h"

#include <algorithm>
#include <cstdint>

#define UNKNOW_ORIENTATION 0

namespace mindspore {
namespace dataset {

template <typename T>
T parse_bytes(const uint8_t *buf, bool intel_align);

template <>
uint8_t parse_bytes(const uint8_t *buf, bool intel_align) {
  return *buf;
}

template <>
uint16_t parse_bytes(const uint8_t *buf, bool intel_align) {
  if (buf == nullptr) {
    return 0;
  }

  uint16_t res = 0;
  if (intel_align) {
    res = (static_cast<uint16_t>(buf[1]) << 8) | buf[0];
  } else {
    res = (static_cast<uint16_t>(buf[0]) << 8) | buf[1];
  }
  return res;
}

template <>
uint32_t parse_bytes(const uint8_t *buf, bool intel_align) {
  if (buf == nullptr) {
    return 0;
  }

  uint32_t res = 0;
  if (intel_align) {
    res = (static_cast<uint32_t>(buf[3]) << 24) | (static_cast<uint32_t>(buf[2]) << 16) |
          (static_cast<uint32_t>(buf[1]) << 8) | buf[0];
  } else {
    res = (static_cast<uint32_t>(buf[0]) << 24) | (static_cast<uint32_t>(buf[1]) << 16) |
          (static_cast<uint32_t>(buf[2]) << 8) | buf[3];
  }
  return res;
}

int parseExif(const uint8_t *buf, uint32_t len) {
  bool intel_align = true;
  uint32_t offset = 0;
  if (!buf || len < 6) {
    return UNKNOW_ORIENTATION;
  }

  if (!std::equal(buf, buf + 6, "Exif\0\0")) {
    return UNKNOW_ORIENTATION;
  }
  offset += 6;

  if (offset + 8 > len) {
    return UNKNOW_ORIENTATION;
  }
  if (buf[offset] == 'I' && buf[offset + 1] == 'I') {
    intel_align = true;
  } else if (buf[offset] == 'M' && buf[offset + 1] == 'M') {
    intel_align = false;
  } else {
    return UNKNOW_ORIENTATION;
  }

  offset += 2;
  if (parse_bytes<uint16_t>(buf + offset, intel_align) != 0x2a) {
    return UNKNOW_ORIENTATION;
  }
  offset += 2;
  uint32_t first_ifd_offset = parse_bytes<uint32_t>(buf + offset, intel_align);
  offset += first_ifd_offset - 4;
  if (offset >= len || offset + 2 > len) {
    return UNKNOW_ORIENTATION;
  }

  int num_entries = parse_bytes<uint16_t>(buf + offset, intel_align);
  if (offset + 6 + 12 * num_entries > len) {
    return UNKNOW_ORIENTATION;
  }
  offset += 2;
  while (num_entries > 0) {
    uint16_t tag = parse_bytes<uint16_t>(buf + offset, intel_align);
    if (tag == 0x112) {
      uint16_t format = parse_bytes<uint16_t>(buf + offset + 2, intel_align);
      uint32_t length = parse_bytes<uint32_t>(buf + offset + 4, intel_align);
      if (format == 3 && length) {
        uint16_t orient = parse_bytes<uint16_t>(buf + offset + 8, intel_align);
        return static_cast<int>(orient);
      }
    }
    offset += 12;
    num_entries--;
  }
  return UNKNOW_ORIENTATION;
}

int ExifInfo::parseOrientation(const unsigned char *data, unsigned len) {
  constexpr int64_t len_size = 4;
  constexpr int64_t len_min = 2;
  constexpr int64_t offset_factor = 4;
  constexpr int64_t section_length_min = 16;
  if (!data || len < len_size) return UNKNOW_ORIENTATION;

  if (data[0] != 0xFF || data[1] != 0xD8) return UNKNOW_ORIENTATION;

  while (len > len_min) {
    if (data[len - 1] == 0xD9 && data[len - 2] == 0xFF) break;
    len--;
  }
  if (len <= len_min) return UNKNOW_ORIENTATION;

  unsigned int offset = 0;
  for (; offset < len - 1; offset++) {
    if (data[offset] == 0xFF && data[offset + 1] == 0xE1) break;
  }
  if (offset + offset_factor > len) return UNKNOW_ORIENTATION;
  offset += 2;
  uint16_t section_length = parse_bytes<uint16_t>(data + offset, false);
  if (offset + section_length > len || section_length < section_length_min) return UNKNOW_ORIENTATION;
  offset += 2;

  return parseExif(data + offset, len - offset);
}
}  // namespace dataset
}  // namespace mindspore
