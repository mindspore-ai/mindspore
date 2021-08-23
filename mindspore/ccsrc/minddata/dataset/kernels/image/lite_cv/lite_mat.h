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

#ifndef MINI_MAT_H_
#define MINI_MAT_H_

#include <string>
#include <memory>

namespace mindspore {
namespace dataset {

#define ALIGN 16
#define MAX_DIMS 3

template <typename T>
struct Chn1 {
  Chn1(T c1) : c1(c1) {}
  T c1;
};

template <typename T>
struct Chn2 {
  Chn2(T c1, T c2) : c1(c1), c2(c2) {}
  T c1;
  T c2;
};

template <typename T>
struct Chn3 {
  Chn3(T c1, T c2, T c3) : c1(c1), c2(c2), c3(c3) {}
  T c1;
  T c2;
  T c3;
};

template <typename T>
struct Chn4 {
  Chn4(T c1, T c2, T c3, T c4) : c1(c1), c2(c2), c3(c3), c4(c4) {}
  T c1;
  T c2;
  T c3;
  T c4;
};

struct Point {
  float x;
  float y;
  Point() : x(0), y(0) {}
  Point(float _x, float _y) : x(_x), y(_y) {}
};

typedef struct imageToolsImage {
  int w;
  int h;
  int stride;
  int dataType;
  void *image_buff;
} imageToolsImage_t;

using BOOL_C1 = Chn1<bool>;
using BOOL_C2 = Chn2<bool>;
using BOOL_C3 = Chn3<bool>;
using BOOL_C4 = Chn4<bool>;

using UINT8_C1 = Chn1<uint8_t>;
using UINT8_C2 = Chn2<uint8_t>;
using UINT8_C3 = Chn3<uint8_t>;
using UINT8_C4 = Chn4<uint8_t>;

using INT8_C1 = Chn1<int8_t>;
using INT8_C2 = Chn2<int8_t>;
using INT8_C3 = Chn3<int8_t>;
using INT8_C4 = Chn4<int8_t>;

using UINT16_C1 = Chn1<uint16_t>;
using UINT16_C2 = Chn2<uint16_t>;
using UINT16_C3 = Chn3<uint16_t>;
using UINT16_C4 = Chn4<uint16_t>;

using INT16_C1 = Chn1<int16_t>;
using INT16_C2 = Chn2<int16_t>;
using INT16_C3 = Chn3<int16_t>;
using INT16_C4 = Chn4<int16_t>;

using UINT32_C1 = Chn1<uint32_t>;
using UINT32_C2 = Chn2<uint32_t>;
using UINT32_C3 = Chn3<uint32_t>;
using UINT32_C4 = Chn4<uint32_t>;

using INT32_C1 = Chn1<int32_t>;
using INT32_C2 = Chn2<int32_t>;
using INT32_C3 = Chn3<int32_t>;
using INT32_C4 = Chn4<int32_t>;

using UINT64_C1 = Chn1<uint64_t>;
using UINT64_C2 = Chn2<uint64_t>;
using UINT64_C3 = Chn3<uint64_t>;
using UINT64_C4 = Chn4<uint64_t>;

using INT64_C1 = Chn1<int64_t>;
using INT64_C2 = Chn2<int64_t>;
using INT64_C3 = Chn3<int64_t>;
using INT64_C4 = Chn4<int64_t>;

using FLOAT32_C1 = Chn1<float>;
using FLOAT32_C2 = Chn2<float>;
using FLOAT32_C3 = Chn3<float>;
using FLOAT32_C4 = Chn4<float>;

using FLOAT64_C1 = Chn1<double>;
using FLOAT64_C2 = Chn2<double>;
using FLOAT64_C3 = Chn3<double>;
using FLOAT64_C4 = Chn4<double>;

enum LPixelType {
  BGR = 0,
  RGB = 1,
  RGBA = 2,
  RGBA2GRAY = 3,
  RGBA2BGR = 4,
  RGBA2RGB = 5,
  NV212BGR = 6,
  NV122BGR = 7,
};

enum WARP_BORDER_MODE { WARP_BORDER_MODE_CONSTANT };

class LDataType {
 public:
  enum Type : uint8_t {
    UNKNOWN = 0,
    BOOL,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    DOUBLE,
    NUM_OF_TYPES
  };

  LDataType() : type_(UNKNOWN) {}

  LDataType(Type d) : type_(d) {}

  ~LDataType() = default;

  inline Type Value() const { return type_; }
  inline bool operator==(const LDataType &ps) const { return this->type_ == ps.type_; }

  inline bool operator!=(const LDataType &ps) const { return this->type_ != ps.type_; }

  uint8_t SizeInBytes() const {
    if (type_ < LDataType::NUM_OF_TYPES)
      return SIZE_IN_BYTES[type_];
    else
      return 0;
  }

 public:
  static inline const uint8_t SIZE_IN_BYTES[] = {
    0,  // UNKNOWN
    1,  // BOOL
    1,  // INT8
    1,  // UINT8
    2,  // INT16
    2,  // UINT16
    4,  // INT32
    4,  // UINT32
    8,  // INT64
    8,  // UINT64
    2,  // FLOAT16
    4,  // FLOAT32
    8,  // FLOAT64
    8,  // DOUBLE
  };

  Type type_;
};

class LiteMat {
  // Class that represents a lite Mat of a Image.
 public:
  LiteMat();

  explicit LiteMat(int width, LDataType data_type = LDataType::UINT8);

  LiteMat(int width, int height, LDataType data_type = LDataType::UINT8);

  LiteMat(int width, int height, void *p_data, LDataType data_type = LDataType::UINT8);

  LiteMat(int width, int height, int channel, LDataType data_type = LDataType::UINT8);

  LiteMat(int width, int height, int channel, void *p_data, LDataType data_type = LDataType::UINT8);

  ~LiteMat();

  LiteMat(const LiteMat &m);

  void Init(int width, LDataType data_type = LDataType::UINT8);

  void Init(int width, int height, LDataType data_type = LDataType::UINT8);

  void Init(int width, int height, void *p_data, LDataType data_type = LDataType::UINT8);

  // Perform Init operation on given LiteMat, will malloc memory for it.
  // @param width set width for given LiteMat.
  // @param height set height for given LiteMat.
  // @param channel set channel for given LiteMat.
  // @param data_type set data type for given LiteMat.
  // @param align_memory whether malloc align memory or not, default is true, which is better for doing acceleration.
  void Init(int width, int height, int channel, LDataType data_type = LDataType::UINT8, bool align_memory = true);

  void Init(int width, int height, int channel, void *p_data, LDataType data_type = LDataType::UINT8);

  bool GetROI(int x, int y, int w, int h, LiteMat &dst);  // NOLINT

  bool IsEmpty() const;

  void Release();

  LiteMat &operator=(const LiteMat &m);

  template <typename T>
  operator T *() {
    return reinterpret_cast<T *>(data_ptr_);
  }

  template <typename T>
  operator const T *() const {
    return reinterpret_cast<const T *>(data_ptr_);
  }

  template <typename T>
  inline T *ptr(int w) const {
    if (IsEmpty()) {
      return nullptr;
    }
    return reinterpret_cast<T *>(reinterpret_cast<unsigned char *>(data_ptr_) + steps_[0] * w);
  }

 private:
  /// \brief Apply for memory alignment
  /// \param[in] size The size of the requested memory alignment.
  void *AlignMalloc(unsigned int size);

  /// \brief Free memory
  /// \param[in] ptr Pointer to free memory.
  void AlignFree(void *ptr);

  /// \brief Initialize the element size of different types of data.
  /// \param[in] data_type Type of data.
  void InitElemSize(LDataType data_type);

  /// \brief Add value of reference count.
  /// \param[in] p The point of references count.
  /// \param[in] value The value of new added references.
  /// \return return reference count.
  int addRef(int *p, int value);

  /// \brief Set the step size of the pixels in the Litemat array.
  /// \param[in] c0 The number used to set teh value of step[0].
  /// \param[in] c1 The number used to set teh value of step[1].
  /// \param[in] c2 The number used to set teh value of step[2].
  void setSteps(int c0, int c1, int c2);

  bool CheckLiteMat();

 public:
  void *data_ptr_ = nullptr;
  int elem_size_;
  int width_;
  int height_;
  int channel_;
  int c_step_;
  int dims_;
  size_t size_;
  LDataType data_type_;
  int *ref_count_;
  size_t steps_[MAX_DIMS];
  bool release_flag;
};

/// \brief Calculates the difference between the two images for each element
bool Subtract(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst);

/// \brief Calculates the division between the two images for each element
bool Divide(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst);

/// \brief Calculates the multiply between the two images for each element
bool Multiply(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst);

#define RETURN_FALSE_IF_LITEMAT_EMPTY(_m) \
  do {                                    \
    if ((_m).IsEmpty()) {                 \
      return false;                       \
    }                                     \
  } while (false)

#define RETURN_IF_LITEMAT_EMPTY(_m) \
  do {                              \
    if ((_m).IsEmpty()) {           \
      return;                       \
    }                               \
  } while (false)

}  // namespace dataset
}  // namespace mindspore
#endif  // MINI_MAT_H_
