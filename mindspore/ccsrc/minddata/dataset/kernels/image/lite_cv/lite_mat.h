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
  BGR = 0,       /**< Pixel in BGR type. */
  RGB = 1,       /**< Pixel in RGB type. */
  RGBA = 2,      /**< Pixel in RGBA type. */
  RGBA2GRAY = 3, /**< Convert image from RGBA to GRAY. */
  RGBA2BGR = 4,  /**< Convert image from RGBA to BGR. */
  RGBA2RGB = 5,  /**< Convert image from RGBA to RGB. */
  NV212BGR = 6,  /**< Convert image from NV21 to BGR. */
  NV122BGR = 7,  /**< Convert image from NV12 to BGR. */
};

enum WARP_BORDER_MODE { WARP_BORDER_MODE_CONSTANT };

class LDataType {
 public:
  enum Type : uint8_t {
    UNKNOWN = 0, /**< Unknown data type. */
    BOOL,        /**< BOOL data type. */
    INT8,        /**< INT8 data type. */
    UINT8,       /**< UINT8 data type. */
    INT16,       /**< INT16 data type. */
    UINT16,      /**< UINT16 data type. */
    INT32,       /**< INT32 data type. */
    UINT32,      /**< UINT32 data type. */
    INT64,       /**< INT64 data type. */
    UINT64,      /**< UINT64 data type. */
    FLOAT16,     /**< FLOAT16 data type. */
    FLOAT32,     /**< FLOAT32 data type. */
    FLOAT64,     /**< FLOAT64 data type. */
    DOUBLE,      /**< DOUBLE data type. */
    NUM_OF_TYPES /**< number of types. */
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
    0, /**< Unknown size. */
    1, /**< Size of BOOL. */
    1, /**< Size of INT8. */
    1, /**< Size of UINT8. */
    2, /**< Size of INT16. */
    2, /**< Size of UINT16. */
    4, /**< Size of INT32. */
    4, /**< Size of UINT32. */
    8, /**< Size of INT64. */
    8, /**< Size of UINT64. */
    2, /**< Size of FLOAT16. */
    4, /**< Size of FLOAT32. */
    8, /**< Size of FLOAT64. */
    8, /**< Size of DOUBLE. */
  };

  Type type_;
};

class LiteMat {
  // Class that represents a lite Mat of a Image.
 public:
  /// \brief Constructor
  LiteMat();

  /// \brief Function to create an LiteMat object.
  /// \param[in] width The width of the input object.
  /// \param[in] data_type The data type of the input object.
  explicit LiteMat(int width, LDataType data_type = LDataType::UINT8);

  /// \brief Function to create an LiteMat object.
  /// \param[in] width The width of the input object.
  /// \param[in] height The height of the input object.
  /// \param[in] data_type The data type of the input object.
  LiteMat(int width, int height, LDataType data_type = LDataType::UINT8);

  /// \brief Function to create an LiteMat object.
  /// \param[in] width The width of the input object.
  /// \param[in] height The height of the input object.
  /// \param[in] p_data The pointer data of the input object.
  /// \param[in] data_type The data type of the input object.
  LiteMat(int width, int height, void *p_data, LDataType data_type = LDataType::UINT8);

  /// \brief Function to create an LiteMat object.
  /// \param[in] width The width of the input object.
  /// \param[in] height The height of the input object.
  /// \param[in] channel The channel of the input object.
  /// \param[in] data_type The data type of the input object.
  LiteMat(int width, int height, int channel, LDataType data_type = LDataType::UINT8);

  /// \brief Function to create an LiteMat object.
  /// \param[in] width The width of the input object.
  /// \param[in] height The height of the input object.
  /// \param[in] channel The channel of the input object.
  /// \param[in] p_data The pointer data of the input object.
  /// \param[in] data_type The data type of the input object.
  LiteMat(int width, int height, int channel, void *p_data, LDataType data_type = LDataType::UINT8);

  /// \brief Destructor of LiteMat.
  ~LiteMat();

  LiteMat(const LiteMat &m);

  /// \brief Perform Init operation on given LiteMat
  /// \param[in] width Set width for given LiteMat.
  /// \param[in] data_type Set data type for given LiteMat.
  void Init(int width, LDataType data_type = LDataType::UINT8);

  /// \brief Perform Init operation on given LiteMat
  /// \param[in] width Set width for given LiteMat.
  /// \param[in] height Set height for given LiteMat.
  /// \param[in] data_type Set data type for given LiteMat.
  void Init(int width, int height, LDataType data_type = LDataType::UINT8);

  /// \brief Perform Init operation on given LiteMat
  /// \param[in] width Set width for given LiteMat.
  /// \param[in] height Set height for given LiteMat.
  /// \param[in] p_data Set pointer data for given LiteMat.
  /// \param[in] data_type Set data type for given LiteMat.
  void Init(int width, int height, void *p_data, LDataType data_type = LDataType::UINT8);

  /// \brief Perform Init operation on given LiteMat
  /// \param[in] width Set width for given LiteMat.
  /// \param[in] height Set height for given LiteMat.
  /// \param[in] channel Set channel for given LiteMat.
  /// \param[in] data_type Set data type for given LiteMat.
  /// \param[in] align_memory Whether malloc align memory or not, default is true,
  ///     which is better for doing acceleration.
  void Init(int width, int height, int channel, const LDataType &data_type = LDataType::UINT8,
            bool align_memory = true);

  /// \brief Perform Init operation on given LiteMat
  /// \param[in] width Set width for given LiteMat.
  /// \param[in] height Set height for given LiteMat.
  /// \param[in] channel Set channel for given LiteMat.
  /// \param[in] p_data Set pointer data for given LiteMat.
  /// \param[in] data_type Set data type for given LiteMat.
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
    if (w >= height_) {
      return nullptr;
    }
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
