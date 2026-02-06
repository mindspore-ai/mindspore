// Additional modificated from
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/GridSamplerKernel.cpp by Huawei Technologies
// Co., Ltd in 2021-2022.
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_2D_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_2D_GRAD_CPU_KERNEL_H_
#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <array>
#include <functional>
#include <utility>
#include <tuple>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "mindspore/ops/infer/ops_func_impl/grid_sampler_2d_grad.h"

namespace mindspore {
const int64_t hZero = 0;
const int64_t hOne = 1;
const int64_t hTwo = 2;
const int64_t hThree = 3;
const int64_t hFour = 4;
const int64_t hFive = 5;
const int64_t hSix = 6;
const int64_t hSeven = 7;
const int64_t hEight = 8;
namespace kernel {
namespace grid_sampler_2d_grad_cpu {
enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

class GridSampler2DGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  GridSampler2DGradCpuKernelMod() = default;
  ~GridSampler2DGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
                                                     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeFloat64)};
    return support_list;
  }

 private:
  ShapeVector grad_shape_;
  ShapeVector x_shape_;
  ShapeVector grid_shape_;
  ShapeVector dx_shape_;
  ShapeVector dgrid_shape_;
  int64_t interpolation_mode_;
  int64_t padding_mode_;
  bool align_corners_;
  size_t dx_size_;
  size_t grid_size_;
  TypeId dtype_{kTypeUnknown};

  template <typename T>
  void ComputeTask(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
};

// *******************VEC256***********************

namespace vec256 {
template <size_t n>
struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t)     \
  template <>                         \
  struct int_of_size<sizeof(int_t)> { \
    using type = int_t;               \
  }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;
template <class T>
struct Vec256 {
 private:
  T values[32 / sizeof(T)];  // 32

 public:
  using value_type = T;
  static constexpr int size() { return 32 / sizeof(T); }
  Vec256() : values{0} {}
  explicit Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }

  template <typename... Args, typename = typename std::enable_if<(sizeof...(Args) == size()), void>::type>
  explicit Vec256(Args... vals) {
    values = {vals...};
  }

  Vec256<T> trunc() const { return map(std::trunc); }
  static Vec256<T> LoadU(const void *ptr) {
    Vec256 vec;
    size_t count = 32;
    auto cp_ret = memcpy_s(static_cast<void *>(vec.values), count, ptr, count);
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. errorno is: " << cp_ret;
    }
    return vec;
  }
  static Vec256<T> LoadU(const void *ptr, int64_t count) {
    Vec256 vec;
    auto cp_ret = memcpy_s(static_cast<void *>(vec.values), sizeof(vec.values), ptr, count * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. errorno is: " << cp_ret;
    }
    return vec;
  }
  void store(void *ptr, int count = size()) const {
    auto cp_ret = memcpy_s(ptr, count * sizeof(T), values, count * sizeof(T));
    if (cp_ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed. errorno is: " << cp_ret;
    }
  }
  const T &operator[](int idx) const { return values[idx]; }
  T &operator[](int idx) { return values[idx]; }
  int zero_mask() const {
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vec256<T> map(T (*f)(T)) const {
    Vec256<T> ret;
    for (size_t i = 0; i != IntToSize(size()); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  template <typename other_t_abs = T,
            typename std::enable_if<!std::is_floating_point<other_t_abs>::value, int>::type = 0>
  Vec256<T> abs() const {
    static_assert(std::is_same<other_t_abs, T>::value, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  template <typename float_t_abs = T,
            typename std::enable_if<std::is_floating_point<float_t_abs>::value, int>::type = 0>
  Vec256<T> abs() const {
    static_assert(std::is_same<float_t_abs, T>::value, "float_t_abs must be T");
    return map(std::abs);
  }
  static Vec256<T> blendv(const Vec256<T> &a, const Vec256<T> &b, const Vec256<T> &mask) {
    Vec256 vec;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
    for (size_t i = 0; i < IntToSize(size()); i++) {
      if (buffer[i] & 0x01) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<T> arange(T base = static_cast<T>(0), T step = static_cast<T>(1)) {
    Vec256 vec;
    for (size_t i = 0; i < IntToSize(size()); i++) {
      vec.values[i] = base + i * step;
    }
    return vec;
  }
  static Vec256<T> set(const Vec256<T> &a, const Vec256<T> &b, int64_t count = size()) {
    Vec256 vec;
    for (size_t i = 0; i < IntToSize(size()); i++) {
      if (i < LongToSize(count)) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  Vec256<T> floor() const {
    Vec256<T> ret;
    for (size_t i = 0; i != IntToSize(size()); i++) {
      ret[i] = std::floor(values[i]);
    }
    return ret;
  }

  inline T round_impl(const T z) { return std::nearbyint(z); }
  Vec256<T> round() const {
    Vec256<T> ret;
    for (size_t i = 0; i != IntToSize(size()); i++) {
      ret[i] = std::nearbyint(values[i]);
    }
    return ret;
  }
  Vec256<T> operator==(const Vec256<T> &other) const { return binary_pred(other, std::equal_to<T>()); }
  Vec256<T> operator!=(const Vec256<T> &other) const { return binary_pred(other, std::not_equal_to<T>()); }
  Vec256<T> operator>=(const Vec256<T> &other) const { return binary_pred(other, std::greater_equal<T>()); }
  Vec256<T> operator<=(const Vec256<T> &other) const { return binary_pred(other, std::less_equal<T>()); }
  Vec256<T> operator>(const Vec256<T> &other) const { return binary_pred(other, std::greater<T>()); }
  Vec256<T> operator<(const Vec256<T> &other) const { return binary_pred(other, std::less<T>()); }

 private:
  template <typename Op>
  inline Vec256<T> binary_pred(const Vec256<T> &other, Op op) const {
    Vec256<T> vec;
    for (int64_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        auto ret = memset_s(static_cast<void *>(vec.values + i), sizeof(T), 0xFF, sizeof(T));
        if (ret != 0) {
          MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
        }
      } else {
        auto ret = memset_s(static_cast<void *>(vec.values + i), sizeof(T), 0, sizeof(T));
        if (ret != 0) {
          MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
        }
      }
    }
    return vec;
  }
};

template <class T>
Vec256<T> inline operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}
template <class T>
Vec256<T> operator-(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}
template <class T>
Vec256<T> inline operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}
template <class T>
Vec256<T> inline operator/(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}
template <class T>
inline Vec256<T> operator&(const Vec256<T> &a, const Vec256<T> &b) {
  return bitwise_binary_op(a, b, std::bit_and<int_same_size_t<T>>());
}
template <class T>
inline Vec256<T> operator^(const Vec256<T> &a, const Vec256<T> &b) {
  return bitwise_binary_op(a, b, std::bit_xor<int_same_size_t<T>>());
}
#define RETURN_TYPE std::enable_if<scale == hOne || scale == hTwo || scale == hFour || scale == hEight, Vec256<T>>::type
template <int64_t scale = hOne, typename T = void>
typename RETURN_TYPE inline GatherVec(T const *base_addr, const Vec256<int_same_size_t<T>> &vindex) {
  static constexpr int kSize = Vec256<T>::size();
  int_same_size_t<T> index_arr[kSize];
  vindex.store(static_cast<void *>(index_arr));
  T buffer[kSize];
  for (int64_t i = 0; i < kSize; i++) {
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
  }
  return Vec256<T>::LoadU(static_cast<void *>(buffer));
}
template <int64_t scale = hOne, typename T = void>
typename RETURN_TYPE inline MaskGather(const Vec256<T> &src, T const *base_addr,
                                       const Vec256<int_same_size_t<T>> &vindex, Vec256<T> *mask) {
  static constexpr int kSize = Vec256<T>::size();
  T src_arr[kSize];
  int_same_size_t<T> mask_arr[kSize];  // use int type so we can logical and
  int_same_size_t<T> index_arr[kSize];
  src.store(static_cast<void *>(src_arr));
  mask->store(static_cast<void *>(mask_arr));
  vindex.store(static_cast<void *>(index_arr));
  T buffer[kSize];
  for (int64_t i = 0; i < kSize; i++) {
    if (mask_arr[i] & 0x01) {
      buffer[i] = base_addr[static_cast<size_t>(index_arr[i] * static_cast<size_t>(scale) / sizeof(T))];
    } else {
      buffer[i] = src_arr[i];
    }
  }
  *mask = Vec256<T>();
  return Vec256<T>::LoadU(static_cast<void *>(buffer));
}
template <typename T>
inline Vec256<int_same_size_t<T>> ConvertToIntOfSameSize(const Vec256<T> &src) {
  static constexpr int kSize = Vec256<T>::size();
  T src_arr[kSize];
  src.store(static_cast<void *>(src_arr));
  int_same_size_t<T> buffer[kSize];
  for (int64_t i = 0; i < kSize; i++) {
    buffer[i] = static_cast<int_same_size_t<T>>(src_arr[i]);
  }
  return Vec256<int_same_size_t<T>>::LoadU(static_cast<void *>(buffer));
}
template <typename dst_t, typename src_t>
struct CastImpl {
  static inline Vec256<dst_t> apply(const Vec256<src_t> &src) {
    src_t src_arr[Vec256<src_t>::size()];
    src.store(static_cast<void *>(src_arr));
    return Vec256<dst_t>::LoadU(static_cast<const void *>(src_arr));
  }
};
template <typename T>
struct CastImpl<T, T> {
  static inline Vec256<T> apply(const Vec256<T> &src) { return src; }
};
template <typename T>
inline typename std::enable_if<Vec256<T>::size() % hTwo == hZero, std::pair<Vec256<T>, Vec256<T>>>::type deinterleave2(
  const Vec256<T> &a, const Vec256<T> &b) {
  static constexpr int kSize = Vec256<T>::size();
  static constexpr int half_size = kSize / 2;
  T a_arr[kSize];
  T b_arr[kSize];
  T buffer1[kSize];
  T buffer2[kSize];
  a.store(static_cast<void *>(a_arr));
  b.store(static_cast<void *>(b_arr));
  for (int64_t i = 0; i < half_size; i++) {
    buffer1[i] = a_arr[i * hTwo];
    buffer1[half_size + i] = b_arr[i * hTwo];
    buffer2[i] = a_arr[i * hTwo + hOne];
    buffer2[half_size + i] = b_arr[i * hTwo + hOne];
  }
  return std::make_pair(Vec256<T>::LoadU(static_cast<void *>(buffer1)), Vec256<T>::LoadU(static_cast<void *>(buffer2)));
}
template <typename T>
inline typename std::enable_if<Vec256<T>::size() % hTwo == hZero, std::pair<Vec256<T>, Vec256<T>>>::type interleave2(
  const Vec256<T> &a, const Vec256<T> &b) {
  static constexpr int kSize = Vec256<T>::size();
  static constexpr int half_size = kSize / 2;
  T a_arr[kSize];
  T b_arr[kSize];
  T buffer1[kSize];
  T buffer2[kSize];
  a.store(static_cast<void *>(a_arr));
  b.store(static_cast<void *>(b_arr));
  for (int64_t i = 0; i < half_size; i++) {
    buffer1[i * hTwo] = a_arr[i];
    buffer1[i * hTwo + hOne] = b_arr[i];
    buffer2[i * hTwo] = a_arr[half_size + i];
    buffer2[i * hTwo + hOne] = b_arr[half_size + i];
  }
  return std::make_pair(Vec256<T>::LoadU(static_cast<void *>(buffer1)), Vec256<T>::LoadU(static_cast<void *>(buffer2)));
}

template <typename dst_t, typename src_t>
inline Vec256<dst_t> cast(const Vec256<src_t> &src) {
  return CastImpl<dst_t, src_t>::apply(src);
}

template <typename T>
inline bool _isnan(T val) {
  return std::isnan(T(val));
}
template <class T>
Vec256<T> inline maximum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      c[i] = a[i];
    }
  }
  return c;
}
template <typename T>
Vec256<T> inline minimum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      c[i] = a[i];
    }
  }
  return c;
}
template <class T, typename Op>
static inline Vec256<T> bitwise_binary_op(const Vec256<T> &a, const Vec256<T> &b, Op op) {
  using iT = int_same_size_t<T>;
  iT buffer[Vec256<T>::size()];
  for (int i = 0; i != Vec256<T>::size(); i++) {
    auto a_val = a[i];
    auto b_val = b[i];
    iT *i_a_ptr = reinterpret_cast<iT *>(&a_val);
    iT *i_b_ptr = reinterpret_cast<iT *>(&b_val);
    buffer[i] = op(*i_a_ptr, *i_b_ptr);
  }
  return Vec256<T>::LoadU(buffer);
}
}  // namespace vec256

template <typename T, size_t N>
class TensorAcc {
 public:
  TensorAcc(T *data_, int64_t *sizes_, int64_t *strides_) : dataptr(data_), sizes(sizes_), strides(strides_) {}
  TensorAcc(const TensorAcc<T, 4> &tacc) { TensorAcc(tacc.dataptr, tacc.sizes, tacc.strides); }
  int64_t stride(int64_t i) const { return strides[i]; }
  int64_t size(int64_t i) const { return sizes[i]; }
  T *data() { return dataptr; }
  const T *data() const { return dataptr; }
  TensorAcc<T, N - 1> operator[](const int64_t i) {
    return TensorAcc<T, N - 1>(this->dataptr + this->strides[0] * i, this->sizes + 1, this->strides + 1);
  }

  const TensorAcc<T, N - 1> operator[](int64_t i) const {
    return TensorAcc<T, N - 1>(this->dataptr + this->strides[0] * i, this->sizes + 1, this->strides + 1);
  }
  ~TensorAcc() {}

 private:
  T *dataptr;
  int64_t *sizes;
  int64_t *strides;
};

template <typename T, size_t N>
TensorAcc<T, N> accessor(T *data_ptr, std::vector<int64_t> sizess) {
  static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
  int64_t stride_tmp = 1;
  int64_t *strid = new int64_t[N];
  for (int64_t i = N - 1; i > -1; --i) {
    strid[i] = stride_tmp;
    stride_tmp *= static_cast<int64_t>(sizess[i]);
  }
  int64_t *sizes = new int64_t[N];
  for (size_t k = 0; k < N; ++k) sizes[k] = sizess[k];
  return TensorAcc<T, N>(data_ptr, sizes, strid);
}

bool GeometryIsContiguous(std::array<int64_t, hFour> sizes, std::array<int64_t, hFour> strides) {
  int64_t dim = sizes.size();
  int64_t expected_stride = 1;
  bool contig_if_nonempty = true;
  for (int64_t i = dim - 1; i >= 0; i--) {
    if (sizes[LongToSize(i)] == 0) {
      return true;
    }
    if (contig_if_nonempty) {
      if (sizes[LongToSize(i)] != 1 && strides[LongToSize(i)] != expected_stride) {
        contig_if_nonempty = false;
      }
      expected_stride *= sizes[i];
    }
  }
  return contig_if_nonempty;
}

template <typename T, bool align_corners>
struct ComputeLocationBase;

template <typename T>
struct ComputeLocationBase<T, true> {
  using Vec = vec256::Vec256<T>;
  const T max_val;
  const T scaling_factor;
  const T low;
  const T twice_span;
  const bool empty;

  explicit ComputeLocationBase(int64_t size)
      : max_val(static_cast<T>(size - 1)),
        scaling_factor(static_cast<T>(size - hOne) / hTwo),
        low(static_cast<T>(0)),
        twice_span(static_cast<T>(size - hOne) * hTwo),
        empty(size <= 0) {}

  inline Vec unnormalize(const Vec &in) const { return (in + Vec(1)) * Vec(scaling_factor); }

  inline Vec clip_coordinates(const Vec &in) const { return minimum(Vec(max_val), maximum(in, Vec(0))); }
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = vec256::int_same_size_t<T>;
    auto bounded_lo = maximum(in, Vec(0));
    auto in_bound_lo = vec256::cast<T>(cast<int_t>(bounded_lo) != vec256::cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = vec256::cast<T>(cast<int_t>(res) != vec256::cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  inline vec256::Vec256<T> reflect_coordinates(const vec256::Vec256<T> &in) const {
    if (empty) {
      return Vec(0);
    }
    Vec two_span_vec(twice_span);
    auto abs_in = in.abs();
    auto double_flips = (abs_in / two_span_vec).trunc();
    auto extra = abs_in - double_flips * two_span_vec;
    return minimum(extra, two_span_vec - extra);
  }

  inline std::pair<vec256::Vec256<T>, vec256::Vec256<T>> reflect_coordinates_get_grad(
    const vec256::Vec256<T> &in) const {
    if (empty) {
      return std::make_pair(Vec(0), Vec(0));
    }
    auto neg_in = in < Vec(0);
    auto abs_in = in.abs();
    Vec twice_span_vec(twice_span);
    auto extra = abs_in - (abs_in / twice_span_vec).trunc() * twice_span_vec;
    auto reflected_extra = twice_span_vec - extra;
    auto one_flip = extra > reflected_extra;

    return std::make_pair(Vec::blendv(extra, reflected_extra, one_flip),
                          Vec::blendv(Vec(1), Vec(-1), one_flip ^ neg_in));
  }
};

template <typename T>
struct ComputeLocationBase<T, false> {
  using Vec = vec256::Vec256<T>;
  const T max_val;
  const T scaling_factor;
  const T low;
  const T twice_span;
  const bool empty;  // only used when align_corners=True

  explicit ComputeLocationBase(int64_t size)
      : max_val(static_cast<T>(size - 1)),
        scaling_factor(static_cast<T>(size) / 2),
        low(static_cast<T>(-0.5)),
        twice_span(static_cast<T>(size) * 2),
        empty(size <= 0) {}

  inline Vec unnormalize(const Vec &in) const { return (in + Vec(1)) * Vec(scaling_factor) - Vec(0.5); }

  inline Vec clip_coordinates(const Vec &in) const { return minimum(Vec(max_val), maximum(in, Vec(0))); }
  inline std::pair<Vec, Vec> clip_coordinates_get_grad(const Vec &in) const {
    using int_t = vec256::int_same_size_t<T>;
    auto bounded_lo = maximum(in, Vec(0));
    auto in_bound_lo = vec256::cast<T>(vec256::cast<int_t>(bounded_lo) != vec256::cast<int_t>(Vec(0)));
    auto res = minimum(bounded_lo, Vec(max_val));
    auto in_bound_hi = vec256::cast<T>(vec256::cast<int_t>(res) != vec256::cast<int_t>(Vec(max_val)));
    return std::make_pair(res, in_bound_lo & in_bound_hi);
  }

  inline Vec reflect_coordinates(const Vec &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    auto abs_in = (in - low_vec).abs();
    auto fdouble_flips = abs_in / twice_span_vec;
    auto double_flips = fdouble_flips.trunc();
    auto extra = abs_in - double_flips * twice_span_vec;
    return minimum(extra, twice_span_vec - extra) + low_vec;
  }

  inline std::pair<vec256::Vec256<T>, vec256::Vec256<T>> reflect_coordinates_get_grad(
    const vec256::Vec256<T> &in) const {
    Vec twice_span_vec(twice_span), low_vec(low);
    Vec in_minus_low = in - low_vec;
    auto neg_i = in_minus_low < Vec(0);
    auto abs_i = in_minus_low.abs();
    auto extra = abs_i - (abs_i / twice_span_vec).trunc() * twice_span_vec;
    auto reflected_extra = twice_span_vec - extra;
    auto one_flip = extra > reflected_extra;
    auto boolex = one_flip ^ neg_i;
    return std::make_pair(Vec::blendv(extra, reflected_extra, one_flip) + low_vec,
                          Vec::blendv(Vec(1), Vec(-1), boolex));
  }
};

template <typename T, GridSamplerPadding padding, bool align_corners>
struct ComputeLocation;

template <typename T, bool align_corners>
struct ComputeLocation<T, GridSamplerPadding::Zeros, align_corners> : ComputeLocationBase<T, align_corners> {
  using Vec = vec256::Vec256<T>;
  using ComputeLocationBase<T, align_corners>::unnormalize;
  using ComputeLocationBase<T, align_corners>::scaling_factor;

  using ComputeLocationBase<T, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const { return unnormalize(in); }

  inline std::pair<Vec, Vec> ApplyGetGrad(const Vec &in) const {
    return std::make_pair(unnormalize(in), Vec(scaling_factor));
  }
};

template <typename T, bool align_corners>
struct ComputeLocation<T, GridSamplerPadding::Border, align_corners> : ComputeLocationBase<T, align_corners> {
  using Vec = vec256::Vec256<T>;
  using ComputeLocationBase<T, align_corners>::unnormalize;
  using ComputeLocationBase<T, align_corners>::clip_coordinates;
  using ComputeLocationBase<T, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<T, align_corners>::scaling_factor;

  using ComputeLocationBase<T, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const { return clip_coordinates(unnormalize(in)); }

  inline std::pair<Vec, Vec> ApplyGetGrad(const Vec &in) const {
    Vec res, grad_clip;
    std::tie(res, grad_clip) = clip_coordinates_get_grad(unnormalize(in));
    return std::make_pair(res, grad_clip & Vec(scaling_factor));
  }
};

template <typename T, bool align_corners>
struct ComputeLocation<T, GridSamplerPadding::Reflection, align_corners> : ComputeLocationBase<T, align_corners> {
  using Vec = vec256::Vec256<T>;
  using ComputeLocationBase<T, align_corners>::unnormalize;
  using ComputeLocationBase<T, align_corners>::clip_coordinates;
  using ComputeLocationBase<T, align_corners>::clip_coordinates_get_grad;
  using ComputeLocationBase<T, align_corners>::reflect_coordinates;
  using ComputeLocationBase<T, align_corners>::reflect_coordinates_get_grad;
  using ComputeLocationBase<T, align_corners>::scaling_factor;

  using ComputeLocationBase<T, align_corners>::ComputeLocationBase;

  inline Vec apply(const Vec &in) const {
    auto res = reflect_coordinates(unnormalize(in));
    res = clip_coordinates(res);
    return res;
  }

  inline std::pair<Vec, Vec> ApplyGetGrad(const Vec &in) const {
    Vec result;
    Vec grad_refl;
    Vec grad_clip;
    Vec grad(scaling_factor);
    std::tie(result, grad_refl) = reflect_coordinates_get_grad(unnormalize(in));
    grad = grad_refl * grad;
    std::tie(result, grad_clip) = clip_coordinates_get_grad(result);
    grad = grad_clip & grad;
    return std::make_pair(result, grad);
  }
};

template <typename T>
static inline void MaskScatterAdd(const T *src, T *base_addr, const vec256::int_same_size_t<T> *offsets,
                                  const vec256::int_same_size_t<T> *mask, int64_t len) {
  for (int64_t i = 0; i < len; i++) {
    if (mask[i] & 0x01) {
      base_addr[offsets[i]] += src[i];
    }
  }
}

template <typename T, int spatial_dim, GridSamplerInterpolation interp, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample2D;

template <typename T, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample2D<T, hTwo, GridSamplerInterpolation::Bilinear, padding, align_corners> {
  using Vec = vec256::Vec256<T>;
  using integer_t = vec256::int_same_size_t<T>;
  using iVec = vec256::Vec256<integer_t>;

  const int64_t InpH;
  const int64_t InpW;
  const int64_t InpSH;
  const int64_t InpSW;
  const int64_t C;
  const int64_t InpSC;
  const ComputeLocation<T, padding, align_corners> ComputeH;
  const ComputeLocation<T, padding, align_corners> ComputeW;
  const bool MustInBound = padding != GridSamplerPadding::Zeros;

  explicit ApplyGridSample2D(const TensorAcc<T, 4> &input)
      : InpH(input.size(2)),
        InpW(input.size(3)),
        InpSH(input.stride(2)),
        InpSW(input.stride(3)),
        C(input.size(1)),
        InpSC(input.stride(1)),
        ComputeH(input.size(2)),
        ComputeW(input.size(3)) {}
  inline std::tuple<Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, iVec, iVec> ComputeInterpParams(
    const Vec &X, const Vec &Y) const {
    auto XW = X.floor();
    auto YN = Y.floor();
    auto W = X - XW;
    auto E = Vec(1) - W;
    auto N = Y - YN;
    auto S = Vec(1) - N;
    auto NW = S * E;
    auto NE = S * W;
    auto SW = N * E;
    auto SE = N * W;
    auto IXW = vec256::ConvertToIntOfSameSize(XW);
    auto IYN = vec256::ConvertToIntOfSameSize(YN);
    auto IXE = IXW + iVec(1);
    auto IYS = IYN + iVec(1);
    auto WMask = MustInBound ? iVec(-1) : (IXW > iVec(-1)) & (IXW < iVec(InpW));
    auto NMask = MustInBound ? iVec(-1) : (IYN > iVec(-1)) & (IYN < iVec(InpH));
    auto EMask = MustInBound ? (IXE < iVec(InpW)) : (IXE > iVec(-1)) & (IXE < iVec(InpW));
    auto SMask = MustInBound ? (IYS < iVec(InpH)) : (IYS > iVec(-1)) & (IYS < iVec(InpH));
    auto NWMask = vec256::cast<T>(MustInBound ? iVec(-1) : (WMask & NMask));
    auto NEMask = vec256::cast<T>(EMask & NMask);
    auto SWMask = vec256::cast<T>(WMask & SMask);
    auto SEMask = vec256::cast<T>(EMask & SMask);

    return std::make_tuple(N, S, W, E, NW, NE, SW, SE, NWMask, NEMask, SWMask, SEMask, IYN, IXW);
  }

  inline void Backward(TensorAcc<T, 3> *GInpSlice, TensorAcc<T, 3> *GGridSlice, const TensorAcc<T, 3> &GOutSlice,
                       const TensorAcc<T, 3> &InpSlice, int64_t offset, const Vec &grid_x, const Vec &grid_y,
                       int64_t len) const {
    Vec X, Y, GxMult, GyMult;
    std::tie(X, GxMult) = ComputeW.ApplyGetGrad(grid_x);
    std::tie(Y, GyMult) = ComputeH.ApplyGetGrad(grid_y);

    iVec IYN, IXW;
    Vec N, S, W, E, NW, NE, SW, SE, NWMask, NEMask, SWMask, SEMask;

    std::tie(N, S, W, E, NW, NE, SW, SE, NWMask, NEMask, SWMask, SEMask, IYN, IXW) = ComputeInterpParams(X, Y);

    auto INWOffset = IYN * iVec(InpSH) + IXW * iVec(InpSW);
    auto INEOffset = iVec(InpSW) + INWOffset;
    auto ISWOffset = iVec(InpSH) + INWOffset;
    auto ISEOffset = iVec(InpSW) + ISWOffset;

    auto IGInpNWOffset = IYN * iVec(InpW) + IXW;
    auto IGInpNEOffset = IGInpNWOffset + iVec(1);
    auto IGInpSWOffset = IGInpNWOffset + iVec(InpW);
    auto IGInpSEOffset = IGInpSWOffset + iVec(1);
    static constexpr int kSize = iVec::size();
    integer_t IGInpNWOffsetArr[kSize];
    integer_t IGInpNEOffsetArr[kSize];
    integer_t IGInpSWOffsetArr[kSize];
    integer_t IGInpSEOffsetArr[kSize];
    IGInpNWOffset.store(IGInpNWOffsetArr);
    IGInpNEOffset.store(IGInpNEOffsetArr);
    IGInpSWOffset.store(IGInpSWOffsetArr);
    IGInpSEOffset.store(IGInpSEOffsetArr);

    integer_t INWMaskArr[kSize], INEMaskArr[kSize], ISWMaskArr[kSize], ISEMaskArr[kSize];
    NWMask.store(INWMaskArr);
    NEMask.store(INEMaskArr);
    SWMask.store(ISWMaskArr);
    SEMask.store(ISEMaskArr);

    T GInpCornerArr[Vec::size()];

    auto GX = Vec(hZero), GY = Vec(hZero);
    int64_t i = 0;
    while (i < C) {
      auto InpSliceCPtr = InpSlice[i].data();
      auto GInpSliceCPtr = (*GInpSlice)[i].data();
      auto GOut = Vec::LoadU(offset + GOutSlice[i].data(), len);

      (NW * GOut).store(GInpCornerArr);
      MaskScatterAdd(GInpCornerArr, GInpSliceCPtr, IGInpNWOffsetArr, INWMaskArr, len);
      (NE * GOut).store(GInpCornerArr);
      MaskScatterAdd(GInpCornerArr, GInpSliceCPtr, IGInpNEOffsetArr, INEMaskArr, len);
      (SW * GOut).store(GInpCornerArr);
      MaskScatterAdd(GInpCornerArr, GInpSliceCPtr, IGInpSWOffsetArr, ISWMaskArr, len);
      (SE * GOut).store(GInpCornerArr);
      MaskScatterAdd(GInpCornerArr, GInpSliceCPtr, IGInpSEOffsetArr, ISEMaskArr, len);
      Vec NWMaskCopy = NWMask;
      Vec NEMaskCopy = NEMask;
      Vec SWMaskCopy = SWMask;
      Vec SEMaskCopy = SEMask;
      auto NWVal = vec256::MaskGather<sizeof(T)>(Vec(0), InpSliceCPtr, INWOffset, &NWMaskCopy);
      auto NEVal = vec256::MaskGather<sizeof(T)>(Vec(0), InpSliceCPtr, INEOffset, &NEMaskCopy);
      auto SWVal = vec256::MaskGather<sizeof(T)>(Vec(0), InpSliceCPtr, ISWOffset, &SWMaskCopy);
      auto SEVal = vec256::MaskGather<sizeof(T)>(Vec(0), InpSliceCPtr, ISEOffset, &SEMaskCopy);

      GX = GX + (S * (NEVal - NWVal) + N * (SEVal - SWVal)) * GOut;
      GY = GY + (E * (SWVal - NWVal) + W * (SEVal - NEVal)) * GOut;
      ++i;
    }

    GX = GX * GxMult;
    GY = GY * GyMult;

    constexpr int64_t step = Vec::size();
    auto InterleavedGGrid = interleave2(GX, GY);
    auto GGridPtr = (*GGridSlice)[0].data() + offset * 2;
    std::get<0>(InterleavedGGrid).store(GGridPtr, std::min(len * hTwo, step));
    std::get<1>(InterleavedGGrid).store(GGridPtr + step, std::max(static_cast<int64_t>(0), len * hTwo - step));
  }
};

template <typename T, GridSamplerPadding padding, bool align_corners>
struct ApplyGridSample2D<T, hTwo, GridSamplerInterpolation::Nearest, padding, align_corners> {
  using Vec = vec256::Vec256<T>;
  using integer_t = vec256::int_same_size_t<T>;
  using iVec = vec256::Vec256<integer_t>;

  const int64_t InpH;
  const int64_t InpW;
  const int64_t InpSH;
  const int64_t InpSW;
  const int64_t C;
  const int64_t InpSC;
  const ComputeLocation<T, padding, align_corners> ComputeH;
  const ComputeLocation<T, padding, align_corners> ComputeW;
  const bool MustInBound = padding != GridSamplerPadding::Zeros;

  explicit ApplyGridSample2D(const TensorAcc<T, 4> &input)
      : InpH(input.size(2)),
        InpW(input.size(3)),
        InpSH(input.stride(2)),
        InpSW(input.stride(3)),
        C(input.size(1)),
        InpSC(input.stride(1)),
        ComputeH(input.size(2)),
        ComputeW(input.size(3)) {}

  inline void Backward(TensorAcc<T, 3> *GInpSlice, TensorAcc<T, 3> *GGridSlice, const TensorAcc<T, 3> &GOutSlice,
                       const TensorAcc<T, 3> &, int64_t offset, const Vec &grid_x, const Vec &grid_y,
                       int64_t len) const {
    auto X = ComputeW.apply(grid_x);
    auto XNearest = X.round();
    auto IXNearest = vec256::ConvertToIntOfSameSize<T>(XNearest);
    auto Y = ComputeH.apply(grid_y);
    auto YNearest = Y.round();
    auto IYNearest = vec256::ConvertToIntOfSameSize<T>(YNearest);

    auto IMask = MustInBound ? iVec(-1)
                             : (IXNearest > iVec(-1)) & (IXNearest < iVec(InpW)) & (IYNearest > iVec(-1)) &
                                 (IYNearest < iVec(InpH));

    auto IGInpOffset = IXNearest + iVec(InpW) * IYNearest;  // gInp is contiguous
    static constexpr int kSize = iVec::size();
    integer_t MaskArr[kSize], GInpOffsetArr[kSize];
    IMask.store(MaskArr);
    IGInpOffset.store(GInpOffsetArr);

    int64_t i = 0;
    while (i < C) {
      MaskScatterAdd(GOutSlice[i].data() + offset, (*GInpSlice)[i].data(), GInpOffsetArr, MaskArr, len);
      ++i;
    }
    auto GGridPtr = (*GGridSlice)[0].data() + offset * 2;
    auto ret = memset_s(static_cast<void *>(GGridPtr), sizeof(T) * len * hTwo, 0, sizeof(T) * len * hTwo);
    if (ret != 0) {
      MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
    }
  }
};

template <typename T, typename ApplyFn>
static inline void GridSampler2DGridSliceIterator(const TensorAcc<T, 3> &GridSlice, const ApplyFn &ApplyFN) {
  int64_t OutH = GridSlice.size(0);
  int64_t OutW = GridSlice.size(1);
  int64_t GridSH = GridSlice.stride(0);
  int64_t GridSW = GridSlice.stride(1);
  int64_t GridSCoor = GridSlice.stride(2);
  auto GridPtr = GridSlice.data();

  using Vec = vec256::Vec256<T>;
  using iVec = vec256::Vec256<vec256::int_same_size_t<T>>;
  constexpr int64_t step = Vec::size();

  if (GeometryIsContiguous({OutH, OutW, 2}, {GridSH, GridSW, GridSCoor})) {
    int64_t tSize, spatial_offset;
    tSize = OutH * OutW;
    spatial_offset = 0;
    while (spatial_offset < tSize) {
      int64_t grid_offset, len;
      grid_offset = spatial_offset * hTwo;
      len = std::min(step, tSize - spatial_offset);
      auto vec1 = Vec::LoadU(GridPtr + grid_offset, std::min(step, len * 2));
      auto vec2 = Vec::LoadU(GridPtr + grid_offset + step, std::max(static_cast<int64_t>(0), len * 2 - step));
      auto PairVecXY = deinterleave2(vec1, vec2);

      auto Y = std::get<1>(PairVecXY);
      auto X = std::get<0>(PairVecXY);
      if (len < step) {
        X = Vec::set(Vec(0), X, len);
        Y = Vec::set(Vec(0), Y, len);
      }

      ApplyFN(X, Y, spatial_offset, len);
      spatial_offset += step;
    }
  } else if (GridSW == hOne || OutW == hOne) {
    auto LineFn = [&ApplyFN, &step](const T *grid_ptr_x, const T *grid_ptr_y, int64_t out_base_offset, int64_t tSize) {
      int64_t i = 0;
      while (i < tSize) {
        int64_t len;
        len = std::min(step, tSize - i);
        auto X = Vec::LoadU(grid_ptr_x + i, len);
        auto Y = Vec::LoadU(grid_ptr_y + i, len);
        // make sure that X and Y are valid grid sample locations
        if (len < step) {
          X = Vec::set(Vec(0), X, len);
          Y = Vec::set(Vec(0), Y, len);
        }
        ApplyFN(X, Y, out_base_offset + i, len);
        i += step;
      }
    };

    if (GeometryIsContiguous({OutH, OutW}, {GridSH, GridSW})) {
      LineFn(GridPtr, GridPtr + GridSCoor, 0, OutH * OutW);
    } else {
      auto grid_ptr_NH = GridPtr;
      int64_t h = 0;
      while (h < OutH) {
        LineFn(grid_ptr_NH, grid_ptr_NH + GridSCoor, h * OutW, OutW);
        grid_ptr_NH += GridSH;
        h++;
      }
    }
  } else {
    auto spatial_offset = 0;
    auto i_offsets_delta = iVec(GridSW * step);
    int64_t h = 0;
    while (h < OutH) {
      auto grid_ptr_x = h * GridSH + GridPtr;
      auto grid_ptr_y = GridSCoor + grid_ptr_x;
      auto i_offsets = iVec::arange(0, LongToInt(GridSW));
      int64_t w = 0;
      while (w < OutW) {
        auto len = std::min(step, OutW - w);
        if (len < step) {
          i_offsets = iVec::set(iVec(0), i_offsets, len);
        }
        ApplyFN(vec256::GatherVec<sizeof(T)>(grid_ptr_x, i_offsets),
                vec256::GatherVec<sizeof(T)>(grid_ptr_y, i_offsets), spatial_offset, len);

        i_offsets = i_offsets + i_offsets_delta;
        spatial_offset += len;
        w += step;
      }
      h++;
    }
  }
}
}  // namespace grid_sampler_2d_grad_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_GRID_SAMPLER_2D_GRAD_CPU_KERNEL_H_
