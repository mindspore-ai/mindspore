/* Copyright 2017-2022 The ml_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * The code implementation refers to the following files from ml_dtypes:
 * - https://github.com/jax-ml/ml_dtypes/tree/v0.4.0/ml_dtypes/_src/custom_float.h
 * - https://github.com/jax-ml/ml_dtypes/tree/v0.4.0/ml_dtypes/_src/ufuncs.h
 * - https://github.com/jax-ml/ml_dtypes/tree/v0.4.0/ml_dtypes/_src/common.h
 * - https://github.com/jax-ml/ml_dtypes/tree/v0.4.0/ml_dtypes/_src/dtypes.cc
 * Additional modifications are made by Huawei Technologies Co., Ltd in 2024.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "include/utils/np_dtypes.h"
#include <algorithm>
#include <string>
#include <memory>
#include <limits>
#include <utility>
#include <vector>
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "base/float16.h"
#include "base/bfloat16.h"
#include "utils/log_adapter.h"

#if NPY_API_VERSION < 0x0000000d
#error Current Numpy version is too low, the required version is not less than 1.19.3.
#endif

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace mindspore {
namespace np_dtypes {
// A safe PyObject pointer which can decrement the references automatically when destructing.
struct PyObjDeleter {
  void operator()(PyObject *object) const { Py_DECREF(object); }
};
using PyObjectPtr = std::unique_ptr<PyObject, PyObjDeleter>;
PyObjectPtr SafePtr(PyObject *object) { return PyObjectPtr(object); }

// Representation of a custom Python type.
template <typename T>
struct PyType {
  PyObject_HEAD;
  T value;
};

// Description of a numpy type.
template <typename T>
struct NpTypeBaseDescr {
  static int Dtype() { return np_type_num; }
  static PyTypeObject *TypePtr() { return np_type_ptr; }
  static int np_type_num;
  static PyTypeObject *np_type_ptr;
  static PyArray_Descr *np_descr;
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto descr_proto;
  static PyNumberMethods number_methods;
};

template <typename T>
int NpTypeBaseDescr<T>::np_type_num = NPY_NOTYPE;
template <typename T>
PyTypeObject *NpTypeBaseDescr<T>::np_type_ptr = nullptr;
template <typename T>
PyArray_Descr *NpTypeBaseDescr<T>::np_descr = nullptr;
template <typename T>
PyArray_ArrFuncs NpTypeBaseDescr<T>::arr_funcs;
template <typename T>
PyArray_DescrProto NpTypeBaseDescr<T>::descr_proto;

template <typename T>
struct NpTypeDescr {
  static int Dtype() { return np_type_num; }
  static int np_type_num;
};

template <>
struct NpTypeDescr<bfloat16> : NpTypeBaseDescr<bfloat16> {
  static constexpr const char *type_name = "bfloat16";
  static constexpr const char *type_doc = "BFloat16 type for numpy";
  static constexpr char kind = 'V';
  static constexpr char type = 'E';
  static constexpr char byte_order = '=';
};

template <>
int NpTypeDescr<unsigned char>::np_type_num = NPY_UBYTE;
template <>
int NpTypeDescr<unsigned short>::np_type_num = NPY_USHORT;
template <>
int NpTypeDescr<unsigned int>::np_type_num = NPY_UINT;
template <>
int NpTypeDescr<unsigned long>::np_type_num = NPY_ULONG;
template <>
int NpTypeDescr<unsigned long long>::np_type_num = NPY_ULONGLONG;
template <>
int NpTypeDescr<char>::np_type_num = NPY_BYTE;
template <>
int NpTypeDescr<short>::np_type_num = NPY_SHORT;
template <>
int NpTypeDescr<int>::np_type_num = NPY_INT;
template <>
int NpTypeDescr<long>::np_type_num = NPY_LONG;
template <>
int NpTypeDescr<long long>::np_type_num = NPY_LONGLONG;
template <>
int NpTypeDescr<bool>::np_type_num = NPY_BOOL;
template <>
int NpTypeDescr<float16>::np_type_num = NPY_HALF;
template <>
int NpTypeDescr<float>::np_type_num = NPY_FLOAT;
template <>
int NpTypeDescr<double>::np_type_num = NPY_ULONG;
template <>
int NpTypeDescr<long double>::np_type_num = NPY_LONGDOUBLE;

inline bool IsFloatEqual(const float &a, const float &b) {
  return (std::fabs(a - b) <= std::numeric_limits<float>::epsilon());
}

// Check if object is specific numpy custom type.
template <typename T>
bool PyType_CheckType(PyObject *object) {
  return PyObject_IsInstance(object, reinterpret_cast<PyObject *>(NpTypeDescr<T>::TypePtr()));
}

// Get value in the Python type object.
template <typename T>
T PyType_GetValue(PyObject *object) {
  return reinterpret_cast<PyType<T> *>(object)->value;
}

// Create PyTypeObject<T> data from T value.
template <typename T>
PyObjectPtr PyTypeFromValue(T value) {
  PyTypeObject *np_type_p = NpTypeDescr<T>::TypePtr();
  PyObjectPtr npy_data_p = SafePtr(np_type_p->tp_alloc(np_type_p, 0));
  PyType<T> *data_p = reinterpret_cast<PyType<T> *>(npy_data_p.get());
  if (data_p) {
    data_p->value = value;
  }
  return npy_data_p;
}

template <typename T>
PyObject *PyType_Add(PyObject *a, PyObject *b) {
  if (PyType_CheckType<T>(a) && PyType_CheckType<T>(b)) {
    return PyTypeFromValue<T>(PyType_GetValue<T>(a) + PyType_GetValue<T>(b)).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject *PyType_Subtract(PyObject *a, PyObject *b) {
  if (PyType_CheckType<T>(a) && PyType_CheckType<T>(b)) {
    return PyTypeFromValue<T>(PyType_GetValue<T>(a) - PyType_GetValue<T>(b)).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject *PyType_Multiply(PyObject *a, PyObject *b) {
  if (PyType_CheckType<T>(a) && PyType_CheckType<T>(b)) {
    return PyTypeFromValue<T>(PyType_GetValue<T>(a) * PyType_GetValue<T>(b)).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject *PyType_Divide(PyObject *a, PyObject *b) {
  if (PyType_CheckType<T>(a) && PyType_CheckType<T>(b)) {
    return PyTypeFromValue<T>(PyType_GetValue<T>(a) / PyType_GetValue<T>(b)).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

template <typename T>
PyObject *PyType_Negative(PyObject *self) {
  return PyTypeFromValue<T>(-PyType_GetValue<T>(self)).release();
}

template <typename T>
PyObject *PyType_Int(PyObject *self) {
  T value = PyType_GetValue<T>(self);
  return PyLong_FromLong(static_cast<long>(static_cast<float>(value)));
}

template <typename T>
PyObject *PyType_Float(PyObject *self) {
  T value = PyType_GetValue<T>(self);
  return PyFloat_FromDouble(static_cast<double>(static_cast<float>(value)));
}

template <typename T>
PyNumberMethods NpTypeBaseDescr<T>::number_methods = {
  PyType_Add<T>,       // nb_add
  PyType_Subtract<T>,  // nb_subtract
  PyType_Multiply<T>,  // nb_multiply
  nullptr,             // nb_remainder
  nullptr,             // nb_divmod
  nullptr,             // nb_power
  PyType_Negative<T>,  // nb_negative
  nullptr,             // nb_positive
  nullptr,             // nb_absolute
  nullptr,             // nb_nonzero
  nullptr,             // nb_invert
  nullptr,             // nb_lshift
  nullptr,             // nb_rshift
  nullptr,             // nb_and
  nullptr,             // nb_xor
  nullptr,             // nb_or
  PyType_Int<T>,       // nb_int
  nullptr,             // reserved
  PyType_Float<T>,     // nb_float
  nullptr,             // nb_inplace_add
  nullptr,             // nb_inplace_subtract
  nullptr,             // nb_inplace_multiply
  nullptr,             // nb_inplace_remainder
  nullptr,             // nb_inplace_power
  nullptr,             // nb_inplace_lshift
  nullptr,             // nb_inplace_rshift
  nullptr,             // nb_inplace_and
  nullptr,             // nb_inplace_xor
  nullptr,             // nb_inplace_or
  nullptr,             // nb_floor_divide
  PyType_Divide<T>,    // nb_true_divide
  nullptr,             // nb_inplace_floor_divide
  nullptr,             // nb_inplace_true_divide
  nullptr,             // nb_index
};

template <typename TypeIn, typename TypeOut, typename Func>
struct UnaryUFunc {
  static std::vector<int> Types() { return {NpTypeDescr<TypeIn>::Dtype(), NpTypeDescr<TypeOut>::Dtype()}; }
  static void Fn(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const char *arg_p = args[0];
    char *out_p = args[1];
    for (npy_intp d = 0; d < *dimensions; d++) {
      auto arg = *reinterpret_cast<const TypeIn *>(arg_p);
      *reinterpret_cast<TypeOut *>(out_p) = Func()(arg);
      arg_p += steps[0];
      out_p += steps[1];
    }
  }
};

template <typename TypeIn, typename TypeOut, typename TypeOut2, typename Func>
struct UnaryUFunc2 {
  static std::vector<int> Types() {
    return {NpTypeDescr<TypeIn>::Dtype(), NpTypeDescr<TypeOut>::Dtype(), NpTypeDescr<TypeOut2>::Dtype()};
  }
  static void Fn(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const char *arg_p = args[0];
    char *out0_p = args[1];
    char *out1_p = args[2];
    for (npy_intp d = 0; d < *dimensions; d++) {
      auto arg = *reinterpret_cast<const TypeIn *>(arg_p);
      std::tie(*reinterpret_cast<TypeOut *>(out0_p), *reinterpret_cast<TypeOut2 *>(out1_p)) = Func()(arg);
      arg_p += steps[0];
      out0_p += steps[1];
      out1_p += steps[2];
    }
  }
};

template <typename TypeIn, typename TypeOut, typename Func>
struct BinaryUFunc {
  static std::vector<int> Types() {
    return {NpTypeDescr<TypeIn>::Dtype(), NpTypeDescr<TypeIn>::Dtype(), NpTypeDescr<TypeOut>::Dtype()};
  }
  static void Fn(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const char *arg0_p = args[0];
    const char *arg1_p = args[1];
    char *out_p = args[2];
    for (npy_intp d = 0; d < *dimensions; d++) {
      auto arg0 = *reinterpret_cast<const TypeIn *>(arg0_p);
      auto arg1 = *reinterpret_cast<const TypeIn *>(arg1_p);
      *reinterpret_cast<TypeOut *>(out_p) = Func()(arg0, arg1);
      arg0_p += steps[0];
      arg1_p += steps[1];
      out_p += steps[2];
    }
  }
};

template <typename TypeIn, typename TypeIn2, typename TypeOut, typename Func>
struct BinaryUFunc2 {
  static std::vector<int> Types() {
    return {NpTypeDescr<TypeIn>::Dtype(), NpTypeDescr<TypeIn2>::Dtype(), NpTypeDescr<TypeOut>::Dtype()};
  }
  static void Fn(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const char *arg0_p = args[0];
    const char *arg1_p = args[1];
    char *out_p = args[2];
    for (npy_intp d = 0; d < *dimensions; d++) {
      auto arg0 = *reinterpret_cast<const TypeIn *>(arg0_p);
      auto arg1 = *reinterpret_cast<const TypeIn2 *>(arg1_p);
      *reinterpret_cast<TypeOut *>(out_p) = Func()(arg0, arg1);
      arg0_p += steps[0];
      arg1_p += steps[1];
      out_p += steps[2];
    }
  }
};
namespace ufuncs {
// Implementation of Numpy universal functions.
template <typename T>
struct Add {
  T operator()(T a, T b) { return a + b; }
};
template <typename T>
struct Subtract {
  T operator()(T a, T b) { return a - b; }
};
template <typename T>
struct Multiply {
  T operator()(T a, T b) { return a * b; }
};
template <typename T>
struct Divide {
  T operator()(T a, T b) { return a / b; }
};
inline std::pair<float, float> divmod(float a, float b) {
  if (IsFloatEqual(b, 0.0f)) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (IsFloatEqual(mod, 0.0f)) {
    mod = std::copysign(0.0f, b);
  } else if ((b < 0.0f) != (mod < 0.0f)) {
    mod += b;
    div -= 1.0f;
  }
  float floor_div;
  if (!IsFloatEqual(div, 0.0f)) {
    floor_div = std::floor(div);
    if (div - floor_div > 0.5f) {
      floor_div += 1.0f;
    }
  } else {
    floor_div = std::copysign(0.0f, a / b);
  }
  return {floor_div, mod};
}
template <typename T>
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {NpTypeDescr<T>::Dtype(), NpTypeDescr<T>::Dtype(), NpTypeDescr<T>::Dtype(), NpTypeDescr<T>::Dtype()};
  }
  static void Fn(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const char *arg0_p = args[0];
    const char *arg1_p = args[1];
    char *out0_p = args[2];
    char *out1_p = args[3];
    for (npy_intp d = 0; d < *dimensions; d++) {
      T arg0 = *reinterpret_cast<const T *>(arg0_p);
      T arg1 = *reinterpret_cast<const T *>(arg1_p);
      float floordiv, mod;
      std::tie(floordiv, mod) = divmod(static_cast<float>(arg0), static_cast<float>(arg1));
      *reinterpret_cast<T *>(out0_p) = T(floordiv);
      *reinterpret_cast<T *>(out1_p) = T(mod);
      arg0_p += steps[0];
      arg1_p += steps[1];
      out0_p += steps[2];
      out1_p += steps[3];
    }
  }
};
template <typename T>
struct FloorDivide {
  T operator()(T a, T b) { return T(divmod(static_cast<float>(a), static_cast<float>(b)).first); }
};
template <typename T>
struct Remainder {
  T operator()(T a, T b) { return T(divmod(static_cast<float>(a), static_cast<float>(b)).second); }
};
template <typename T>
struct Fmod {
  T operator()(T a, T b) { return T(std::fmod(static_cast<float>(a), static_cast<float>(b))); }
};
template <typename T>
struct Negative {
  T operator()(T a) { return -a; }
};
template <typename T>
struct Positive {
  T operator()(T a) { return a; }
};
template <typename T>
struct Power {
  T operator()(T a, T b) { return pow(a, b); }
};
template <typename T>
struct Abs {
  T operator()(T a) { return abs(a); }
};
template <typename T>
struct Cbrt {
  T operator()(T a) { return T(std::cbrt(static_cast<float>(a))); }
};
template <typename T>
struct Ceil {
  T operator()(T a) { return ceil(a); }
};
template <typename T>
struct CopySign {
  T operator()(T a, T b) { return T(std::copysign(static_cast<float>(a), static_cast<float>(b))); }
};
template <typename T>
struct Exp {
  T operator()(T a) { return exp(a); }
};
template <typename T>
struct Exp2 {
  T operator()(T a) { return T(std::exp2(static_cast<float>(a))); }
};
template <typename T>
struct Expm1 {
  T operator()(T a) { return T(std::expm1(static_cast<float>(a))); }
};
template <typename T>
struct Floor {
  T operator()(T a) { return floor(a); }
};
template <typename T>
struct Frexp {
  std::pair<T, int> operator()(T a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {T(f), exp};
  }
};
template <typename T>
struct Heaviside {
  T operator()(T x, T h0) {
    if (isnan(x)) {
      return x;
    }
    if (x < T(0)) {
      return T(0);
    }
    if (x > T(0)) {
      return T(1);
    }
    return h0;
  }
};
template <typename T>
struct Conjugate {
  T operator()(T a) { return a; }
};
template <typename T>
struct IsFinite {
  bool operator()(T a) { return isfinite(a); }
};
template <typename T>
struct IsInf {
  bool operator()(T a) { return isinf(a); }
};
template <typename T>
struct IsNan {
  bool operator()(T a) { return isnan(a); }
};
template <typename T>
struct Ldexp {
  T operator()(T a, int exp) { return T(std::ldexp(static_cast<float>(a), exp)); }
};
template <typename T>
struct Log {
  T operator()(T a) { return log(a); }
};
template <typename T>
struct Log1p {
  T operator()(T a) { return T(std::log1p(static_cast<float>(a))); }
};
template <typename T>
struct Log2 {
  T operator()(T a) { return T(std::log2(static_cast<float>(a))); }
};
template <typename T>
struct Log10 {
  T operator()(T a) { return T(std::log10(static_cast<float>(a))); }
};
template <typename T>
struct LogAddExp {
  T operator()(T a, T b) {
    float x = static_cast<float>(a);
    float y = static_cast<float>(b);
    if (IsFloatEqual(x, y)) {
      return T(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return T(out);
  }
};
template <typename T>
struct LogAddExp2 {
  T operator()(T a, T b) {
    float x = static_cast<float>(a);
    float y = static_cast<float>(b);
    if (IsFloatEqual(x, y)) {
      return T(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return T(out);
  }
};
template <typename T>
struct Modf {
  std::pair<T, T> operator()(T a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {T(f), T(integral)};
  }
};
template <typename T>
struct Reciprocal {
  T operator()(T a) { return T(1.f / static_cast<float>(a)); }
};
template <typename T>
struct Rint {
  T operator()(T a) { return T(std::rint(static_cast<float>(a))); }
};
template <typename T>
struct Sign {
  T operator()(T a) {
    if (isnan(a)) {
      return a;
    }
    if (a < T(0)) {
      return T(-1);
    }
    if (a > T(0)) {
      return T(1);
    }
    return a;
  }
};
template <typename T>
struct SignBit {
  bool operator()(T a) { return std::signbit(static_cast<float>(a)); }
};
template <typename T>
struct Sqrt {
  T operator()(T a) { return T(std::sqrt(static_cast<float>(a))); }
};
template <typename T>
struct Square {
  T operator()(T a) {
    float f(a);
    return T(f * f);
  }
};
template <typename T>
struct Trunc {
  T operator()(T a) { return T(std::trunc(static_cast<float>(a))); }
};
// Trigonometric functions
template <typename T>
struct Sin {
  T operator()(T a) { return sin(a); }
};
template <typename T>
struct Cos {
  T operator()(T a) { return cos(a); }
};
template <typename T>
struct Tan {
  T operator()(T a) { return tan(a); }
};
template <typename T>
struct Arcsin {
  T operator()(T a) { return T(std::asin(static_cast<float>(a))); }
};
template <typename T>
struct Arccos {
  T operator()(T a) { return T(std::acos(static_cast<float>(a))); }
};
template <typename T>
struct Arctan {
  T operator()(T a) { return T(std::atan(static_cast<float>(a))); }
};
template <typename T>
struct Arctan2 {
  T operator()(T a, T b) { return T(std::atan2(static_cast<float>(a), static_cast<float>(b))); }
};
template <typename T>
struct Hypot {
  T operator()(T a, T b) { return T(std::hypot(static_cast<float>(a), static_cast<float>(b))); }
};
template <typename T>
struct Sinh {
  T operator()(T a) { return T(std::sinh(static_cast<float>(a))); }
};
template <typename T>
struct Cosh {
  T operator()(T a) { return T(std::cosh(static_cast<float>(a))); }
};
template <typename T>
struct Tanh {
  T operator()(T a) { return tanh(a); }
};
template <typename T>
struct Arcsinh {
  T operator()(T a) { return T(std::asinh(static_cast<float>(a))); }
};
template <typename T>
struct Arccosh {
  T operator()(T a) { return T(std::acosh(static_cast<float>(a))); }
};
template <typename T>
struct Arctanh {
  T operator()(T a) { return T(std::atanh(static_cast<float>(a))); }
};
template <typename T>
struct Deg2rad {
  T operator()(T a) {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float RADIANS_PER_DEGREE = PI / 180.0f;
    return T(static_cast<float>(a) * RADIANS_PER_DEGREE);
  }
};
template <typename T>
struct Rad2deg {
  T operator()(T a) {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float DEGREES_PER_RADIAN = 180.0f / PI;
    return T(static_cast<float>(a) * DEGREES_PER_RADIAN);
  }
};
template <typename T>
struct Eq {
  npy_bool operator()(T a, T b) { return a == b; }
};
template <typename T>
struct Ne {
  npy_bool operator()(T a, T b) { return a != b; }
};
template <typename T>
struct Lt {
  npy_bool operator()(T a, T b) { return a < b; }
};
template <typename T>
struct Le {
  npy_bool operator()(T a, T b) { return a <= b; }
};
template <typename T>
struct Gt {
  npy_bool operator()(T a, T b) { return a > b; }
};
template <typename T>
struct Ge {
  npy_bool operator()(T a, T b) { return a >= b; }
};
template <typename T>
struct Maximum {
  T operator()(T a, T b) { return isnan(a) || a > b ? a : b; }
};
template <typename T>
struct Minimum {
  T operator()(T a, T b) { return isnan(a) || a < b ? a : b; }
};
template <typename T>
struct Fmax {
  T operator()(T a, T b) { return isnan(b) || a > b ? a : b; }
};
template <typename T>
struct Fmin {
  T operator()(T a, T b) { return isnan(b) || a < b ? a : b; }
};
template <typename T>
struct LogicalNot {
  npy_bool operator()(T a) { return !static_cast<bool>(a); }
};
template <typename T>
struct LogicalAnd {
  npy_bool operator()(T a, T b) { return static_cast<bool>(a) && static_cast<bool>(b); }
};
template <typename T>
struct LogicalOr {
  npy_bool operator()(T a, T b) { return static_cast<bool>(a) || static_cast<bool>(b); }
};
template <typename T>
struct LogicalXor {
  npy_bool operator()(T a, T b) { return static_cast<bool>(a) ^ static_cast<bool>(b); }
};
// Get unsigned integer type with same size of T.
template <int kNumBytes>
struct GetUnsignedInteger;
template <>
struct GetUnsignedInteger<1> {
  using uint_type = uint8_t;
};
template <>
struct GetUnsignedInteger<2> {
  using uint_type = uint16_t;
};
template <>
struct GetUnsignedInteger<4> {
  using uint_type = uint32_t;
};
template <typename T>
using UIntType = typename GetUnsignedInteger<sizeof(T)>::uint_type;
template <typename TypeIn, typename TypeOut>
TypeOut bit_cast(TypeIn value) {
  static_assert(sizeof(TypeIn) == sizeof(TypeOut), "For bit_cast, types must match size.");
  TypeOut out = TypeOut(0);
  errno_t ret = memcpy_s(&out, sizeof(TypeOut), &value, sizeof(TypeIn));
  if (ret != EOK) {
    PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d", ret);
    return out;
  }
  return out;
}
template <typename T>
struct NextAfter {
  T operator()(T from, T to) {
    if (isnan(from) || isnan(to)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    UIntType<T> from_uint = bit_cast<T, UIntType<T>>(from);
    UIntType<T> to_uint = bit_cast<T, UIntType<T>>(to);
    if (from_uint == to_uint) {
      return to;
    }
    UIntType<T> sign_mask = UIntType<T>(1) << (sizeof(T) * CHAR_BIT - 1);
    UIntType<T> from_uint_abs = bit_cast<T, UIntType<T>>(abs(from));
    UIntType<T> from_uint_sign = from_uint & sign_mask;
    UIntType<T> to_uint_abs = bit_cast<T, UIntType<T>>(abs(to));
    UIntType<T> to_uint_sign = to_uint & sign_mask;
    if (from_uint_abs == 0) {
      if (to_uint_abs == 0) {
        return to;
      } else {
        // Minimum non-zero value with sign bit of `to`.
        return bit_cast<UIntType<T>, T>(static_cast<UIntType<T>>(0x01 | to_uint_sign));
      }
    }
    UIntType<T> next_step = (from_uint_abs > to_uint_abs || from_uint_sign != to_uint_sign)
                              ? static_cast<UIntType<T>>(-1)
                              : static_cast<UIntType<T>>(1);
    UIntType<T> out_uint = from_uint + next_step;
    return bit_cast<UIntType<T>, T>(out_uint);
  }
};
}  // namespace ufuncs

// Cast input object to Python type T.
template <typename T>
bool CastToPyType(PyObject *obj, T *output) {
  // object is an instance of NpTypeDescr
  if (PyType_CheckType<T>(obj)) {
    *output = PyType_GetValue<T>(obj);
    return true;
  }
  // object is an instance of int
  if (PyLong_Check(obj)) {
    long value = PyLong_AsLong(obj);
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(value);
    return true;
  }
  // object is an instance of float
  if (PyFloat_Check(obj)) {
    double value = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
      return false;
    }
    *output = T(value);
    return true;
  }
  // object is an instance of scalar float16
  if (PyArray_IsScalar(obj, Half)) {
    float16 value;
    PyArray_ScalarAsCtype(obj, &value);
    *output = T(value);
    return true;
  }
  // object is an instance of scalar float
  if (PyArray_IsScalar(obj, Float)) {
    float value;
    PyArray_ScalarAsCtype(obj, &value);
    *output = T(value);
    return true;
  }
  // object is an instance of scalar double
  if (PyArray_IsScalar(obj, Double)) {
    double value;
    PyArray_ScalarAsCtype(obj, &value);
    *output = T(value);
    return true;
  }
  // object is an instance of scalar long double
  if (PyArray_IsScalar(obj, LongDouble)) {
    long double value;
    PyArray_ScalarAsCtype(obj, &value);
    *output = T(value);
    return true;
  }
  // object is an instance of 0-dim array
  if (PyArray_IsZeroDim(obj)) {
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);
    // cast value in array to type T
    if (PyArray_TYPE(arr) != NpTypeDescr<T>::Dtype()) {
      PyObjectPtr new_arr = SafePtr(PyArray_Cast(arr, NpTypeDescr<T>::Dtype()));
      if (PyErr_Occurred()) {
        return false;
      }
      arr = reinterpret_cast<PyArrayObject *>(new_arr.get());
    }
    *output = *reinterpret_cast<T *>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

// Constructs a new Python type.
template <typename T>
PyObject *PyType_New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_Format(PyExc_TypeError, "No keyword arguments should be provided when constructing %s",
                 NpTypeDescr<T>::type_name);
    return nullptr;
  }
  Py_ssize_t arg_num = PyTuple_Size(args);
  if (arg_num != 1) {
    PyErr_Format(PyExc_TypeError, "One argument is expected when constructing %s, but got %d.",
                 NpTypeDescr<T>::type_name, arg_num);
    return nullptr;
  }
  PyObject *arg = PyTuple_GetItem(args, 0);
  T value;
  // If arg is already NpTypeDescr<T>, just return it.
  if (PyType_CheckType<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  }
  // If arg can be casted to T value, create NpTypeDescr<T> from the value.
  if (CastToPyType<T>(arg, &value)) {
    return PyTypeFromValue<T>(value).release();
  }
  // If arg is an array, cast it to NpTypeDescr<T>
  if (PyArray_Check(arg)) {
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(arg);
    if (PyArray_TYPE(arr) != NpTypeDescr<T>::Dtype()) {
      return PyArray_Cast(arr, NpTypeDescr<T>::Dtype());
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  // If arg is unicodes or bytes, convert it from string to float, then cast the float to T value,
  // and then create NpTypeDescr<T> from the value.
  if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    PyObject *value_f = PyFloat_FromString(arg);
    if (CastToPyType<T>(value_f, &value)) {
      return PyTypeFromValue<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "Only number argument is expected when constructing %s, but got %s.",
               NpTypeDescr<T>::type_name, Py_TYPE(arg)->tp_name);
  return nullptr;
}

// Implementation of repr() for PyType.
template <typename T>
PyObject *PyType_Repr(PyObject *self) {
  T value = reinterpret_cast<PyType<T> *>(self)->value;
  std::string value_str = std::to_string(static_cast<float>(value));
  return PyUnicode_FromString(value_str.c_str());
}

// Overload function _Py_HashDouble to support Python version over 3.10.
inline Py_hash_t HashDouble_(Py_hash_t (*hash_double)(PyObject *, double), PyObject *self, double value) {
  return hash_double(self, value);
}

inline Py_hash_t HashDouble_(Py_hash_t (*hash_double)(double), PyObject *self, double value) {
  return hash_double(value);
}

// Implementation of hash() for PyType.
template <typename T>
Py_hash_t PyType_Hash(PyObject *self) {
  T value = reinterpret_cast<PyType<T> *>(self)->value;
  return HashDouble_(&_Py_HashDouble, self, static_cast<double>(value));
}

// Implementation of str() for PyType.
template <typename T>
PyObject *PyType_Str(PyObject *self) {
  T value = reinterpret_cast<PyType<T> *>(self)->value;
  std::string value_str = std::to_string(static_cast<float>(value));
  return PyUnicode_FromString(value_str.c_str());
}

// Implementation of Comparisons for PyType.
template <typename T>
PyObject *PyType_RichCompare(PyObject *a, PyObject *b, int op) {
  if (!PyType_CheckType<T>(a) || !PyType_CheckType<T>(b)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  T x = PyType_GetValue<T>(a);
  T y = PyType_GetValue<T>(b);
  bool result;
  switch (op) {
    case Py_EQ:
      result = (x == y);
      break;
    case Py_NE:
      result = (x != y);
      break;
    case Py_LT:
      result = (x < y);
      break;
    case Py_LE:
      result = (x <= y);
      break;
    case Py_GT:
      result = (x > y);
      break;
    case Py_GE:
      result = (x >= y);
      break;
    default:
      PyErr_Format(PyExc_ValueError, "Got invalid op type %d when comparing %s", op, NpTypeDescr<T>::type_name);
      return nullptr;
  }
  PyObject *ret = PyBool_FromLong(result);
  Py_INCREF(ret);
  return ret;
}

// Implementations of NumPy array methods for PyType.
template <typename T>
PyObject *NpType_GetItem(void *data, void *arr) {
  T value;
  errno_t ret = memcpy_s(&value, sizeof(T), data, sizeof(T));
  if (ret != EOK) {
    PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d.", ret);
    return nullptr;
  }
  return PyTypeFromValue(value).release();
}

template <typename T>
int NpType_SetItem(PyObject *item, void *data, void *arr) {
  T value;
  if (!CastToPyType<T>(item, &value)) {
    PyErr_Format(PyExc_TypeError, "Only number argument is expected for SetItem %s, but got %s.",
                 NpTypeDescr<T>::type_name, Py_TYPE(item)->tp_name);
    return -1;
  }
  errno_t ret = memcpy_s(data, sizeof(T), &value, sizeof(T));
  if (ret != EOK) {
    PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d.", ret);
    return -1;
  }
  return 0;
}

template <typename T>
int NpType_Compare(const void *d1, const void *d2, void *arr) {
  T x = *reinterpret_cast<const T *>(d1);
  T y = *reinterpret_cast<const T *>(d2);
  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  if (!isnan(x) && isnan(y)) {
    return -1;
  }
  if (isnan(x) && !isnan(y)) {
    return 1;
  }
  return 0;
}

template <typename T>
void NpType_CopySwapN(void *dest, npy_intp dstride, void *src, npy_intp sstride, npy_intp n, int swap, void *arr) {
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t), "Swap is not supported");
  char *dst_p = reinterpret_cast<char *>(dest);
  char *src_p = reinterpret_cast<char *>(src);
  if (!src_p) {
    return;
  }
  if (swap && sizeof(T) == sizeof(int16_t)) {
    for (npy_intp i = 0; i < n; i++) {
      *reinterpret_cast<T *>(dst_p + dstride * i) = *reinterpret_cast<T *>(src_p + sstride * i);
      char *r = dst_p + dstride * i;
      std::swap(r[0], r[1]);
    }
  } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
    errno_t ret = memcpy_s(dst_p, n * sizeof(T), src_p, n * sizeof(T));
    if (ret != EOK) {
      PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d.", ret);
      return;
    }
  } else {
    if (sstride == 0) {
      T value = *reinterpret_cast<T *>(src_p);
#pragma omp parallel for
      for (npy_intp i = 0; i < n; i++) {
        *reinterpret_cast<T *>(dst_p + dstride * i) = value;
      }
    } else {
      for (npy_intp i = 0; i < n; i++) {
        *reinterpret_cast<T *>(dst_p + dstride * i) = *reinterpret_cast<T *>(src_p + sstride * i);
      }
    }
  }
}

template <typename T>
void NpType_CopySwap(void *dest, void *src, int swap, void *arr) {
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t), "Swap is not supported");
  if (!src) {
    return;
  }
  errno_t ret = memcpy_s(dest, sizeof(T), src, sizeof(T));
  if (ret != EOK) {
    PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d.", ret);
    return;
  }
  if (swap && (sizeof(T) == sizeof(int16_t))) {
    char *p = reinterpret_cast<char *>(dest);
    std::swap(p[0], p[1]);
  }
}

template <typename T>
npy_bool NpType_NonZero(void *data, void *arr) {
  T value;
  errno_t ret = memcpy_s(&value, sizeof(T), data, sizeof(T));
  if (ret != EOK) {
    PyErr_Format(PyExc_MemoryError, "memcpy_s failed: %d.", ret);
    return false;
  }
  return value != static_cast<T>(0);
}

template <typename T>
int NpType_Fill(void *data, npy_intp length, void *arr) {
  T *const buffer = reinterpret_cast<T *>(data);
  const T start(buffer[0]);
  const T delta = static_cast<T>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; i++) {
    buffer[i] = static_cast<T>(start + T(i) * delta);
  }
  return 0;
}

template <typename T>
void NpType_Dot(void *ip1, npy_intp is1, void *ip2, npy_intp is2, void *op, npy_intp n, void *arr) {
  char *p1 = reinterpret_cast<char *>(ip1);
  char *p2 = reinterpret_cast<char *>(ip2);
  T acc = T(0);
  for (npy_intp i = 0; i < n; i++) {
    T *const a = reinterpret_cast<T *>(p1);
    T *const b = reinterpret_cast<T *>(p2);
    acc += static_cast<T>(*a) * static_cast<T>(*b);
    p1 += is1;
    p2 += is2;
  }
  T *out = reinterpret_cast<T *>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
int NpType_ArgMax(void *data, npy_intp n, npy_intp *max_ind, void *arr) {
  const T *data_p = reinterpret_cast<const T *>(data);
  T max_val = static_cast<T>(data_p[0]);
  *max_ind = 0;
  for (npy_intp i = 0; i < n; i++) {
    T val = static_cast<T>(data_p[i]);
    if (isnan(val) || val > max_val) {
      max_val = val;
      *max_ind = i;
      // NumPy stops at the first NaN.
      if (isnan(val)) {
        break;
      }
    }
  }
  return 0;
}

template <typename T>
int NpType_ArgMin(void *data, npy_intp n, npy_intp *min_ind, void *arr) {
  const T *data_p = reinterpret_cast<const T *>(data);
  T min_val = static_cast<T>(data_p[0]);
  *min_ind = 0;
  for (npy_intp i = 1; i < n; i++) {
    T val = static_cast<T>(data_p[i]);
    if (isnan(val) || val < min_val) {
      min_val = val;
      *min_ind = i;
      // NumPy stops at the first NaN.
      if (isnan(val)) {
        break;
      }
    }
  }
  return 0;
}

template <typename T>
PyArray_DescrProto GetNpDescrProto() {
  return {
    PyObject_HEAD_INIT(nullptr)         // PyObject_HEAD
    nullptr,                            // typeobj
    NpTypeDescr<T>::kind,               // kind
    NpTypeDescr<T>::type,               // type
    NpTypeDescr<T>::byte_order,         // byteorder
    NPY_NEEDS_PYAPI | NPY_USE_SETITEM,  // flags
    0,                                  // type_num
    sizeof(T),                          // elsize
    alignof(T),                         // alignment
    nullptr,                            // subarray
    nullptr,                            // fields
    nullptr,                            // names
    &NpTypeDescr<T>::arr_funcs,         // f
    nullptr,                            // metadata
    nullptr,                            // c_metadata
    -1,                                 // hash
  };
}

// Cast a numpy array from type 'From' to 'To'.
template <typename From, typename To>
void NpyCast(void *from, void *to, npy_intp n, void *from_arr, void *to_arr) {
  const From *from_ptr = static_cast<From *>(from);
  To *to_ptr = static_cast<To *>(to);
  for (npy_intp i = 0; i < n; i++) {
    to_ptr[i] = static_cast<To>(from_ptr[i]);
  }
}

// Register a cast between T and other numpy type Y.
template <typename T, typename Y>
bool RegisterNpTypeCast(int np_type, bool scalar_castable) {
  PyArray_Descr *descr = PyArray_DescrFromType(np_type);
  if (PyArray_RegisterCastFunc(descr, NpTypeDescr<T>::Dtype(), NpyCast<Y, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(NpTypeDescr<T>::np_descr, np_type, NpyCast<T, Y>) < 0) {
    return false;
  }
  if (scalar_castable && PyArray_RegisterCanCast(NpTypeDescr<T>::np_descr, np_type, NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

// Register casts between T and other numpy types.
template <typename T>
bool RegisterNpTypeCasts() {
  if (!RegisterNpTypeCast<T, bool>(NPY_BOOL, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, float16>(NPY_HALF, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, float>(NPY_FLOAT, true)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, double>(NPY_DOUBLE, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, long double>(NPY_LONGDOUBLE, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, unsigned char>(NPY_UBYTE, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, unsigned short>(NPY_USHORT, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, unsigned int>(NPY_UINT, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, unsigned long>(NPY_ULONG, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, unsigned long long>(NPY_ULONGLONG, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, char>(NPY_BYTE, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, short>(NPY_SHORT, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, int>(NPY_INT, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, long>(NPY_LONG, false)) {
    return false;
  }
  if (!RegisterNpTypeCast<T, long long>(NPY_LONGLONG, false)) {
    return false;
  }
  // Complexs are not support yet.
  return true;
}

// Register a Numpy universal function.
template <typename UFunc, typename T>
bool RegisterNpTypeUFunc(PyObject *numpy, const char *fn_name) {
  std::vector<int> types = UFunc::Types();
  PyUFuncGenericFunction fn = reinterpret_cast<PyUFuncGenericFunction>(UFunc::Fn);
  PyObjectPtr ufunc_p = SafePtr(PyObject_GetAttrString(numpy, fn_name));
  if (!ufunc_p) {
    return false;
  }
  PyUFuncObject *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_p.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError, "The ufunc %s need %d arguments, but got %lu.", fn_name, ufunc->nargs,
                 types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, NpTypeDescr<T>::Dtype(), fn, const_cast<int *>(types.data()), nullptr) < 0) {
    return false;
  }
  return true;
}

// Register Numpy universal functions of type T.
template <typename T>
bool RegisterNpTypeUFuncs(PyObject *numpy) {
  // Math operations
  bool ok = RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Add<T>>, T>(numpy, "add") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Subtract<T>>, T>(numpy, "subtract") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Multiply<T>>, T>(numpy, "multiply") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Divide<T>>, T>(numpy, "divide") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::LogAddExp<T>>, T>(numpy, "logaddexp") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::LogAddExp2<T>>, T>(numpy, "logaddexp2") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Negative<T>>, T>(numpy, "negative") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Positive<T>>, T>(numpy, "positive") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Divide<T>>, T>(numpy, "true_divide") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::FloorDivide<T>>, T>(numpy, "floor_divide") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Power<T>>, T>(numpy, "power") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Remainder<T>>, T>(numpy, "remainder") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Remainder<T>>, T>(numpy, "mod") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Fmod<T>>, T>(numpy, "fmod") &&
            RegisterNpTypeUFunc<ufuncs::DivmodUFunc<T>, T>(numpy, "divmod") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Abs<T>>, T>(numpy, "absolute") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Abs<T>>, T>(numpy, "fabs") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Rint<T>>, T>(numpy, "rint") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Sign<T>>, T>(numpy, "sign") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Heaviside<T>>, T>(numpy, "heaviside") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Conjugate<T>>, T>(numpy, "conjugate") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Exp<T>>, T>(numpy, "exp") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Exp2<T>>, T>(numpy, "exp2") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Expm1<T>>, T>(numpy, "expm1") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Log<T>>, T>(numpy, "log") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Log1p<T>>, T>(numpy, "log1p") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Log2<T>>, T>(numpy, "log2") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Log10<T>>, T>(numpy, "log10") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Sqrt<T>>, T>(numpy, "sqrt") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Square<T>>, T>(numpy, "square") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Cbrt<T>>, T>(numpy, "cbrt") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Reciprocal<T>>, T>(numpy, "reciprocal") &&
            // Trigonometric functions
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Sin<T>>, T>(numpy, "sin") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Cos<T>>, T>(numpy, "cos") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Tan<T>>, T>(numpy, "tan") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arcsin<T>>, T>(numpy, "arcsin") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arccos<T>>, T>(numpy, "arccos") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arctan<T>>, T>(numpy, "arctan") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Arctan2<T>>, T>(numpy, "arctan2") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Hypot<T>>, T>(numpy, "hypot") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Sinh<T>>, T>(numpy, "sinh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Cosh<T>>, T>(numpy, "cosh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Tanh<T>>, T>(numpy, "tanh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arcsinh<T>>, T>(numpy, "arcsinh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arccosh<T>>, T>(numpy, "arccosh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Arctanh<T>>, T>(numpy, "arctanh") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Deg2rad<T>>, T>(numpy, "deg2rad") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Rad2deg<T>>, T>(numpy, "rad2deg") &&
            // Comparison functions
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Eq<T>>, T>(numpy, "equal") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Ne<T>>, T>(numpy, "not_equal") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Lt<T>>, T>(numpy, "less") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Le<T>>, T>(numpy, "less_equal") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Gt<T>>, T>(numpy, "greater") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::Ge<T>>, T>(numpy, "greater_equal") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Maximum<T>>, T>(numpy, "maximum") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Minimum<T>>, T>(numpy, "minimum") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Fmax<T>>, T>(numpy, "fmax") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::Fmin<T>>, T>(numpy, "fmin") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::LogicalAnd<T>>, T>(numpy, "logical_and") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::LogicalOr<T>>, T>(numpy, "logical_or") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, bool, ufuncs::LogicalXor<T>>, T>(numpy, "logical_xor") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, bool, ufuncs::LogicalNot<T>>, T>(numpy, "logical_not") &&
            // Floating point functions
            RegisterNpTypeUFunc<UnaryUFunc<T, bool, ufuncs::IsFinite<T>>, T>(numpy, "isfinite") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, bool, ufuncs::IsInf<T>>, T>(numpy, "isinf") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, bool, ufuncs::IsNan<T>>, T>(numpy, "isnan") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, bool, ufuncs::SignBit<T>>, T>(numpy, "signbit") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::CopySign<T>>, T>(numpy, "copysign") &&
            RegisterNpTypeUFunc<UnaryUFunc2<T, T, T, ufuncs::Modf<T>>, T>(numpy, "modf") &&
            RegisterNpTypeUFunc<BinaryUFunc2<T, int, T, ufuncs::Ldexp<T>>, T>(numpy, "ldexp") &&
            RegisterNpTypeUFunc<UnaryUFunc2<T, T, int, ufuncs::Frexp<T>>, T>(numpy, "frexp") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Floor<T>>, T>(numpy, "floor") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Ceil<T>>, T>(numpy, "ceil") &&
            RegisterNpTypeUFunc<UnaryUFunc<T, T, ufuncs::Trunc<T>>, T>(numpy, "trunc") &&
            RegisterNpTypeUFunc<BinaryUFunc<T, T, ufuncs::NextAfter<T>>, T>(numpy, "nextafter");
  return ok;
}

template <typename T>
bool RegisterNumpyType() {
  // Check if current type is already initialized.
  if (NpTypeDescr<T>::Dtype() != NPY_NOTYPE) {
    return true;
  }

  // Import Python modules
  import_array1(false);
  import_umath1(false);
  PyObjectPtr numpy_str = SafePtr(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  PyObjectPtr numpy_obj = SafePtr(PyImport_Import(numpy_str.get()));
  if (!numpy_obj) {
    return false;
  }
  // Initializes the NumPy type.
  PyHeapTypeObject *heap_type = reinterpret_cast<PyHeapTypeObject *>(PyType_Type.tp_alloc(&PyType_Type, 0));
  if (!heap_type) {
    return false;
  }
  PyObjectPtr name = SafePtr(PyUnicode_FromString(NpTypeDescr<T>::type_name));
  PyObjectPtr qualname = SafePtr(PyUnicode_FromString(NpTypeDescr<T>::type_name));
  heap_type->ht_name = name.release();
  heap_type->ht_qualname = qualname.release();
  PyTypeObject *py_type = &heap_type->ht_type;
  py_type->tp_name = NpTypeDescr<T>::type_name;
  py_type->tp_basicsize = sizeof(PyType<T>);
  py_type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  py_type->tp_base = &PyGenericArrType_Type;
  py_type->tp_new = PyType_New<T>;
  py_type->tp_repr = PyType_Repr<T>;
  py_type->tp_hash = PyType_Hash<T>;
  py_type->tp_str = PyType_Str<T>;
  py_type->tp_doc = const_cast<char *>(NpTypeDescr<T>::type_doc);
  py_type->tp_richcompare = PyType_RichCompare<T>;
  py_type->tp_as_number = &NpTypeDescr<T>::number_methods;
  if (PyType_Ready(py_type) < 0) {
    return false;
  }
  NpTypeDescr<T>::np_type_ptr = py_type;

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs &arr_funcs = NpTypeDescr<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NpType_GetItem<T>;
  arr_funcs.setitem = NpType_SetItem<T>;
  arr_funcs.compare = NpType_Compare<T>;
  arr_funcs.copyswapn = NpType_CopySwapN<T>;
  arr_funcs.copyswap = NpType_CopySwap<T>;
  arr_funcs.nonzero = NpType_NonZero<T>;
  arr_funcs.fill = NpType_Fill<T>;
  arr_funcs.dotfunc = NpType_Dot<T>;
  arr_funcs.argmax = NpType_ArgMax<T>;
  arr_funcs.argmin = NpType_ArgMin<T>;

  // Before NumPy 2.0, we allocate and manage the lifetime of descriptor, and Numpy only stores the pointer.
  // After NumPy 2.0, NumPy allocates and manages the lifetime of the descriptor.
  PyArray_DescrProto &descr_proto = NpTypeDescr<T>::descr_proto;
  descr_proto = GetNpDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = py_type;

  NpTypeDescr<T>::np_type_num = PyArray_RegisterDataType(&descr_proto);
  if (NpTypeDescr<T>::Dtype() < 0) {
    return false;
  }
  NpTypeDescr<T>::np_descr = PyArray_DescrFromType(NpTypeDescr<T>::Dtype());

  // Support numpy.dtype(type_name)
  PyObjectPtr np_type_dict = SafePtr(PyObject_GetAttrString(numpy_obj.get(), "sctypeDict"));
  if (!np_type_dict) {
    return false;
  }
  if (PyDict_SetItemString(np_type_dict.get(), NpTypeDescr<T>::type_name,
                           reinterpret_cast<PyObject *>(NpTypeDescr<T>::TypePtr())) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(reinterpret_cast<PyObject *>(NpTypeDescr<T>::TypePtr()), "dtype",
                             reinterpret_cast<PyObject *>(NpTypeDescr<T>::np_descr)) < 0) {
    return false;
  }

  // Register casts
  if (!RegisterNpTypeCasts<T>()) {
    return false;
  }

  // Register UFuncs
  if (!RegisterNpTypeUFuncs<T>(numpy_obj.get())) {
    return false;
  }

  return true;
}

std::string GetNumpyRunningVersion() {
  static std::string version_str = "";
  if (!version_str.empty()) {
    return version_str;
  }
  PyObjectPtr numpy_str = SafePtr(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return version_str;
  }
  PyObjectPtr numpy_obj = SafePtr(PyImport_Import(numpy_str.get()));
  if (!numpy_obj) {
    return version_str;
  }
  PyObject *numpy_dict = PyModule_GetDict(numpy_obj.get());
  if (!numpy_dict) {
    return version_str;
  }
  PyObject *numpy_version = PyDict_GetItemString(numpy_dict, "__version__");
  if (!numpy_version || !PyUnicode_Check(numpy_version)) {
    return version_str;
  }
  const char *version_c = PyUnicode_AsUTF8(numpy_version);
  if (!version_c) {
    return version_str;
  }
  version_str = version_c;
  MS_LOG(DEBUG) << "Current numpy version:" << version_str;
  return version_str;
}

std::string GetNumpyCompiledVersion() {
  switch (NPY_API_VERSION) {
    case 0x0000000d:  // 1.19.3+
      return "1.19";
    case 0x0000000e:  // 1.20/1.21
      return "1.20/1.21";
    case 0x0000000f:  // 1.22
      return "1.22";
    case 0x00000010:  // 1.23/1.24
      return "1.23/1.24";
    case 0x00000011:  // 1.25/1.26
      return "1.25/1.26";
    case 0x00000012:  // 2.0
      return "2.0";
    default:  // Values that exceed the macro definition limit.
      return (NPY_API_VERSION < 0x0000000d) ? "<1.19" : ">2.0";
  }
}

bool NumpyVersionValid() {
  // Get current numpy versions
  std::string curr_version = np_dtypes::GetNumpyRunningVersion();
  if (curr_version.empty()) {
    return false;
  }
  MS_LOG(DEBUG) << "current Numpy version :" << curr_version
                << ", Numpy version when the mindspore is compiled:" << GetNumpyCompiledVersion();
  std::replace(curr_version.begin(), curr_version.end(), '.', ' ');
  std::istringstream iss(curr_version);
  std::vector<int> version_parts(3);
  // version_parts[i] will be 0 if string is invalid.
  iss >> version_parts[0] >> version_parts[1] >> version_parts[2];
  // NPY_API_VERSION indices compiled numpy versions
  switch (NPY_API_VERSION) {
    case 0x0000000d:  // compiled: 1.19, running support: 1.19-1.26
      return (version_parts[0] == 1) && (version_parts[1] >= 19);
    case 0x0000000e:  // compiled: 1.20/1.21, running support: 1.20-1.26
      return (version_parts[0] == 1) && (version_parts[1] >= 20);
    case 0x0000000f:  // compiled: 1.22, running support: 1.22-1.26
      return (version_parts[0] == 1) && (version_parts[1] >= 22);
    case 0x00000010:  // compiled: 1.23/1.24, running support: 1.23-1.26
      return (version_parts[0] == 1) && (version_parts[1] >= 23);
    case 0x00000011:  // compiled: 1.25/1.26, running support: 1.20-1.26
      return (version_parts[0] == 1) && (version_parts[1] >= 20);
    case 0x00000012:  // compiled: 2.0, running support: 1.20-1.26, 2.0+
      return ((version_parts[0] == 1) && (version_parts[1] >= 20)) ||
             ((version_parts[0] == 2) && (version_parts[1] >= 0));
    default:
      if (NPY_API_VERSION < 0x0000000d) {
        // compiled: <1.19, running support: 1.19-1.26
        return (version_parts[0] == 1) && (version_parts[1] >= 19);
      }
      // compiled: >2.0, running support: 1.20-1.26, 2.0+
      return ((version_parts[0] == 1) && (version_parts[1] >= 20)) ||
             ((version_parts[0] == 2) && (version_parts[1] >= 0));
  }
}

bool IsNumpyVersionValid() {
  // use static bool to cache the result of NumpyVersionValid
  static bool is_valid = NumpyVersionValid();
  return is_valid;
}

void RegisterNumpyTypes() {
  if (!IsNumpyVersionValid()) {
    MS_LOG(INFO) << "The Numpy bfloat16 data type is not supported now, current Numpy version :"
                 << GetNumpyRunningVersion()
                 << ", Numpy version when the mindspore is compiled:" << GetNumpyCompiledVersion();
    return;
  }
  if (!RegisterNumpyType<bfloat16>()) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    MS_LOG(EXCEPTION) << "Failed to register BFloat16 type!";
  }
}
}  // namespace np_dtypes

int GetBFloat16NpDType() { return np_dtypes::NpTypeDescr<bfloat16>::Dtype(); }

bool IsCustomNumpyTypeValid(bool show_warning) {
  if (!np_dtypes::IsNumpyVersionValid()) {
    if (show_warning) {
      MS_LOG(WARNING) << "The Numpy bfloat16 data type is supported only when the "
                      << "current Numpy version is not less than the version when the mindspore "
                      << "is compiled, and the major versions must be same, "
                      << "but got current Numpy version :" << np_dtypes::GetNumpyRunningVersion()
                      << ", Numpy version when the mindspore is compiled:" << np_dtypes::GetNumpyCompiledVersion();
    }
    return false;
  }
  return true;
}

void RegNumpyTypes(py::module *m) {
  np_dtypes::RegisterNumpyTypes();
  auto m_sub = m->def_submodule("np_dtypes", "types of numpy");
  m_sub.add_object("bfloat16", reinterpret_cast<PyObject *>(np_dtypes::NpTypeDescr<bfloat16>::TypePtr()));
  (void)m_sub.def("np_dtype_valid", &IsCustomNumpyTypeValid, "Check whether custom numpy types are valid");
}
}  // namespace mindspore

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif
