/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#ifndef AI_CPU_RANDOM_DISTRIBUTIONS_H
#define AI_CPU_RANDOM_DISTRIBUTIONS_H

#include <securec.h>
#include <string.h>
#include <cmath>
#include "utils.h"
#include "utils/philox_random.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
namespace random {
/* A class that generates unit normal distribution random numbers from the
 underlying random integer generator.
 Arguments:
   Generator: a generator type that returns a number of uint32 upon each
              each invocation. It needs to define kResultElementCount for the
              sample count for each invocation, and ResultType for actual
              returned sample type.
   RealInputType: the data type of the real numbers that will be returned by the
             distribution. This could be either half or float or double for
             now.
 This class is meant to be implemented through specialization. The default
 is not defined by design. */
template <class Generator, typename RealType>
class NormalDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// There's nothing to gain from working in half internally.
template <class Generator>
class NormalDistribution<Generator, Eigen::half> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<Eigen::half, kResultElementCount>;

  void operator()(Generator *gen, const Eigen::half *input, Eigen::half *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      BoxMullerFloat(sample[i], sample[i + 1], &f[0], &f[1]);
      output[i] = static_cast<Eigen::half>(f[0]);
      count++;
      if (count == size) {
        return;
      }
      output[i + 1] = static_cast<Eigen::half>(f[1]);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator>
class NormalDistribution<Generator, float> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<float, kResultElementCount>;

  void operator()(Generator *gen, const float *input, float *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      BoxMullerFloat(sample[i], sample[i + 1], &f[0], &f[1]);
      output[i] = f[0];
      count++;
      if (count == size) {
        return;
      }
      output[i + 1] = f[1];
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator>
class NormalDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount / 2;
  using ResultType = Array<double, kResultElementCount>;

  void operator()(Generator *gen, const double *input, double *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    double f[2];
    int count = 0;
    typename Generator::ResultType sample = (*gen)();
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      // For the double type, the algorithm requires four inputs and produces two outputs
      BoxMullerDouble(sample[i2], sample[i2 + 1], sample[i2 + 2], sample[i2 + 3], &f[0], &f[1]);
      output[i] = f[0];
      count++;
      if (count == size) {
        return;
      }
      output[i + 1] = f[1];
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

/* A class that generates unit tensor inut bernoulli distribution random numbers from the
 underlying random integer generator.
 Arguments:
   Generator: a generator type that returns a number of uint32 upon each
              each invocation. It needs to define kResultElementCount for the
              sample count for each invocation, and ResultType for actual
              returned sample type.
   RealInputType: the data type of the real numbers that will be returned by the
             distribution. This could be either half or float or double for
             now.
 This class is meant to be implemented through specialization. The default
 is not defined by design. */
template <class Generator, typename UnrealOutputType, typename RealInputType>
class BernoulliTensorDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// There's nothing to gain from working in half internally.
template <class Generator, typename UnrealOutputType>
class BernoulliTensorDistribution<Generator, UnrealOutputType, Eigen::half> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<Eigen::half, kResultElementCount>;

  void operator()(Generator *gen, const Eigen::half *input, UnrealOutputType *output, const int64_t &size,
                  bool *ptr_flag) {
    const float epsilon = 1.0e-7f;
    const Eigen::half prob_up_limit = static_cast<Eigen::half>(1.0);
    const Eigen::half prob_down_limit = static_cast<Eigen::half>(0.0);
    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      f[0] = Uint32ToFloat(sample[i]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (input[i] > prob_up_limit || input[i] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(static_cast<Eigen::half>(f[0]) <= input[i]);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      if (input[i + 1] > prob_up_limit || input[i + 1] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i + 1] = static_cast<UnrealOutputType>(static_cast<Eigen::half>(f[1]) <= input[i + 1]);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator, typename UnrealOutputType>
class BernoulliTensorDistribution<Generator, UnrealOutputType, float> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<float, kResultElementCount>;

  void operator()(Generator *gen, const float *input, UnrealOutputType *output, const int64_t &size, bool *ptr_flag) {
    const float epsilon = 1.0e-7f;
    const float prob_up_limit = 1.0;
    const float prob_down_limit = 0.0;
    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      f[0] = Uint32ToFloat(sample[i]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (input[i] > prob_up_limit || input[i] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(f[0] <= input[i]);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      if (input[i + 1] > prob_up_limit || input[i + 1] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i + 1] = static_cast<UnrealOutputType>(f[1] <= input[i + 1]);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator, typename UnrealOutputType>
class BernoulliTensorDistribution<Generator, UnrealOutputType, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount / 2;
  using ResultType = Array<double, kResultElementCount>;

  void operator()(Generator *gen, const double *input, UnrealOutputType *output, const int64_t &size, bool *ptr_flag) {
    const double epsilon = 1.0e-7f;
    const double prob_up_limit = 1.0;
    const double prob_down_limit = 0.0;
    typename Generator::ResultType sample = (*gen)();
    double f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      // For the double type, the algorithm requires four inputs and produces two outputs
      f[0] = Uint64ToDouble(sample[i2], sample[i2 + 1]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (input[i] > prob_up_limit || input[i] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(f[0] <= input[i]);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint64ToDouble(sample[i2 + 2], sample[i2 + 3]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      if (input[i + 1] > prob_up_limit || input[i + 1] < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i + 1] = static_cast<UnrealOutputType>(f[1] <= input[i + 1]);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

/* A class that generates unit scalar inut bernoulli distribution random numbers from the
 underlying random integer generator.
 Arguments:
   Generator: a generator type that returns a number of uint32 upon each
              each invocation. It needs to define kResultElementCount for the
              sample count for each invocation, and ResultType for actual
              returned sample type.
   RealInputType: the data type of the real numbers that will be returned by the
             distribution. This could be either half or float or double for
             now.
 This class is meant to be implemented through specialization. The default
 is not defined by design. */
template <class Generator, typename UnrealOutputType, typename RealInputType>
class BernoulliScalarDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// There's nothing to gain from working in half internally.
template <class Generator, typename UnrealOutputType>
class BernoulliScalarDistribution<Generator, UnrealOutputType, Eigen::half> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<Eigen::half, kResultElementCount>;

  void operator()(Generator *gen, const Eigen::half *input, UnrealOutputType *output, const int64_t &size,
                  bool *ptr_flag) {
    const float epsilon = 1.0e-7f;
    const Eigen::half prob_up_limit = static_cast<Eigen::half>(1.0);
    const Eigen::half prob_down_limit = static_cast<Eigen::half>(0.0);
    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      f[0] = Uint32ToFloat(sample[i]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (*input > prob_up_limit || *input < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(static_cast<Eigen::half>(f[0]) <= *input);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      output[i + 1] = static_cast<UnrealOutputType>(static_cast<Eigen::half>(f[1]) <= *input);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator, typename UnrealOutputType>
class BernoulliScalarDistribution<Generator, UnrealOutputType, float> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<float, kResultElementCount>;

  void operator()(Generator *gen, const float *input, UnrealOutputType *output, const int64_t &size, bool *ptr_flag) {
    const float epsilon = 1.0e-7f;
    const float prob_up_limit = 1.0;
    const float prob_down_limit = 0.0;
    typename Generator::ResultType sample = (*gen)();
    float f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      f[0] = Uint32ToFloat(sample[i]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (*input > prob_up_limit || *input < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(f[0] <= *input);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      output[i + 1] = static_cast<UnrealOutputType>(f[1] <= *input);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator, typename UnrealOutputType>
class BernoulliScalarDistribution<Generator, UnrealOutputType, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount / 2;
  using ResultType = Array<double, kResultElementCount>;

  void operator()(Generator *gen, const double *input, UnrealOutputType *output, const int64_t &size, bool *ptr_flag) {
    const double epsilon = 1.0e-7f;
    const double prob_up_limit = 1.0;
    const double prob_down_limit = 0.0;
    typename Generator::ResultType sample = (*gen)();
    double f[2];
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      // For the double type, the algorithm requires four inputs and produces two outputs
      f[0] = Uint64ToDouble(sample[i2], sample[i2 + 1]);
      if (f[0] < epsilon) {
        f[0] = epsilon;
      }
      if (*input > prob_up_limit || *input < prob_down_limit) {
        *ptr_flag = false;
        return;
      }
      output[i] = static_cast<UnrealOutputType>(f[0] <= *input);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint64ToDouble(sample[i2 + 2], sample[i2 + 3]);
      if (f[1] < epsilon) {
        f[1] = epsilon;
      }
      output[i + 1] = static_cast<UnrealOutputType>(f[1] <= *input);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};
/* A class that generates unit tensor inut uniform distribution random numbers from the
 underlying random integer generator.
 This class is meant to be implemented through specialization. The default
 is not defined by design. */
template <class Generator, typename RealType>
class UniformDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// There's nothing to gain from working in half internally.
template <class Generator>
class UniformDistribution<Generator, Eigen::half> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<Eigen::half, kResultElementCount>;

  void operator()(Generator *gen, Eigen::half *input, Eigen::half *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    typename Generator::ResultType sample = (*gen)();
    float f[2];
    const int two = 2;
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += two) {
      f[0] = Uint32ToFloat(sample[i]);
      output[i] = static_cast<Eigen::half>(f[0]);
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      output[i + 1] = static_cast<Eigen::half>(f[1]);
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator>
class UniformDistribution<Generator, float> {
 public:
  // The number of elements that will be returned. default value is 4 for philox
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount;
  using ResultType = Array<float, kResultElementCount>;

  void operator()(Generator *gen, float *input, float *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    typename Generator::ResultType sample = (*gen)();
    float f[2];
    const int two = 2;
    int count = 0;
    for (int32_t i = 0; i < kResultElementCount; i += two) {
      f[0] = Uint32ToFloat(sample[i]);
      output[i] = f[0];
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint32ToFloat(sample[i + 1]);
      output[i + 1] = f[1];
      count++;
      if (count == size) {
        return;
      }
    }
  }
};

template <class Generator>
class UniformDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int32_t kResultElementCount = Generator::kResultElementCount / 2;
  using ResultType = Array<double, kResultElementCount>;

  void operator()(Generator *gen, double *input, double *output, const int64_t &size, bool *ptr_flag) {
    (void)input;
    (void)ptr_flag;

    double f[2];
    const int two = 2;
    const int three = 3;
    int count = 0;
    typename Generator::ResultType sample = (*gen)();
    for (int32_t i = 0; i < kResultElementCount; i += two) {
      const int i2 = 2 * i;
      // For the double type, the algorithm requires four inputs and produces two outputs
      f[0] = Uint64ToDouble(sample[i2], sample[i2 + 1]);
      output[i] = f[0];
      count++;
      if (count == size) {
        return;
      }
      f[1] = Uint64ToDouble(sample[i2 + two], sample[i2 + three]);
      output[i + 1] = f[1];
      count++;
      if (count == size) {
        return;
      }
    }
  }
};
}  // namespace random
}  // namespace aicpu

#endif