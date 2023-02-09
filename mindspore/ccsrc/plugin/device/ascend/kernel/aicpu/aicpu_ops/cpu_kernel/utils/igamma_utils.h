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

#ifndef AICPU_KERNELS_NORMALIZED_IGAMMA_UTILS_H_
#define AICPU_KERNELS_NORMALIZED_IGAMMA_UTILS_H_

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

template <typename T>
T IgammaSeries(const T &ax, const T &x, const T &a, const T &enabled, const int &mode);

template <typename T>
T IgammacContinuedFraction(const T &ax, const T &x, const T &a, const T &enabled, const int &mode);

// Computes an approximation of the lgamma function.
template <typename T>
T Lgamma(const T &input);

// Computes an approximation of the digamma function.
template <typename T>
T Digamma(const T &input);

template <typename T>
T IgammaSingle(const T &a, const T &x);

// Computes an approximation of the incomplete gamma function.
template <typename T>
void Igamma(T *a, T *x, T *output, int size);

template <typename T>
T IgammaGradASingle(const T &a, const T &x);

/** an approximation of the derivative of the incomplete gamma function
 * with respect to a.
 */
template <typename T>
void IgammaGradA(T *a, T *x, T *output, int size);

template <typename T>
T IgammacSingle(const T &a, const T &x);

// Computes an approximation of the complementary incomplete gamma function.
template <typename T>
void Igammac(T *a, T *x, T *output, int size);

#endif