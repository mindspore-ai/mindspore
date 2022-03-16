/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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

package com.mindspore.flclient.compression;

import java.util.List;

/**
 * Compress Weight Bean
 *
 * @since 2021-12-21
 */
public class CompressWeight {
    private String weightFullname;
    private List<Byte> compressData;
    private float minValue;
    private float maxValue;

    public CompressWeight() {
    }

    public CompressWeight(String weightFullname, List<Byte> compressData, float minValue, float maxValue) {
        this.weightFullname = weightFullname;
        this.compressData = compressData;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    public String getWeightFullname() {
        return weightFullname;
    }

    public void setWeightFullname(String weightFullname) {
        this.weightFullname = weightFullname;
    }

    public List<Byte> getCompressData() {
        return compressData;
    }

    public void setCompressData(List<Byte> compressData) {
        this.compressData = compressData;
    }

    public float getMinValue() {
        return minValue;
    }

    public void setMinValue(float minValue) {
        this.minValue = minValue;
    }

    public float getMaxValue() {
        return maxValue;
    }

    public void setMaxValue(float maxValue) {
        this.maxValue = maxValue;
    }

    @Override
    public String toString() {
        return "CompressWeight{" +
                "weightFullname='" + weightFullname + '\'' +
                ", compressData=" + compressData +
                ", minValue=" + minValue +
                ", maxValue=" + maxValue +
                '}';
    }
}
