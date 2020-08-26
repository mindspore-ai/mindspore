/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.lite;

import android.util.Log;

import java.nio.ByteBuffer;

public class MSTensor {
    private long tensorPtr;

    public MSTensor() {
        this.tensorPtr = 0;
    }

    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
    }

    public boolean init(int dataType, int[] shape) {
        this.tensorPtr = createMSTensor(dataType, shape, shape.length);
        return this.tensorPtr != 0;
    }

    public int[] getShape() {
        return this.getShape(this.tensorPtr);
    }

    public void setShape(int[] shape) {
        this.setShape(this.tensorPtr, shape, shape.length);
    }

    public int getDataType() {
        return this.getDataType(this.tensorPtr);
    }

    public void setDataType(int dataType) {
        this.setDataType(this.tensorPtr, dataType);
    }

    public byte[] getBtyeData() {
        return this.getByteData(this.tensorPtr);
    }

    public float[] getFloatData() {
        return this.getFloatData(this.tensorPtr);
    }

    public int[] getIntData() {
        return this.getIntData(this.tensorPtr);
    }

    public long[] getLongData() {
        return this.getLongData(this.tensorPtr);
    }

    public void setData(byte[] data) {
        this.setData(this.tensorPtr, data, data.length);
    }

    public void setData(ByteBuffer data) {
        this.setByteBufferData(this.tensorPtr, data);
    }

    public long size() {
        return this.size(this.tensorPtr);
    }

    public int elementsNum() {
        return this.elementsNum(this.tensorPtr);
    }

    public void free() {
        this.free(this.tensorPtr);
        this.tensorPtr = 0;
    }

    private float[] decodeBytes(byte[] bytes) {
        if (bytes.length % 4 != 0) {
            Log.e("MS_LITE", "Length of bytes should be multi of 4 ");
            return null;
        }
        int size = bytes.length / 4;
        float[] ret = new float[size];
        for (int i = 0; i < size; i = i + 4) {
            int accNum = 0;
            accNum = accNum | (bytes[i] & 0xff) << 0;
            accNum = accNum | (bytes[i + 1] & 0xff) << 8;
            accNum = accNum | (bytes[i + 2] & 0xff) << 16;
            accNum = accNum | (bytes[i + 3] & 0xff) << 24;
            ret[i / 4] = Float.intBitsToFloat(accNum);
        }
        return ret;
    }

    private native long createMSTensor(int dataType, int[] shape, int shapeLen);

    private native int[] getShape(long tensorPtr);

    private native boolean setShape(long tensorPtr, int[] shape, int shapeLen);

    private native int getDataType(long tensorPtr);

    private native boolean setDataType(long tensorPtr, int dataType);

    private native byte[] getByteData(long tensorPtr);

    private native long[] getLongData(long tensorPtr);

    private native int[] getIntData(long tensorPtr);

    private native float[] getFloatData(long tensorPtr);

    private native boolean setData(long tensorPtr, byte[] data, long dataLen);

    private native boolean setByteBufferData(long tensorPtr, ByteBuffer buffer);

    private native long size(long tensorPtr);

    private native int elementsNum(long tensorPtr);

    private native void free(long tensorPtr);
}
