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

package com.mindspore.lite;

import java.nio.ByteBuffer;

public class MSTensor {
    private long tensorPtr;

    public MSTensor() {
        this.tensorPtr = 0;
    }

    public MSTensor(long tensorPtr) {
        this.tensorPtr = tensorPtr;
    }

    public int[] getShape() {
        return this.getShape(this.tensorPtr);
    }

    public int getDataType() {
        return this.getDataType(this.tensorPtr);
    }

    public byte[] getByteData() {
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

    public String tensorName() {
        return this.tensorName(this.tensorPtr);
    }

    protected long getMSTensorPtr() {
        return tensorPtr;
    }

    private native int[] getShape(long tensorPtr);

    private native int getDataType(long tensorPtr);

    private native byte[] getByteData(long tensorPtr);

    private native long[] getLongData(long tensorPtr);

    private native int[] getIntData(long tensorPtr);

    private native float[] getFloatData(long tensorPtr);

    private native boolean setData(long tensorPtr, byte[] data, long dataLen);

    private native boolean setByteBufferData(long tensorPtr, ByteBuffer buffer);

    private native long size(long tensorPtr);

    private native int elementsNum(long tensorPtr);

    private native void free(long tensorPtr);

    private native String tensorName(long tensorPtr);
}
