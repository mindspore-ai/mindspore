package com.mindspore.lite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Model {
    static {
        System.loadLibrary("mindspore-lite-jni");
    }

    private long modelPtr;

    public Model() {
        this.modelPtr = 0;
    }

    public long getModelPtr() {
        return modelPtr;
    }

    public void setModelPtr(long modelPtr) {
        this.modelPtr = modelPtr;
    }

    public boolean loadModel(Context context, String modelName) {
        FileInputStream fis = null;
        AssetFileDescriptor fileDescriptor = null;
        boolean ret = false;
        try {
            fileDescriptor = context.getAssets().openFd(modelName);
            fis = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = fis.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLen = fileDescriptor.getDeclaredLength();
            MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLen);
            this.modelPtr = loadModel(buffer);
            ret = this.modelPtr != 0;
        } catch (IOException e) {
            this.modelPtr = 0;
            Log.e("MS_LITE", "Load model failed: " + e.getMessage());
            ret = false;
        } finally {
            if (null != fis) {
                try {
                    fis.close();
                } catch (IOException e) {
                    Log.e("MS_LITE", "Close file failed: " + e.getMessage());
                }
            }
            if (null != fileDescriptor) {
                try {
                    fileDescriptor.close();
                } catch (IOException e) {
                    Log.e("MS_LITE", "Close fileDescriptor failed: " + e.getMessage());
                }
            }
        }
        return ret;
    }

    public void free() {
        this.free(this.modelPtr);
        this.modelPtr = 0;
    }

    private native long loadModel(MappedByteBuffer buffer);

    private native void free(long modelPtr);
}
