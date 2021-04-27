/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
package com.mindspore.common.utils;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class ImageUtils {

    private static final String TAG = "BitmapUtils";


    private static Bitmap loadBitmapFromView(View v, boolean isNeedCrop) {
        int w = v.getWidth();
        int h = v.getHeight();

        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);
        v.layout((int) v.getX(), (int) v.getY(),
                (int) v.getX() + w, (int) v.getY() + h);
        v.draw(c);
        return isNeedCrop ? Bitmap.createBitmap(bmp, 0, 0, w / 10 * 9, h / 10 * 9, null, false) : bmp;
    }


    private static Uri sharePic(Context context, Bitmap cachebmp, String child) {
        final File qrImage = new File(Environment.getExternalStorageDirectory(), child + ".jpg");
        if (qrImage.exists()) {
            qrImage.delete();
        }
        try {
            qrImage.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileOutputStream fOut = null;
        try {
            fOut = new FileOutputStream(qrImage);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        if (cachebmp == null) {
            return null;
        }
        cachebmp.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
        try {
            fOut.flush();
            fOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Uri photoUri = FileProvider.getUriForFile(
                context, context.getPackageName() + ".fileprovider", qrImage);
        return photoUri;
    }


    // Save the picture to the system album and refresh it.
    public static Uri saveToAlbum(Context context, View view, String child, boolean isNeedCrop) {
        Bitmap cachebmp = loadBitmapFromView(view, isNeedCrop);
        File file = null;
        String fileName = TextUtils.isEmpty(child) ? System.currentTimeMillis() + ".jpg" : child + ".jpg";
        File root = new File(Environment.getExternalStorageDirectory().getAbsoluteFile(), context.getPackageName());
        File dir = new File(root, "image");
        if (dir.mkdirs() || dir.isDirectory()) {
            file = new File(dir, fileName);
        }
        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            cachebmp.compress(Bitmap.CompressFormat.JPEG, 100, os);
            os.flush();

        } catch (FileNotFoundException e) {
            Log.e(TAG, e.getMessage());
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        } finally {
            try {
                if (os != null) {
                    os.close();
                }
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
        }
        if (file == null) {
            return null;
        }
        // Gallery refresh.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            String path = null;
            try {
                path = file.getCanonicalPath();
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
            MediaScannerConnection.scanFile(context, new String[]{path}, null,
                    (path1, uri) -> {
                        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                        mediaScanIntent.setData(uri);
                        context.sendBroadcast(mediaScanIntent);
                    });
        } else {
            String relationDir = file.getParent();
            File file1 = new File(relationDir);
            context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_MOUNTED, Uri.fromFile(file1.getAbsoluteFile())));
        }

        Uri photoUri = FileProvider.getUriForFile(
                context, context.getPackageName() + ".fileprovider", file);
        return photoUri;
    }

}
