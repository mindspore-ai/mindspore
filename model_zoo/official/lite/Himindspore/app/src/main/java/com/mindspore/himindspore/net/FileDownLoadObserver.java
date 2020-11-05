package com.mindspore.himindspore.net;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import io.reactivex.observers.DefaultObserver;
import okhttp3.ResponseBody;

public abstract class FileDownLoadObserver<T> extends DefaultObserver<T> {

    @Override
    public void onNext(T t) {
        onDownLoadSuccess(t);
    }

    @Override
    public void onError(Throwable e) {
        onDownLoadFail(e);
    }

    @Override
    public void onComplete() {
    }

    public abstract void onDownLoadSuccess(T t);

    public abstract void onDownLoadFail(Throwable throwable);

    public abstract void onProgress(int progress, long total);


    public File saveFile(ResponseBody responseBody, String destFileDir, String destFileName) throws IOException {
        InputStream is = null;
        byte[] buf = new byte[2048];
        int len = 0;
        FileOutputStream fos = null;
        try {
            is = responseBody.byteStream();
            final long total = responseBody.contentLength();
            long sum = 0;

            File dir = new File(destFileDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            File file = new File(dir, destFileName);
            fos = new FileOutputStream(file);
            while ((len = is.read(buf)) != -1) {
                sum += len;
                fos.write(buf, 0, len);
                final long finalSum = sum;
                onProgress((int) (finalSum * 100 / total), total);
            }
            fos.flush();

            return file;

        } finally {
            try {
                if (is != null) is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                if (fos != null) fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}