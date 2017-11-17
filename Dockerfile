FROM alpine:3.6
MAINTAINER Rosco Pecoltran <https://github.com/roscopeoltran>

# Buidl arguments
ARG GOSU_VERSION=${GOSU_VERSION:-"1.10"}
ARG BAZEL_VERSION=${BAZEL_VERSION:-"0.7.0"}
ARG TENSORFLOW_VERSION=${TENSORFLOW_VERSION:-1.4.0"}
ARG APK_RUNTIME=${APK_RUNTIME:-"python3 python3-tkinter py3-numpy py3-numpy-f2py freetype libpng libjpeg-turbo imagemagick graphviz git"}
ARG APK_BUILD=${APK_BUILD:-"bash cmake curl freetype-dev nano g++ libjpeg-turbo-dev libpng-dev linux-headers make musl-dev openblas-dev \
                            openjdk8 patch perl python3-dev py-numpy-dev rsync sed swig zip"}

# Global env variables
ENV JAVA_HOME=/usr/lib/jvm/java-1.8-openjdk \
    LOCAL_RESOURCES=2048,.5,1.0

# Install Gosu to /usr/local/bin/gosu
ADD https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-amd64 /usr/local/sbin/gosu

# Install runtime dependencies & create runtime user
RUN chmod +x /usr/local/sbin/gosu \
    && apk add --no-cache --no-progress ${APK_RUNTIME} \
    && apk add --no-cache --virtual=.build-deps ${APK_BUILD} \
    && : prepare for building TensorFlow \
    && : install wheel python module \
    && cd /tmp \
    && pip3 install --no-cache-dir wheel \
    && : \
    && : add python symlink to avoid python detection error \
    && $(cd /usr/bin && ln -s python3 python) \
    && : install Bazel to build TensorFlow \
    && curl -SLO https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-dist.zip \
    && mkdir bazel-${BAZEL_VERSION} \
    && unzip -qd bazel-${BAZEL_VERSION} bazel-${BAZEL_VERSION}-dist.zip \
    && cd bazel-${BAZEL_VERSION} \
    && : add -fpermissive compiler option to avoid compilation failure \
    && sed -i -e '/"-std=c++0x"/{h;s//"-fpermissive"/;x;G}' tools/cpp/cc_configure.bzl \
    && : add '#include <sys/stat.h>' to avoid mode_t type error \
    && sed -i -e '/#endif  \/\/ COMPILER_MSVC/{h;s//#else/;G;s//#include <sys\/stat.h>/;G;}' third_party/ijar/common.h \
    && bash compile.sh \
    && cp -p output/bazel /usr/bin/ \
    && : \
    && : build TensorFlow pip package \
    && cd /tmp \
    && curl -SL https://github.com/tensorflow/tensorflow/archive/v${TENSORFLOW_VERSION}.tar.gz \
        | tar xzf - \
    && cd tensorflow-${TENSORFLOW_VERSION} \
    && : musl-libc does not have "secure_getenv" function \
    && sed -i -e '/JEMALLOC_HAVE_SECURE_GETENV/d' third_party/jemalloc.BUILD \
    && PYTHON_BIN_PATH=/usr/bin/python \
        PYTHON_LIB_PATH=/usr/lib/python3.6/site-packages \
        CC_OPT_FLAGS="-march=native" \
        TF_NEED_JEMALLOC=1 \
        TF_NEED_GCP=0 \
        TF_NEED_HDFS=0 \
        TF_NEED_S3=0 \
        TF_ENABLE_XLA=0 \
        TF_NEED_GDR=0 \
        TF_NEED_VERBS=0 \
        TF_NEED_OPENCL=0 \
        TF_NEED_CUDA=0 \
        TF_NEED_MPI=0 \
        bash configure \
    && bazel build -c opt --local_resources ${LOCAL_RESOURCES} //tensorflow/tools/pip_package:build_pip_package \
    && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
    && : \
    && : install python modules including TensorFlow \
    && cd \
    && pip3 install --no-cache-dir /tmp/tensorflow_pkg/tensorflow-${TENSORFLOW_VERSION}-cp36-cp36m-linux_x86_64.whl \
    && pip3 install --no-cache-dir pandas scipy jupyter \
    && pip3 install --no-cache-dir scikit-learn matplotlib Pillow \
    && pip3 install --no-cache-dir google-api-python-client \
    && : \
    && : clean up unneeded packages and files \
    && apk del .build-deps \
    && adduser -D app -h /data -s /bin/sh \
    && rm -f /usr/bin/bazel \
    && rm -rf /tmp/* /root/.cache

RUN jupyter notebook --generate-config --allow-root \
    && sed -i -e "/c\.NotebookApp\.ip/a c.NotebookApp.ip = '*'" \
        -e "/c\.NotebookApp\.open_browser/a c.NotebookApp.open_browser = False" \
            /root/.jupyter/jupyter_notebook_config.py
RUN ipython profile create \
    && sed -i -e "/c\.InteractiveShellApp\.matplotlib/a c.InteractiveShellApp.matplotlib = 'inline'" \
            /root/.ipython/profile_default/ipython_kernel_config.py

ADD shared/docker/scripts/init.sh /usr/local/bin/init.sh
RUN chmod u+x /usr/local/bin/init.sh
EXPOSE 8888
CMD ["/usr/local/bin/init.sh"]
