FROM alpine:latest
MAINTAINER Rosco Pecoltran <https://github.com/roscopecoltran>

ARG BLAS_VERSION=${BLAS_VERSION:-"3.6.0"}
ARG LAPACK_VERSION=${LAPACK_VERSION:-"3.6.1"}
ARG APK_RUNTIME=${APK_RUNTIME:-"wget py-pip python-dev musl-dev linux-headers g++ make gfortran ca-certificates"}

WORKDIR /fasttext

COPY ./fasttext_app.py .

RUN apk add --no-cache --update ${APK_RUNTIME} \
	&& wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip \
	&& ln -s /usr/include/locale.h /usr/include/xlocale.h \
	&& mkdir -p /tmp/build \
	&& cd /tmp/build \
	&& wget http://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz \
	&& wget http://www.netlib.org/lapack/lapack-${LAPACK_VERSION}.tgz \
	&& tar xzf blas-${BLAS_VERSION}.tgz \
	&& tar xzf lapack-${LAPACK_VERSION}.tgz \
	&& cd /tmp/build/BLAS-${BLAS_VERSION}/ \
	&& gfortran -O3 -std=legacy -m64 -fno-second-underscore -fPIC -c *.f \
	&& ar r libfblas.a *.o \
	&& ranlib libfblas.a \
	&& mv libfblas.a /tmp/build/. \
	&& cd /tmp/build/lapack-${LAPACK_VERSION}/ \
	&& sed -e "s/frecursive/fPIC/g" -e "s/ \.\.\// /g" -e "s/^CBLASLIB/\#CBLASLIB/g" make.inc.example > make.inc \
	&& make lapacklib \
	&& make clean \
	&& mv liblapack.a /tmp/build/. \
	&& cd /fasttext \
	&& export BLAS=/tmp/build/libfblas.a \
	&& export LAPACK=/tmp/build/liblapack.a \
	&& pip --no-cache-dir install --upgrade pip cython \
	&& pip --no-cache-dir install --upgrade flask gensim fasttext spacy \
	&& rm -rf /tmp/build

EXPOSE 5000

CMD ["python", "./fasttext_app.py", "-p", "5000", "-l", "en", "-m", "wiki.en.bin"]