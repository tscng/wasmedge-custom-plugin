# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM rust:1.91.1 AS buildbase
WORKDIR /src
RUN <<EOT bash
    set -ex
    apt-get update
    apt-get install -y \
        git \
        clang
    rustup target add wasm32-wasip1
EOT

FROM buildbase AS build
COPY Cargo.toml .
COPY src ./src
# Build the Wasm binary
RUN RUSTFLAGS="--cfg wasmedge --cfg tokio_unstable" cargo build --target wasm32-wasip1 --release

FROM scratch
ENTRYPOINT [ "/app.wasm" ]
COPY --link --from=build /src/target/wasm32-wasip1/release/app.wasm /app.wasm