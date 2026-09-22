#pragma once

#include <cstddef>

using cudaStream_t = void*;
using cudaError_t = int;
using cudaHostFn_t = void (*)(void*);
using cudaStreamCaptureStatus = int;

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaStreamCaptureStatus cudaStreamCaptureStatusNone = 0;

inline cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void*) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus* status, unsigned long long*) {
  *status = cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }
