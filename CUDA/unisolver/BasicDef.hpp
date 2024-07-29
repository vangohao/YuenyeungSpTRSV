#pragma once

#define ALWAYS_INLINE __attribute__((always_inline))

#ifdef UNI_ENABLE_GPU
#define UNI_DEVICE __device__
#define UNI_HOST_DEVICE __host__ __device__
#else
#define UNI_DEVICE
#define UNI_HOST_DEVICE
#endif
