#pragma once
// Force Eigen to use MKL if Grid has been configured with --enable-mkl
#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#if defined __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifdef __ve__
    #warning "building for NEC SX-Aurora"
    
    // SX-Aurora doesn't allow alignment > 8 byte
    // Disable any compiler attributes to solve this independent from Eigen source
    #define __attribute__(n)

    // replace prefetch function for SX-Aurora
    #define __builtin_prefetch(addr) __builtin_vprefetch(addr, 1)

    #include <Grid/Eigen/Dense>

    #undef __attribute__
    #undef __builtin_prefetch
#else
    #include <Grid/Eigen/Dense>
#endif

#if defined __GNUC__
    #pragma GCC diagnostic pop
#endif
