#ifndef GRID_VECTOR_TYPES_ARITH
#define GRID_VECTOR_TYPES_ARITH

#include "Grid_vector_types_base.h"
#include "veintrin.h"

#if GEN_SIMD_WIDTH != 2048
#error "intrinsics only implemented for 2048 byte simd width"
#endif

namespace Grid 
{
    namespace Opt = Grid::Optimization;
    
    // MULT
    // ====
    
    // V = V * V
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmuld_vvv( _ve_vld_vss(8, lhs->v.v), _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]);
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]);
        
        __vr ret_re = _ve_vfmsbd_vvvv( _ve_vfmuld_vvv( lr_im, rr_im ), lr_re, rr_re );
        __vr ret_im = _ve_vfmadd_vvvv( _ve_vfmuld_vvv( lr_im, rr_re ), lr_re, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    
    // V = S * V
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmuld_vsv(*lhs, _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = reinterpret_cast<float_t(&)[2]>(lhs)[0]; auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]);
        auto lr_im = reinterpret_cast<float_t(&)[2]>(lhs)[1]; auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]);
        
        __vr ret_re = _ve_vfmsbd_vvsv( _ve_vfmuld_vsv( lr_im, rr_im ), lr_re, rr_re );
        __vr ret_im = _ve_vfmadd_vvsv( _ve_vfmuld_vsv( lr_im, rr_re ), lr_re, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    
    // V = V * S
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmuld_vsv(*rhs, _ve_vld_vss(8, lhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs) 
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = reinterpret_cast<float_t(&)[2]>(rhs)[0]; 
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = reinterpret_cast<float_t(&)[2]>(rhs)[1]; 
        
        __vr ret_re = _ve_vfmsbd_vvsv( _ve_vfmuld_vsv( rr_im, lr_im ), rr_re, lr_re );
        __vr ret_im = _ve_vfmadd_vvsv( _ve_vfmuld_vsv( rr_im, lr_re ), rr_re, lr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    // MAC
    // ===
    
    // V = V * V
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmadd_vvvv( _ve_vld_vss(8, ret->v.v), _ve_vld_vss(8, lhs->v.v), _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    };
    
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]); auto ret_re = _ve_vld_vss(16, &ret->v.v[0]);
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]); auto ret_im = _ve_vld_vss(16, &ret->v.v[1]);
        
        ret_re = _ve_vfmsbd_vvvv( _ve_vfmsbd_vvvv( ret_re, lr_im, rr_im ), lr_re, rr_re );
        ret_im = _ve_vfmadd_vvvv( _ve_vfmadd_vvvv( ret_im, lr_im, rr_re ), lr_re, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    };
    
    // V = S * V
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmadd_vvsv( _ve_vld_vss(8, ret->v.v), *lhs, _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = reinterpret_cast<float_t(&)[2]>(lhs)[0]; auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]); auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = reinterpret_cast<float_t(&)[2]>(lhs)[1]; auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]); auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfmsbd_vvsv( _ve_vfmsbd_vvsv( ret_re, lr_im, rr_im ), lr_re, rr_re );
        ret_im = _ve_vfmadd_vvsv( _ve_vfmadd_vvsv( ret_im, lr_im, rr_re ), lr_re, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    // V = V * S
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfmadd_vvsv( _ve_vld_vss(8, ret->v.v), *lhs, _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs) 
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = reinterpret_cast<float_t(&)[2]>(rhs)[0]; auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = reinterpret_cast<float_t(&)[2]>(rhs)[1]; auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfmsbd_vvsv( _ve_vfmsbd_vvsv( ret_re, rr_im, lr_im ), rr_re, lr_re );
        ret_im = _ve_vfmadd_vvsv( _ve_vfmadd_vvsv( ret_im, rr_im, lr_re ), rr_re, lr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    // ADD
    // ===
    
    template <class num_t, class float_t>
    inline void add(Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfaddd_vvv( _ve_vld_vss(8, lhs->v.v), _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void add(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfaddd_vsv( *lhs, _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void add(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = reinterpret_cast<float_t(&)[2]>(lhs)[0]; auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]); auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = reinterpret_cast<float_t(&)[2]>(lhs)[1]; auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]); auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfaddd_vsv( lr_re, rr_re );
        ret_im = _ve_vfaddd_vsv( lr_im, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] ); 
    }

    template <class float_t>
    inline void add(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfaddd_vsv( *rhs, _ve_vld_vss(8, lhs->v.v) ), 8, ret->v.v );
    }
        
    template <class float_t>
    inline void add(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = reinterpret_cast<float_t(&)[2]>(rhs)[0]; auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = reinterpret_cast<float_t(&)[2]>(rhs)[1]; auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfaddd_vsv( rr_re, lr_re );
        ret_im = _ve_vfaddd_vsv( rr_im, lr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
    // SUB
    // ===
    
    template <class num_t, class float_t>
    inline void sub(Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfsubd_vvv( _ve_vld_vss(8, lhs->v.v), _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void sub(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfsubd_vsv( *lhs, _ve_vld_vss(8, rhs->v.v) ), 8, ret->v.v );
    }
    
    template <class float_t>
    inline void sub(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = reinterpret_cast<float_t(&)[2]>(lhs)[0]; auto rr_re = _ve_vld_vss(16, &rhs->v.v[0]); auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = reinterpret_cast<float_t(&)[2]>(lhs)[1]; auto rr_im = _ve_vld_vss(16, &rhs->v.v[1]); auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfsubd_vsv( lr_re, rr_re );
        ret_im = _ve_vfsubd_vsv( lr_im, rr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] ); 
    }

    template <class float_t>
    inline void sub(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::r);
        
        _ve_vst_vss( _ve_vfsubd_vsv( *rhs, _ve_vld_vss(8, lhs->v.v) ), 8, ret->v.v );
    }
        
    template <class float_t>
    inline void sub(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs)
    {
        _ve_lvl(Opt::W<float_t>::c);
        
        auto lr_re = _ve_vld_vss(16, &lhs->v.v[0]); auto rr_re = reinterpret_cast<float_t(&)[2]>(rhs)[0]; auto ret_re = _ve_vld_vss(16, &ret->v.v[0]); 
        auto lr_im = _ve_vld_vss(16, &lhs->v.v[1]); auto rr_im = reinterpret_cast<float_t(&)[2]>(rhs)[1]; auto ret_im = _ve_vld_vss(16, &ret->v.v[1]); 
        
        ret_re = _ve_vfsubd_vsv( rr_re, lr_re );
        ret_im = _ve_vfsubd_vsv( rr_im, lr_im );
        
        _ve_vst_vss( ret_re, 16, &ret->v.v[0] );
        _ve_vst_vss( ret_im, 16, &ret->v.v[1] );
    }
    
}

#endif
