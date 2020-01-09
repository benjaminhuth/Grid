    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/simd/Grid_generic.h

    Copyright (C) 2018

Author: Nils Meyer          <nils.meyer@ur.de>

    Copyright (C) 2015
    Copyright (C) 2017

Author: Antonin Portelli <antonin.portelli@me.com>
        Andrew Lawson    <andrew.lawson1991@gmail.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */


// Ensure compilation with correct compiler
#if !defined(__clang__) && !defined(__ve__)
#error "Must be compiled with clang on the NEC SX-Aurora"
#endif

#warning "Compile intrinsic layer for NEC SX-Aurora"

#define GEN_SIMD_WIDTH 4096

#include "Grid_generic_types.h"
#include "velintrin.h"

namespace Grid {
namespace Optimization {

    struct Vsplat{
        // Complex
        template <typename T>
        inline vec<T> operator()(const T a, const T b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 2)
        {
            out.v[i]   = a;
            out.v[i+1] = b;
        }

        return out;
        }

        // Real
        template <typename T>
        inline vec<T> operator()(const T a){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 1)
        {
            out.v[i] = a;
        }

        return out;
        }
    };

    struct Vstore{
        // Real
        template <typename T>
        inline void operator()(const vec<T> &a, T *D){
        *((vec<T> *)D) = a;
        }
    };

    struct Vstream{
        // Real
        template <typename T>
        inline void operator()(T * a, const vec<T> &b){
        *((vec<T> *)a) = b;
        }
    };

    struct Vset{
        // Complex
        template <typename T>
        inline vec<T> operator()(const std::complex<T> *a){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            out.v[2*i]   = a[i].real();
            out.v[2*i+1] = a[i].imag();
        }

        return out;
        }

        // Real
        template <typename T>
        inline vec<T> operator()(const T *a){
        vec<T> out;

        out = *((vec<T> *)a);

        return out;
        }
    };

    /////////////////////////////////////////////////////
    // Arithmetic operations
    /////////////////////////////////////////////////////
    struct Sum{
        // Complex/Real
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 1)
        {
            out.v[i] = a.v[i] + b.v[i];
        }

        return out;
        }
    };

    struct Sub{
        // Complex/Real
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 1)
        {
            out.v[i] = a.v[i] - b.v[i];
        }

        return out;
        }
    };

    struct Mult{
        // Real
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 1)
        {
            out.v[i] = a.v[i]*b.v[i];
        }

        return out;
        }
    };

#define cmul(a, b, c, i)\
c[i]   = a[i]*b[i]   - a[i+1]*b[i+1];\
c[i+1] = a[i]*b[i+1] + a[i+1]*b[i];

    struct MultRealPart{
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            out.v[2*i]   = a.v[2*i]*b.v[2*i];
            out.v[2*i+1] = a.v[2*i]*b.v[2*i+1];
        }
        return out;
        }
    };

    struct MaddRealPart{
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b, const vec<T> &c){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            out.v[2*i]   = a.v[2*i]*b.v[2*i] + c.v[2*i];
            out.v[2*i+1] = a.v[2*i]*b.v[2*i+1] + c.v[2*i+1];
        }
        return out;
        }
    };

    struct MultComplex{
        // Complex
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            cmul(a.v, b.v, out.v, 2*i);
        }

        return out;
        }
    };

#undef cmul

    struct Div{
        // Real
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::r, 1)
        {
            out.v[i] = a.v[i]/b.v[i];
        }

        return out;
        }
    };

#define conj(a, b, i)\
b[i]   = a[i];\
b[i+1] = -a[i+1];

    struct Conj{
        // Complex
        template <typename T>
        inline vec<T> operator()(const vec<T> &a){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            conj(a.v, out.v, 2*i);
        }

        return out;
        }
    };

#undef conj

#define timesmi(a, b, i)\
b[i]   = a[i+1];\
b[i+1] = -a[i];

    struct TimesMinusI{
        // Complex
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            timesmi(a.v, out.v, 2*i);
        }

        return out;
        }
    };

#undef timesmi

#define timesi(a, b, i)\
b[i]   = -a[i+1];\
b[i+1] = a[i];

    struct TimesI{
        // Complex
        template <typename T>
        inline vec<T> operator()(const vec<T> &a, const vec<T> &b){
        vec<T> out;

        VECTOR_FOR(i, W<T>::c, 1)
        {
            timesi(a.v, out.v, 2*i);
        }

        return out;
        }
    };

#undef timesi
    
    /////////////////////////////////////////////////////
    // operator() specialized for vec<double>
    /////////////////////////////////////////////////////
    
    
    template <>
    inline vec<double> Sum::operator()<double>(const vec<double> &a, const vec<double> &b){
        vec<double> out;
        
        auto a1 = _vel_vld_vssl(8, &a.v[0],   256);
        auto a2 = _vel_vld_vssl(8, &a.v[256], 256);
        
        auto b1 = _vel_vld_vssl(8, &b.v[0],   256);
        auto b2 = _vel_vld_vssl(8, &b.v[256], 256);
        
        auto res1 = _vel_vfaddd_vvvl(a1, b1, 256);
        auto res2 = _vel_vfaddd_vvvl(a2, b2, 256);
        
        _vel_vst_vssl(res1, 8, &out.v[0],   256);
        _vel_vst_vssl(res2, 8, &out.v[256], 256);
        
        return out;
    };
        
    template <>
    inline vec<double> Sub::operator()<double>(const vec<double> &a, const vec<double> &b){
        vec<double> out;
        
        auto a1 = _vel_vld_vssl(8, &a.v[0],   256);
        auto a2 = _vel_vld_vssl(8, &a.v[256], 256);
        
        auto b1 = _vel_vld_vssl(8, &b.v[0],   256);
        auto b2 = _vel_vld_vssl(8, &b.v[256], 256);
        
        auto res1 = _vel_vfsubd_vvvl(a1, b1, 256);
        auto res2 = _vel_vfsubd_vvvl(a2, b2, 256);
        
        _vel_vst_vssl(res1, 8, &out.v[0],   256);
        _vel_vst_vssl(res2, 8, &out.v[256], 256);
        
        return out;
    };
    
    template <>
    inline vec<double> Mult::operator()<double>(const vec<double> &a, const vec<double> &b){
        vec<double> out;
        
        auto a1 = _vel_vld_vssl(8, &a.v[0],   256);
        auto a2 = _vel_vld_vssl(8, &a.v[256], 256);
        
        auto b1 = _vel_vld_vssl(8, &b.v[0],   256);
        auto b2 = _vel_vld_vssl(8, &b.v[256], 256);
        
        auto res1 = _vel_vfmuld_vvvl(a1, b1, 256);
        auto res2 = _vel_vfmuld_vvvl(a2, b2, 256);
        
        _vel_vst_vssl(res1, 8, &out.v[0],   256);
        _vel_vst_vssl(res2, 8, &out.v[256], 256);
        
        return out;
    };
        
//     template <>
//     inline vec<double> MultRealPart::operator()<double>(const vec<double> &a, const vec<double> &b){
//         vec<double> out;
//         
//         return out;
//     }
    
//     template <>
//     inline vec<double> MaddRealPart::operator()<double>(const vec<double> &a, const vec<double> &b, const vec<double> &c){
//         vec<double> out;
//         
//         return out;
//     }
    
    template <>
    inline vec<double> MultComplex::operator()<double>(const vec<double> &a, const vec<double> &b){
        vec<double> out;
    
        auto a_re = _vel_vld_vssl(16, &a.v[0], 256);
        auto a_im = _vel_vld_vssl(16, &a.v[1], 256);
        
        auto b_re = _vel_vld_vssl(16, &b.v[0], 256);
        auto b_im = _vel_vld_vssl(16, &b.v[1], 256);
        
        auto res_re = _vel_vfmsbd_vvvvl( _vel_vfmuld_vvvl( a_im, b_im, 256 ), a_re, b_re, 256 );
        auto res_im = _vel_vfmadd_vvvvl( _vel_vfmuld_vvvl( a_im, b_re, 256 ), a_re, b_im, 256 );
        
        _vel_vst_vssl(res_re, 16, &out.v[0], 256);
        _vel_vst_vssl(res_im, 16, &out.v[1], 256);
        
        return out;
    }
    
//     template <>
//     inline vec<double> Div::operator()<double>(const vec<double> &a, const vec<double> &b){
//         vec<double> out;
//         
//         return out;
//     }
    
//     template <>
//     inline vec<double> Conj::operator()<double>(const vec<double> &a){
//         vec<double> out;
//         
//         return out;
//     }
    
//     template <>
//     inline vec<double> TimesMinusI::operator()<double>(const vec<double> &a, const vec<double> &b){
//         vec<double> out;
//         
//         return out;
//     }
    
//     template <>
//     inline vec<double> TimesI::operator()<double>(const vec<double> &a, const vec<double> &b){
//         vec<double> out;
//         
//         return out;
//     }
    
    /////////////////////////////////////////////////////
    // PrecisionChange imitated by converting to normal array
    /////////////////////////////////////////////////////

    struct PrecisionChange {
        static inline vech StoH (const vecf &a, const vecf &b) {
            vech ret;
#ifdef USE_FP16
            vech *ha = (vech *)&a;
            vech *hb = (vech *)&b;
            const int nf = W<float>::r;
            //      VECTOR_FOR(i, nf,1){ ret.v[i]    = ( (uint16_t *) &a.v[i])[1] ; }
            //      VECTOR_FOR(i, nf,1){ ret.v[i+nf] = ( (uint16_t *) &b.v[i])[1] ; }
            VECTOR_FOR(i, nf,1){ ret.v[i]    = ha->v[2*i+1]; }
            VECTOR_FOR(i, nf,1){ ret.v[i+nf] = hb->v[2*i+1]; }
#else
            assert(0);
#endif
            return ret;
        }
        static inline void  HtoS (const vech &h, vecf &sa,vecf &sb) {
#ifdef USE_FP16
            const int nf = W<float>::r;
            const int nh = W<uint16_t>::r;
            vech *ha = (vech *)&sa;
            vech *hb = (vech *)&sb;
            VECTOR_FOR(i, nf, 1){ sb.v[i]= sa.v[i] = 0; }
            //      VECTOR_FOR(i, nf, 1){ ( (uint16_t *) (&sa.v[i]))[1] = h.v[i];}
            //      VECTOR_FOR(i, nf, 1){ ( (uint16_t *) (&sb.v[i]))[1] = h.v[i+nf];}
            VECTOR_FOR(i, nf, 1){ ha->v[2*i+1]=h.v[i]; }
            VECTOR_FOR(i, nf, 1){ hb->v[2*i+1]=h.v[i+nf]; }
#else
            assert(0);
#endif
        }
        static inline vecf DtoS (const vecd &a, const vecd &b) {
            const int nd = W<double>::r;
            const int nf = W<float>::r;
            vecf ret;
            VECTOR_FOR(i, nd,1){ ret.v[i]    = a.v[i] ; }
            VECTOR_FOR(i, nd,1){ ret.v[i+nd] = b.v[i] ; }
            return ret;
        }
        static inline void StoD (const vecf &s, vecd &a, vecd &b) {
            const int nd = W<double>::r;
            VECTOR_FOR(i, nd,1){ a.v[i] = s.v[i] ; }
            VECTOR_FOR(i, nd,1){ b.v[i] = s.v[i+nd] ; }
        }
        static inline vech DtoH (const vecd &a, const vecd &b, const vecd &c, const vecd &d) {
            vecf sa,sb;
            sa = DtoS(a,b);
            sb = DtoS(c,d);
            return StoH(sa,sb);
        }
        static inline void HtoD (const vech &h, vecd &a, vecd &b, vecd &c, vecd &d) {
            vecf sa,sb;
            HtoS(h,sa,sb);
            StoD(sa,a,b);
            StoD(sb,c,d);
        }
    };

    //////////////////////////////////////////////
    // Exchange support
    struct Exchange{

        template <typename T,int n>
        static inline void ExchangeN(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2)
        {
            assert(("exchange shouldn't be called in implementation for SX-Aurora", 0));
        }
        
        template <typename T>
        static inline void Exchange0(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,0>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange1(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,1>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange2(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,2>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange3(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,3>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange4(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,4>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange5(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,5>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange6(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,6>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange7(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,7>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange8(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,8>(out1,out2,in1,in2);
        };
        template <typename T>
        static inline void Exchange9(vec<T> &out1, vec<T> &out2, const vec<T> &in1, const vec<T> &in2){
        ExchangeN<T,9>(out1,out2,in1,in2);
        };
    };


    //////////////////////////////////////////////
    // Some Template specialization
    #define perm(a, b, n, w)\
    unsigned int _mask = w >> (n + 1);\
    VECTOR_FOR(i, w, 1)\
    {\
        b[i] = a[i^_mask];\
    }

    #define DECL_PERMUTE_N(n)\
    template <typename T>\
    static inline vec<T> Permute##n(vec<T> in) {\
        assert(("permute shouldn't be called in implementation for SX-Aurora", 0));\
        return in;\
    }

    struct Permute{
        DECL_PERMUTE_N(0);
        DECL_PERMUTE_N(1);
        DECL_PERMUTE_N(2);
        DECL_PERMUTE_N(3);
        DECL_PERMUTE_N(4);
        DECL_PERMUTE_N(5);
        DECL_PERMUTE_N(6);
        DECL_PERMUTE_N(7);
        DECL_PERMUTE_N(8);
        DECL_PERMUTE_N(9);
    };

    #undef perm
    #undef DECL_PERMUTE_N

    #define rot(a, b, n, w)\
    VECTOR_FOR(i, w, 1)\
    {\
        b[i] = a[(i + n)%w];\
    }

    struct Rotate{

        template <int n, typename T> static inline vec<T> tRotate(const vec<T> &in){
            return rotate(in, n);
        }

        template <typename T>
        static inline vec<T> rotate(const vec<T> &in, const int n){
            assert(("rotate shouldn't be called in implementation for SX-Aurora", 0));
            return in;
        }
        
        template <typename T>
        static inline vec<T> splitRotate(const vec<T> &in, const int n, const int split)
        {
            vec<T> out;
            
            auto w = W<T>::r / split;
            
            VECTOR_FOR(i, W<T>::r, 1)
            {
                out.v[i] = in.v[(i+n) % w + (i/w)*w];
            }
            
            return out;
        }
        
        template <typename T>
        static inline vec<T> splitRotateComplex(const vec<T> &in, const int n, const int split)
        {
            return splitRotate(in, 2*n, split);
        }
    };
    
    // method for real numbers not so easy with intrinsics and 4096 bytes VL, and also this obviously does perform badly
//     template <> 
//     inline vec<double> Rotate::splitRotateComplex<double>(const vec<double> &in, const int n, const int split)
//     {
//         // Bit-magic to compute logarithms
//         const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };
//         const static uint16_t de_bruijn = 2479;
//         const static uint16_t max16bit = 65535;
//         
//         uint16_t logs = table[ uint16_t(split * de_bruijn) >> 12 ];
//         
//         uint16_t w = 256 >> logs;
//         
//         uint16_t logw = table[ uint16_t(w * de_bruijn) >> 12 ];
//         
//         uint16_t mask = max16bit << logw;
//         
//         // Compute split rotate
//         
//         // Implement index calculation: out[i] = in[(i+r) - ((i+r) & mask) + (i & mask)];
//         auto indices = _vel_vseq_vl(256);
//         
//         auto i_plus_r =          _vel_vaddul_vsvl(n, indices, 256);
//         auto i_plus_r_and_mask = _vel_vand_vsvl(mask, i_plus_r, 256);
//         auto i_and_mask =        _vel_vand_vsvl(mask, indices, 256);
//         
//         indices = _vel_vsubul_vvvl(i_plus_r, _vel_vaddul_vvvl(i_plus_r_and_mask, i_and_mask, 256), 256);
//         
//         // compute addresses
//         auto start_address = (unsigned long)&in.v[0];
//         
//         auto real_addresses = _vel_vaddul_vsvl(start_address, _vel_vmulul_vsvl(sizeof(double), indices, 256), 256);
//         auto imag_addresses = _vel_vaddul_vsvl(sizeof(double), real_addresses, 256);
//         
//         // gathers
//         vec<double> out;
//         
//         auto out_real = _vel_vgt_vvssl(real_addresses, start_address, start_address+512, 256);
//         auto out_imag = _vel_vgt_vvssl(imag_addresses, start_address, start_address+512, 256);
//         
//         _vel_vst_vssl(out_real, 16, &out.v[0], 256);
//         _vel_vst_vssl(out_imag, 16, &out.v[1], 256);
//         
//         return out;
//     }
    
    #undef rot

    #define acc(v, a, off, step, n)\
    for (unsigned int i = off; i < n; i += step)\
    {\
        a += v[i];\
    }

    template <typename Out_type, typename In_type>
    struct Reduce{
        //Need templated class to overload output type
        //General form must generate error if compiled
        inline Out_type operator()(const In_type &in){
        printf("Error, using wrong Reduce function\n");
        exit(1);
        return 0;
        }
    };

    //Complex float Reduce
    template <>
    inline Grid::ComplexF Reduce<Grid::ComplexF, vecf>::operator()(const vecf &in){
        float a = 0.f, b = 0.f;

        acc(in.v, a, 0, 2, W<float>::r);
        acc(in.v, b, 1, 2, W<float>::r);

        return Grid::ComplexF(a, b);
    }

    //Real float Reduce
    template<>
    inline Grid::RealF Reduce<Grid::RealF, vecf>::operator()(const vecf &in){
        float a = 0.;

        acc(in.v, a, 0, 1, W<float>::r);

        return a;
    }

    //Complex double Reduce
    template<>
    inline Grid::ComplexD Reduce<Grid::ComplexD, vecd>::operator()(const vecd &in){        
        auto re = _vel_vld_vssl(16, &in.v[0], 256);
        auto im = _vel_vld_vssl(16, &in.v[1], 256);
        
        double sre = _vel_lvsd_svs(_vel_vfsumd_vvl(re, 256), 0);
        double sim = _vel_lvsd_svs(_vel_vfsumd_vvl(im, 256), 0);

        return std::complex<double>(sre,sim);
    }

    //Real double Reduce
    template<>
    inline Grid::RealD Reduce<Grid::RealD, vecd>::operator()(const vecd &in){
        auto s1 = _vel_vld_vssl(8, &in.v[0],   256);
        auto s2 = _vel_vld_vssl(8, &in.v[256], 256);

        return _vel_lvsd_svs(_vel_vfsumd_vvl(s1, 256), 0) + _vel_lvsd_svs(_vel_vfsumd_vvl(s2, 256), 0);
    }

    //Integer Reduce
    template<>
    inline Integer Reduce<Integer, veci>::operator()(const veci &in){
        Integer a = 0;

        acc(in.v, a, 0, 1, W<Integer>::r);

        return a;
    }

    #undef acc  // EIGEN compatibility
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Here assign types

    typedef Optimization::vech SIMD_Htype; // Reduced precision type
    typedef Optimization::vecf SIMD_Ftype; // Single precision type
    typedef Optimization::vecd SIMD_Dtype; // Double precision type Optimization::vec<double> { __vr }
    typedef Optimization::veci SIMD_Itype; // Integer type

    // prefetch utilities
    inline void v_prefetch0(int size, const char *ptr){};
    inline void prefetch_HINT_T0(const char *ptr){};

    // Function name aliases
    typedef Optimization::Vsplat   VsplatSIMD;
    typedef Optimization::Vstore   VstoreSIMD;
    typedef Optimization::Vset     VsetSIMD;
    typedef Optimization::Vstream  VstreamSIMD;
    template <typename S, typename T> using ReduceSIMD = Optimization::Reduce<S,T>;

    // Arithmetic operations
    typedef Optimization::Sum         SumSIMD;
    typedef Optimization::Sub         SubSIMD;
    typedef Optimization::Div         DivSIMD;
    typedef Optimization::Mult        MultSIMD;
    typedef Optimization::MultComplex MultComplexSIMD;
    typedef Optimization::MultRealPart MultRealPartSIMD;
    typedef Optimization::MaddRealPart MaddRealPartSIMD;
    typedef Optimization::Conj        ConjSIMD;
    typedef Optimization::TimesMinusI TimesMinusISIMD;
    typedef Optimization::TimesI      TimesISIMD;
}
