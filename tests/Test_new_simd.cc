/*
 * Q&A:
 * 
 * Q: Why new test for SIMD?
 * A: For performance reasons on SX-Aurora, there is 
 *    introduced a new SIMD layer, which does not rely 
 *    on Grid_generic.h, but directely implements the 
 *    functions as memberfunctions of Grid_vector_types.h 
 *    (to workaround obviously bad inlining for nc++)
 */ 

#include <random>
#include <vector>
#include <functional>

#include <Grid/Grid.h>

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);

template<typename T>
void random_fill(std::vector<T> &v)
{
    std::generate(v.begin(), v.end(), [&](){ return uniform_dist(rng); });
}

template<typename T>
void random_fill(std::vector<std::complex<T>> &v)
{
    std::generate(v.begin(), v.end(), [&](){ return std::complex<T>(uniform_dist(rng), uniform_dist(rng)); });
}

template<class vector_t, class scalar_t>
void test_unary(std::function<void(scalar_t &, const scalar_t &)> scalar_func,
                std::function<void(vector_t &, const vector_t &)> vector_func,
                std::string name)
{
    std::cout << "Testing " << name << ": ";
    
    int Nsimd = vector_t::Nsimd();
    
    std::vector<scalar_t> result(Nsimd);
    std::vector<scalar_t> reference(Nsimd);
    std::vector<scalar_t> input(Nsimd); random_fill(input);
    vector_t vinput, vresult;
    
    for(int i=0; i<Nsimd; ++i)
        scalar_func(reference[i], input[i]);
        
    Grid::merge<vector_t,scalar_t>(vinput,input);
    
    vector_func(vresult, vinput);
    
    Grid::extract<vector_t,scalar_t>(vresult,result);
    
    int failed=0;
    for(int i=0;i<Nsimd;i++)
    {
        if ( std::abs(reference[i]-result[i])>1.0e-6)
        {
            std::cout << "**** " << name << "(" << input2[i] << ") -> " << result[i] << "  vs.  " << reference[i] << std::endl;
            failed++;
        }
    }
    
    if ( failed > 0 )
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "OK" << std::endl;
}
    
template<class vector_t, class scalar_t>
void test_binary(std::function<void(scalar_t &, const scalar_t&, const scalar_t&)> scalar_func,
                 std::function<void(vector_t &, const vector_t&, const vector_t&)> vector_func,
                 std::string name)
{
    std::cout << "Testing " << name << ": ";
    
    int Nsimd = vector_t::Nsimd();
    
    std::vector<scalar_t> result(Nsimd); random_fill(result);
    std::vector<scalar_t> reference(result);
    std::vector<scalar_t> input1(Nsimd); random_fill(input1);
    std::vector<scalar_t> input2(Nsimd); random_fill(input2);
    vector_t vinput1, vinput2, vresult;
    
    for(int i=0; i<Nsimd; ++i)
        scalar_func(reference[i], input1[i], input2[i]);
        
    Grid::merge<vector_t,scalar_t>(vinput1,input1);
    Grid::merge<vector_t,scalar_t>(vinput2,input2);
    Grid::merge<vector_t,scalar_t>(vresult,result);
    
    vector_func(vresult, vinput1, vinput2);
    
    Grid::extract<vector_t,scalar_t>(vresult,result);
    
    int failed=0;
    for(int i=0;i<Nsimd;i++)
    {
        if ( std::abs(reference[i]-result[i])>1.0e-6)
        {
            std::cout << "**** " << name << "(" << input1[i] << "," << input2[i] << ") -> " << result[i] << "  vs.  " << reference[i] << std::endl;
            failed++;
        }
    }
    
    if ( failed > 0 )
        std::cout << "FAILED" << std::endl;
    else
        std::cout << "OK" << std::endl;
}
    

template<class vector_t, class scalar_t>
void test_arithmetic()
{
    test_binary<vector_t, scalar_t>([](auto &a, const auto &b, const auto &c){ a = b+c; },
                                    [](auto &a, const auto &b, const auto &c){ add(&a, &b, &c); },
                                    "add");
    test_binary<vector_t, scalar_t>([](auto &a, const auto &b, const auto &c){ a = b-c; },
                                    [](auto &a, const auto &b, const auto &c){ sub(&a, &b, &c); },
                                    "sub");
    test_binary<vector_t, scalar_t>([](auto &a, const auto &b, const auto &c){ a = b*c; },
                                    [](auto &a, const auto &b, const auto &c){ mult(&a, &b, &c); },
                                    "mult");
    test_binary<vector_t, scalar_t>([](auto &a, const auto &b, const auto &c){ a = a+b*c; },
                                    [](auto &a, const auto &b, const auto &c){ mac(&a, &b, &c); },
                                    "mac");
}
    
using namespace Grid;

int main()
{
    std::cout << "vRealF:" << std::endl; test_arithmetic<vRealF, RealF>(); std::cout << std::endl;
    std::cout << "vRealD:" << std::endl; test_arithmetic<vRealD, RealD>(); std::cout << std::endl;
    std::cout << "vComplexF:" << std::endl; test_arithmetic<vComplexF, ComplexF>(); std::cout << std::endl;
    std::cout << "vComplexD:" << std::endl; test_arithmetic<vComplexD, ComplexD>(); std::cout << std::endl;
}
    
