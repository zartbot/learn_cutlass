
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;



#define PRINT_TUPLE(name, content)      \
    print(name);                  \
    print(" : ");                 \
    print(content);               \
    print(" rank: ");             \
    print(cute::rank(content));   \
    print(" depth: ");            \
    print(cute::depth(content));  \
    print(" size: ");             \
    print(cute::size(content));   \
    print("\n");


int main()
{
    //动态整型
    auto dynamic_var = int{2};
    dynamic_var = 4;

    bool is_dynamic = cute::is_integral<decltype(dynamic_var)>();
    printf("%d \n",is_dynamic);
    
    //静态整型
    auto static_var = Int<3>{};
    is_dynamic = cute::is_integral<decltype(static_var)>();
    printf("%d \n",is_dynamic);  
    // static_var  -= 3 , compile error

    
    //复合运算
    auto var = Int<8>{} + static_var + max (_4{}, _3{}) - abs(_m4{}) * dynamic_var;
    printf ("var= %d \n",var);

    auto a =  make_tuple(uint16_t{42}, int{7});
    PRINT_TUPLE("a",a);
    auto b =  make_tuple(uint16_t{4}, int{8},Int<9>{} );
    PRINT_TUPLE("b",b);
    auto c = make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), b);
    PRINT_TUPLE("c",c);
    PRINT_TUPLE("c",get<2>(c));


}