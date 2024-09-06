#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

#define MAXN 128*128

using namespace cute;


template<class T1,class T2>
void print_compatible(T1 l1, T2 l2) {
    print(l1);
    printf(" -> ");
    print(l2);
    printf(" is ");
    if (is_compatible<decltype(l1),decltype(l2)>()) {
        printf("compatible\n");
    } else {
        printf("NOT compatible\n");
    }
}


int main()
{
    
    auto s1 = make_shape(_24{});
    
    printf("reflexive\n");
    print_compatible(s1,s1);

    printf("\n\ntransitive\n");
    auto s3 = make_shape(make_tuple(_4{},_6{}));    
    auto s5 = make_shape(make_tuple(make_tuple(_2{},_2{}),_6{}));
    print_compatible(s1,s3);
    print_compatible(s3,s5);
    print_compatible(s1,s5);

    printf("\n\nantisymetric\n");
    auto s2 = make_shape(make_tuple(_24{}));
    print_compatible(s1,s2);
    print_compatible(s2,s1);
    print_compatible(s1,s3);
    print_compatible(s3,s1);

    printf("\n\nothers\n");
    auto s4 = make_shape(make_tuple(_2{},_3{}),_4{});
    auto s6 = make_shape(make_tuple(_2{},_3{},_4{}));
    print_compatible(s1,s4);
    print_compatible(s1,s6);


    auto s7 = make_shape(make_tuple(make_tuple(_2{},_3{}),_4{}));
    print_compatible(s1,s7);
    print_compatible(s3,s7);






}
