#include<cuda.h>

__global__ void kernel(float* D, uint64_t desc_a, uint64_t desc_b, const int scaleA, const int scaleB, int scale_D, const int tnspA,const int tnspB) {
     float d[16];

     for (int i = 0 ; i < 16 ; ++i ) {
       d[i]=0;
     }
    
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      " %8,"
      " %9,"
      " p,   1, 1 , 0 , 0; \n"
    "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(int32_t(scale_D)));
    
    //防止编译器优化
    desc_a++;
    desc_b++;
    scale_D=1;

    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      " %8,"
      " %9,"
      " p,   1, 1 , 0 , 0; \n"
    "}\n"
      : "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(int32_t(scale_D)));

    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");         

    //store to GMEM
    for(int i = 0 ; i < 16 ; ++i ) {
      D[i] = d[i];
    }
}