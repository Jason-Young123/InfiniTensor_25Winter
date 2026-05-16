#include <iostream>
#include <vector>


class mydouble{
private:
    uint32_t _hi32;//高32bit
    uint32_t _lo32;//低32bit

public:
/* basics */
    __host__ __device__
    mydouble(int val){//构造函数1: int构造
        double d_val = static_cast<double>(val);
        uint64_t double_bits = *reinterpret_cast<uint64_t*>(&d_val);
        _hi32 = static_cast<uint32_t>((double_bits >> 32) & 0xFFFFFFFF); // 右移32位取高32位
        _lo32 = static_cast<uint32_t>(double_bits & 0xFFFFFFFF);         // 直接取低32位
    }

    __host__ __device__
    mydouble(float val){//构造函数2: float构造
        double d_val = static_cast<double>(val);
        uint64_t double_bits = *reinterpret_cast<uint64_t*>(&d_val);
        _hi32 = static_cast<uint32_t>((double_bits >> 32) & 0xFFFFFFFF);
        _lo32 = static_cast<uint32_t>(double_bits & 0xFFFFFFFF);
    }

    __host__ __device__
    mydouble(const mydouble& other) {//拷贝构造
        // 直接复制成员变量
        _hi32 = other._hi32;
        _lo32 = other._lo32;
    }

    __host__ __device__
    mydouble& operator=(const mydouble& other) {//拷贝赋值
        if (this != &other) {
            _hi32 = other._hi32;
            _lo32 = other._lo32;
        }
        return *this;
    }

    __host__ __device__
    ~mydouble(){}


/* main functions */
    __host__ __device__
    mydouble mul(mydouble op1, mydouble op2){
        return mydouble(0);
    }

    mydouble add(mydouble op1, mydouble op2){
        return mydouble(0);
    }

    /*__host__ __device__
    mydouble div(){
        
    }*/


/* auxiliary functions */
    __host__ __device__
    void info(){
        printf("hi32 = %08x, lo32 = %08x\n", _hi32, _lo32);
    }

};



__global__ void test_mydouble(){
    mydouble a(float(3.141593));
    a.info();
}



int main(){
    test_mydouble<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}