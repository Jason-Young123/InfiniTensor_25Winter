#include <iostream>

int main()
{
    int a = 3;
    double x[a] = {0};
    //int *c = new int[3];
    std::cout << __FILE__ << ":" << __LINE__ << ":" << x[2] << std::endl;
    return 0;
}