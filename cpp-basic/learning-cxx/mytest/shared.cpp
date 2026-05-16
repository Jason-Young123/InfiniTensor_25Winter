#include <iostream>
#include <memory>

//假设shared一开始指向地址addr1,内容是(int)10
int main(){
	auto shared = std::make_shared<int>(10);//shared -> addr1
	std::shared_ptr<int> ptrs[]{shared, shared, shared};//ptrs[0/1/2] -> addr1
	//总计数 = 4

	std::weak_ptr<int> observer = shared;//weak_ptr不增加计数次数,因此仍为4
	std::cout << observer.use_count() << std::endl;//分别是shard, ptrs[0], ptrs[1], ptrs[2]
	
	std::ignore = std::move(ptrs[0]);//该赋值被优化掉,等于没执行,ptrs[0]没变
	std::cout << ptrs[0] << std::endl;//输出addr1
	std::cout << observer.use_count() << std::endl;//仍然是shard, ptrs[0], ptrs[1], ptrs[2]
	
	ptrs[1] = std::move(ptrs[2]);//此时ptrs[1]接管ptrs[2],指向addr1, ptrs[2]变为nullptr
	std::cout << ptrs[2] << std::endl;
	std::cout << observer.use_count() << std::endl;//分别是shared, ptrs[0], ptrs[1] (ptrs[2]变为了nullptr)

	/*std::cout << "addr of shared: " << shared << std::endl;
	std::cout << "ptrs[0]: " << ptrs[0] << std::endl;
	std::cout << "ptrs[1]: " << ptrs[1] << std::endl;
	std::cout << "ptrs[2]: " << ptrs[2] << std::endl;

	ptrs[0].reset();
	std::cout << "ptrs[0]: " << ptrs[0] << std::endl;
	std::cout << observer.use_count() << std::endl;

	ptrs[1] = nullptr;
	std::cout << "ptrs[1]: " << ptrs[1] << std::endl;
	std::cout << observer.use_count() << std::endl;

	ptrs[2] = std::make_shared<int>(*shared);
	std::cout << "ptrs[2]: " << ptrs[2] << std::endl;
	std::cout << observer.use_count() << std::endl;

	
	ptrs[0] = shared; ptrs[1] = shared;
	std::cout << "ptrs[0]: " << ptrs[0] << std::endl;
	std::cout << "ptrs[1]: " << ptrs[1] << std::endl;
    std::cout << observer.use_count() << std::endl;

	
	ptrs[2] = std::move(shared);
	std::cout << "ptrs[2]: " << ptrs[2] << std::endl;
    std::cout << observer.use_count() << std::endl;
	*/

	return 0;
}






