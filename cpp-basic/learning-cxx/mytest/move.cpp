#include <iostream>
#include <vector>

int main(){
	std::vector<int> a{1,2};
	std::cout << "the addr of a.data is: " << a.data() << std::endl;
	std::cout << "size of a is: " << a.size() << std::endl;
	std::cout << "capacity of a is: " << a.capacity() << std::endl;

	a.push_back(1);
	a.push_back(1);
	std::cout << "the addr of a.data is: " << a.data() << std::endl;
	std::cout << "size of a is: " << a.size() << std::endl;
	std::cout << "capacity of a is: " << a.capacity() << std::endl;

	std::vector<int> b;
	std::cout << "the addr of b.data is: " << b.data() << std::endl;
	std::cout << "size of b is: " << b.size() << std::endl;
	std::cout << "capacity of b is: " << b.capacity() << std::endl;
	
	b.push_back(2);
	std::cout << "the addr of b.data is: " << b.data() << std::endl;
	std::cout << "size of b is: " << b.size() << std::endl;
	std::cout << "capacity of b is: " << b.capacity() << std::endl;


	std::vector<int> c;
	c.reserve(10);
	std::cout << "the addr of c.data is: " << c.data() << std::endl;
	std::cout << "size of c is: " << c.size() << std::endl;
	std::cout << "capacity of c is: " << c.capacity() << std::endl;

	c.push_back(3);
	std::cout << "the addr of c.data is: " << c.data() << std::endl;
	std::cout << "size of c is: " << c.size() << std::endl;
	std::cout << "capacity of c is: " << c.capacity() << std::endl;
	
	
	c = std::move(b);
	std::cout << "the addr of b.data is: " << b.data() << std::endl;
	std::cout << "size of b is: " << b.size() << std::endl;
	std::cout << "capacity of b is: " << b.capacity() << std::endl;
	std::cout << "the addr of c.data is: " << c.data() << std::endl;
	std::cout << "size of c is: " << c.size() << std::endl;
	std::cout << "capacity of c is: " << c.capacity() << std::endl;


	
	return 0;

}
