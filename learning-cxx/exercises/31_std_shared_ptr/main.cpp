#include "../exercise.h"
#include <memory>

// READ: `std::shared_ptr` <https://zh.cppreference.com/w/cpp/memory/shared_ptr>
// READ: `std::weak_ptr` <https://zh.cppreference.com/w/cpp/memory/weak_ptr>

// TODO: 将下列 `?` 替换为正确的值
int main(int argc, char **argv) {
    auto shared = std::make_shared<int>(10);//shared -> 0x1234
    std::shared_ptr<int> ptrs[]{shared, shared, shared};//ptrs[0/1/2] -> 0x1234

    std::weak_ptr<int> observer = shared;//observer -> 0x1234
    ASSERT(observer.use_count() == 4, "");

    ptrs[0].reset();
    ASSERT(observer.use_count() == 3, "");//ptrs[0] -> 0

    ptrs[1] = nullptr;
    ASSERT(observer.use_count() == 2, "");//ptrs[1] -> 0

    ptrs[2] = std::make_shared<int>(*shared);//ptrs[2] -> 0x5678(内容也是10)
    ASSERT(observer.use_count() == 1, "");

    ptrs[0] = shared;//ptrs[0] -> 0x1234
    ptrs[1] = shared;//ptrs[1] -> 0x1234
    ptrs[2] = std::move(shared);//shared -> 0, ptrs[2] -> 0x1234(原本shared指向的空间)
    ASSERT(observer.use_count() == 3, "");

    std::ignore = std::move(ptrs[0]);//ptrs[0] -> 0, std::ignore -> 0x1234
	ptrs[1] = std::move(ptrs[1]);//ptrs[1] -> 0, ptrs[1] -> 0x1234
    ptrs[1] = std::move(ptrs[2]);//ptrs[2] -> 0, ptrs[1] -> 0x1234
	ASSERT(observer.use_count() == 2, "");

    shared = observer.lock();//检查对象是否还存在,现在ptrs[1]仍指向0x1234,因此存在,则让shared指向该地址、并令引用值计数+1
    ASSERT(observer.use_count() == 3, "");

    shared = nullptr;
    for (auto &ptr : ptrs) ptr = nullptr;
    ASSERT(observer.use_count() == 0, "");

    shared = observer.lock();//若对象已经不存在,则返回nullptr
    ASSERT(observer.use_count() == 0, "");

    return 0;
}
