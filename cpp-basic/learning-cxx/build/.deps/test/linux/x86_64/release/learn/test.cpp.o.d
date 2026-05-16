{
    depfiles_format = "gcc",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-O3",
            "-std=c++17",
            "-D__XMAKE__=\"/home/jason/InfiniTensor_25Winter/cpp-basic/learning-cxx\"",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    },
    depfiles = "test.o: learn/test.cpp learn/test.h\
",
    files = {
        "learn/test.cpp"
    }
}