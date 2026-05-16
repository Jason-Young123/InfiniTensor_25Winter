{
    depfiles = "main.o: 00_hello_world/main.cpp 00_hello_world/../exercise.h\
",
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
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    },
    files = {
        "00_hello_world/main.cpp"
    }
}