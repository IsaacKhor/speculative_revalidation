add_rules("mode.debug", "mode.release", "mode.releasedbg")

add_requires("boost", "zstd", "fmt", "abseil")
add_packages("boost", "zstd", "fmt", "abseil")

add_requires("fmt", {system = true})
add_packages("fmt")

set_languages("c++20")
set_rundir('$(projectdir)')

-- add asan when in debug mode
if is_mode("debug") then
    add_cflags("-fsanitize=address", "-fno-omit-frame-pointer")
    add_cxxflags("-fsanitize=address", "-fno-omit-frame-pointer")
    add_ldflags("-fsanitize=address")
    -- fix libboost python dependency
    add_ldflags("-L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lpython3.12 -ldl  -lm")
end

target('sim')
    set_kind('binary')
    add_files('sim/sim.cpp')

target('oracle_backpass')
    set_kind('binary')
    add_files('sim/oracle_backpass.cpp')
