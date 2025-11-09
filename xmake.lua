add_rules("mode.debug", "mode.release", "mode.releasedbg")

add_requires("boost", "zstd", "fmt", "abseil")
add_packages("boost", "zstd", "fmt", "abseil")

add_requires("fmt", {system = true})
add_packages("fmt")

set_languages("c++20")

target('sim')
    set_kind('binary')
    set_rundir('$(projectdir)')
    add_files('sim/sim.cpp')
