#pragma once

// C standard headers
#include <string>
#include <vector>
#include <any>
#include <map>
#include <functional>
#include <typeindex>
#include <cstring>

// C headers should be included with their C++ versions
#include <cstddef>
#include <cstdint>

// function metadata to store type information
struct FunctionMetadata {
    std::vector<std::type_index> argument_types;
    std::type_index return_type;
    size_t total_size;

    // construct to properly initialize type_index
    FunctionMetadata() : return_type(typeid(void)) {}

    FunctionMetadata(
        const std::vector<std::type_index>& args,
        const std::type_index& ret,
        size_t size
    ) : argument_types(args), return_type(ret), total_size(size) {}
};

class FunctionRegistry {
    private:
        template<typename Func, typename... Args, size_t... I>
        static void invoke_helper(Func func, void* args, void* result, std::index_sequence<I...>) {
            using ArgsTuple = std::tuple<Args...>;
            char* args_ptr = static_cast<char*>(args);

            size_t offsets[] = {0, sizeof(std::tuple_element_t<I, ArgsTuple>)...};
            for (size_t i = 1; i < sizeof...(Args); ++i) {
                offsets[i] += offsets[i-1];
            }

            using RetType = std::invoke_result_t<Func, Args...>;
            if constexpr (std::is_void_v<RetType>) {
                func(*(std::tuple_element_t<I, ArgsTuple>*)(args_ptr + offsets[I])...);
            } else {
                *static_cast<RetType*>(result) = func(*(std::tuple_element_t<I, ArgsTuple>*)(args_ptr + offsets[I])...);
            }
        }

        std::map<std::string, std::function<void(void*, void*)>> functions;
        std::map<std::string, FunctionMetadata> function_metadata;

    public:
        template<typename Ret, typename... Args>
        void register_function(const std::string& name, Ret(*func)(Args...)) {
            FunctionMetadata metadata (
                {std::type_index(typeid(Args))...},
                std::type_index(typeid(Ret)),
                (sizeof(Args) + ...) + (std::is_void_v<Ret> ? 0 : sizeof(Ret))
            );

            function_metadata[name]   = metadata;
            functions[name]           = [func](void* args, void* result) {
                invoke_helper(func, args, result, std::index_sequence_for<Args...>{});
            };
        }

        const FunctionMetadata* get_metadata(const std::string& name) const {
            auto it = function_metadata.find(name);
            return it != function_metadata.end() ? &it->second : nullptr;
        }

        bool execute(const std::string& name, void* args, void* result) {
            auto it = functions.find(name);
            if (it != functions.end()) {
                it->second(args, result);
                return true;
            }
            return false;
        }
};
