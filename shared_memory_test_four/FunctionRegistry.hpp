#pragma once
#include <string>
#include <vector>
#include <any>
#include <map>
#include <functional>
#include <typeindex>

struct FunctionMetadata {
    std::vector<std::type_index> argument_types;
    std::type_index return_type;
    size_t total_size;
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
            FunctionMetadata metadata;
            metadata.argument_types   = {std::type_index(typeid(Args))...};
            metadata.return_type      =  std::type_index(typeid(Ret));
            metadata.total_size       = (sizeof(Args) + ...) + sizeof(Ret);

            function_metadata[name]   = metadata;
            function[name]            = [func](void* args, void* result) {
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
}
