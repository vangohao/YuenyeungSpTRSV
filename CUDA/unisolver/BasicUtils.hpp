#pragma once
#include <algorithm>
#include <cassert>
#ifndef UNI_ENABLE_SW
#include <charconv>
#endif
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "BasicDef.hpp"

#define Assert(x, str)                     \
    {                                      \
        if (!(x)) {                        \
            std::cerr << str << std::endl; \
            assert(x);                     \
            __builtin_unreachable();       \
        }                                  \
    }

namespace uni {

template <class T>
inline ALWAYS_INLINE constexpr T myPow(T x, unsigned int p);

static inline auto GetFileName(std::string fullname) {
    auto pos = fullname.rfind('/');
    return fullname.substr(pos + 1, fullname.length() - pos - 1);
}

static inline auto GetFileNameWithNoExt(std::string fullname) {
    auto pos = fullname.rfind('/');
    auto pos2 = fullname.rfind('.');
    return fullname.substr(pos + 1, pos2 - pos - 1);
}

namespace detail {

template <class T, T... inds, class F>
ALWAYS_INLINE constexpr void loop(std::integer_sequence<T, inds...>,
                                  F f [[maybe_unused]]) {
    (f(std::integral_constant<T, inds>{}), ...);  // C++17 fold expression
}

template <class T, T... inds, class F>
ALWAYS_INLINE constexpr auto loop_sum(std::integer_sequence<T, inds...>,
                                      F f [[maybe_unused]]) {
    return (... +
            f(std::integral_constant<T, inds>{}));  // C++17 fold expression
}

}  // namespace detail

template <class T, T count, class F>
ALWAYS_INLINE constexpr void loop(F f) {
    detail::loop(std::make_integer_sequence<T, count>{}, f);
}

template <class T, T count, class F>
ALWAYS_INLINE constexpr auto loop_sum(F f) {
    return detail::loop_sum(std::make_integer_sequence<T, count>{}, f);
}

template <class T>
ALWAYS_INLINE const void* pointer(T buf) {
    return static_cast<const void*>(buf);
}

template <typename T>
ALWAYS_INLINE constexpr int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Note that in C, (-1)/7=-1; However in Python, (-1)//7=-2.
// What we here do is like Python.
template <class T>
ALWAYS_INLINE constexpr T ceil_div(T a, T b) {
    assert(b > 0);
    if (a >= 0)
        return (a + b - 1) / b;
    else
        return a / b;
}
template <class T>
ALWAYS_INLINE constexpr T floor_div(T a, T b) {
    assert(b > 0);
    if (a >= 0)
        return a / b;
    else
        return (a - b + 1) / b;
}

template <class T>
ALWAYS_INLINE constexpr void my_swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

inline int string_to_int(const std::string& str) {
    int value = 0;
#ifndef UNI_ENABLE_SW
    std::from_chars_result res =
        std::from_chars(str.data(), str.data() + str.size(), value);

    if (res.ec == std::errc::invalid_argument ||
        res.ec == std::errc::result_out_of_range) {
        return 0;
    }

    return value;
#else
    try {
        value = std::stoi(str);
    } catch (const std::invalid_argument& e) {
        return 0;
    } catch (const std::out_of_range& e) {
        return 0;
    }
#endif
}

template <typename Iterator1, typename Iterator2, typename Function>
constexpr Function for_each_pair(Iterator1 first1, Iterator1 last1,
                                 Iterator2 first2, Function f) {
    for (; first1 != last1; ++first1, (void)++first2) {
        f(*first1, *first2);
    }
    return f;
}

template <typename Iterator1, typename Iterator2, typename Iterator3,
          typename Function>
constexpr Function for_each_tuple(Iterator1 first1, Iterator1 last1,
                                  Iterator2 first2, Iterator3 first3,
                                  Function f) {
    for (; first1 != last1; ++first1, (void)++first2, (void)++first3) {
        f(*first1, *first2, *first3);
    }
    return f;
}

template <typename T1, typename T2, typename F>
constexpr F for_vector_pair(std::vector<T1>& v1, std::vector<T2>& v2, F f) {
    return for_each_pair(v1.begin(), v1.end(), v2.begin(), f);
}

template <typename T1, typename T2, typename T3, typename F>
constexpr F for_vector_tuple(std::vector<T1>& v1, std::vector<T2>& v2,
                             std::vector<T3>& v3, F f) {
    return for_each_tuple(v1.begin(), v1.end(), v2.begin(), v3.begin(), f);
}
}  // namespace uni
