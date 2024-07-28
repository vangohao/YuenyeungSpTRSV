#pragma once
#include "BasicDef.hpp"
#ifdef UNI_ENABLE_SW
#pragma swuc push infer
#endif
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "BasicUtils.hpp"

namespace uni {

template <class T>
inline ALWAYS_INLINE constexpr T myPow(T x, unsigned int p);

template <class STREAM, size_t I, class T>
UNI_HOST_DEVICE STREAM& operator<<(STREAM& s, std::array<T, I> tpl) {
    s << "(";
    loop<size_t, I>([&](auto i) { s << std::get<i>(tpl) << ","; });
    s << ")";
    return s;
}

template <size_t I, class T>
UNI_HOST_DEVICE std::string array_to_string(std::array<T, I> tpl) {
    return "(" + loop_sum<size_t, I>([&](auto i) {
               return std::to_string(std::get<i>(tpl)) + ",";
           }) +
           ")";
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#if __cplusplus < 202002L
// template <class T, std::size_t N>
// ALWAYS_INLINE constexpr bool operator==(const std::array<T, N>& lhs,
//                                         const std::array<T, N>& rhs) {
//     bool result = true;
//     loop<size_t, N>([&](auto i) ALWAYS_INLINE {
//         if (std::get<i>(lhs) != std::get<i>(rhs)) result = false;
//     });
//     return result;
// }
// template <class T, std::size_t N>
// ALWAYS_INLINE constexpr bool operator!=(const std::array<T, N>& lhs,
//                                         const std::array<T, N>& rhs) {
//     return !(lhs == rhs);
// }
#endif
#endif

template <std::size_t N>
constexpr std::array<bool, N> array_element_wise_logical_and(
    const std::array<bool, N>& lhs, const std::array<bool, N>& rhs) {
    std::array<bool, N> result = {};
    loop<size_t, N>([&](auto i) {
        std::get<i>(result) = (std::get<i>(lhs) && std::get<i>(rhs));
    });
    return result;
}

template <class T, std::size_t N>
constexpr std::array<bool, N> array_element_wise_equal(
    const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
    std::array<bool, N> result = {};
    loop<size_t, N>([&](auto i) {
        std::get<i>(result) = (std::get<i>(lhs) == std::get<i>(rhs));
    });
    return result;
}

template <class T, std::size_t N>
constexpr std::array<bool, N> array_element_wise_equal(
    const std::array<T, N>& lhs, T rhs) {
    std::array<bool, N> result = {};
    loop<size_t, N>(
        [&](auto i) { std::get<i>(result) = (std::get<i>(lhs) == rhs); });
    return result;
}

template <std::size_t N>
ALWAYS_INLINE constexpr bool array_reduce_or(const std::array<bool, N>& a) {
    return std::apply([](auto... args) { return (... || args); }, a);
}

template <std::size_t N>
ALWAYS_INLINE constexpr bool array_reduce_and(const std::array<bool, N>& a) {
    return std::apply([](auto... args) { return (... && args); }, a);
}

namespace detail {
template <typename T, std::size_t... Is>
ALWAYS_INLINE constexpr std::array<T, sizeof...(Is)> constant_array(
    T value, std::index_sequence<Is...>) {
    // cast Is to void to remove the warning: unused value
    return {{(static_cast<void>(Is), value)...}};
}
}  // namespace detail

template <typename T, std::size_t N>
ALWAYS_INLINE constexpr std::array<T, N> constant_array(const T& value) {
    return detail::constant_array(value, std::make_index_sequence<N>());
}

namespace detail {

template <class Function, std::size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
    -> std::array<typename std::result_of<Function(std::size_t)>::type,
                  sizeof...(Indices)> {
    return {{f(Indices)...}};
}

template <typename T, std::size_t... Indices>
constexpr auto make_array_sequence_helper(std::index_sequence<Indices...>)
    -> std::array<T, sizeof...(Indices)> {
    return {{Indices...}};
}

}  // namespace detail

template <size_t N, class Function>
constexpr auto make_array(Function f)
    -> std::array<typename std::invoke_result_t<Function, size_t>, N> {
    return detail::make_array_helper(f, std::make_index_sequence<N>{});
}

template <size_t N, typename T = int>
constexpr auto make_array_sequence() -> std::array<T, N> {
    return detail::make_array_sequence_helper<T>(std::make_index_sequence<N>{});
}

template <size_t I, class T>
void vector2array(std::vector<T>& v_src, std::array<T, I>& a_dst) {
    std::copy(v_src.begin(), v_src.begin() + std::min(v_src.size(), I),
              a_dst.begin());
}

/**
 * @brief minmod批量操作
 *
 * @param a 实数数组
 * @param b 实数数组
 * @return minmod(a[i],b[i])
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> minmod(
    const std::array<T, I>& a, const std::array<T, I>& b) {
    std::array<T, I> result;
    loop<size_t, I>([&](auto i) {
        std::get<i>(result) = 0;
        if (std::get<i>(a) > 0 && std::get<i>(b) > 0)
            std::get<i>(result) = std::min(std::get<i>(a), std::get<i>(b));
        if (std::get<i>(a) < 0 && std::get<i>(b) < 0)
            std::get<i>(result) = std::max(std::get<i>(a), std::get<i>(b));
    });
    return result;
}

template <size_t I, class T, typename = std::enable_if_t<std::is_scalar_v<T>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T minmod(const T& a, const T& b) {
    T result(0);
    if (a > T(0) && b > T(0)) result = std::min(a, b);
    if (a < T(0) && b < T(0)) result = std::max(a, b);

    return result;
}

/**
 * @brief maxmod批量操作
 *
 * @param a 实数数组
 * @param b 实数数组
 * @return maxmod(a[i],b[i])
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> maxmod(
    const std::array<T, I>& a, const std::array<T, I>& b) {
    std::array<T, I> result;
    loop<size_t, I>([&](auto i) {
        std::get<i>(result) = 0;
        if (std::get<i>(a) > 0 && std::get<i>(b) > 0)
            std::get<i>(result) = std::max(std::get<i>(a), std::get<i>(b));
        if (std::get<i>(a) < 0 && std::get<i>(b) < 0)
            std::get<i>(result) = std::min(std::get<i>(a), std::get<i>(b));
    });
    return result;
}

template <size_t I, class T, typename = std::enable_if_t<std::is_scalar_v<T>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T maxmod(const T& a, const T& b) {
    T result(0);
    if (a > T(0) && b > T(0)) result = std::max(a, b);
    if (a < T(0) && b < T(0)) result = std::min(a, b);

    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> choose_by_mask(
    const std::array<T, I>& a, const std::array<T, I>& b,
    const std::array<bool, I>& mask) {
    std::array<T, I> result;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) =
            std::get<i>(mask) ? std::get<i>(a) : std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    operator+(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) + std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator+=(
    std::array<T1, I>& a, const std::array<T2, I>& b) {
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(a) = std::get<i>(a) + std::get<i>(b);
    });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    operator-(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) - std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator-=(
    std::array<T1, I>& a, const std::array<T2, I>& b) {
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(a) = std::get<i>(a) - std::get<i>(b);
    });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    operator*(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) * std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator*=(
    std::array<T1, I>& a, const std::array<T2, I>& b) {
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(a) = std::get<i>(a) * std::get<i>(b);
    });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    operator/(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) / std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator/=(
    std::array<T1, I>& a, const std::array<T2, I>& b) {
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(a) = std::get<i>(a) / std::get<i>(b);
    });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    operator%(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) % std::get<i>(b);
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator%=(
    std::array<T1, I>& a, const std::array<T2, I>& b) {
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(a) = std::get<i>(a) % std::get<i>(b);
    });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE
    ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, I>
    array_ceil_div(const std::array<T1, I>& a, const std::array<T2, I>& b) {
    std::array<std::common_type_t<T1, T2>, I> result = {};
    loop<size_t, I>([&](auto i) {
        std::get<i>(result) = ceil_div(std::get<i>(a), std::get<i>(b));
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator+(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) + b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator+=(
    std::array<T1, I>& a, T2 b) {
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { std::get<i>(a) = std::get<i>(a) + b; });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator-(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) - b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator-=(
    std::array<T1, I>& a, T2 b) {
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { std::get<i>(a) = std::get<i>(a) - b; });
    return a;
}

template <size_t I, class T1>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator-(
    const std::array<T1, I>& a) {
    std::array<T1, I> result = {};
    loop<size_t, I>(
        [&](auto i) ALWAYS_INLINE { std::get<i>(result) = -std::get<i>(a); });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator*(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) * b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator*=(
    std::array<T1, I>& a, T2 b) {
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { std::get<i>(a) = std::get<i>(a) * b; });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator*(
    T2 b, const std::array<T1, I>& a) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) * b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator/(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) / b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator/=(
    std::array<T1, I>& a, T2 b) {
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { std::get<i>(a) = std::get<i>(a) / b; });
    return a;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator&(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) & b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator^(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) ^ b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> operator%(
    const std::array<T1, I>& a, T2 b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) % b;
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I>& operator%=(
    std::array<T1, I>& a, T2 b) {
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { std::get<i>(a) = std::get<i>(a) % b; });
    return a;
}

template <size_t I, class T1, class T2,
          typename = std::enable_if_t<std::is_constructible_v<T1, T2>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<bool, I> operator>(
    const std::array<T1, I>& a, T2 b) {
    std::array<bool, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) > T1(b);
    });
    return result;
}

template <size_t I, class T1, class T2,
          typename = std::enable_if_t<std::is_constructible_v<T1, T2>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<bool, I> operator>=(
    const std::array<T1, I>& a, T2 b) {
    std::array<bool, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) >= T1(b);
    });
    return result;
}

template <size_t I, class T1, class T2,
          typename = std::enable_if_t<std::is_constructible_v<T1, T2>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<bool, I> operator<(
    const std::array<T1, I>& a, T2 b) {
    std::array<bool, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) < T1(b);
    });
    return result;
}

template <size_t I, class T1, class T2,
          typename = std::enable_if_t<std::is_constructible_v<T1, T2>>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<bool, I> operator<=(
    const std::array<T1, I>& a, T2 b) {
    std::array<bool, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::get<i>(a) <= T1(b);
    });
    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> abs(
    const std::array<T, I>& a) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::abs(std::get<i>(a));
    });
    return result;
}

template <size_t I, class T1, class T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> power(
    T1 a, const std::array<T2, I>& b) {
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = myPow(a, std::get<i>(b));
    });
    return result;
}

/**
 * @brief clamp(a, min, max)
 *        if (min <= a <= max) return a;
 *        elif (a < min) return min;
 *        else return max;
 *
 * @tparam I std::array length
 * @tparam T element type
 * @param a
 * @param min
 * @param max
 * @return std::array<T, I>
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> clamp(
    const std::array<T, I>& a, const std::array<T, I>& min,
    const std::array<T, I>& max) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::min(
            std::max(std::get<i>(a), std::get<i>(min)), std::get<i>(max));
    });
    return result;
}

/**
 * @brief clamp(a, min, max)
 *        if (min <= a <= max) return a;
 *        elif (a < min) return min;
 *        else return max;
 *
 * @tparam I std::array length
 * @tparam T element type
 * @param a
 * @param min
 * @param max
 * @return std::array<T, I>
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> clamp(
    const std::array<T, I>& a, const T& min, const T& max) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::min(std::max(std::get<i>(a), min), max);
    });
    return result;
}

/**
 * @brief clamp(a, min, max)
 *        if (min <= a <= max) return a;
 *        elif (a < min) return min;
 *        else return max;
 *
 * @tparam T element type
 */
template <class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T clamp(const T& a, const T& min,
                                                const T& max) {
    return std::min(std::max(a, min), max);
}

/**
 * @brief get location of a in periodic boundary
 *
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> mod_periodic(
    const std::array<T, I>& a, const std::array<T, I>& offset,
    const std::array<T, I>& size, const std::array<bool, I>& mask) {
    std::array<T, I> result = a;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        if (std::get<i>(mask))
            std::get<i>(result) =
                (std::get<i>(a) - std::get<i>(offset) + std::get<i>(size)) %
                    std::get<i>(size) +
                std::get<i>(offset);
    });
    return result;
}

/**
 * @brief in_range(a, min, max)
 *        if (min <= a <= max) return true;
 *        else return false;
 *
 * @tparam I std::array length
 * @tparam T element type
 * @param a
 * @param min
 * @param max
 * @return bool
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr bool in_range(
    const std::array<T, I>& a, const std::array<T, I>& min,
    const std::array<T, I>& max) {
    return a == clamp(a, min, max);
}
/**
 * @brief in_range(a, min, max)
 *        if (min <= a <= max) return true;
 *        else return false;
 *
 * @tparam I std::array length
 * @tparam T element type
 * @param a
 * @param min
 * @param max
 * @return bool
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr bool in_range(const std::array<T, I>& a,
                                                      const T& min,
                                                      const T& max) {
    return a == clamp(a, min, max);
}

/**
 * @brief in_range_with_mask(a, min, max, mask)
 *        On all masked-0 dims, (min <= a <= max), return true;
 *        Otherwise return false.
 *
 * @tparam I std::array length
 * @tparam T element type
 * @param a
 * @param min
 * @param max
 * @param bc
 * @return bool
 */
template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr bool in_range_with_mask(
    const std::array<T, I>& a, const std::array<T, I>& min,
    const std::array<T, I>& max, const std::array<bool, I>& mask) {
    return array_satisfy(
        [&](T a, T min, T max, bool mask) {
            return mask || (min <= a && a <= max);
        },
        a, min, max, mask);
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> normalize(
    const std::array<T, I>& a) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        if (std::get<i>(a) > 0)
            std::get<i>(result) = 1;
        else if (std::get<i>(a) < 0)
            std::get<i>(result) = -1;
        else
            std::get<i>(result) = 0;
    });
    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> maximum(
    const std::array<T, I>& a, const std::array<T, I>& b) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::max(std::get<i>(a), std::get<i>(b));
    });
    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> minimum(
    const std::array<T, I>& a, const std::array<T, I>& b) {
    std::array<T, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = std::min(std::get<i>(a), std::get<i>(b));
    });
    return result;
}

template <size_t I, class... T, class F>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr bool array_satisfy(
    F&& f, const std::array<T, I>&... a) {
    bool result = true;
    loop<size_t, I>(
        [&](auto i) ALWAYS_INLINE { result = result && f(std::get<i>(a)...); });
    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T max(const std::array<T, I>& a) {
    T result{};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        auto tmp = std::get<i>(a);
        if (tmp > result) result = tmp;
    });
    return result;
}

template <size_t I, class T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T min(const std::array<T, I>& a) {
    T result = std::numeric_limits<T>::max();
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        auto tmp = std::get<i>(a);
        if (tmp < result) result = tmp;
    });
    return result;
}

template <class T1, class T2, size_t I>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T1, I> array_astype(
    const std::array<T2, I>& a) {
    // TODO: Add a warning.
    std::array<T1, I> result = {};
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        std::get<i>(result) = static_cast<T1>(std::get<i>(a));
    });
    return result;
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T sum(std::array<T, I> tpl) {
    T total = 0;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE { total += std::get<i>(tpl); });
    return total;
}

template <size_t I, typename T1, typename T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::common_type_t<T1, T2> dot_product(
    std::array<T1, I> a, std::array<T2, I> b) {
#ifndef UNI_ENABLE_TIANHE
    return loop_sum<size_t, I>([&](auto i)
                                   ALWAYS_INLINE -> std::common_type_t<T1, T2> {
                                       return std::get<i>(a) * std::get<i>(b);
                                   });
#else
    return loop_sum<size_t, I>([&](auto i) -> std::common_type_t<T1, T2> {
        return std::get<i>(a) * std::get<i>(b);
    });
#endif
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T l2norm2(std::array<T, I> tpl) {
    T total = 0;
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { total += myPow(std::get<i>(tpl), 2); });
    return total;
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T l2norm(std::array<T, I> tpl) {
    T total = 0;
    loop<size_t, I>([&](auto i)
                        ALWAYS_INLINE { total += myPow(std::get<i>(tpl), 2); });
    return sqrt(total);
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T Product(
    [[maybe_unused]] std::array<T, I> tpl) {
    if constexpr (I == 0)
        return 1;
    else
        return std::apply(
            [](auto... args) ALWAYS_INLINE { return (... * args); }, tpl);
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> FlatToCart(
    T idx, std::array<T, I> shape) {
    if constexpr (I == 0) {
        return {};
    } else {
        auto cart = std::array<T, I>();
        loop<size_t, I - 1>([&](auto i) ALWAYS_INLINE {
            std::get<i>(cart) = idx % std::get<i>(shape);
            idx /= std::get<i>(shape);
        });
        std::get<I - 1>(cart) = idx;
        return cart;
    }
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::array<T, I> BaseToArray(T p,
                                                                     T idx) {
    Assert(myPow(p, I) > idx && idx >= 0,
           "Unable to represent the number by such base size.\n");
    return FlatToCart(idx, constant_array<T, I>(p));
}

template <size_t I, typename T1, typename T2>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::common_type_t<T1, T2> CartToFlat(
    std::array<T1, I> cart, std::array<T2, I> shape) {
    std::common_type_t<T1, T2> idx = 0;
    std::common_type_t<T1, T2> mul = 1;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        idx += mul * std::get<i>(cart);
        mul *= std::get<i>(shape);
    });
    return idx;
}

template <size_t I, typename T>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr T ArrayToBase(T p,
                                                      std::array<T, I> array) {
    return CartToFlat(array, constant_array<T, I>(p));
}

template <size_t I, typename T1, typename T2,
          typename = std::enable_if_t<I >= 1>>
UNI_HOST_DEVICE ALWAYS_INLINE constexpr std::common_type_t<T1, T2> CartToFlat(
    std::array<T1, I> cart, std::array<T2, I - 1> shape) {
    std::common_type_t<T1, T2> idx = 0;
    std::common_type_t<T1, T2> mul = 1;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        // if (i > 0)
        //     idx *= std::get<I - i>(shape);
        // idx += std::get<i>(cart);

        idx += mul * std::get<i>(cart);
        mul *= std::get<i>(shape);
    });
    return idx;
}

template <size_t I, typename T, class F>
UNI_HOST_DEVICE void NestedLoop(std::array<T, I> offset, std::array<T, I> size,
                                F f) {
    for (T i = 0; i < Product(size); i++) {
        auto loc = offset + FlatToCart(i, size);
        f(loc);
    }
}

template <size_t I, typename T, class F>
void NestedLoopUnpack(std::array<T, I> offset, std::array<T, I> size, F f) {
    for (T i = 0; i < Product(size); i++) {
        auto loc = offset + FlatToCart(i, size);
        std::apply(f, loc);
    }
}

template <typename T, std::size_t S>
ALWAYS_INLINE constexpr std::array<T, S> array_reverse(std::array<T, S> a) {
    std::array<T, S> result = {};
    loop<size_t, S>([&](auto i) ALWAYS_INLINE {
        std::get<S - 1 - i>(result) = std::get<i>(a);
    });
    return result;
}

namespace detail {
template <class T1, class T2, size_t N1, size_t... I1, size_t N2, size_t... I2>
ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, N1 + N2>
array_concat(const std::array<T1, N1>& a1, const std::array<T2, N2>& a2,
             std::integer_sequence<size_t, I1...>,
             std::integer_sequence<size_t, I2...>) {
    return {a1[I1]..., a2[I2]...};
}
}  // namespace detail

template <class T1, class T2, size_t N1, size_t N2>
ALWAYS_INLINE constexpr std::array<std::common_type_t<T1, T2>, N1 + N2>
array_concat(std::array<T1, N1> a1, std::array<T2, N2> a2) {
    return detail::array_concat(a1, a2,
                                std::make_integer_sequence<size_t, N1>{},
                                std::make_integer_sequence<size_t, N2>{});
}

template <class T1, class T2, size_t N2>
ALWAYS_INLINE constexpr auto array_concat(T1 a1, std::array<T2, N2> a2) {
    return array_concat(std::array<T1, 1>{a1}, a2);
}

template <class T1, class T2, size_t N1>
ALWAYS_INLINE constexpr auto array_concat(std::array<T1, N1> a1, T2 a2) {
    return array_concat(a1, std::array<T2, 1>{a2});
}

template <class T, size_t N>
class array_rless {
public:
    ALWAYS_INLINE
    constexpr bool operator()(const std::array<T, N>& a1,
                              const std::array<T, N>& a2) const {
        for (auto i = N; i-- > 0;) {
            if (a1[i] != a2[i]) {
                return a1[i] < a2[i];
            }
        }
        return false;
    }
};

template <typename... P>
ALWAYS_INLINE constexpr auto forward_ref_and_value_as_tuple(P&&... params) {
    return std::tuple<P...>(std::forward<P>(params)...);
}

template <typename tuple_t>
ALWAYS_INLINE constexpr auto get_array_from_tuple(tuple_t&& tuple) {
    constexpr auto get_array = [](auto&&... x) ALWAYS_INLINE {
        return std::array<std::common_type_t<decltype(x)...>, sizeof...(x)>{
            std::forward<decltype(x)>(x)...};
    };
    return std::apply(get_array, std::forward<tuple_t>(tuple));
}

namespace detail {
template <std::size_t Ofst, class Tuple, std::size_t... I>
ALWAYS_INLINE constexpr auto tuple_slice_impl(Tuple&& t,
                                              std::index_sequence<I...>) {
    return std::forward_as_tuple(std::get<I + Ofst>(std::forward<Tuple>(t))...);
}

template <std::size_t Ofst, std::size_t Step, class Tuple, std::size_t... I>
ALWAYS_INLINE constexpr auto array_slice_impl(Tuple&& t,
                                              std::index_sequence<I...>) {
    return std::array<typename std::remove_reference_t<Tuple>::value_type,
                      sizeof...(I)>(
        {std::get<I * Step + Ofst>(std::forward<Tuple>(t))...});
}
}  // namespace detail

template <std::size_t I1, std::size_t I2, class Cont>
ALWAYS_INLINE constexpr auto tuple_slice(Cont&& t) {
    static_assert(I2 >= I1, "invalid slice");
    static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
                  "slice index out of bounds");

    return detail::tuple_slice_impl<I1>(std::forward<Cont>(t),
                                        std::make_index_sequence<I2 - I1>{});
}

template <std::size_t I1, std::size_t I2, std::size_t Step = 1, class Cont>
UNI_HOST_DEVICE constexpr auto ALWAYS_INLINE array_slice(Cont&& t) {
    static_assert(I2 >= I1, "invalid slice");
    static_assert(Step > 0, "invalid step");
    static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
                  "slice index out of bounds");

    return detail::array_slice_impl<I1, Step>(
        std::forward<Cont>(t),
        std::make_index_sequence<(I2 - I1 + Step - 1) / Step>{});
}

template <class Tuple, size_t... IS>
ALWAYS_INLINE constexpr auto reorder_tuple(std::index_sequence<IS...>,
                                           Tuple tpl) {
    return std::forward_as_tuple(std::get<IS>(tpl)...);
}

template <size_t Element, size_t... IS>
ALWAYS_INLINE constexpr size_t find_index_sequence(std::index_sequence<IS...>) {
    int cnt = 0;
    int result = sizeof...(IS);
    ((IS == Element ? (result = cnt) : (cnt++)), ...);
    return result;
}

namespace detail {
template <size_t... IS, size_t... NS>
ALWAYS_INLINE constexpr auto get_reindex_sequence(
    std::index_sequence<IS...> find_in, std::index_sequence<NS...>) {
    return std::index_sequence<find_index_sequence<NS>(find_in)...>{};
}
}  // namespace detail

template <size_t... IS>
ALWAYS_INLINE constexpr auto get_reindex_sequence(
    std::index_sequence<IS...> find_in) {
    return detail::get_reindex_sequence(
        find_in, std::make_index_sequence<sizeof...(IS)>{});
}

/**
 * @brief reorder an std::array
 *
 * @tparam dim
 * @tparam INT_TYPE
 * @tparam IS
 * @param order indexs of original dim, from inside to outside
 * @param a array to reorder
 * @return constexpr auto reordered array
 */
template <size_t dim, class INT_TYPE, size_t... IS>
ALWAYS_INLINE constexpr auto array_reorder(std::index_sequence<IS...>,
                                           std::array<INT_TYPE, dim> a) {
    static_assert(sizeof...(IS) == dim,
                  "length of order must equal to array dim");
    std::array<INT_TYPE, dim> new_a = {std::get<IS>(a)...};
    return new_a;
}

template <size_t N, typename F, size_t... indexes>
ALWAYS_INLINE constexpr auto make_sequence_helper(
    F f, std::index_sequence<indexes...>) {
    return std::integer_sequence<size_t, std::get<indexes>(f())...>{};
}

template <typename F>
ALWAYS_INLINE constexpr auto make_sequence(F f) {
    constexpr size_t N = std::tuple_size<std::invoke_result_t<F>>();
    using indexes = std::make_index_sequence<N>;
    return make_sequence_helper<N>(f, indexes{});
}

template <size_t I>
inline ALWAYS_INLINE constexpr std::array<double, I> array_stdpow(
    std::array<double, I> x, double p) {
    std::array<double, I> result;
    loop<size_t, I>(
        [&](auto i) ALWAYS_INLINE { result[i] = std::pow(std::get<i>(x), p); });
    return result;
}

template <class T, size_t I>
inline ALWAYS_INLINE constexpr std::array<T, I> array_pow(std::array<T, I> x,
                                                          unsigned int p) {
    std::array<T, I> result = constant_array<T, I>(1);
    std::array<T, I> current{x};
    constexpr auto bits = sizeof(unsigned int) * 8;
    unsigned int i = 0;
    while (i < bits && (1u << i) <= p) {
        if (p & (1u << i)) result = result * current;
        current = current * current;
        i++;
    }
    return result;
}

template <class T>
inline ALWAYS_INLINE constexpr T myPow(T x, unsigned int p) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type");
    T result{1};
    T current{x};
    constexpr auto bits = sizeof(unsigned int) * 8;
    unsigned int i = 0;
    while (i < bits && (1u << i) <= p) {
        if (i > 0) current *= current;
        if (p & (1u << i)) result *= current;
        i++;
    }
    return result;
}

// p static version of myPow
template <unsigned int P, class T>
inline ALWAYS_INLINE constexpr T myPow(T x) {
    T result{1};
    T current{x};
    constexpr auto bits = sizeof(unsigned int) * 8;

    loop<size_t, bits>([&](auto i) ALWAYS_INLINE {
        if (i > 0) current *= current;
        if (P & (1u << i)) result *= current;
    });

    return result;
}

template <size_t skip, size_t Dim, class T>
inline ALWAYS_INLINE constexpr std::array<T, Dim - 1> array_skip(
    std::array<T, Dim> a) {
    static_assert(0 <= skip && skip < Dim, "skip dim invalid");
    return array_concat(array_slice<0, skip>(a), array_slice<skip + 1, Dim>(a));
}

template <size_t Dim, class T, typename = std::enable_if_t<Dim >= 1>>
inline ALWAYS_INLINE constexpr T det(std::array<std::array<T, Dim>, Dim> mat) {
    if constexpr (Dim == 1) {
        return mat[0][0];
    } else {
        return loop_sum<size_t, Dim>([&](auto i) ALWAYS_INLINE {
            std::array<std::array<T, Dim - 1>, Dim - 1> submat;
            loop<size_t, Dim - 1>([&](auto j) ALWAYS_INLINE {
                submat[j] = array_skip<i>(mat[j + 1]);
            });

            return myPow(-1, i) * mat[0][i] * det(submat);
        });
    }
}

template <size_t Dim, class T, typename = std::enable_if_t<Dim >= 2>>
inline ALWAYS_INLINE constexpr std::array<T, Dim> outer_product(
    std::array<std::array<T, Dim>, Dim - 1> mat) {
    std::array<T, Dim> result;
    loop<size_t, Dim>([&](auto i) ALWAYS_INLINE {
        std::array<std::array<T, Dim - 1>, Dim - 1> submat;
        loop<size_t, Dim - 1>(
            [&](auto j) ALWAYS_INLINE { submat[j] = array_skip<i>(mat[j]); });

        result[i] = myPow(-1, i) * det(submat);
    });
    return result;
}

template <size_t Dim, class T>
UNI_HOST_DEVICE inline ALWAYS_INLINE constexpr T inner_product(
    std::array<T, Dim> u, std::array<T, Dim> v) {
    T result = 0;
    loop<size_t, Dim>([&](auto i) ALWAYS_INLINE { result += u[i] * v[i]; });
    return result;
}

template <size_t Dim1, size_t Dim2, class T1, class T2>
inline ALWAYS_INLINE constexpr std::array<T1, Dim1 * Dim2> Kronnecker_product(
    std::array<T1, Dim1> u, std::array<T2, Dim2> v) {
    std::array<T1, Dim1 * Dim2> result;
    loop<size_t, Dim1>([&](auto i) ALWAYS_INLINE {
        loop<size_t, Dim2>(
            [&](auto j) ALWAYS_INLINE { result[i * Dim1 + j] = u[i] * v[j]; });
    });
    return result;
}

template <size_t sDim, size_t Dim, class T>
inline ALWAYS_INLINE constexpr std::array<T, sDim> array_cut(
    std::array<T, Dim> u, size_t offset) {
    Assert(offset + sDim <= Dim, "Out of Range.\n");
    std::array<T, sDim> result;
    loop<size_t, sDim>([&](auto i)
                           ALWAYS_INLINE { result[i] = u[offset + i]; });
    return result;
}

template <class T>
inline ALWAYS_INLINE constexpr std::array<T, 3> cross_product(
    std::array<T, 3> a, std::array<T, 3> b) {
    std::array<T, 3> ret = {};
    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];
    return ret;
}

/**
 * @brief
 * 通过一个constexpr的std::array创建一个等长的在某些对应位置相等的constexprstd::array，其余位置为0，如取第1号元素相等：{1,2,3}->{0,2,0}
 *
 * @tparam T std::array中变量类型
 * @tparam N std::array长度
 * @tparam I 选取位置的index
 * @param a 原std::array
 * @return ALWAYS_INLINE constexpr 新std::array
 */
template <class T, class TI, size_t N, TI... I>
inline ALWAYS_INLINE constexpr auto create_array_by_some_indices(
    const std::array<T, N>& a, std::integer_sequence<TI, I...>) {
    std::array<T, N> res{};
    for (auto i : {I...}) res[i] = a[i];
    return res;
}

template <class T, T value, T... IS>
auto make_constant_interger_sequence_helper(std::integer_sequence<T, IS...>) {
    return std::integer_sequence<T,
                                 (IS, value)...>();  // 使用逗号运算符返回value
}

template <class T, T value, T N>
auto make_constant_interger_sequence() {
    return make_constant_interger_sequence_helper<T, value>(
        std::make_integer_sequence<T, N>());
}
// Primary template
template <typename T>
struct is_std_array : std::false_type {};

// Specialization for std::array
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

template <class T, size_t I>
auto find_first_not_zero(std::array<T, I> a) {
    size_t idx = I;
    bool found = false;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        if (a[i] != 0 && !found) {
            idx = i;
            found = true;
        }
    });
    return idx;
}

template <class T, size_t... Ns>
auto concat_arrays(const std::array<T, Ns>&... arrays) {
    constexpr size_t newSize = (Ns + ...);
    std::array<T, newSize> result;
    // size_t indices[] = {Ns...};
    size_t offset = 0;
    ((std::copy(arrays.begin(), arrays.end(), result.begin() + offset),
      offset += Ns),
     ...);
    return result;
}

template <typename T, size_t N, size_t I = 0>
constexpr T accumulate(const std::array<T, N>& arr) {
    if constexpr (I < N) {
        return arr[I] + accumulate<T, N, I + 1>(arr);
    } else {
        return T();
    }
}

template <size_t I, class T>
std::array<std::array<T, I>, I> outer_product1(std::array<T, I> a,
                                               std::array<T, I> b) {
    std::array<std::array<T, I>, I> result;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE {
        loop<size_t, I>([&](auto j)
                            ALWAYS_INLINE { result[i][j] = a[i] * b[j]; });
    });
    return result;
}

template <size_t I, class T>
std::array<T, I> outer_product1(std::array<T, I> a, T b) {
    std::array<T, I> result;
    loop<size_t, I>([&](auto i) ALWAYS_INLINE { result[i] = a[i] * b; });
    return result;
}

}  // namespace uni
#ifdef UNI_ENABLE_SW
#pragma swuc pop
#endif
