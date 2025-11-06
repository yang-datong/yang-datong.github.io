---
layout: default
---

# Armv8中在16bit数据中使用Dotprod的方法

**定理：对于任意一个 N 位（N-bit）的整数 A，都可以将其分解为一个 M 位的高位部分 A_h 和一个 (N-M) 位的低位部分 A_l，使得以下等式恒成立：**
$$
A = (A_h << (N-M)) + A_l
$$

- 其中 A_h = A >> (N-M)
- 其中 A_l = A & mask  (mask 是一个低 (N-M) 位全为1的掩码)

> 如 N = 16时，对于任意一个 16bit 类型的变量 A，都必然存在其高8位 A_h 和低8位 A_l，使得 A == (A_h << 8) + A_l 这个等式成立。

```cpp
// g++ -w
#include <cstdint>
#include <iostream>

int main() {
#if 0
  uint16_t A;
  uint8_t A_h = (A >> 8) & 0xff;
  uint8_t A_l = A & 0xff;
  bool eq = A == (A_h << 8) + A_l;
#else
  int16_t A;
  int8_t A_h = (A >> 8) & 0xff;
  uint8_t A_l = A & 0xff;
  bool eq = A == (A_h << 8) + A_l;
#endif
  std::cout << "eq:" << eq << std::endl;

  return 0;
}
```

## 8-bit

点积指令的基本操作是将两个向量中的对应元素相乘，然后将乘积累加。具体来说，这些指令在 128 位向量寄存器中对多个 8 位整数进行操作，并将结果累加到 32 位整数中。

其操作可以概括为以下步骤：

1.  **数据分组**：指令将一个 128 位的向量寄存器视为四个 32 位的元素通道。
2.  **逐元素相乘**：在每个通道内，将第一个源向量中的四个 8 位元素与第二个源向量中对应的四个 8 位元素进行相乘。
3.  **内部累加**：将这四个 8 位乘法的结果相加，得到一个 32 位的和。
4.  **累加到目标寄存器**：最后，将这个 32 位的和累加到目标向量寄存器中对应的 32 位元素上。

**举例说明 `SDOT` 指令的操作：**
$$
C_{0} = C_{0} + \sum_{j=0}^{3} \bigg( a_j b_j \bigg)
$$

*   **$C_{0}$**：代表目标累加寄存器中对应32位元素的 **初始值**。
*   **$a_j$** 和 **$b_j$**：分别代表两个源向量寄存器中对应32位元素内的四个8位整数。`j` 的范围从0到3，表示这四个8位元素。
*   **$\sum_{j=0}^{3} \bigg( a_j b_j \bigg)$**：这部分表示点积运算。它计算四个8位元素对 (`a₀*b₀`, `a₁*b₁`, `a₂*b₂`, `a₃*b₃`) 的乘积，然后将这四个乘积相加。
*   **$C_{0}$**：代表运算结束后写入目标寄存器该32位元素的 **最终结果**。

这只是一次的点积运算，一般在NEON中会有4个点积同时进行，如:

```asm
sdot v0.4s, v1.16b, v2.16b
```

它表示为：

- **通道 0**: v0.s[0] += dot_product(v1.b[0..3], v2.b[0..3])
- **通道 1**: v0.s[1] += dot_product(v1.b[4..7], v2.b[4..7])
- **通道 2**: v0.s[2] += dot_product(v1.b[8..11], v2.b[8..11])
- **通道 3**: v0.s[3] += dot_product(v1.b[12..15], v2.b[12..15])

所以这从根本上就限制了只能处理 8 位输入源。

> 在256位SVE硬件上，sdot z0.d, z1.h, z2.h这样可以表示输入源为16bit.

## 16-bit

对于8bit的A、B其点积表示为：$Sum(A*B)$

而对于16bit的A、B其点积表示为：
$$
A      = (A_h << 8) + A_l                             \\
A * B  = (( A_h << 8 ) + A_l) *  (( B_h << 8 ) + B_l) \\
A * B =
		\begin{cases}
			A_h B_h << 16 +  (A_h B_l +  A_l B_h) << 8  + A_l B_l , & A \neq B \\
			A_h^{2} << 16 + ( A_h A_l << 9 ) + A_l^{2} ,            & A = B
		\end{cases} \\
Sum( A * B ) = Sum( A_h B_h) << 16 + [Sum( A_h B_l ) + Sum(A_l B_h)] << 8 + Sum( A_l B_l )
$$

其中，$A_h,A_l$分别为$A$的高、低8bit，当A为有符号数值时，$A_h$同样是有符号的int8，而$A_l$是uint8

计算无符号16位点积需要 **4个 udot 指令**来计算四个部分的累加和，计算有符号16位点积需要 **1个sdot，2个usdot，和1个udot** 指令组合。



## 溢出条件

### 8bit

在sdot指令中，对有符号位8-bit容器，其取值范围在$a_j, b_j \in [-0x80,0x7f]$，其单次点积表示为32-bit的$C_0$，则有：
$$
C_0 \in 4 * [-0x80,0x7f]^2 = 4 * [-0x80 * 0x7f, (-0x80) * (-0x80)] = [-0xfe00,0x10000]
$$
这个值完全可以使用32-bit的容器装下，而对于$N$次点积可以表示为$C_{N-1} \in 4N * (a_j,b_j)$，其$N$的最大值可以表示为：
$$
C_{N-1} \in 4N *[-0x80,0x7f]^2 \subset [-0x80000000,0x7fffffff] \\
N \leq \frac{-0x80000000}{-0x80*0x7f} * \frac{1}{4} \approx 0x8102  \quad (\text{负溢出})\\
N \leq \frac{0x7fffffff}{(0x80)^2} * \frac{1}{4} \approx  0x7fff  \quad (\text{正溢出}) \\
$$
这时候继续点积则有可能会导致溢出，，所以安全迭代次数为N = 0x7fff

同理，对于在udot指令中，对无符号位8-bit容器，其取值范围在$a_j, b_j \in [0xff,0xff]$，其单次点积表示为32-bit的$C_0$，则有：
$$
C_0 \in 4 * [0 , (0xff)^2] = [0, 0x3f804]
$$
安全迭代次数为N = 0xffffffff / 0x3f804 $\approx$ 0x4080

> 注意，累加器32bit的符号位由输入源决定，输入源存在符号则累加器必须也是有符号。

### 16bit

对于8-bit的A、B其点积表示为：$Sum(A*B)$

则对于16-bit的A、B其点积表示为：
$$
Sum( A * B ) = Sum( A_h B_h) << 16 + [Sum(B_l A_h) + Sum(A_l B_h)] << 8 + Sum( A_l B_l )
$$

在实际使用中，可以使用3个或4个 NEON 寄存器来维护累加结果，使用3个可以减少寄存器的使用，但累加的数值范围就相对变小，对于4个累加器的情况：

- sdot(int8 * int8)，单次点积取值范围是[-0xfe00, 0x10000]
- udot(uint8 * uint8)，单次点积取值范围是[0, 0x3f804]
- usdot(uint8 * int8)，单次点积取值范围是[-0x1fe00, 0x1fa04]

可知udot会最快产生正向溢出（N = 0x7fffffff / 0x3f804 $\approx$ 0x2040)，usdot会最快产生负向溢出（N = -0x80000000 / -0x1fe00 $\approx$ 0x4040)，那么安全迭代次数为 N = 0x2040

对于3个累加器的情况：将 $Sum(A_h*B_l)$ 和 $Sum(A_l*B_h)$ 这两个 usdot 的结果累加到同一个32位寄存器中，则 usdot 单次点积取值范围是 [0xff * (-0x80) * 8,0xff * 0x7f * 8] = [-0x3fc00, 0x3f408]，那么安全迭代次数为 N = min{0x2020, 0x2060} = 0x2020



## 代码实现

demo.c

```c
// ME: gcc demo.c demo.S -g3 -O3 -march=armv8.4-a+dotprod -w && ./a.out
// ME: gcc demo.c demo.S -g3 -O3 -march=armv8.6-a+i8mm -w && ./a.out
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 定义数据的位深度，8表示8 bit, 16表示16 bit
#define HIGHT_BIT_DEPTH 16
// 定义符号位, 1表示有符号数, 0表示无符号数
#define SIGN 0

// 定义数组的长度，16的倍数，以便NEON指令高效处理(0x2020是最大的安全值，对于16 bit)
#define VECTOR_LENGTH 0x2020
/*#define VECTOR_LENGTH 8*/

__attribute__((noinline)) int32_t dot_product_s8_c(const int8_t *vec1,
                                                   const int8_t *vec2,
                                                   int length) {
  int32_t n = 0;
  for (int i = 0; i < length; ++i)
    n += vec1[i] * vec2[i];

  return n;
}

__attribute__((noinline)) uint32_t dot_product_u8_c(const uint8_t *vec1,
                                                    const uint8_t *vec2,
                                                    int length) {
  uint32_t n = 0;
  for (int i = 0; i < length; ++i)
    n += vec1[i] * vec2[i];

  return n;
}

__attribute__((noinline)) int64_t dot_product_s16_c(const int16_t *vec1,
                                                    const int16_t *vec2,
                                                    int length) {
  int64_t n = 0;
  for (int i = 0; i < length; ++i)
    n += vec1[i] * vec2[i];

  return n;
}

__attribute__((noinline)) uint64_t dot_product_u16_c(const uint16_t *vec1,
                                                     const uint16_t *vec2,
                                                     int length) {
  uint64_t n = 0;
  for (int i = 0; i < length; ++i)
    n += vec1[i] * vec2[i];

  return n;
}

// ./demo.S:25
int32_t dot_product_s8_dotprod(const int8_t *vec1, const int8_t *vec2,
                               int length);
uint32_t dot_product_u8_dotprod(const uint8_t *vec1, const uint8_t *vec2,
                                int length);
// ./demo.S:67
int64_t dot_product_s16_dotprod(const int16_t *vec1, const int16_t *vec2,
                                int length);
int64_t dot_product_u16_dotprod(const uint16_t *vec1, const uint16_t *vec2,
                                int length);

void foo() {
#if HIGHT_BIT_DEPTH == 8
#if SIGN == 1
  int8_t vector_a[VECTOR_LENGTH], vector_b[VECTOR_LENGTH];
#else
  uint8_t vector_a[VECTOR_LENGTH], vector_b[VECTOR_LENGTH];
#endif
#elif HIGHT_BIT_DEPTH == 16
#if SIGN == 1
  int16_t vector_a[VECTOR_LENGTH], vector_b[VECTOR_LENGTH];
#else
  uint16_t vector_a[VECTOR_LENGTH], vector_b[VECTOR_LENGTH];
#endif
#endif

  srand(time(NULL));

  // Test data
  for (int i = 0; i < VECTOR_LENGTH; ++i)
    /*vector_a[i] = i % 20, vector_b[i] = 7 * i % 100;*/
    vector_a[i] = (rand() % 201), vector_b[i] = (rand() % 301);

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
#if HIGHT_BIT_DEPTH == 8
#if SIGN == 1
  int32_t result = dot_product_s8_c(vector_a, vector_b, VECTOR_LENGTH);
#else
  uint32_t result = dot_product_u8_c(vector_a, vector_b, VECTOR_LENGTH);
#endif
#elif HIGHT_BIT_DEPTH == 16
#if SIGN == 1
  int64_t result = dot_product_s16_c(vector_a, vector_b, VECTOR_LENGTH);
#else
  uint64_t result = dot_product_u16_c(vector_a, vector_b, VECTOR_LENGTH);
#endif
#endif
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);

  long long start_ns =
      (long long)start_time.tv_sec * 1000000000 + start_time.tv_nsec;
  long long end_ns = (long long)end_time.tv_sec * 1000000000 + end_time.tv_nsec;
  long elapsed_ns = end_ns - start_ns;
  printf("C Number of ns -> %ld,r:%lld\n", elapsed_ns, result);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
#if HIGHT_BIT_DEPTH == 8
#if SIGN == 1
  result = dot_product_s8_dotprod(vector_a, vector_b, VECTOR_LENGTH);
#else
  result = dot_product_u8_dotprod(vector_a, vector_b, VECTOR_LENGTH);
#endif
#elif HIGHT_BIT_DEPTH == 16
#if SIGN == 1
  result = dot_product_s16_dotprod(vector_a, vector_b, VECTOR_LENGTH);
#else
  result = dot_product_u16_dotprod(vector_a, vector_b, VECTOR_LENGTH);
#endif
#endif
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
  start_ns = (long long)start_time.tv_sec * 1000000000 + start_time.tv_nsec;
  end_ns = (long long)end_time.tv_sec * 1000000000 + end_time.tv_nsec;
  elapsed_ns = end_ns - start_ns;
  printf("Dotprod Number of ns -> %ld,r:%lld\n", elapsed_ns, result);
}

int main() {
  printf("HIGHT_BIT_DEPTH:%d,SIGN:%d\n", HIGHT_BIT_DEPTH, SIGN);
  for (int i = 0; i < 10; ++i)
    foo();
  return 0;
}
```

demo.S

```asm
// ME: gcc demo.c demo.S -g3 -O3 -march=armv8.4-a+dotprod && ./a.out
// ME: gcc demo.c demo.S -g3 -O3 -march=armv8.6-a+i8mm && ./a.out

// .arch armv8.4-a+dotprod
.arch armv8.6-a+i8mm

#if defined(__APPLE__) && defined(__MACH__)
	#define NAME(name) _##name
#else
	#define NAME(name) name
#endif

.macro movrel rd, val, offset=0
#if defined(__APPLE__)
	adrp \rd, \val+(\offset)@PAGE
	add  \rd, \rd, \val+(\offset)@PAGEOFF
#elif defined(__linux__)
	adrp \rd, \val+(\offset)
	add  \rd, \rd, :lo12:\val+(\offset)
#endif
.endm

// int32_t dot_product_s8_dotprod(const int8_t *vec1, const int8_t *vec2, int length)
// uint32_t dot_product_u8_dotprod(const uint8_t *vec1, const uint8_t *vec2, int length)
.macro dot_product_dotprod sign
NAME(dot_product_\sign\()_dotprod):
0:
	movi v0.4s, #0

1:
	ldr  q1, [x0], #0x10
	ldr  q2, [x1], #0x10

.ifc \sign, s8
	sdot v0.4s, v1.16b, v2.16b
.else
	udot v0.4s, v1.16b, v2.16b
.endif

	subs x2, x2, #0x10
	b.gt 1b

2:
	addv s0, v0.4s
	fmov w0, s0
	ret
.endm

.global NAME(dot_product_s8_dotprod)
.global NAME(dot_product_u8_dotprod)

dot_product_dotprod s8
dot_product_dotprod u8

.purgem dot_product_dotprod

// NOTE: s16需要用到usdot，这是属于I8MM扩展指令集的，需要在arch中指定(u16不需要i8mm)
// int16_t A = 0xfedc; int8_t A_h = A >> 8; uint8_t A_l = A & 0xff;
// uint16_t A = 0xfedc; uint8_t A_h = A >> 8; uint8_t A_l = A & 0xff;
// bool iseq = (A == ((int16_t)(A_h << 8) + A_l));
// Sum( A * B ) = Sum( A_h B_h) << 16 + Sum( A_h B_l + A_l B_h ) << 8 + Sum( A_l B_l )

// int64_t dot_product_s16_dotprod(const int16_t *vec1, const int16_t *vec2, int length);
// uint64_t dot_product_u16_dotprod(const uint16_t *vec1, const uint16_t *vec2, int length);
.macro dot_product_dotprod sign
NAME(dot_product_\sign\()_dotprod):
0:
	movi v0.4s, #0
	movi v1.4s, #0
	movi v2.4s, #0
	lsl  x8, x2, #1

1:
	ldp q3, q4, [x0], #0x20
	ldp q5, q6, [x1], #0x20

	subs x8, x8, #0x20

	uzp1 v7.16b, v3.16b, v4.16b  // v7 = A_l (uint8_t|uint8_t)
	uzp2 v8.16b, v3.16b, v4.16b  // v8 = A_h (int8_t|uint8_t)
	uzp1 v9.16b, v5.16b, v6.16b  // v9 = B_l (uint8_t|uint8_t)
	uzp2 v10.16b, v5.16b, v6.16b // v10 = B_h (int8_t|uint8_t)

	// Sum(A_l * B_l)
	udot v0.4s, v7.16b, v9.16b
.ifc \sign, s16
	// Sum(A_h * B_l + A_l * B_h)
	usdot v1.4s, v9.16b, v8.16b  // B_l * A_h
	usdot v1.4s, v7.16b, v10.16b // A_l * B_h
	// Sum(A_h * B_h)
	sdot v2.4s, v8.16b, v10.16b
.else
	udot v1.4s, v9.16b, v8.16b
	udot v1.4s, v7.16b, v10.16b
	udot v2.4s, v8.16b, v10.16b
.endif

	b.gt 1b

2:
	uaddlv d0, v0.4s  // Sum(A_l * B_l)
.ifc   \sign, s16
	saddlv d1, v1.4s  // Sum(A_h * B_l + A_l * B_h)
	saddlv d2, v2.4s  // Sum(A_h * B_h)
.else
	uaddlv d1, v1.4s
	uaddlv d2, v2.4s
.endif

	fmov x0, d0 // sum_l
	fmov x1, d1 // sum_mid
	fmov x2, d2 // sum_h

	// dotprod = (sum_h << 16) + (sum_mid << 8) + sum_l
	lsl x2, x2, #16
	lsl x1, x1, #8
	add x0, x0, x1
	add x0, x0, x2

	ret
.endm

.global NAME(dot_product_s16_dotprod)
.global NAME(dot_product_u16_dotprod)

dot_product_dotprod s16
dot_product_dotprod u16
```



## 性能提升

测试环境：

```sh
OS: Android 16 aarch64
Host: HONOR MTG-AN00
Kernel: 6.12.23-android16-5-g4bb20b674611-abogki361520241-4k
CPU: SM8850 (8) @ 2.380GHz
Memory: 5200MiB / 11293MiB
```

测试情况：

```sh
$ gcc demo.c demo.S -g3 -O3 -march=armv8.6-a+i8mm -w && ./a.out
HIGHT_BIT_DEPTH:16,SIGN:0
C Number of ns -> 6771,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5729,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5729,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5729,r:123988895
Dotprod Number of ns -> 1250,r:123988895
C Number of ns -> 5625,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5677,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5677,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5729,r:123988895
Dotprod Number of ns -> 1302,r:123988895
C Number of ns -> 5677,r:123988895
Dotprod Number of ns -> 1355,r:123988895
C Number of ns -> 5729,r:123988895
Dotprod Number of ns -> 1355,r:123988895
```

性能大约是C语言版本的 4.4 倍
