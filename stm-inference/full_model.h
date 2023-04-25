#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_141_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_141(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       4000
#define CONV_FILTERS        2
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_154_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_154(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      2
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_154_bias[CONV_FILTERS] = {0, 5}
;

const int16_t conv1d_154_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-114, -118, -241}
}
, {{30, -381, -103}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  2
#define INPUT_SAMPLES   3998
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_142_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_142(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      2
#define INPUT_SAMPLES       999
#define CONV_FILTERS        4
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_155_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_155(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    2
#define CONV_FILTERS      4
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_155_bias[CONV_FILTERS] = {1, 2, -4, 1}
;

const int16_t conv1d_155_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-28, -266, 263}
, {264, -158, -43}
}
, {{-68, -201, -27}
, {-170, 237, 117}
}
, {{-82, 91, -286}
, {-281, -86, 166}
}
, {{-107, -244, -252}
, {-7, 70, 18}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  4
#define INPUT_SAMPLES   997
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_143_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_143(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      4
#define INPUT_SAMPLES       249
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_156_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_156(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    4
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_156_bias[CONV_FILTERS] = {0, 5, 0, 2, 10, 0, 4, 4}
;

const int16_t conv1d_156_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-71, -116, -157}
, {-195, -107, 55}
, {-98, -116, -9}
, {124, -66, 1}
}
, {{-39, -32, -161}
, {-49, 117, -1}
, {-82, 73, 89}
, {-164, 57, 135}
}
, {{-158, 204, -154}
, {-95, -158, -66}
, {-26, -54, -91}
, {-134, 30, 85}
}
, {{54, -105, 139}
, {-159, 75, 105}
, {209, 149, -159}
, {159, 95, -118}
}
, {{-2, -21, 46}
, {147, -133, 62}
, {-116, -3, -24}
, {-210, -50, 89}
}
, {{43, 126, -151}
, {-74, -2, -11}
, {-2, 116, -195}
, {95, -70, 31}
}
, {{-5, 137, 104}
, {-119, -88, -201}
, {49, 62, 6}
, {-17, 28, -206}
}
, {{-158, -26, 66}
, {-78, 43, -178}
, {-174, -108, 13}
, {161, 93, -34}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   247
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_144_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_144(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       61
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_157_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_157(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_157_bias[CONV_FILTERS] = {5, -2, 5, -4, 5, 1, -3, 6, 9, 4, 11, 8, 7, 4, 6, 5}
;

const int16_t conv1d_157_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{100, 110, 102}
, {-11, 141, -124}
, {-131, -132, -135}
, {-53, 69, -39}
, {-110, 104, -127}
, {62, 87, 117}
, {-82, -139, -56}
, {-39, -142, -105}
}
, {{3, 24, -87}
, {-140, 33, -132}
, {-99, -115, -60}
, {37, 35, -70}
, {-104, -132, -95}
, {0, 133, -145}
, {6, -121, -30}
, {142, 125, -83}
}
, {{-148, 20, -82}
, {-111, 64, 68}
, {97, -19, 85}
, {-88, 12, -118}
, {13, -29, -86}
, {124, -61, -134}
, {-42, 39, 19}
, {134, -23, 148}
}
, {{125, 46, 13}
, {-4, 7, -116}
, {-99, 32, 12}
, {18, -25, 10}
, {-19, -104, -54}
, {-50, 116, -91}
, {-15, 22, 50}
, {-121, -15, -89}
}
, {{53, 87, -62}
, {126, 139, 121}
, {111, -133, -141}
, {-55, -116, 85}
, {14, -2, 129}
, {-141, 105, 36}
, {-140, -23, 32}
, {-66, -99, -62}
}
, {{-25, 25, -131}
, {-130, 4, 75}
, {92, -116, 76}
, {-10, -108, 0}
, {-23, -55, -137}
, {76, -133, -110}
, {-139, 123, 73}
, {-29, 101, -108}
}
, {{-37, -119, -74}
, {91, 98, 32}
, {61, 74, 23}
, {22, -94, 65}
, {-120, 133, -42}
, {32, 140, -66}
, {-37, 142, 147}
, {-96, -24, -143}
}
, {{50, -72, -148}
, {-65, -70, -125}
, {30, 14, -133}
, {128, 99, -96}
, {114, 156, 25}
, {40, -111, 148}
, {-123, 153, 133}
, {-31, 108, 133}
}
, {{130, 143, 119}
, {-50, -11, 136}
, {11, -13, -104}
, {-20, 21, -103}
, {93, 100, 8}
, {42, -33, -34}
, {76, -56, -116}
, {142, 60, 124}
}
, {{-7, -112, 110}
, {92, 64, -26}
, {150, -117, 144}
, {105, 91, -84}
, {-15, 33, 121}
, {-53, -7, 99}
, {-55, -125, 37}
, {75, 77, -108}
}
, {{140, -148, -81}
, {108, -131, -134}
, {-4, 134, -130}
, {-64, 137, -34}
, {107, 76, -127}
, {98, 151, -103}
, {109, 135, -11}
, {88, 109, -15}
}
, {{-132, 44, 5}
, {-119, -12, 96}
, {94, -113, -30}
, {35, 62, -129}
, {-92, -116, 20}
, {69, 108, -45}
, {9, 53, -14}
, {145, 57, -39}
}
, {{84, 131, 78}
, {113, -122, 15}
, {101, -75, 92}
, {110, -94, -18}
, {-23, 102, 70}
, {-50, -49, -108}
, {-20, 37, -41}
, {-57, 87, 115}
}
, {{-84, 50, 81}
, {-113, -120, -102}
, {124, -21, -52}
, {-56, 110, 15}
, {151, 112, -92}
, {-137, 75, 74}
, {153, -7, 152}
, {108, -94, 144}
}
, {{1, -118, -112}
, {23, 6, 118}
, {-109, -94, 64}
, {-110, -3, 112}
, {87, 112, 92}
, {33, -72, -96}
, {49, -115, 126}
, {-115, 81, 11}
}
, {{-118, -13, 45}
, {-91, -59, -57}
, {-13, 24, 123}
, {145, -37, 123}
, {-13, -53, 138}
, {101, -25, -88}
, {-131, -25, 11}
, {-134, -54, -20}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       59
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_158_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_158(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_158_bias[CONV_FILTERS] = {2, 8, 11, 4, 6, 10, 0, -3, 7, 5, 9, 9, 2, -6, 4, 7, 0, 1, 6, 7, 8, 10, 5, -3, 3, 7, 8, 8, -5, 8, 5, -1}
;

const int16_t conv1d_158_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{76, -58, -10}
, {-12, 11, 103}
, {24, 106, -50}
, {-79, -22, 49}
, {-41, 35, 101}
, {-70, 71, -10}
, {5, 25, -57}
, {93, 93, 0}
, {-23, 78, -6}
, {-16, 36, -34}
, {-12, 30, 36}
, {15, 33, 95}
, {32, 108, 67}
, {67, -34, 60}
, {24, -102, 84}
, {-49, -71, 45}
}
, {{-100, 38, -88}
, {40, -91, -16}
, {47, 80, -3}
, {97, 81, 82}
, {102, 2, -40}
, {85, -8, 58}
, {102, 35, -16}
, {-76, 53, -27}
, {25, -69, 67}
, {61, 103, 88}
, {-27, -55, -75}
, {-90, -91, -24}
, {-96, -18, 23}
, {64, -25, -77}
, {59, -7, 14}
, {-77, -69, 35}
}
, {{42, 59, 95}
, {-69, -104, 101}
, {24, 95, 63}
, {-89, -17, -55}
, {67, 86, -51}
, {-47, -6, -61}
, {-95, -15, -45}
, {76, 58, -66}
, {-11, 51, -69}
, {-26, 27, -13}
, {44, -37, 69}
, {-8, -69, -4}
, {-35, -81, -75}
, {-90, 75, 43}
, {-34, -97, 79}
, {41, 36, -9}
}
, {{25, -83, -99}
, {59, 37, -86}
, {-80, -61, -72}
, {97, 82, -14}
, {44, -58, -3}
, {-43, -92, 102}
, {21, 0, 39}
, {-16, 102, 7}
, {17, -27, 70}
, {-54, -87, 105}
, {16, -94, 2}
, {-43, 40, 81}
, {47, -41, -85}
, {65, -101, -100}
, {-66, 47, 36}
, {-76, -95, -68}
}
, {{34, 101, 66}
, {94, 45, 68}
, {27, -83, -30}
, {-25, 1, -50}
, {73, 19, -91}
, {91, 77, 76}
, {94, 49, 108}
, {34, 106, -74}
, {21, 83, -81}
, {57, -43, -70}
, {-16, 17, 105}
, {43, 74, -34}
, {-32, 102, 75}
, {-14, -24, 92}
, {-23, 80, 31}
, {96, 51, -99}
}
, {{-22, -80, 29}
, {97, 22, -89}
, {-80, 50, 15}
, {-13, 93, 59}
, {-55, -2, 58}
, {83, -63, -32}
, {-33, -33, -18}
, {-61, 45, 65}
, {17, 85, 89}
, {90, 14, -4}
, {-57, 111, 75}
, {-5, 96, -68}
, {-79, -48, -28}
, {62, 109, 18}
, {-90, 12, -26}
, {-31, -32, -40}
}
, {{-65, 70, 41}
, {8, -100, 86}
, {-69, 42, 43}
, {-34, -89, -37}
, {-16, -87, -98}
, {-65, 3, 16}
, {-78, -70, -95}
, {-87, -103, -56}
, {63, 4, -2}
, {-63, -36, 29}
, {-24, 47, -19}
, {-102, 14, 9}
, {59, -106, -91}
, {99, -47, -21}
, {-75, 74, -88}
, {73, -1, 63}
}
, {{-65, -88, 6}
, {82, 27, 21}
, {69, -99, 48}
, {-64, -87, -73}
, {-56, -99, 62}
, {96, 80, 64}
, {52, 5, 50}
, {-85, -61, -66}
, {57, -107, -76}
, {33, -18, -51}
, {-8, 87, 71}
, {-96, -55, 67}
, {38, -70, 16}
, {-15, 29, -34}
, {-7, 65, -80}
, {-108, -91, 44}
}
, {{50, -69, -57}
, {-92, -43, 5}
, {54, 90, -63}
, {69, -7, 4}
, {15, 21, 58}
, {102, -12, 70}
, {13, 95, 104}
, {-53, 44, 74}
, {16, 93, 90}
, {26, 56, -69}
, {57, 113, -79}
, {-50, 29, 0}
, {8, -61, 71}
, {25, 107, -69}
, {52, 78, -47}
, {35, 54, 58}
}
, {{-96, -51, 46}
, {-62, -94, 8}
, {103, -57, 0}
, {-8, 25, 27}
, {88, -97, 28}
, {76, 100, -35}
, {-92, -21, -9}
, {-27, 57, -18}
, {65, 37, 14}
, {88, -93, 39}
, {43, 52, -81}
, {73, 51, 57}
, {91, -80, 24}
, {-26, -63, 42}
, {-87, 21, 73}
, {61, -70, 35}
}
, {{72, 71, -10}
, {-3, -23, 29}
, {32, 79, -36}
, {92, -19, -14}
, {-5, 99, 67}
, {-61, -20, 63}
, {83, -79, -100}
, {-63, -59, 84}
, {-90, -53, -63}
, {96, 18, 4}
, {74, -10, -12}
, {-22, 10, 11}
, {30, 85, 82}
, {60, -30, -90}
, {-62, 0, -4}
, {90, -90, -54}
}
, {{-38, 101, -53}
, {-13, -73, -25}
, {74, 69, 58}
, {-83, -4, -26}
, {-76, -85, 26}
, {-98, -44, -67}
, {34, -44, -32}
, {-64, 49, -38}
, {52, 98, -39}
, {-38, -71, -42}
, {-51, 28, -87}
, {76, 54, 12}
, {25, -70, -61}
, {-49, -24, 40}
, {54, 73, 32}
, {50, 20, -56}
}
, {{-74, 45, -64}
, {-30, 74, 81}
, {10, -40, -16}
, {44, -58, 85}
, {58, -71, 76}
, {-30, 70, 8}
, {73, -73, -12}
, {-85, -61, -27}
, {54, -68, 102}
, {58, 30, -13}
, {91, -7, 105}
, {102, -50, -64}
, {14, 66, -34}
, {-56, -96, 7}
, {70, -30, 20}
, {3, -72, 91}
}
, {{96, 27, 8}
, {-93, 18, 23}
, {-11, -73, -12}
, {46, -12, 44}
, {8, -83, 8}
, {30, -100, 80}
, {-3, 96, -5}
, {-61, 91, 18}
, {-12, -11, -7}
, {-12, 72, 43}
, {0, -70, -19}
, {96, 43, 89}
, {-50, -61, -102}
, {45, 27, -54}
, {-64, 105, 76}
, {-17, -24, -62}
}
, {{-101, 82, -98}
, {6, -87, 67}
, {-102, 6, 99}
, {89, 19, -54}
, {71, -103, 93}
, {-32, -101, -62}
, {105, 99, -43}
, {60, -28, -40}
, {-87, -61, 110}
, {8, 102, -38}
, {-40, -13, 68}
, {-52, 92, 53}
, {60, 53, -50}
, {-83, -58, -76}
, {-88, -50, 61}
, {81, 59, 87}
}
, {{73, -63, -66}
, {66, 16, -32}
, {-45, 71, -24}
, {89, -75, -63}
, {36, -68, 26}
, {50, -45, -54}
, {21, -29, 3}
, {112, -10, -27}
, {-72, 102, 5}
, {-27, 93, -34}
, {-41, -32, -59}
, {92, 49, 5}
, {1, -92, 84}
, {73, 97, 13}
, {106, 8, -5}
, {-34, -62, 0}
}
, {{-42, -69, 44}
, {10, 39, 91}
, {-3, 22, -59}
, {65, 76, 52}
, {74, -54, 46}
, {9, -92, 0}
, {-72, -48, 29}
, {-52, -25, 57}
, {-62, 87, 103}
, {-29, -94, 73}
, {-102, -43, 6}
, {-81, -16, 27}
, {96, -98, 3}
, {-99, -71, -88}
, {105, 68, 34}
, {-6, -96, 29}
}
, {{-62, 25, 22}
, {22, -48, 81}
, {-95, -64, 26}
, {-37, -44, -31}
, {37, 75, 92}
, {-93, 34, 60}
, {-74, 43, -13}
, {-80, 51, 35}
, {-83, -56, 85}
, {-40, 38, 29}
, {35, 80, 69}
, {27, 4, 12}
, {-40, 93, 38}
, {27, 22, 49}
, {-91, -77, -114}
, {-3, 45, -87}
}
, {{84, 57, 13}
, {94, -75, 6}
, {56, -38, -28}
, {-97, -75, 95}
, {51, 23, 25}
, {23, -64, -91}
, {-95, -75, -30}
, {61, -7, 27}
, {2, -60, -67}
, {-23, -53, 57}
, {-31, 61, 34}
, {31, -65, 75}
, {-65, 14, 29}
, {-89, -29, 18}
, {37, -77, -100}
, {56, -31, 4}
}
, {{-104, 62, 12}
, {51, 11, -54}
, {-29, -56, -80}
, {-67, -65, -35}
, {62, -2, 3}
, {-50, -83, 8}
, {-53, 27, -28}
, {23, 92, -30}
, {87, -63, 55}
, {-7, -41, 19}
, {79, -17, 85}
, {50, -49, 85}
, {-54, 66, -70}
, {83, 11, 92}
, {55, -49, 9}
, {66, -15, -25}
}
, {{49, 12, -76}
, {-2, 68, 19}
, {-90, 18, 22}
, {-83, 33, 64}
, {5, 31, -39}
, {60, -100, -67}
, {52, 41, -49}
, {-68, 17, -89}
, {22, -24, 105}
, {21, -66, 91}
, {-58, 26, 8}
, {6, 75, -13}
, {53, 2, 31}
, {-52, -50, 96}
, {19, -78, 83}
, {-84, -15, 18}
}
, {{15, 23, 49}
, {60, -72, -62}
, {14, -4, 66}
, {-55, -68, -96}
, {44, 71, 50}
, {-38, 59, 3}
, {-21, 33, -72}
, {-86, 6, 10}
, {1, 0, 66}
, {-101, 5, 5}
, {53, 86, 102}
, {77, -32, 105}
, {0, 68, 9}
, {56, -44, 100}
, {108, -96, 0}
, {-11, -37, 16}
}
, {{18, -42, -61}
, {24, -79, 39}
, {-17, -21, -91}
, {48, -29, -35}
, {74, -58, 33}
, {-88, -96, 96}
, {-17, 44, -20}
, {22, 26, -18}
, {-56, 25, 55}
, {-95, 7, -6}
, {-67, -26, 17}
, {-73, -8, -37}
, {23, -11, 57}
, {87, 16, 69}
, {-40, 43, -16}
, {19, 84, 38}
}
, {{-99, -98, 0}
, {15, 37, 29}
, {-9, -74, -18}
, {37, 70, 3}
, {28, -29, 86}
, {74, -48, -48}
, {45, -1, 38}
, {-5, 32, -21}
, {65, 12, 28}
, {44, 93, 90}
, {39, -36, 69}
, {-29, -65, 84}
, {-4, -98, 24}
, {86, 8, -69}
, {82, -84, -44}
, {-4, 84, 19}
}
, {{13, -15, -6}
, {80, -76, 100}
, {-33, 71, 1}
, {-95, 73, 64}
, {13, 7, 21}
, {-3, -60, 45}
, {-23, -39, 38}
, {-78, 12, -7}
, {-53, 53, 1}
, {-36, -6, -52}
, {94, -60, -37}
, {2, -57, -75}
, {-6, -2, 62}
, {-75, 68, -21}
, {-40, -9, -17}
, {27, -91, -51}
}
, {{19, -86, 104}
, {-56, -97, -78}
, {102, 14, 40}
, {-91, -79, 61}
, {-43, -42, -96}
, {75, 67, 11}
, {24, -61, -51}
, {-12, -74, 35}
, {-55, 24, 82}
, {15, 79, 37}
, {-55, 77, 63}
, {26, -75, -9}
, {-94, -14, -19}
, {-79, 49, -45}
, {89, 1, -50}
, {-35, 59, 94}
}
, {{51, 87, 59}
, {-87, 91, -51}
, {95, -103, 83}
, {-5, -47, -72}
, {11, 104, -71}
, {-12, -23, 93}
, {78, -3, -4}
, {77, 60, 33}
, {-15, -53, -14}
, {-88, 5, 35}
, {-45, 83, -52}
, {-97, 97, 67}
, {40, 47, 7}
, {4, -18, 0}
, {38, 94, 72}
, {36, 82, 44}
}
, {{0, 58, -37}
, {66, 51, -15}
, {103, 42, 54}
, {100, 96, -43}
, {64, -21, -28}
, {101, 5, -56}
, {-12, 12, 40}
, {4, -54, -26}
, {34, -37, 108}
, {75, 76, 67}
, {-62, -13, 49}
, {-50, -72, 96}
, {-8, 1, -85}
, {2, -36, 1}
, {29, -86, -65}
, {-43, -61, 106}
}
, {{-6, 100, -15}
, {-98, -97, -33}
, {23, 25, 103}
, {-80, -55, 20}
, {37, 65, -57}
, {-21, 85, 7}
, {28, -60, -5}
, {54, 37, -94}
, {-78, 89, -72}
, {-32, 84, 33}
, {-90, 86, 6}
, {-97, -71, 18}
, {102, 8, 40}
, {42, 20, 77}
, {-51, -24, 41}
, {39, 81, -25}
}
, {{98, 31, 12}
, {32, 36, -96}
, {87, 89, 95}
, {17, 88, 9}
, {-83, 94, -50}
, {37, -36, -82}
, {-100, -83, -12}
, {20, 52, 107}
, {-34, -26, -2}
, {90, -14, -58}
, {-45, -73, 90}
, {15, -17, -39}
, {-65, -70, -52}
, {46, -91, -19}
, {-91, -67, -100}
, {-56, 100, 84}
}
, {{-78, 8, -14}
, {77, 20, 26}
, {100, 39, -55}
, {-3, 76, 71}
, {72, 84, -22}
, {97, -5, 81}
, {79, -53, 17}
, {60, 40, -79}
, {53, -93, 23}
, {51, 35, 1}
, {-19, -75, -29}
, {-25, -6, -51}
, {47, 10, -92}
, {-101, 30, -94}
, {15, 64, 41}
, {51, 44, -28}
}
, {{25, -62, -45}
, {10, -25, -54}
, {46, -44, -93}
, {66, 79, 68}
, {103, -51, -102}
, {41, -85, 63}
, {-9, 86, 10}
, {108, 98, -39}
, {-19, -49, -31}
, {18, 89, 47}
, {50, 92, 45}
, {-100, -3, -86}
, {69, -36, 85}
, {53, -41, 101}
, {47, -52, -41}
, {78, -67, 106}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       57
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_159_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_159(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_159_bias[CONV_FILTERS] = {12, 4, -2, -6, 2, 4, 1, -2, 4, 1, -6, -3, 7, 9, 8, 9, 0, -2, 5, -2, 2, 7, -1, -5, 3, 1, 3, 4, 0, 8, 4, 7, 11, 9, 2, -2, -2, 9, 6, 9, 6, 9, -1, 10, -2, -3, 4, 7, 8, 2, 3, 8, 0, -1, 4, 8, 0, 1, 3, 6, 2, 10, 8, 0}
;

const int16_t conv1d_159_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-36, 15, -49}
, {-60, 43, -57}
, {-51, 0, 14}
, {-35, 66, 51}
, {-25, -60, 59}
, {-2, -66, 60}
, {-25, -41, 24}
, {-29, -68, 24}
, {55, 23, -11}
, {-23, 82, -11}
, {69, 0, 34}
, {-46, -34, 26}
, {-5, 41, -17}
, {-68, -31, 59}
, {6, 9, 19}
, {9, -7, 13}
, {57, 14, -55}
, {-39, -23, -37}
, {-31, -20, 3}
, {24, -37, 20}
, {3, 57, 26}
, {14, -11, -45}
, {55, 24, -59}
, {-7, -1, 25}
, {24, 23, -2}
, {33, -40, -51}
, {-49, 8, 63}
, {49, 40, -68}
, {53, 19, -13}
, {65, -26, 37}
, {-35, 17, 10}
, {-34, -49, -56}
}
, {{-35, 3, 68}
, {66, 1, 76}
, {68, -55, -31}
, {-50, -54, -55}
, {-2, 8, 28}
, {-4, 39, -68}
, {30, 50, 39}
, {-6, 40, 26}
, {37, -25, 64}
, {-56, 69, 13}
, {70, 65, -26}
, {31, 4, 33}
, {-29, 73, -7}
, {-5, -46, -47}
, {-51, -55, 42}
, {-12, -62, 7}
, {-10, 8, -50}
, {-32, -48, -18}
, {-48, -56, 25}
, {-27, 68, 30}
, {-45, -5, -44}
, {69, 11, -45}
, {-26, 33, -43}
, {31, 66, 30}
, {18, -43, 64}
, {77, 42, 40}
, {21, 39, 40}
, {-64, -22, 75}
, {-74, -62, 60}
, {16, -2, 39}
, {55, 56, -71}
, {-19, 13, -34}
}
, {{75, -18, 22}
, {29, 8, -69}
, {76, 39, 66}
, {52, 17, -72}
, {76, -68, -8}
, {-18, 41, 5}
, {-48, -40, -29}
, {-26, 23, -23}
, {-62, 14, -45}
, {23, 47, 26}
, {-7, -22, 3}
, {4, 50, -20}
, {33, 20, 70}
, {76, 39, 33}
, {-32, 33, 31}
, {-66, -48, 17}
, {29, -67, 5}
, {34, -12, 1}
, {16, -3, 37}
, {-7, 67, 29}
, {-67, -10, -2}
, {-30, 13, -22}
, {-66, -6, -69}
, {53, 33, 64}
, {-37, -27, -55}
, {67, -45, -58}
, {-47, 14, 11}
, {-46, -33, 39}
, {8, 45, 11}
, {39, -35, -48}
, {31, -33, 18}
, {-40, -48, -10}
}
, {{-64, -6, 61}
, {-35, -46, -68}
, {2, 37, 8}
, {1, 0, 30}
, {20, -28, 14}
, {-50, -7, 33}
, {-11, 67, 52}
, {-66, 54, 4}
, {5, -57, -14}
, {-72, 31, -57}
, {27, -62, -69}
, {-5, 5, 43}
, {-41, 24, -37}
, {38, 8, 11}
, {50, 8, -6}
, {-26, -56, -73}
, {-65, -31, -46}
, {33, -70, -18}
, {0, 1, 50}
, {-43, 63, 44}
, {-1, -33, -10}
, {17, 50, 20}
, {-27, -21, 23}
, {73, -55, -36}
, {69, -30, -41}
, {14, -63, 73}
, {-69, 70, -31}
, {20, 12, -26}
, {-9, 53, -21}
, {35, -58, -19}
, {59, -46, -11}
, {2, 56, 39}
}
, {{59, 48, -41}
, {-3, 0, 0}
, {79, 38, 38}
, {49, -58, 51}
, {11, -47, -11}
, {78, -18, 27}
, {2, 35, 23}
, {33, -12, 0}
, {68, 57, 0}
, {67, -20, 52}
, {-56, -3, -8}
, {-46, 61, -77}
, {-31, -62, -33}
, {76, -60, -58}
, {32, 47, -20}
, {80, 19, 49}
, {-22, -30, 39}
, {69, -15, -51}
, {-64, -74, -37}
, {12, -56, 46}
, {16, -57, -20}
, {-51, 27, 68}
, {68, -5, 46}
, {-6, -18, 8}
, {-7, 51, -52}
, {67, 51, 56}
, {81, 81, 43}
, {69, -59, 35}
, {-15, 67, -44}
, {2, 33, 18}
, {-57, -3, 41}
, {76, -59, 14}
}
, {{72, 75, 71}
, {29, 57, -18}
, {47, -6, -19}
, {-44, -14, -15}
, {66, -22, -57}
, {30, 76, 69}
, {49, -23, 10}
, {47, 5, 9}
, {-61, 23, 62}
, {-57, 54, 18}
, {-37, -4, -62}
, {63, 0, 2}
, {23, 36, 43}
, {-4, 1, -67}
, {-38, 3, -44}
, {-60, 6, 84}
, {40, -16, 61}
, {58, 45, -50}
, {1, -68, 62}
, {36, 84, 72}
, {34, -21, 35}
, {18, 8, -14}
, {-7, 9, -25}
, {62, 56, 25}
, {-56, 47, -16}
, {22, 66, 0}
, {21, -63, 75}
, {31, -65, -16}
, {15, -42, 34}
, {-42, 26, -39}
, {48, 15, 76}
, {52, -28, 48}
}
, {{32, -64, -68}
, {-4, -36, -42}
, {23, -29, 20}
, {-49, 62, 45}
, {-12, 65, -27}
, {-8, 0, 70}
, {-14, 62, 40}
, {-39, -14, 56}
, {-55, 51, 49}
, {6, 71, -54}
, {-2, -21, 20}
, {-54, -49, -67}
, {-11, 4, -23}
, {-31, 66, 7}
, {-17, 4, 29}
, {68, -57, -27}
, {-55, 60, -48}
, {23, 47, -71}
, {-59, -40, -61}
, {11, -54, -70}
, {-37, -59, -29}
, {44, 33, -70}
, {-24, -45, 21}
, {33, -66, -59}
, {-69, 71, 51}
, {-44, -1, -39}
, {-35, -34, 10}
, {12, 62, 46}
, {58, -17, 0}
, {59, -71, 31}
, {-61, -5, -49}
, {-36, -64, -64}
}
, {{39, -6, 66}
, {-52, -27, 32}
, {19, 8, -40}
, {-4, 22, -18}
, {-55, 4, 4}
, {-55, -10, -20}
, {-30, 35, -18}
, {-59, 15, 68}
, {55, 34, -10}
, {45, 11, 2}
, {-55, 31, -41}
, {-40, 9, -65}
, {28, 18, 41}
, {7, -26, 32}
, {-24, 76, 78}
, {-28, 54, 18}
, {5, 47, 7}
, {30, 47, 35}
, {51, 76, -48}
, {66, 17, 28}
, {-49, 44, 10}
, {40, -18, -67}
, {-63, -30, -46}
, {-3, 37, -18}
, {-8, 67, 18}
, {-19, 14, 14}
, {57, 55, 32}
, {67, -11, -33}
, {-68, -18, 39}
, {12, 22, -63}
, {76, 66, 5}
, {-15, 53, -47}
}
, {{-69, 0, 1}
, {42, -65, -21}
, {-37, 11, -72}
, {72, 58, -15}
, {54, -2, 3}
, {38, 63, -61}
, {-47, 33, -10}
, {40, -49, 31}
, {-21, -25, -26}
, {-45, 56, -53}
, {55, -25, 61}
, {-75, -26, 68}
, {28, 46, -37}
, {50, 57, -34}
, {73, 14, 6}
, {62, -67, 28}
, {46, -39, 64}
, {75, -36, -32}
, {-48, 76, 35}
, {-53, 47, 34}
, {-68, -49, 22}
, {56, -51, 40}
, {54, 24, 58}
, {-18, -71, -29}
, {-18, 35, -45}
, {-28, -65, -33}
, {-36, 54, -37}
, {-3, 9, -16}
, {-50, 27, -15}
, {76, 26, 22}
, {-11, 34, -54}
, {-38, 27, 37}
}
, {{-65, -11, 49}
, {53, -16, -7}
, {3, 17, -10}
, {-32, -20, 53}
, {-28, 2, 67}
, {-8, 12, 52}
, {16, 38, -67}
, {26, -31, 61}
, {-4, 7, -24}
, {25, 40, -18}
, {-60, -39, 16}
, {60, -53, -57}
, {-69, -44, 60}
, {73, -44, -68}
, {25, 49, 15}
, {20, -66, 29}
, {14, 61, -29}
, {30, -21, 57}
, {59, 30, 48}
, {-5, 25, 70}
, {40, -37, -50}
, {-62, -21, -8}
, {63, 8, 23}
, {-3, 48, 57}
, {-17, -46, -20}
, {-61, 71, 51}
, {52, -15, -65}
, {65, 27, 57}
, {-45, 65, 11}
, {26, 19, 55}
, {-1, 52, 35}
, {59, 70, -13}
}
, {{3, -37, 48}
, {-53, -7, -1}
, {0, 24, -41}
, {-63, -29, -14}
, {-28, 65, 27}
, {-26, -29, 62}
, {-46, -70, 62}
, {-6, 12, -2}
, {6, 22, -45}
, {-50, 49, 9}
, {9, -18, -45}
, {47, 14, -72}
, {-43, -77, 29}
, {52, -20, -45}
, {-34, 53, 40}
, {26, 45, -36}
, {57, 23, -45}
, {-15, 44, -53}
, {-13, -22, -14}
, {17, 5, 11}
, {-47, -47, -57}
, {-48, -16, 45}
, {42, 18, -5}
, {-47, -67, -45}
, {71, -46, -26}
, {30, 35, 58}
, {67, 16, 3}
, {68, 66, -9}
, {43, 6, -55}
, {60, 53, -70}
, {-19, 14, -55}
, {28, -44, -56}
}
, {{15, -18, -28}
, {58, -5, -24}
, {-50, -58, 5}
, {60, -54, 51}
, {18, 25, -46}
, {-14, -14, -49}
, {1, 58, 6}
, {-13, -18, 9}
, {30, -32, -44}
, {-57, 55, -60}
, {46, -40, -15}
, {-2, -29, -65}
, {-51, -19, -73}
, {-13, -24, -54}
, {-14, 32, -56}
, {-45, 4, -65}
, {-54, 6, -67}
, {-43, -56, 60}
, {64, -24, -6}
, {42, -23, 13}
, {39, 60, 0}
, {4, -19, 25}
, {19, 48, 49}
, {50, 64, -26}
, {-18, -13, -75}
, {-44, -56, -46}
, {72, -1, 72}
, {-46, 14, 9}
, {-17, 16, -62}
, {-69, 66, -5}
, {-2, -54, -9}
, {-40, -66, 59}
}
, {{-31, -45, 71}
, {68, 19, -14}
, {72, -16, 33}
, {4, -26, 67}
, {-67, 76, -12}
, {57, -26, 53}
, {-33, -37, 27}
, {-5, -13, 22}
, {-67, 42, 71}
, {-42, 38, 57}
, {-56, 10, 64}
, {30, -47, -30}
, {-47, 31, -16}
, {-66, -70, 21}
, {-36, 68, 33}
, {3, 20, -64}
, {-56, 44, 6}
, {-67, 46, 56}
, {63, 24, -66}
, {-15, -17, -20}
, {-49, -39, 74}
, {35, 49, 43}
, {-71, -24, -33}
, {-38, 67, 9}
, {-51, 19, -49}
, {41, 41, -31}
, {-13, 38, -66}
, {-53, 38, -65}
, {45, 19, 59}
, {62, 26, 22}
, {-23, -60, 15}
, {-25, 4, -58}
}
, {{-69, -29, 62}
, {-50, 0, 63}
, {-73, 58, -9}
, {22, -57, -5}
, {-51, -24, -65}
, {-61, 30, -66}
, {-60, 6, 3}
, {-41, 35, 34}
, {-9, -23, 70}
, {-12, -11, -42}
, {-36, 59, 28}
, {54, -29, -55}
, {-23, -58, -21}
, {-42, -8, -4}
, {41, 46, -58}
, {-71, 47, -6}
, {15, 5, -54}
, {42, 70, 69}
, {6, -22, -57}
, {40, 30, 3}
, {19, -1, -39}
, {-5, 74, -2}
, {66, -10, 39}
, {-65, -44, 18}
, {32, -24, -22}
, {-28, -2, 23}
, {62, 7, -7}
, {39, 27, 66}
, {-33, -46, 10}
, {50, -26, 3}
, {71, 58, 13}
, {-25, -39, 44}
}
, {{59, 46, 53}
, {-40, 30, -15}
, {-35, -56, 17}
, {10, 49, -39}
, {32, 40, 0}
, {-6, -49, -27}
, {-45, 12, -14}
, {-56, -65, -49}
, {-21, 64, 10}
, {11, 4, -45}
, {-68, -27, 55}
, {-59, 9, 21}
, {16, -46, -26}
, {-16, -65, -3}
, {-3, -55, -11}
, {-26, 62, 70}
, {47, 35, 27}
, {-52, 61, -52}
, {50, 4, -59}
, {29, 0, 30}
, {2, 73, 63}
, {-43, 55, 60}
, {-26, -62, -24}
, {19, 46, 0}
, {6, 41, -68}
, {13, 61, 5}
, {5, 39, 68}
, {0, 36, 67}
, {-36, -64, -53}
, {-13, -29, -1}
, {-4, -36, 53}
, {-60, 54, 51}
}
, {{-52, 67, 36}
, {-64, 57, 48}
, {-9, 26, -7}
, {26, 49, 44}
, {17, 74, -66}
, {66, 1, -9}
, {0, 6, -34}
, {33, 13, -22}
, {-3, 3, -2}
, {-35, 24, 14}
, {68, 1, -59}
, {-28, 32, 72}
, {-30, -73, -18}
, {36, -52, 9}
, {20, -24, -26}
, {34, -21, -54}
, {63, -24, -65}
, {66, -27, -19}
, {43, -64, -34}
, {13, -34, 45}
, {54, -38, -4}
, {-47, -24, 54}
, {64, 42, -42}
, {-10, 41, -61}
, {-45, 56, 14}
, {-61, 60, 9}
, {68, 33, -13}
, {11, -62, -11}
, {-4, -49, 61}
, {-20, -10, -61}
, {26, -58, 66}
, {-36, 3, 42}
}
, {{17, -41, -65}
, {-41, 21, -35}
, {-11, 57, -6}
, {-57, 56, 10}
, {-40, 22, 63}
, {-23, -43, 8}
, {42, 40, 37}
, {11, 8, -8}
, {-29, -64, -55}
, {49, 7, 47}
, {-27, 23, -14}
, {4, -68, -70}
, {45, -64, -67}
, {63, 71, -70}
, {-66, 18, 0}
, {-43, 61, -37}
, {55, 40, 0}
, {2, -62, 20}
, {11, -32, -24}
, {13, 55, -27}
, {57, 31, -52}
, {-11, 27, 13}
, {-46, 14, 0}
, {48, -39, 22}
, {36, 44, -15}
, {68, -9, -67}
, {-37, -31, 63}
, {60, -59, -48}
, {-17, 22, -3}
, {5, -48, -43}
, {63, 73, -35}
, {49, -3, -58}
}
, {{14, 22, -31}
, {-46, -1, 8}
, {-67, 62, 35}
, {-17, -72, 68}
, {48, 31, -52}
, {19, -46, -32}
, {-27, -41, -38}
, {32, 41, -18}
, {-42, 28, 38}
, {-28, 14, 1}
, {-67, -61, 42}
, {-43, 24, 14}
, {38, -1, -36}
, {-57, 43, 10}
, {49, 42, 50}
, {-46, -33, -54}
, {20, -44, -15}
, {21, -34, -22}
, {19, 62, -37}
, {20, 53, -15}
, {-12, -9, -36}
, {-11, 62, 41}
, {-19, 71, 31}
, {-7, 54, -33}
, {21, 34, -31}
, {68, -67, -71}
, {15, -53, 50}
, {-64, 53, -14}
, {61, -4, 9}
, {-50, -51, 66}
, {-3, 59, -24}
, {48, -59, 31}
}
, {{58, 69, 62}
, {-16, -63, 64}
, {-27, 20, 10}
, {-60, -24, 74}
, {75, 59, 49}
, {10, -6, -55}
, {16, 53, -61}
, {-10, 42, 16}
, {42, -49, 42}
, {30, 59, 68}
, {-35, -22, 24}
, {-43, 65, -18}
, {11, 0, 65}
, {-64, -3, 36}
, {-19, 72, 22}
, {-10, -36, 19}
, {45, 37, -52}
, {-58, -7, -14}
, {-42, 33, 28}
, {-17, -48, 26}
, {75, -1, 14}
, {-54, 33, 31}
, {59, -54, 18}
, {23, -46, 56}
, {3, -35, 66}
, {-63, -29, 32}
, {-42, 38, 2}
, {-51, -48, -74}
, {-25, 18, -46}
, {9, -54, 7}
, {61, 9, -42}
, {37, 68, -49}
}
, {{75, -30, -65}
, {14, -70, 14}
, {26, 18, 62}
, {-13, 49, 1}
, {43, -41, -4}
, {33, 40, 60}
, {17, -66, -66}
, {-12, 46, -5}
, {6, 34, 29}
, {-15, 0, 20}
, {-35, -9, -40}
, {-12, -7, -55}
, {-12, 42, 5}
, {15, -31, 49}
, {-35, 70, -48}
, {68, 44, 7}
, {-70, 65, 13}
, {-9, 56, 17}
, {-38, -20, 42}
, {-17, 51, -45}
, {21, 22, 51}
, {40, -7, 42}
, {-31, 46, -1}
, {41, -26, 1}
, {27, -20, -22}
, {21, 69, 49}
, {66, -22, -35}
, {-75, -47, -15}
, {28, 57, -49}
, {3, -30, -65}
, {-10, -26, -44}
, {22, -16, 9}
}
, {{-41, 1, 46}
, {38, 77, -19}
, {6, 26, 68}
, {-22, -69, -18}
, {-61, 26, 42}
, {-72, 21, 19}
, {68, -47, -27}
, {11, -52, 32}
, {-55, 58, 25}
, {63, 20, -56}
, {-9, 16, -68}
, {12, 47, -44}
, {28, 9, 31}
, {-50, -43, 24}
, {1, 51, -20}
, {-28, -5, 36}
, {50, 37, 31}
, {3, 11, 22}
, {24, -19, -11}
, {35, 65, 20}
, {64, -11, -45}
, {-5, -36, -50}
, {51, 22, -64}
, {-72, -55, -26}
, {38, -26, 65}
, {-15, 28, -11}
, {-28, -64, 62}
, {50, -11, -55}
, {-58, 65, 31}
, {-45, -30, 49}
, {10, 32, -48}
, {-76, 4, 32}
}
, {{-26, -3, -24}
, {72, 40, 47}
, {18, 36, 54}
, {63, -25, 53}
, {27, 56, -50}
, {18, 62, 17}
, {71, -28, 64}
, {41, -60, 2}
, {34, -51, 0}
, {-19, 0, -31}
, {59, 22, -65}
, {-42, -35, -58}
, {28, -60, 19}
, {-57, 28, -29}
, {-45, 75, 70}
, {3, 76, -58}
, {-14, 35, -49}
, {2, 12, 31}
, {-6, 18, -6}
, {63, 64, 25}
, {-15, 52, -16}
, {17, -19, -24}
, {15, -59, -31}
, {-35, 71, -68}
, {-37, -50, -43}
, {-45, 40, 41}
, {8, -48, -5}
, {-43, -22, 58}
, {63, 55, -63}
, {-39, -53, -1}
, {21, -22, -32}
, {-30, -11, -55}
}
, {{39, 39, 8}
, {3, -59, 65}
, {63, 6, 0}
, {53, -26, -30}
, {25, -11, 56}
, {8, -66, -72}
, {-38, -32, -47}
, {70, -1, 17}
, {12, -46, -1}
, {29, -68, -30}
, {-32, 42, -50}
, {43, 22, 35}
, {41, 18, -62}
, {-31, 28, 26}
, {18, 30, -4}
, {-66, -33, -54}
, {-43, 11, -17}
, {57, 20, -42}
, {-41, -50, -34}
, {-7, 17, -39}
, {54, -31, -58}
, {-72, 16, -60}
, {-38, 41, 20}
, {16, 68, 53}
, {39, 9, 62}
, {-3, 30, -39}
, {35, -69, -6}
, {6, -52, -12}
, {-45, 66, 40}
, {55, -71, -47}
, {-19, -33, -45}
, {-10, -54, 10}
}
, {{-70, 15, -6}
, {-53, -31, -42}
, {-17, -21, 0}
, {43, -68, -28}
, {-43, -61, 14}
, {19, 29, -19}
, {-2, -23, 71}
, {57, 33, 45}
, {-33, -44, 51}
, {-21, 21, 50}
, {13, -51, 63}
, {50, -31, -52}
, {7, -41, -37}
, {39, 2, 72}
, {-14, -27, -26}
, {60, 40, -70}
, {-20, 70, 56}
, {-51, 65, -22}
, {-51, 57, 70}
, {50, -42, 49}
, {-9, -30, 3}
, {-9, -72, 46}
, {-25, 61, 17}
, {-2, -72, 33}
, {-71, -35, 15}
, {49, 35, -16}
, {38, -53, 60}
, {-3, 7, 64}
, {-56, -67, 62}
, {33, -57, -71}
, {-30, 12, 19}
, {-37, -43, -24}
}
, {{4, 18, -35}
, {17, 32, 52}
, {-31, -50, -32}
, {-46, 27, -70}
, {20, 62, 81}
, {76, -55, 64}
, {40, -41, 56}
, {55, 0, 24}
, {5, 37, 65}
, {-20, -66, -55}
, {-7, 33, -58}
, {-32, 52, -18}
, {-34, 62, -43}
, {-21, -24, 44}
, {52, 60, 31}
, {40, -37, 4}
, {32, -67, -23}
, {-31, -41, 66}
, {-45, 70, -67}
, {-31, 53, 24}
, {35, 45, -14}
, {-15, -47, 38}
, {50, 68, -45}
, {14, 65, -6}
, {-44, -69, 30}
, {-12, -68, -30}
, {15, -12, 59}
, {-35, -33, 51}
, {-43, 68, 66}
, {71, -54, -44}
, {14, 49, 64}
, {81, -35, -17}
}
, {{-52, -69, -63}
, {65, -52, -66}
, {5, 15, 31}
, {59, -34, 53}
, {-27, 0, 33}
, {-65, 3, -57}
, {59, 37, -71}
, {0, -56, -26}
, {-40, 68, 5}
, {-72, 45, 55}
, {-28, 31, -13}
, {31, -16, 14}
, {-54, 34, -32}
, {-40, -69, 41}
, {-21, 13, -73}
, {10, -45, 56}
, {-21, 11, -25}
, {8, 72, -40}
, {-54, -11, 24}
, {-4, -7, -38}
, {0, 26, -63}
, {-47, -29, -67}
, {-22, -3, -58}
, {-56, 68, 20}
, {-64, 18, 56}
, {-4, -54, -72}
, {-52, -14, 7}
, {-25, -71, -39}
, {-69, -54, 41}
, {70, -58, -48}
, {3, -53, -65}
, {70, -48, 47}
}
, {{48, 62, -34}
, {-5, 55, -7}
, {50, 4, 4}
, {19, -13, 12}
, {-41, 60, 47}
, {49, -6, 10}
, {-43, -1, 59}
, {-15, 10, -53}
, {66, -50, 60}
, {-41, -4, 36}
, {35, -50, 0}
, {-20, -61, 8}
, {60, -29, -35}
, {-57, -46, 45}
, {64, 12, 4}
, {40, 0, 16}
, {23, -6, -54}
, {20, -41, -16}
, {-6, 39, 11}
, {8, 28, -42}
, {21, 11, -74}
, {-50, 38, 42}
, {-35, -29, 24}
, {-24, 70, 14}
, {-38, -3, -29}
, {-75, 48, 31}
, {-56, -57, -64}
, {-68, 59, 67}
, {59, -39, -39}
, {61, -31, -67}
, {-23, 41, -73}
, {51, 37, -7}
}
, {{-42, -36, -42}
, {53, -42, -49}
, {-38, 44, -6}
, {56, -45, -61}
, {76, 20, 6}
, {38, -11, -2}
, {-32, 68, -23}
, {53, 59, -24}
, {60, 72, 31}
, {66, 45, -21}
, {40, -28, -9}
, {9, 1, -12}
, {74, -46, 57}
, {-52, 4, 59}
, {-64, 59, -39}
, {-25, -44, 67}
, {-34, -28, -22}
, {-38, -62, 56}
, {1, -26, -25}
, {6, 56, 15}
, {46, 27, -13}
, {73, -27, 51}
, {-43, 34, 40}
, {-57, -63, 67}
, {-10, -52, -1}
, {-12, 29, 74}
, {70, 74, 36}
, {53, 5, -70}
, {11, 64, 1}
, {70, -40, 43}
, {-40, -15, -11}
, {-32, 57, 4}
}
, {{35, -56, 26}
, {-44, -72, -38}
, {78, 48, 51}
, {-74, -5, -43}
, {-61, -39, 74}
, {67, 46, 5}
, {26, 50, -13}
, {-66, 62, -59}
, {13, -7, -44}
, {64, 10, -13}
, {13, 3, -42}
, {73, -6, -59}
, {-26, 46, -38}
, {19, -54, -48}
, {37, -21, 62}
, {57, 70, 54}
, {27, 64, 61}
, {-43, -44, -19}
, {12, 39, -53}
, {29, 14, 39}
, {-73, 58, 44}
, {36, 17, 69}
, {-5, 12, -32}
, {-32, 32, 70}
, {4, 43, 66}
, {10, 11, -16}
, {66, -19, -52}
, {16, 54, 27}
, {29, 57, -28}
, {-71, 49, -24}
, {18, -70, -26}
, {72, 37, -62}
}
, {{28, -61, -1}
, {2, -25, -52}
, {23, -22, -27}
, {0, -31, 19}
, {-19, -38, 70}
, {11, -30, 43}
, {44, -27, -50}
, {-66, -36, 38}
, {-41, 4, -20}
, {24, 56, 47}
, {15, 27, -12}
, {-48, 58, 45}
, {35, -1, -34}
, {49, -8, 21}
, {-11, -66, -56}
, {-17, 14, 26}
, {24, 57, 31}
, {43, -37, -57}
, {-24, 37, -49}
, {30, -43, 11}
, {-66, -11, 43}
, {-59, 60, 12}
, {38, -62, -33}
, {-65, -13, -59}
, {-41, 66, 72}
, {63, 25, 64}
, {79, 74, 52}
, {38, -39, 7}
, {0, -21, -58}
, {-31, -49, -54}
, {-11, -62, 4}
, {52, 17, 69}
}
, {{-59, -10, 5}
, {32, 40, -59}
, {27, 9, -45}
, {72, 44, -15}
, {-16, 57, -28}
, {-54, 69, 32}
, {63, 0, 0}
, {-17, -37, -44}
, {-6, -6, 3}
, {-49, 61, -34}
, {12, -67, -78}
, {9, -69, -62}
, {66, -19, -39}
, {59, 1, -52}
, {8, 32, 65}
, {-27, 67, 33}
, {-64, 57, -68}
, {12, -50, -53}
, {-50, -68, -31}
, {49, 47, 42}
, {-74, -22, -43}
, {38, -54, -38}
, {5, 38, 1}
, {12, -49, 67}
, {-17, -28, -13}
, {-50, -62, 35}
, {-1, -15, -22}
, {5, -17, 51}
, {-7, 5, -38}
, {-6, -35, -52}
, {38, 33, 3}
, {72, 59, 29}
}
, {{-42, -19, -61}
, {9, -36, -51}
, {39, 37, -60}
, {70, 21, -29}
, {-63, 60, -14}
, {76, 11, -5}
, {-70, 50, 37}
, {-59, -19, -45}
, {65, -57, 29}
, {-70, -59, 45}
, {49, 7, 60}
, {-22, 19, 18}
, {26, 38, 37}
, {34, 71, -59}
, {63, -32, -22}
, {48, -2, -19}
, {75, -69, 52}
, {-11, 67, 4}
, {1, 77, 54}
, {-5, 77, -59}
, {59, 42, 54}
, {55, 28, 13}
, {61, 53, 59}
, {-16, 81, 40}
, {-6, -47, -14}
, {-66, -29, 70}
, {52, 39, -37}
, {-45, 40, -3}
, {-17, 4, 57}
, {23, 31, -31}
, {-52, -30, 52}
, {-61, 25, -39}
}
, {{-54, 9, 70}
, {48, 9, 58}
, {34, 24, 40}
, {25, -56, 11}
, {-71, 0, -53}
, {-26, -10, -33}
, {9, -16, 66}
, {-10, -67, -32}
, {6, -59, 46}
, {-47, -67, 46}
, {69, -4, 38}
, {70, 71, -51}
, {-53, 0, -7}
, {-21, 73, -23}
, {-32, -19, -27}
, {-20, -24, -6}
, {-21, -56, 18}
, {-2, -11, 65}
, {11, 58, 23}
, {0, -15, 63}
, {27, 60, -20}
, {18, 24, 71}
, {-32, 66, -20}
, {-64, -40, 35}
, {-18, 17, 45}
, {-17, -10, -17}
, {-64, 34, 40}
, {-28, 58, -64}
, {-19, -15, 19}
, {18, 21, -4}
, {49, -46, 75}
, {71, -29, -20}
}
, {{16, 18, 16}
, {25, 19, -3}
, {-44, 6, 0}
, {29, -6, 65}
, {-56, -54, 44}
, {17, 72, 66}
, {-25, -50, -60}
, {-26, 27, 18}
, {-1, 28, -6}
, {-19, -32, 72}
, {56, 74, -34}
, {12, -48, 46}
, {12, -29, 3}
, {17, 37, 57}
, {17, 42, -55}
, {22, 71, 29}
, {54, 1, -10}
, {24, 72, -49}
, {-20, 36, -18}
, {-12, 67, -42}
, {-23, -24, 52}
, {42, 61, 75}
, {47, 57, -57}
, {10, 46, 8}
, {72, 36, -15}
, {67, -56, -33}
, {5, 69, -33}
, {67, 68, 33}
, {-42, 0, -40}
, {12, 48, -69}
, {-37, 31, 65}
, {-10, 2, -36}
}
, {{45, 19, -9}
, {40, -9, 68}
, {-17, 62, 57}
, {-55, -20, -44}
, {62, 41, 9}
, {-10, -47, 62}
, {65, 0, 25}
, {37, 66, 72}
, {70, -64, 26}
, {-45, -31, 22}
, {50, -61, 22}
, {46, 54, -20}
, {25, -24, 7}
, {-47, -41, 49}
, {-44, -38, -59}
, {-62, 67, 19}
, {6, -24, 58}
, {14, -63, 52}
, {-52, 57, -22}
, {50, 52, 31}
, {-28, 19, -66}
, {42, -8, -21}
, {-58, 68, -31}
, {-19, 60, 66}
, {-59, -47, -60}
, {-56, -44, -18}
, {41, -32, -13}
, {42, -28, 42}
, {46, 69, 52}
, {-15, -24, 32}
, {21, 69, -20}
, {23, -37, 76}
}
, {{-33, -23, 17}
, {-55, -56, -20}
, {57, 20, 5}
, {51, 2, -2}
, {69, -49, 45}
, {14, -55, -57}
, {-59, -71, 31}
, {-2, -30, 6}
, {13, -13, -9}
, {-56, -37, -6}
, {17, 46, 52}
, {7, 11, -41}
, {-43, -4, 55}
, {24, 17, 40}
, {-10, 68, 15}
, {-5, 17, 71}
, {54, -59, -51}
, {47, 32, -58}
, {20, -53, 34}
, {73, -8, 49}
, {-55, -74, 63}
, {6, -30, 9}
, {0, 31, -5}
, {-52, 5, 66}
, {72, 66, -42}
, {-71, -6, -22}
, {-30, -59, 57}
, {25, -16, -64}
, {35, -68, -2}
, {21, -46, 51}
, {4, -71, -37}
, {-20, 77, 50}
}
, {{47, -33, 23}
, {29, -52, 38}
, {62, -20, -54}
, {-9, -41, 7}
, {53, 79, 43}
, {0, 37, -17}
, {-46, -5, -56}
, {35, 6, 50}
, {77, -60, 47}
, {54, -66, 34}
, {-39, -4, -70}
, {21, -1, -65}
, {40, -56, -6}
, {60, 35, 65}
, {-3, 46, -33}
, {17, -46, 56}
, {-45, -7, 56}
, {-61, -3, -31}
, {52, 10, -34}
, {25, -2, 79}
, {-71, 6, -75}
, {62, -26, 13}
, {14, 61, -48}
, {-17, 70, 73}
, {-1, 50, 49}
, {50, -73, 39}
, {-32, -63, 79}
, {7, -9, 0}
, {62, 57, -4}
, {-44, -68, 66}
, {-50, -36, -31}
, {-31, 0, -12}
}
, {{54, 74, -40}
, {-57, -8, -75}
, {-6, 49, -6}
, {-12, -48, 32}
, {22, -7, 54}
, {18, 42, 44}
, {67, 43, 33}
, {-42, -52, 47}
, {76, 4, 48}
, {20, 38, 9}
, {-50, 19, -46}
, {74, 25, -65}
, {15, 51, -59}
, {-23, -11, -60}
, {-34, -57, -47}
, {-12, -21, 36}
, {41, -9, -24}
, {-10, -10, -59}
, {30, 28, 69}
, {-48, -26, -57}
, {-60, 18, -18}
, {-8, 6, -45}
, {78, 5, 54}
, {-13, 19, -43}
, {38, 3, 12}
, {26, -17, -14}
, {41, -64, 40}
, {-9, -67, 41}
, {-53, 70, -52}
, {63, 39, -59}
, {-31, -52, -31}
, {6, 13, -9}
}
, {{75, -64, 66}
, {-61, -50, -54}
, {27, 52, -45}
, {16, -53, -35}
, {37, 27, 7}
, {1, 55, 26}
, {43, 39, 68}
, {35, -56, -54}
, {-43, 40, 21}
, {34, -32, -67}
, {-20, 44, 47}
, {23, -50, -9}
, {-28, -1, -28}
, {36, 17, -13}
, {11, -26, 70}
, {53, 65, -23}
, {-64, -2, 58}
, {13, -71, -30}
, {41, -57, 31}
, {42, 1, 39}
, {19, -3, 58}
, {-48, -3, -38}
, {-69, -49, -47}
, {38, 22, 66}
, {-44, -15, -34}
, {56, -50, -11}
, {27, -59, -19}
, {-11, 15, 3}
, {-39, 5, 45}
, {-62, -15, -22}
, {-65, -10, 33}
, {41, -29, 61}
}
, {{53, 27, -71}
, {-8, -56, 56}
, {-75, 54, -24}
, {-31, 67, 70}
, {28, -1, 49}
, {45, 51, -67}
, {-50, 36, 25}
, {4, -33, 35}
, {33, 28, 45}
, {-4, -62, 11}
, {-34, -49, -26}
, {67, 53, 27}
, {-19, 16, 65}
, {-51, 61, 58}
, {71, -69, 71}
, {47, -59, -63}
, {-31, -63, 46}
, {20, -7, 15}
, {51, 11, -53}
, {-10, 4, 0}
, {-7, 4, -37}
, {36, 45, -71}
, {41, -37, -38}
, {-11, -42, -39}
, {-26, 41, 1}
, {2, 26, 37}
, {-64, 45, -36}
, {5, 54, -56}
, {11, 16, -15}
, {-28, 3, -64}
, {63, 8, -9}
, {64, -29, -67}
}
, {{26, -30, 47}
, {15, -63, 68}
, {28, -45, -4}
, {-48, 3, 31}
, {0, -44, 64}
, {62, 45, 41}
, {2, 58, 46}
, {-52, 66, 11}
, {-57, -72, 10}
, {42, -38, -63}
, {4, 16, 48}
, {69, -75, 14}
, {-8, 28, 58}
, {-39, -57, -14}
, {26, 0, 30}
, {7, 29, 4}
, {-37, 49, 22}
, {48, -41, 24}
, {64, 57, -7}
, {-70, 62, -42}
, {-15, -60, -12}
, {73, -35, 20}
, {6, 23, 14}
, {-25, -9, -67}
, {-3, -60, -52}
, {-16, 51, 76}
, {-18, -27, 3}
, {-32, -1, 74}
, {2, 54, -34}
, {61, -44, -51}
, {54, 47, 41}
, {-60, 39, 49}
}
, {{-9, -48, -59}
, {-50, 54, -32}
, {53, -13, -3}
, {-41, -69, -17}
, {-53, 51, 32}
, {27, 41, 61}
, {-4, -54, 68}
, {73, 41, -2}
, {-36, -55, 76}
, {35, -43, -9}
, {-27, 29, 19}
, {11, -41, -11}
, {-35, 28, -43}
, {-31, 8, 48}
, {-36, 67, 66}
, {-7, -24, -59}
, {-15, 23, -67}
, {55, -30, 67}
, {-61, 2, -42}
, {46, 0, -47}
, {-36, 59, 77}
, {-46, 58, 25}
, {17, 3, 29}
, {-52, 42, -51}
, {61, -21, 15}
, {-4, 61, -20}
, {-44, -28, 17}
, {32, 18, 29}
, {-10, 16, -52}
, {-72, 64, -43}
, {51, -50, -71}
, {1, 15, 43}
}
, {{-62, -50, 71}
, {-49, -6, -9}
, {53, 0, 32}
, {-10, 11, 43}
, {-68, -62, -12}
, {39, 73, -39}
, {59, 24, 10}
, {-19, -64, 28}
, {-17, -2, 47}
, {17, -35, 22}
, {-27, -37, 52}
, {44, -27, -32}
, {-52, -4, 45}
, {40, 60, 35}
, {-54, 6, -29}
, {6, 9, 37}
, {-62, 6, 31}
, {-63, 41, 40}
, {65, -42, 40}
, {-38, -24, -1}
, {-29, -72, -66}
, {-56, 56, 35}
, {70, 21, 49}
, {46, 11, -29}
, {-18, -30, 3}
, {13, -66, -45}
, {-33, -49, 56}
, {1, -54, -2}
, {-66, -20, -13}
, {-18, 41, 26}
, {-42, -55, -38}
, {63, -42, -47}
}
, {{-11, 15, -17}
, {-40, -43, -42}
, {21, 54, 9}
, {73, 11, -14}
, {64, 66, -69}
, {-66, 35, 66}
, {29, 24, 39}
, {47, -45, 71}
, {-44, -34, -40}
, {44, 10, -60}
, {-48, -2, 31}
, {50, 10, -27}
, {-15, 61, -59}
, {-57, 41, -12}
, {23, -61, -15}
, {42, 6, 28}
, {-52, 67, -39}
, {-49, 12, 6}
, {-10, -24, -41}
, {75, 28, 50}
, {48, 48, -44}
, {28, -22, 4}
, {-35, -47, -25}
, {34, -2, 5}
, {37, 57, -28}
, {0, -26, -53}
, {10, -42, 46}
, {57, 52, 0}
, {-6, -65, 48}
, {-20, 51, 63}
, {-47, 23, -59}
, {-11, -4, 73}
}
, {{71, 31, 0}
, {17, 0, 44}
, {-63, -25, 72}
, {-45, 53, -61}
, {-37, -36, 1}
, {-59, 59, -29}
, {-23, 11, 34}
, {0, 53, 28}
, {-60, 18, -19}
, {50, -47, -20}
, {7, -41, -62}
, {72, -31, 0}
, {-35, 44, 8}
, {55, -28, -47}
, {-61, -27, -9}
, {-17, 73, -59}
, {-37, 56, 41}
, {9, -44, -38}
, {-14, 21, -40}
, {-9, 52, -4}
, {7, 6, 60}
, {63, -60, 42}
, {74, -24, 16}
, {-66, 61, 32}
, {-55, 30, 29}
, {28, 28, -35}
, {70, 5, 55}
, {-58, -41, -44}
, {-45, 23, -6}
, {-41, -1, 24}
, {-49, 1, -19}
, {4, 72, 65}
}
, {{-1, 15, -68}
, {27, 26, 52}
, {-18, 53, 45}
, {45, -65, -6}
, {-32, 72, -29}
, {-46, 25, 61}
, {-54, 6, -36}
, {-68, 16, 6}
, {-60, -64, 42}
, {21, 49, 68}
, {-34, -70, -22}
, {11, 3, -1}
, {-62, -41, 5}
, {5, -11, 27}
, {-60, 18, 41}
, {-67, -39, 54}
, {68, -29, 14}
, {1, -45, 46}
, {-12, 36, -24}
, {-29, -9, -74}
, {5, 7, 15}
, {30, -58, -53}
, {-68, -6, -54}
, {37, -20, -24}
, {-36, 14, -67}
, {70, -15, -64}
, {9, -33, 62}
, {63, 6, 40}
, {-4, -58, -8}
, {3, -31, -74}
, {25, -8, 64}
, {65, 23, -62}
}
, {{-32, -65, 30}
, {-40, -7, 43}
, {13, 53, -36}
, {-66, 3, 43}
, {71, -39, 35}
, {29, -60, 46}
, {-29, -41, -7}
, {48, -66, -65}
, {4, 26, -46}
, {15, 69, -50}
, {-27, 57, 69}
, {16, -60, -39}
, {50, -46, -55}
, {14, 0, 34}
, {-31, 11, -12}
, {-59, -68, 33}
, {18, 37, -46}
, {-64, 68, -4}
, {23, 27, -2}
, {73, 52, 22}
, {-65, 4, 57}
, {15, -31, -70}
, {-21, 28, -9}
, {36, -44, -63}
, {-60, -28, -57}
, {0, -39, 38}
, {-35, 12, 13}
, {-6, 8, 65}
, {-61, 50, 40}
, {2, -33, 18}
, {67, -57, -2}
, {4, -75, 25}
}
, {{-39, 0, 5}
, {-29, 43, 34}
, {1, 62, 12}
, {29, 7, 50}
, {19, -4, 11}
, {-31, 39, -73}
, {-8, -42, -57}
, {-35, 44, -60}
, {-49, 72, 53}
, {-1, 52, -19}
, {-8, 43, -4}
, {-38, -3, -39}
, {66, 19, -3}
, {-43, -40, 20}
, {64, 0, 10}
, {-50, 16, 35}
, {52, -23, -30}
, {40, 41, -8}
, {-6, 1, -62}
, {-70, 14, -20}
, {41, -49, -46}
, {-39, -30, 0}
, {62, 20, -57}
, {45, 59, -20}
, {-11, 57, 29}
, {5, 80, 22}
, {53, -17, -8}
, {71, 38, 16}
, {56, 48, 15}
, {21, -41, -68}
, {47, 14, 21}
, {-6, 22, 14}
}
, {{65, 39, -73}
, {-22, -17, -63}
, {70, 77, -18}
, {-51, -49, -16}
, {76, -52, -33}
, {-61, 53, 66}
, {-35, 16, -51}
, {-23, 66, -2}
, {-45, -63, 13}
, {73, 45, -5}
, {34, 17, 47}
, {0, 75, -27}
, {24, 32, -55}
, {-72, -35, 62}
, {-31, -58, 42}
, {71, 62, -19}
, {-50, 64, 40}
, {54, -19, -23}
, {22, 28, 2}
, {-7, -69, 55}
, {32, -24, 35}
, {-3, 26, -13}
, {7, -28, 43}
, {-40, 13, 71}
, {-68, 62, -20}
, {66, -17, 39}
, {-60, 10, 41}
, {-25, -19, 64}
, {-49, 68, 5}
, {66, 71, -27}
, {-6, -28, -43}
, {-43, 0, -56}
}
, {{-48, 47, 3}
, {73, -8, 73}
, {13, 40, 16}
, {54, -14, 3}
, {-75, 22, -9}
, {-71, 69, 46}
, {-62, 69, 29}
, {-32, -24, -50}
, {-62, -43, -18}
, {17, -54, -29}
, {-16, -62, -17}
, {-29, -4, -60}
, {48, 38, -31}
, {9, 22, -64}
, {-2, 18, -39}
, {7, 52, -32}
, {-32, -67, -59}
, {9, 45, 66}
, {-50, 47, -58}
, {-62, -26, 61}
, {5, 35, 32}
, {-22, 33, -19}
, {40, -36, 55}
, {-31, 37, 72}
, {53, 27, 19}
, {-18, 1, 28}
, {45, -45, 48}
, {70, 2, -65}
, {-12, 50, 62}
, {57, 15, -3}
, {44, -39, 16}
, {-20, -52, -19}
}
, {{13, -39, -30}
, {31, -13, -14}
, {8, 45, -25}
, {-17, -60, 69}
, {39, 46, 47}
, {-44, -32, 48}
, {60, -33, 39}
, {-69, -7, -18}
, {-15, -32, 2}
, {45, -34, 10}
, {-43, -47, 27}
, {41, 59, 61}
, {69, 68, -58}
, {-69, 61, -12}
, {31, -12, -54}
, {-48, 61, 63}
, {0, -27, -39}
, {33, -39, -27}
, {-64, -25, 57}
, {-12, -30, 50}
, {45, 45, 37}
, {24, -28, 12}
, {26, 16, -16}
, {-32, -49, 14}
, {-43, -47, 38}
, {-69, -34, 28}
, {-35, 11, -33}
, {-47, 49, -30}
, {24, 12, 61}
, {39, -1, 62}
, {-53, 11, 68}
, {-55, 30, 29}
}
, {{-66, 56, 37}
, {-59, -7, 0}
, {16, -25, 18}
, {-40, -49, -2}
, {-14, 20, 61}
, {-37, 73, 44}
, {-50, 17, -73}
, {-62, -44, 28}
, {-1, -23, 72}
, {66, 34, -44}
, {-19, -29, 26}
, {62, 45, -18}
, {-70, 23, -3}
, {-52, 77, -22}
, {4, 3, 17}
, {-27, -30, 53}
, {-16, 58, -16}
, {53, 19, -24}
, {43, 63, -37}
, {35, -45, -13}
, {-58, -3, 20}
, {-24, -52, 41}
, {-24, 59, -56}
, {-17, 42, 40}
, {-60, -26, 38}
, {39, -44, 15}
, {-6, 64, -11}
, {-72, 64, 28}
, {-38, -59, -62}
, {15, 70, -48}
, {-50, 31, -56}
, {-11, 0, -68}
}
, {{32, -19, -58}
, {-58, -54, -54}
, {67, -51, -56}
, {-12, 16, 5}
, {-29, 27, 15}
, {-41, -62, -58}
, {46, 35, -12}
, {-43, -45, 25}
, {8, -46, -50}
, {58, 74, 67}
, {71, 34, 24}
, {35, 70, 40}
, {-55, 76, -19}
, {26, -58, 15}
, {-12, 49, 33}
, {28, -43, 44}
, {26, -53, -70}
, {-39, -46, 15}
, {33, 61, 21}
, {20, 31, -60}
, {33, -60, 8}
, {-24, 48, 12}
, {33, 62, 44}
, {12, 12, -45}
, {-68, 20, -67}
, {-27, 6, -54}
, {17, 21, -60}
, {-50, 61, 63}
, {-8, -56, 21}
, {50, 68, 60}
, {-59, 78, -67}
, {7, 71, 56}
}
, {{-24, -69, 54}
, {58, 13, 38}
, {-38, -57, -9}
, {-28, 60, -70}
, {-43, -64, -43}
, {-71, -29, 57}
, {33, 41, -31}
, {-19, 17, 58}
, {63, 9, 55}
, {-18, -14, -54}
, {29, 14, -19}
, {-33, 25, 16}
, {-57, -51, 8}
, {-6, 62, -50}
, {17, -50, 10}
, {-47, 2, 63}
, {8, -9, -34}
, {0, -65, 35}
, {-54, -30, -69}
, {55, -48, -17}
, {12, -9, -5}
, {-70, -45, 47}
, {38, -75, -6}
, {-75, 66, 34}
, {-26, -24, -43}
, {-62, 54, -36}
, {-6, 62, 66}
, {40, 47, 72}
, {31, -40, -46}
, {-26, 63, 21}
, {-3, -10, 70}
, {35, -47, -19}
}
, {{11, -6, -53}
, {-12, 62, 69}
, {-22, 3, 36}
, {71, -29, 6}
, {-8, -31, 30}
, {67, 4, -25}
, {36, -30, 54}
, {0, 71, -17}
, {2, 17, -70}
, {-51, 11, 6}
, {16, -74, -54}
, {22, 23, 51}
, {37, -48, 13}
, {-69, -48, -32}
, {-62, -21, 4}
, {9, -69, 51}
, {54, -55, -33}
, {-19, 49, 28}
, {-16, -67, -34}
, {-35, -2, 43}
, {-32, -3, 56}
, {-40, 45, -9}
, {-51, 69, -34}
, {-68, 68, -30}
, {54, -33, 16}
, {-15, 7, -63}
, {58, 25, 8}
, {65, -29, -69}
, {-48, 47, -5}
, {13, -66, 1}
, {-19, 34, -34}
, {35, -69, 12}
}
, {{-43, 59, -34}
, {-42, 33, -27}
, {52, 13, -30}
, {-7, 33, 29}
, {-16, 40, 42}
, {9, -39, -7}
, {-42, 0, -24}
, {63, 4, -26}
, {34, 32, -9}
, {-41, 49, 48}
, {51, -16, -50}
, {-34, -60, 21}
, {36, 22, 33}
, {20, 54, -29}
, {-39, 62, 56}
, {4, 15, 11}
, {61, -41, 50}
, {-26, -29, 31}
, {32, -29, -68}
, {-39, -6, 41}
, {50, -18, -26}
, {-65, -34, -14}
, {20, 12, 22}
, {6, -17, -66}
, {38, 58, 42}
, {-6, 6, 41}
, {10, 34, -13}
, {13, 36, 3}
, {48, 71, 24}
, {-3, 26, -17}
, {61, 44, -74}
, {-64, 46, -31}
}
, {{-45, -25, 13}
, {-52, 56, -6}
, {42, -54, -68}
, {73, 74, 33}
, {-21, 47, 62}
, {-53, 1, -56}
, {9, -49, 15}
, {-56, -59, 34}
, {66, -50, -56}
, {37, -66, 46}
, {56, -63, 67}
, {56, 45, -41}
, {4, 9, 27}
, {-48, 57, -11}
, {11, -13, 10}
, {-31, 15, -58}
, {45, -43, 73}
, {-29, 10, -70}
, {-31, -24, -66}
, {13, 19, -32}
, {32, -43, 58}
, {12, -28, -51}
, {-70, 2, 58}
, {56, 69, 0}
, {34, 18, 58}
, {28, 62, 65}
, {-17, -36, 28}
, {-1, 40, 60}
, {51, 49, -64}
, {-5, -48, -72}
, {39, -33, 63}
, {45, 20, -23}
}
, {{-10, -33, 59}
, {-42, 32, -69}
, {49, -15, 36}
, {37, -73, -44}
, {-34, 24, -11}
, {20, -40, 49}
, {18, -26, -30}
, {-59, -65, -59}
, {19, -15, -65}
, {1, -59, 19}
, {25, -39, -10}
, {42, -6, -58}
, {7, -69, 19}
, {13, -34, -47}
, {21, 5, 19}
, {-69, 14, 49}
, {-23, -37, -7}
, {-59, 65, 29}
, {-45, 64, -63}
, {-58, -68, -71}
, {-60, 32, -66}
, {-14, -62, -14}
, {-47, -13, 57}
, {36, 52, -18}
, {-60, 1, 70}
, {-66, -25, 40}
, {-48, -51, 44}
, {-68, -38, 14}
, {13, 2, 48}
, {73, -50, 15}
, {-53, -31, 58}
, {60, -52, 20}
}
, {{35, -59, -34}
, {-43, -4, 45}
, {7, -42, 68}
, {6, -31, 38}
, {78, -3, 4}
, {-54, -50, -8}
, {58, -56, 51}
, {20, 6, -4}
, {-55, 47, -63}
, {42, 46, 28}
, {5, 42, 34}
, {12, 29, -70}
, {74, 67, -11}
, {18, 33, -55}
, {-54, 75, 25}
, {-62, -32, -50}
, {38, -21, 68}
, {61, 16, 19}
, {15, 46, -65}
, {-4, -29, -33}
, {42, -41, -69}
, {73, -69, -47}
, {-41, -9, -34}
, {0, 61, -60}
, {-31, 64, 39}
, {-3, -6, 50}
, {-52, 52, -35}
, {56, 26, 25}
, {-57, 43, 38}
, {51, -34, 69}
, {45, -2, 41}
, {60, -28, 14}
}
, {{-33, -1, 34}
, {71, 40, -62}
, {47, 27, 72}
, {44, 18, 64}
, {0, 38, -58}
, {61, -65, -25}
, {62, -28, 38}
, {-57, 46, -14}
, {79, 71, -29}
, {-57, 74, -55}
, {-23, 54, 3}
, {65, -11, -38}
, {-19, 17, 34}
, {-57, 37, -21}
, {63, 66, 48}
, {-11, 13, 42}
, {14, 72, -23}
, {-12, 25, 34}
, {28, -38, -29}
, {40, 18, -41}
, {-42, 0, -8}
, {25, 63, -20}
, {8, 25, -37}
, {12, 26, -27}
, {-55, -21, 62}
, {58, 66, 9}
, {-63, 29, -66}
, {-9, 65, -42}
, {70, -44, -35}
, {-10, 39, 63}
, {-55, -37, 24}
, {-60, -46, 0}
}
, {{-28, -8, 5}
, {-54, -55, -21}
, {50, 10, -62}
, {-34, 47, 15}
, {24, 4, -63}
, {41, -49, -65}
, {15, -7, 74}
, {-11, 26, 41}
, {-57, -3, -49}
, {-58, -12, 72}
, {48, -39, 71}
, {-26, 22, 60}
, {-38, -27, -15}
, {-52, -66, 67}
, {-8, 74, 4}
, {0, 4, -12}
, {-7, -23, -58}
, {-7, 48, -47}
, {40, 75, -18}
, {13, -9, 22}
, {14, 67, -13}
, {60, -51, -31}
, {0, 39, 65}
, {-62, -11, 22}
, {45, 36, 40}
, {70, 42, -15}
, {46, 57, -68}
, {-40, 38, -7}
, {-62, 0, 18}
, {-68, 9, 40}
, {56, 55, -12}
, {23, 56, 38}
}
, {{25, -54, 57}
, {35, -32, 65}
, {22, 69, -36}
, {-5, -21, 38}
, {-61, 75, 10}
, {30, -32, -6}
, {-61, 18, -40}
, {-27, -43, -40}
, {-61, -30, 39}
, {44, -44, -22}
, {-57, 9, -33}
, {-11, 56, 21}
, {29, -55, 37}
, {-26, 20, 19}
, {60, 70, 40}
, {-40, 12, -14}
, {2, -16, 30}
, {18, -50, 56}
, {60, 28, 68}
, {-36, 19, 30}
, {20, 43, 40}
, {58, 40, 57}
, {7, -1, 27}
, {26, -11, 13}
, {-9, 51, 60}
, {-20, 24, 18}
, {-16, 30, 58}
, {26, -61, 71}
, {-51, -73, -17}
, {-34, 66, -21}
, {8, -29, 34}
, {52, -67, -10}
}
, {{71, -37, -37}
, {21, -26, 0}
, {74, 27, 12}
, {38, 35, -55}
, {22, 53, -66}
, {-23, -31, -64}
, {-61, -26, 8}
, {-25, 3, 37}
, {61, -11, -37}
, {78, -40, 4}
, {-55, 10, 66}
, {0, -13, -69}
, {-1, 13, 49}
, {-52, -67, 61}
, {-32, -7, -1}
, {61, -7, 62}
, {43, -38, -48}
, {79, -33, -27}
, {-33, -46, 73}
, {61, 30, -60}
, {-36, -25, 70}
, {-25, -39, -58}
, {9, -6, 68}
, {-13, -5, 5}
, {44, -14, -53}
, {-26, 12, 41}
, {75, -29, 63}
, {51, 22, 44}
, {-39, -51, -65}
, {45, -30, 54}
, {-46, -11, 3}
, {57, -60, -27}
}
, {{-40, 0, 65}
, {-28, -47, -36}
, {-51, 3, -25}
, {-42, 70, 14}
, {-28, 41, -59}
, {68, -65, 68}
, {-6, 14, 35}
, {4, 40, 59}
, {53, 14, 21}
, {-29, 62, 47}
, {72, 26, -25}
, {-25, 17, -49}
, {-38, 3, -48}
, {25, 0, -50}
, {-25, -50, -38}
, {-3, 59, 54}
, {-46, -15, 8}
, {-39, -55, 19}
, {10, -57, 36}
, {39, -42, 44}
, {-58, 48, -52}
, {-45, -52, -66}
, {-60, 54, -29}
, {-54, 65, 14}
, {4, 55, -38}
, {57, 72, 53}
, {-14, -71, 9}
, {-14, 11, -40}
, {-71, 10, -32}
, {51, 47, -1}
, {-39, 31, -63}
, {-6, -63, -34}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   55
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_145_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_145(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [13][64]
#define OUTPUT_DIM 832

//typedef number_t *flatten_20_output_type;
typedef number_t flatten_20_output_type[OUTPUT_DIM];

#define flatten_20 //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 832
#define FC_UNITS 16
#define ACTIVATION_RELU

typedef number_t dense_50_output_type[FC_UNITS];

static inline void dense_50(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 832
#define FC_UNITS 16


const int16_t dense_50_bias[FC_UNITS] = {-1, 6, 5, -2, -4, 4, 7, 7, 7, 8, -1, 8, -1, 3, 4, -2}
;

const int16_t dense_50_kernel[FC_UNITS][INPUT_SAMPLES] = {{10, 1, -9, -8, -23, 30, 13, 35, -9, -31, 0, 42, 7, 38, 17, 27, -17, -39, -22, 39, 39, -32, 22, 29, -5, 13, -10, -21, -24, -8, 41, -24, -9, -17, 26, 31, 19, -24, 16, -35, 28, -31, 21, 28, -26, 12, -32, 0, 24, 19, -28, 20, 27, -12, -10, -7, -26, 15, 13, 22, -11, -7, -26, 5, -29, -15, -30, 15, 41, -29, -6, -27, -35, -42, -32, 11, 6, 33, 19, 10, 33, -35, -37, -9, 14, 15, -19, -26, 20, 13, 38, -14, -9, 27, -28, -39, -42, 33, 31, -2, 21, 16, -4, -20, -22, 11, 7, -33, -33, 24, -13, -19, -5, -24, 35, 23, 21, -10, -14, -32, -33, -14, -10, -41, -16, 14, -13, 20, -19, -7, 8, 15, 10, -8, -36, 1, -3, 6, 22, -14, 8, 23, -30, 19, 23, 29, -35, 18, -27, -25, 32, 29, -8, -21, -6, 9, 39, 15, -37, 8, 11, 0, 37, -32, 2, -1, -24, 8, 18, -24, -1, -26, 1, 15, -7, -1, 39, -30, -15, 34, -5, -33, -5, 30, 13, 41, 10, -7, 13, -6, 24, 32, 31, -24, -4, -10, -43, -6, 14, -26, -31, -10, -3, 34, -19, -12, 2, 32, 19, 29, -24, 5, -37, 36, 37, -36, 21, -4, 21, 16, -39, 31, -32, -30, 13, 6, -11, 29, 8, -32, -33, 21, -19, 40, 9, -27, 38, -11, -3, -26, -1, -38, 25, -43, -37, -39, 29, 15, 4, -9, -24, 2, -19, -37, -42, -14, -42, -27, -11, 37, -22, -37, 22, -33, -7, -31, 31, 38, 40, 40, 31, 39, -31, -41, -3, 6, -25, -27, -24, 5, -23, 30, -34, -2, 24, -12, -40, 35, 10, 3, -16, 0, -6, 18, -43, -12, 12, -41, -15, 5, -38, -13, 34, -23, -29, 33, 17, 22, 2, -12, -32, -27, 40, -21, 37, -32, -17, -43, -21, 26, -26, -14, 32, 31, -25, -8, -26, 16, 0, -33, -20, -7, 0, -32, 39, 35, -31, 9, -7, -14, -22, 22, -28, 26, -26, -10, 7, 32, 17, 40, 26, -2, -28, -24, -31, 13, 18, 38, -9, -24, 19, 10, -27, 21, -2, -1, -1, 14, 25, 24, -7, -34, 2, 29, 30, -39, 2, -21, 8, 19, -10, -43, -20, -5, 35, -4, -37, -37, -3, -7, -34, 34, -14, 33, 15, -17, 18, -21, -35, 0, -12, -31, 2, 27, 4, 10, -26, 26, 0, 2, -37, -7, 15, -36, 5, -9, 39, -39, -18, -40, -13, 12, -3, 19, 3, -10, -23, -22, 11, -14, 12, -44, 25, 41, 26, -16, -15, 32, 39, 16, 32, 42, 38, -34, 41, -42, -14, -18, 5, -13, 6, -7, 2, -41, 17, -42, -15, 41, -40, 3, 12, 19, -39, -19, -30, 15, 0, -11, 40, 13, -18, 34, 28, 8, -18, -17, -43, 18, 3, -18, 19, -37, -14, 22, -31, -11, -12, -20, -37, 2, 0, 24, -30, 42, 27, 18, -41, -23, 33, 16, 22, 26, -20, 15, 26, 5, -29, -14, -28, 23, 13, -32, -21, 41, 24, 39, 31, -22, -40, -11, -33, -10, 9, -37, 28, -8, 19, 21, -22, 19, -27, 0, -42, 19, 15, -9, 2, 38, -16, -22, 17, -41, -39, -20, -12, 28, -10, 21, -15, 28, 0, -30, 11, -3, 2, -21, -12, -2, 26, 31, 19, -29, -41, 18, 34, -40, 20, -30, 31, -43, 23, -41, -13, -14, -12, 38, 0, -38, -8, -34, 37, 6, -12, -27, -29, -1, -38, 16, -30, -22, 19, -37, -20, 13, 16, 10, 32, -3, -35, 37, 15, 16, 39, 17, 35, 40, 14, 33, 38, -22, -15, 4, -30, -6, 23, 26, 29, 0, 36, 16, -41, 5, 13, 35, -33, 31, -15, -9, -39, 28, -29, -18, 11, 15, 5, -5, -33, 6, -19, -41, -36, -32, 23, 0, -21, -28, -14, 39, -26, 17, 35, -27, 37, -28, -28, 30, 19, -36, -39, -12, -33, -14, 18, 22, -9, 3, -23, -8, 33, 8, -21, 26, 32, -39, -14, -32, 15, -9, -26, 41, -13, -24, -41, 21, -38, 3, 0, -20, -38, -40, -6, -33, 33, -35, 29, 40, 32, 21, -18, 31, -13, 24, -36, 38, 40, 11, -15, 0, 7, 0, -29, 22, 4, -39, -8, -26, 6, 41, 39, 23, 0, 6, -6, 4, 12, -1, -10, 38, -32, 26, 36, 33, 41, 29, 25, 32, -37, -20, -43, 41, 42, -13, 29, 41, 4, 1, -38, 11, 9, -29, -14, -13, 4, -37, 7, 31, 30, -5, 27, -33, 26, -9, -24, -5, 14, 28, 18, -29, -25, -33, 0, -42, -26, -13, 0, -21, -40, -18, -22, 36, -15, -26, -38, 30, -14, -8, -3, 23, -25, -39, 30, -4, 34, 21, 3, -17, -35, 18, 20, 22, 18, -1, 39, 11, -33, -33, 24, 9, -6, 13, -24, -12, 8, 25, 25, -24, -18, 4, 16, 10, -18, 37, 34, -7, -20, 27, 18, -37, 0, 31, 42, 11}
, {4, -11, 29, 29, 37, -24, 34, -35, -48, -23, 15, 20, -20, -12, 34, -9, 2, 20, 41, 14, 20, 9, -11, 21, -38, 31, 21, 20, -2, 35, -20, -33, 6, 28, -15, -2, -5, -29, -23, 12, -21, 21, 9, -24, -9, -34, 15, 22, 23, -9, 31, -36, -28, 29, 8, 4, -13, 31, 15, 0, 26, -23, 6, -9, 26, -32, 9, -22, -39, 14, 47, -32, -22, -26, 11, 5, -9, 32, 27, 13, 29, -13, -32, 22, -32, -33, 9, 24, 29, 4, 3, -39, -9, 39, 39, 4, 37, -12, 23, -5, -17, -26, -16, -32, 7, 16, -27, 18, -5, 0, -1, -16, -5, -25, 36, -3, -31, -21, -32, -10, 45, -17, -7, 35, 27, 0, -17, 9, -33, -31, -7, -39, 20, 25, 32, 18, 30, -10, 34, -5, 7, 37, -29, 17, -21, -22, 19, 18, -11, -3, -23, 5, 0, -40, 1, 1, -17, -26, -13, -7, 0, -22, -42, 18, -41, -36, -7, -40, 20, 6, 30, 38, 3, -29, 6, 16, 19, -15, -15, 20, -37, -12, 21, -15, 1, -7, 9, 41, 33, -24, 26, 40, -37, 10, 25, -43, 30, -5, -32, -20, -21, 7, 13, -28, -13, -31, 32, -22, 9, -16, -42, -6, -30, 6, -15, -18, 21, -3, -8, 34, 32, -14, -29, -2, -28, -29, -14, 2, -19, 14, -12, -25, -36, 11, -22, 9, -4, -7, 14, -11, 9, 10, 2, -23, -16, 6, 27, -38, -37, 35, -26, 23, -31, 46, 33, 6, 38, -22, -21, 29, 31, 18, 3, -24, 14, -15, 19, -18, -8, -13, 36, 6, -38, -51, -33, -20, -9, -40, 23, -21, 36, 6, -13, 7, -5, 0, 8, 30, -38, 17, -14, -19, 20, -25, -17, 5, -1, -5, -27, -29, -34, 7, -27, -6, 34, 17, -14, 12, -37, -36, -41, 6, 20, -7, 33, 27, 12, 0, 44, 42, -18, -8, 37, -39, -26, -38, 40, 12, 8, 18, 8, 25, 31, 20, 2, 36, 21, 15, -31, -25, 0, 30, 17, 6, -13, -27, 25, 3, -16, 6, 26, -43, -39, -40, 17, -27, -6, -15, -10, -20, -7, -35, -12, 3, -35, -47, -3, 24, -3, 10, -16, 30, 0, -16, -34, 19, 29, -19, -19, -30, 32, -35, 38, 39, -24, 13, 33, -14, 10, 30, -12, -35, -18, -5, 42, 11, 29, 2, 24, 21, 12, 37, 18, -49, -46, -32, 12, 23, 43, -35, 3, 41, 15, 28, 16, 8, 27, -13, 5, 21, -7, 24, 24, -6, 38, 35, -1, -36, 16, -21, -2, 30, -2, -12, -15, 23, -23, 2, 2, -30, 40, -4, -12, 3, 1, 1, 0, -36, 48, -31, 44, 15, 3, 43, -6, -26, 6, 37, 11, 22, 41, -27, 16, -11, 13, 35, -14, -29, -35, -26, -42, 39, -19, 0, 43, 37, 20, 32, 21, 6, -26, 12, -36, 31, -36, 19, 7, -21, 19, -33, 0, 16, -14, 6, 17, -46, -18, -36, 8, 20, 50, -9, -19, 43, 2, -32, -41, 0, -22, -17, -11, 35, 43, 28, 3, 7, -18, 21, -23, -10, 34, -36, 20, 39, 41, 21, -36, 4, -34, -20, -21, -31, 23, -34, 20, -3, 0, -2, -32, 30, 44, -14, -22, 0, -5, -2, 27, -17, 1, 26, -17, -19, -44, -20, -12, -22, -10, 30, -40, 5, 33, -21, -30, 19, -24, 38, 33, -8, 26, -7, -34, 39, 26, -8, 39, -29, -1, -12, 2, 18, 18, 8, 24, -15, 10, -4, -19, 1, 16, -34, 6, 30, 0, 7, 6, -38, 0, 36, -27, -24, 11, -9, -9, -17, 16, -11, 32, -14, -8, 8, 23, -1, -33, -22, -15, 11, 5, -17, 20, -37, 39, -21, 37, 8, -23, -15, 24, -43, -21, 23, 28, 4, 24, 33, -12, 10, 30, 35, -2, -41, -23, -25, 7, -11, 2, -17, -30, 14, -10, 28, -12, 7, -11, 0, -4, -10, -29, -31, -9, -41, -26, -22, -22, 15, 13, 1, 43, 24, 30, 45, 41, -12, -33, 15, 9, 25, 16, 24, -25, -46, -12, 15, -40, 38, -21, -44, -36, 34, 30, -44, -40, 12, 20, -14, 0, -40, -27, -35, 0, 0, 17, -29, -8, 5, -11, 20, 19, -10, -8, -8, 21, -17, 17, 34, 35, -41, -41, -12, -18, 38, -1, -2, 1, -19, -20, -6, 39, 21, 6, -4, 10, -36, 30, -5, 16, 38, -27, 13, -8, 32, -34, -9, 35, 30, 8, 6, -36, 22, -40, -3, -36, -33, -43, 17, -47, 5, -33, 21, -34, -2, -46, -24, -22, -7, -44, -14, -17, -38, -48, -2, -21, 6, 13, -49, -3, 12, 3, -36, -7, 27, -44, -30, 30, 18, 12, -11, -18, -4, -34, -37, 40, -6, 32, -44, 4, 21, 26, 27, -37, 0, 30, -36, 1, -28, 5, -38, -39, 2, -18, 33, -42, 36, 10, 34, -23, 41, 22, 21, -2, 22, 13, -39, 29, -23, 11, 36, -7, 0, 30, -25, -35, 2}
, {-24, -36, 46, 29, -33, -18, 33, -35, -29, 36, -25, 37, -13, -45, 12, 31, 43, -13, 0, 37, 45, 8, 13, 34, 10, 9, -29, -2, -41, -9, -35, -17, 27, 8, 30, -31, -16, -29, 29, 27, -3, 33, -26, 17, -26, 44, 42, -32, 10, 23, 12, -27, -38, -22, 24, -31, -1, 0, -10, -12, 7, 39, -19, 12, -29, -39, -16, 14, -8, 40, 34, 16, -6, -25, -24, 0, -32, 18, -41, 2, -14, 43, -43, 27, 8, 25, 36, 16, 33, 7, 42, 28, -27, -29, -37, 0, 35, -15, -23, 21, -1, 31, 43, 15, -10, 22, 25, 11, -15, 22, 17, -23, 14, 31, 10, -35, -2, -45, -21, -23, 27, -22, 17, -23, -28, 32, -30, -8, -25, -10, -30, -17, -3, 42, -4, 2, -26, -5, 33, 31, -2, 25, 12, -5, 35, -19, 16, 34, -34, -17, 13, -23, -43, -3, -11, 5, -29, -27, 19, 0, 33, 6, -30, 24, -1, 6, -20, -1, -21, 24, 19, -15, 22, -31, -13, -11, -35, -35, -9, 37, -40, 5, 12, 37, -21, 16, 22, -4, 19, -31, 23, 28, 13, -32, -14, -2, 26, -1, 40, 36, -10, 16, -34, 22, 11, -25, -21, 21, -17, -24, 6, -16, -22, -1, 1, 34, 10, -27, 3, 16, -29, 21, -26, -4, 23, -6, 30, -11, 22, 11, 7, 42, 6, 22, 37, -4, -31, -35, 13, 36, 0, -12, 0, -6, -18, 0, 2, 15, 4, -4, 33, -34, 43, -18, -29, 36, -19, 21, 19, 9, -26, -25, 39, -32, 41, 0, -14, -23, -28, -1, -30, 0, -10, -3, -38, -3, -38, -12, -19, -12, 20, 17, -21, -36, -1, 44, -41, -26, 11, -40, 5, 2, -28, 18, -29, -18, -24, -33, 25, -8, 36, -1, 0, -19, 39, -26, -1, -7, -26, 12, 8, 19, -6, -3, 43, -2, 8, -32, 4, -27, 4, -11, -10, -21, -1, 22, 11, -18, 30, 6, 21, 14, -38, 23, 19, -20, 9, -19, -44, -20, -13, -22, -3, 36, -36, 32, 39, 27, 42, 39, 27, -11, -4, 44, 32, -16, 1, 43, 16, -37, 26, 30, -13, -20, -26, 0, 25, 5, -5, 14, -20, -6, -17, 35, 15, 4, 0, 9, -12, -14, -3, -13, 2, 9, 26, -33, 37, 22, 14, -24, 18, 5, -17, 35, 47, 34, 2, 29, -3, -25, 32, 15, -13, -44, -43, 6, 4, 36, 18, 31, -16, 2, 8, 15, 45, -7, 34, 31, 34, -38, 28, 13, -1, 36, -33, 0, 40, -16, 14, 29, -10, 41, -24, 16, 48, 0, 11, -5, -21, 4, -15, 36, 37, 19, -27, -33, -14, 36, -17, 5, 8, -10, 13, 39, -3, 12, 10, 26, 33, -26, -31, 45, 12, 10, 35, -24, -3, 29, -11, -37, -31, 41, -13, 29, 0, -16, -1, -10, 9, 33, -5, -25, 26, -26, -3, 36, 17, 39, 48, -25, 21, 15, -11, -10, -18, -34, -18, -13, 2, 42, 35, 5, 7, 24, 25, -24, 32, 30, -26, -22, 0, 11, 32, -25, 20, 38, 37, 13, 26, 28, -5, -1, -34, 12, 10, -19, 47, 41, -28, -4, 0, 18, 1, -4, -1, -28, -39, 9, -2, -14, -9, 37, -12, 27, -27, 21, 43, 0, 42, 5, 39, 42, 20, 36, -7, 17, 17, -39, -4, 15, -21, 13, 36, -12, 37, 34, 30, -33, -30, -2, 29, -28, -32, 14, -27, 1, -12, 24, -17, 26, 15, 43, 13, -25, -5, -18, -16, -36, -37, -15, 10, 35, 4, -4, -32, -38, 20, -13, 4, -19, 19, -17, 23, -32, -4, 0, 10, 46, 43, -21, -15, 0, -35, -12, -25, -7, -19, -19, 38, -7, -27, 22, -18, -36, -9, -28, 11, -37, -26, 22, -2, 32, 0, -36, -19, 33, 3, 26, -8, -24, -35, 20, -9, -26, 40, -26, -28, -35, 27, -1, -38, -27, 38, 16, 44, -29, -13, 14, 8, -9, -10, -43, -27, -4, 19, 21, -36, -4, 42, -25, 41, 36, -20, -33, -5, -35, -19, 36, -23, 5, -10, -38, -36, -7, 28, 30, -12, -16, -30, -24, 29, 37, -16, -19, 8, -7, 33, -29, 20, 33, -35, -30, 6, -16, -1, 26, -28, -6, -19, -5, 31, 3, 43, 1, 13, 39, -43, -36, -21, -32, 41, -12, -21, -15, 1, 2, -32, -1, 6, -33, 18, 27, 29, -20, 18, 41, 36, 3, 29, 13, 27, -35, 5, 26, 10, 39, -18, 16, 8, -35, 20, -20, -1, -11, -4, 25, -12, -28, -31, -6, -36, -11, -29, -33, -23, -35, -30, -44, 7, -39, -45, -34, -12, 20, -19, 21, -34, -17, -21, 3, -19, 25, -24, 4, -2, 10, 35, -23, 33, -31, 42, -28, -7, 20, -27, 2, -28, -13, 0, 39, 38, -11, -27, -25, 27, 5, -31, -29, -12, 17, 36, -1, -37, -5, -27, 37, 31, 8, 24, 0, -24, 5, -36, 21, -34, -34, 13, 18, -22, -10, -5, -35}
, {-34, -11, 28, -31, -5, -3, -40, -12, -35, -38, -10, -2, -21, -41, 14, -18, -3, 21, -7, -35, 21, -42, 29, 27, -42, 38, 8, -15, -7, -43, 16, -29, 5, -1, -43, 26, -14, 2, -1, -39, -35, 24, 14, 17, -22, -38, 9, -5, -30, 29, -37, 38, -18, 25, 31, -23, -38, 20, 23, -3, 8, -8, 18, -39, 3, 32, -16, -28, -13, 30, 7, 23, 26, 35, 12, 19, 0, -44, 20, -17, 21, 30, -13, -40, -27, 23, 35, 26, 41, -12, -37, 38, -40, 12, 37, -31, -7, -33, 6, 26, 32, -31, -34, -26, -38, -33, 27, 4, 37, 29, -20, 14, -30, -5, -20, -26, -10, 4, -20, -31, -26, -26, 0, 34, -4, -9, 19, -14, 39, -11, 4, 28, -19, -9, -16, 0, 17, 29, 15, 5, -14, -13, -24, -31, -32, -2, 23, 34, -9, -36, -31, 28, 25, -2, 32, -5, 36, 38, -44, -10, -36, 4, 2, -12, 37, 20, 23, -23, -12, -4, -11, -21, -8, 19, 24, 11, -26, 36, 15, 11, 20, -1, 12, 17, -5, 1, 34, -28, 11, 11, -38, -14, 12, -16, 8, -36, -32, 38, -1, -26, -9, -11, 7, 15, -42, 6, 2, -33, 14, -42, 12, -8, -9, -17, -13, 35, 38, -34, -4, -11, -6, -9, -11, -23, -8, 24, -42, -23, -4, -31, -16, -36, 0, -5, 8, -12, -14, 37, -17, -43, -36, -21, -24, -23, 32, 22, 31, 6, 7, -40, -9, -11, -3, 27, 10, -37, -4, 35, -7, 34, -1, 27, -2, -17, -24, 32, 25, -9, 36, 21, 13, 23, 11, -42, 20, 33, -8, 30, -18, -7, 13, 2, 7, -43, -8, 11, -42, -16, -38, 15, -23, 0, -16, 39, -28, 3, -33, 28, -3, -11, -35, -21, 9, 13, 27, 28, -26, -34, 21, -27, 15, -30, -23, 19, 36, -2, 33, 20, -14, -9, 33, -3, 34, -13, -10, -31, 9, 26, -30, -4, 34, -19, -39, -39, -13, 40, -14, -25, -9, -24, -33, 20, 34, 26, -44, 5, -12, 14, -34, 5, 22, 32, -17, -27, 14, 3, -23, 1, 2, -34, -26, -42, 37, -20, 37, -25, -5, -10, 18, -10, 32, 2, 8, 12, -31, 25, 6, 21, -31, 29, 14, -16, 27, 38, -2, -17, 15, 27, -2, -43, -35, 14, 40, 0, -11, 7, -42, -34, 1, -1, -18, -37, -5, -7, 33, -10, -13, -14, -28, -7, 26, 3, -41, -27, 22, 39, 13, 32, -32, -30, 30, -5, -26, 3, -43, -14, -45, 4, 1, 26, -8, -13, -15, -3, 4, 9, -28, -39, -18, 15, 39, 27, -26, 9, 28, -12, -15, 8, -1, 25, -11, -1, -36, 0, 36, -26, -23, 3, 24, 35, 4, 32, -44, 24, 31, 10, 0, -6, 27, -14, -33, -37, -8, -35, -9, 34, 29, 9, 32, 9, 8, -16, -11, 32, -9, 18, -31, 4, -2, 39, -42, -1, -11, 26, -39, 28, 28, 23, -33, -21, -38, 18, -36, 25, 36, 23, 16, -1, -7, -13, 39, -11, -16, 22, -34, 30, -31, 6, 9, 12, -6, -22, 16, -34, -22, 3, 13, -41, 29, -43, 13, -40, 9, -8, 28, -23, 17, 7, -32, -17, 10, 9, -26, -24, 26, -4, 6, -16, -21, 31, 39, -34, 22, 11, 0, 40, 8, 8, 6, -17, 29, -22, -44, 35, 24, 0, -30, -5, 6, -3, 13, -26, -41, -11, -15, -13, 15, -40, -21, -12, 39, -7, -19, -9, -5, -5, 20, 20, 16, -41, 24, 31, -20, 13, -24, -27, 15, -42, 12, -42, 15, 0, -8, 8, 24, -44, -40, -32, -40, 24, 35, 18, 36, 0, -8, 18, -4, -28, 36, -11, -15, -43, -19, 28, 36, -9, -37, 29, 8, -27, 36, 29, 30, -9, 0, 16, -22, 13, -35, -16, 29, 31, 31, -9, 22, 34, -17, 18, -4, -32, -23, 14, 0, -21, -25, 27, 18, -18, 32, -7, -32, 10, -16, 14, -12, -31, 0, 40, -4, -7, -5, -33, 22, 6, 8, -20, 11, 20, -38, 15, -25, -24, 9, 13, -17, -43, -25, 33, -27, 30, 20, -21, 27, 37, 2, -14, -44, -17, -20, -19, 23, -1, -13, -32, -34, -41, -25, -12, -2, 22, -35, -11, 5, 31, 39, -17, 7, 35, 39, -43, -1, -23, -36, 32, -22, 7, -33, 6, 9, -21, -42, 23, 36, 24, -15, 27, 36, 25, -42, 5, 28, -17, 20, -5, 34, 29, -8, -3, 14, 27, -13, 20, 7, 22, 0, 7, -2, -38, 36, -22, -35, 20, 35, -36, 16, 36, 16, 22, -39, -20, 10, 4, 22, -1, 34, -15, -32, 3, 20, -15, 9, 6, 6, 16, -3, 6, 30, -5, 6, 26, -18, -2, 25, -36, -27, -14, -6, -42, -15, -33, 26, 7, 5, -8, 25, -41, -24, -24, -9, 25, 31, 11, 1, -32, -44, 19, 0, 14, -34, -21, 9, 10, 35, -5, -21, 24, -6, -23, 17, 22, -27, 25}
, {12, 22, -17, -8, -29, -18, -15, 47, 10, 8, 12, -3, 42, -30, -2, -21, -4, -10, 18, -33, -28, 26, 14, -8, 43, -11, 40, 17, -18, 44, 17, 37, -28, 17, -24, -29, 42, 31, 1, 26, -29, 20, 37, 4, -20, 39, -3, -22, -12, -10, -32, -1, 21, 18, 43, 0, 34, -39, -16, 33, 12, 37, -18, 29, 46, 45, 35, 32, -31, -1, -24, 2, 34, 6, 16, -38, -1, 42, -32, 8, -17, 7, 25, -27, 30, -12, -29, 28, 20, -26, -12, -22, 17, 36, 27, -41, 15, 13, 19, -23, -30, -3, -39, -7, 28, -16, 2, 20, 9, -13, 7, -4, -5, 14, 35, -37, 42, -28, 12, -17, 38, 17, 37, -22, 10, 11, 21, -6, 14, 44, 9, 24, 46, 40, -5, 28, -38, 18, 32, -2, -6, -18, 16, -12, -15, 20, 42, 16, -34, 4, 44, 13, 28, 40, 27, -29, 5, -27, -25, -35, 37, -41, 37, 0, 18, 40, -26, 27, -4, -38, -36, -36, 35, -13, -3, -6, 17, 16, 32, -31, -20, 12, 13, -25, -29, -14, -30, -21, -33, -22, 16, 41, -7, -8, 31, 36, 7, 10, -10, -12, -2, 16, 37, 7, -2, -16, 38, -15, 11, 1, 8, 9, 4, 33, -3, -38, 17, 28, 14, -19, 33, -30, -2, -6, -36, 0, 23, -10, 34, 14, -1, 7, 42, 24, -3, 25, 26, -7, 31, -24, -28, -5, 12, -35, -27, 2, -26, 26, 4, -39, 39, -23, 7, 4, 27, 16, 2, 12, 36, -30, 26, -16, 3, 14, 29, 27, 31, 11, 0, 0, -3, -33, 11, 9, 32, 15, -19, -10, -15, -8, 22, 4, 35, 41, 0, 10, 25, -30, 2, -17, -42, 18, 32, 13, -35, 21, 25, 43, -22, 26, 15, -36, 17, 30, 22, -3, 25, -20, 33, -15, 35, 46, 23, 23, 2, 23, 9, 35, 15, -25, 37, 14, 31, -21, 24, 30, 33, -3, -40, -16, 41, -15, -3, 20, -1, -37, 36, 3, 14, 39, 42, 19, -29, -43, 20, -27, 23, 24, -32, -25, -3, 35, 38, -23, 22, -4, -39, -15, -15, 31, -19, 29, 38, -15, -5, 7, -3, -16, -39, -10, 13, -4, -12, -41, -32, -15, -18, -22, -19, 42, -43, 0, 39, -12, -33, -19, -16, 34, -8, -6, 19, 14, 6, -38, -25, -23, 5, 14, 2, 18, -18, 8, -34, 0, -36, 19, -16, 41, -23, 15, -9, 18, 20, 32, -35, -11, 28, 19, 2, 0, -14, -26, 44, -16, 21, -32, 13, 36, 9, 0, 31, 6, -12, 0, -25, 36, -43, -37, -32, 10, 42, 7, -1, 34, 7, 6, 21, -44, -35, 39, 28, -35, 20, 8, 13, 10, 25, 3, 5, 36, 23, 13, -11, 1, -26, 9, 19, -5, 42, 32, -31, -32, -22, 38, 37, -18, 8, 23, -14, 39, -19, 37, -27, -17, 9, 30, -10, -21, 24, 41, -5, 18, -30, 7, -29, -35, -22, 6, 16, -22, -20, 12, 39, 42, -32, -20, 7, -25, 28, 8, -32, -3, 16, -39, -14, -32, -14, 0, -16, 37, -11, -21, 0, -45, 41, -1, 4, 41, -3, 23, -39, -5, -8, 15, -12, 37, 22, 38, -38, -14, -22, 29, -18, -35, -1, 42, 26, 4, 39, 0, 42, -30, 44, 9, -11, -22, 6, 14, 5, -7, 39, 9, -16, -10, 38, -12, -28, -21, 11, -32, 25, 6, 22, 36, -8, 16, 9, -23, 37, 25, 1, -35, -39, 39, 47, -18, 8, -16, 21, 0, 12, -25, -37, 21, 24, 8, -13, 18, 9, -29, 27, -5, -32, -32, 8, -40, 15, -32, -13, 38, 10, -36, -13, -17, 17, 30, -26, 9, -23, 17, 2, 45, -32, 1, -38, 12, 25, 16, -19, 35, 22, -31, -19, 27, 19, 44, -39, -38, 18, -5, 10, 42, -30, -28, 27, 32, -13, -27, -21, 12, 28, 1, -28, 19, 34, 24, -12, 1, 27, 16, 11, -29, 15, -1, -26, -30, -8, 8, 1, -43, 17, 21, -38, 37, -37, -25, -23, -19, 35, 0, 31, 38, -1, -7, 39, 9, -2, -33, 22, 35, 3, -40, 2, 8, -24, -28, 6, -5, -7, 42, -14, 9, 41, -33, 29, -46, -31, -9, 19, -44, 26, -29, 14, 16, -7, -19, -1, 8, 22, -4, -32, -35, 13, -33, -35, -11, -3, -27, -5, 36, 0, 6, 44, -23, -20, 18, -31, 1, 33, 16, 18, -17, 33, 5, -7, -1, 11, -14, 29, 8, 11, -19, 9, -35, 31, -11, -17, -26, -37, 44, -39, -9, -20, -34, 12, 35, -15, -3, 40, -34, -34, -29, 9, 0, 5, 41, 9, -28, -28, 3, -6, -14, 37, -10, 37, 10, 6, 33, 10, -8, 33, -10, -37, 22, -15, 36, 33, -3, 31, -30, -39, -11, 17, -29, 3, 11, -26, 11, 19, -2, -18, -37, -28, 28, -1, 12, -11, 9, -30, -43, -7, 3, -12, -9, 6, -25, 22, 28, 36, -38, 19, 36}
, {27, -20, -21, -13, 19, -37, 37, -19, 7, -2, 31, -27, -31, -20, 43, -20, -32, 25, 27, 20, -31, 6, 20, -21, -17, -24, 41, -1, 4, 3, -21, -32, -1, 1, 5, 43, 39, -11, 32, 0, -17, 0, 20, -2, -1, -10, -37, -39, -15, -26, 7, -28, 18, 27, 46, 13, -14, -33, 12, 19, 14, -19, 37, 38, -17, -16, -29, -18, 45, -17, -20, 19, 41, 38, 25, 3, -36, -11, 16, -4, -17, -18, 14, -17, 40, 16, -9, -22, 8, -6, -22, 27, 33, -21, 20, -28, -11, 24, 8, 47, -5, 19, -26, 6, 24, 9, 22, -35, 45, 38, -36, 3, 33, -1, -25, -2, -30, 35, 17, 8, -33, -24, 42, -35, 18, 12, -28, 33, -34, -5, 18, 8, -19, 3, 47, -27, -22, 46, -29, 0, -8, 14, -15, 31, 21, -33, 6, 31, -3, 8, 48, -15, 31, -11, -18, -22, 20, -5, -27, -7, -19, 21, 15, -28, 15, 37, 24, 8, -10, 27, -2, 8, 20, 46, 0, 43, -18, 12, 7, 27, 11, 0, 51, 7, -11, -17, 28, -6, 45, -13, -12, 32, 44, 30, -4, 35, -4, 18, -31, 1, 14, -2, -26, 13, -22, 27, 48, 17, -31, 27, 10, 16, -26, 8, 23, -3, 37, -4, -31, -16, 36, 42, 46, 38, 28, 41, -9, 32, 41, -5, 3, 22, -35, -13, -23, 3, -16, 21, 7, 23, -13, 32, 15, -23, -5, -7, -17, -7, -19, 38, 2, 27, 17, 15, 9, -19, 4, -28, 0, 22, -2, 42, -15, 19, 44, 37, 39, 46, -9, 29, -26, -21, 11, 18, -17, -6, 3, 47, -32, 11, -18, 30, -35, 14, 33, 19, 37, 29, 28, 37, 29, -11, -8, -35, -12, -6, 19, 33, -20, 6, 27, -9, 16, 29, -7, 38, 12, 44, 14, -34, -34, 26, 47, 41, 37, 9, 13, 17, -19, 39, 3, 35, 34, -4, 27, 42, 10, 11, 13, -39, 14, -17, -38, -38, 19, -14, -3, 23, -10, -23, -2, 17, -11, -31, -17, 43, 29, -28, 3, -11, 29, 43, 6, -5, -31, 37, 4, 33, 45, 28, -1, 18, -15, 8, -3, 38, 23, -18, -5, 25, 1, 47, -20, 2, -24, 28, 31, 32, 17, 10, 44, 0, -23, 30, 17, 45, 34, 37, 0, 35, -32, 48, -20, 10, -11, -36, -37, 1, 2, 37, -23, 15, 29, 30, 45, -4, -16, 22, 39, -11, -34, 39, 34, 27, 23, 0, -14, 38, 7, 28, 18, -29, -21, 14, 10, 43, -18, -18, 16, -7, -2, -25, 18, 18, 32, 45, 1, 36, 18, 4, 23, 1, 48, 33, -9, -14, 0, 10, -21, -27, -9, -19, 41, -8, 26, 50, -28, 15, 21, 11, 1, -28, -30, 28, 41, -1, -33, -16, 48, -19, 32, 33, 39, 6, 38, 15, -19, -34, 39, -33, 35, 29, -11, 22, 34, 6, -2, 40, -25, 9, 22, -22, 18, -20, -11, -34, -2, 7, -18, 27, -33, -8, -5, 12, 37, 11, 8, 10, -11, 30, 41, 18, 36, -5, -25, 0, 23, 18, -17, 30, 45, -24, -1, 39, -16, -9, 46, -18, 28, -15, -11, 3, 33, 10, 10, -5, -4, 10, 0, 38, 7, 19, 37, -19, 9, -3, 42, 38, -24, -1, 18, 31, 47, 30, 0, 12, -18, 47, -15, -14, 3, 45, 18, 35, -8, 1, -24, 40, -24, 25, 15, -4, 20, -17, 40, 40, -14, 17, 8, -7, -1, 47, -7, 43, 28, -15, 23, -1, 18, 1, -12, 0, -12, -13, -13, -30, -14, 21, -10, 32, -34, -19, 32, 24, -31, -37, -15, -23, -23, -9, -3, -25, 9, 37, -2, -14, -3, -26, 37, 18, 30, -30, -20, 37, 0, -35, 31, 40, -7, 42, -2, 39, 17, -20, 32, -20, 15, -26, 26, -27, -3, -26, 0, -30, 31, -20, -18, 4, 39, 45, 24, 19, -1, 25, 26, -37, -5, 46, 43, -15, -13, 35, -16, 20, -5, 40, -1, 37, -33, -20, -27, -17, 8, -34, 28, -15, -15, 7, -10, -30, 33, 37, 12, -30, -21, 18, 0, 28, -2, -33, 27, -11, 14, -36, 41, 17, -25, 11, -36, -1, -25, 23, -5, 4, 37, -21, 26, -3, 6, 25, -10, -23, -33, -16, -35, 28, 48, 6, 31, 12, 9, -32, -2, 4, -29, -5, 47, -2, 1, -35, 2, 8, 24, 10, 19, -25, -19, 2, 27, -29, -18, 39, -9, -16, 29, 34, -5, 1, 19, -24, 30, -24, -28, 16, 30, -27, 12, -37, 3, 29, -35, -24, -26, 2, 16, 42, -28, -23, 49, -12, 34, -23, -10, 16, 46, 32, -13, 40, -7, 23, 23, -23, -7, -26, 1, 23, -8, -26, 17, -25, -12, -21, 15, -28, 41, -23, 24, 23, 47, -31, -12, 4, -31, -9, 43, 28, 12, 35, -3, 22, 2, -28, 26, -22, 2, -34, -29, 7, 39, -37, -34, -4, -19, -9, -23, 3, 32, -14, 25, 34, -7, -7}
, {9, 10, 38, 35, -6, -36, 2, 23, 36, 3, 36, 4, 4, 9, 37, 23, -3, 37, 35, 5, -25, 20, 3, 34, 29, 36, 14, 11, 18, 14, 37, 0, -11, -3, 9, -35, 5, -26, -27, 3, -33, 40, -21, 33, -20, -35, 13, 36, -16, -20, 2, -9, -29, -14, -35, -39, 40, 19, -13, -41, -24, -24, -38, -8, 36, -16, -23, 29, 10, -10, -41, 8, 36, 33, 23, 19, -19, 24, -3, -14, 0, -43, 22, -22, 20, -34, -7, 27, -13, 42, -37, 36, 19, 4, -1, -39, -32, 19, 37, 7, -12, -32, 20, -1, 6, 0, -17, 21, 44, -21, -39, 28, 2, -8, 5, -38, 3, 1, 10, 17, -18, 16, 33, 28, -28, 8, 26, 15, -37, 14, -28, 0, 27, 7, -26, -27, -4, 19, -14, -3, -10, -28, -16, -13, -32, 5, -8, -31, 7, 20, -13, -23, -20, -9, 32, -18, -29, -9, 41, -26, -5, 33, -13, 31, 33, 17, 40, -34, 0, 37, 16, -8, -22, 19, 23, -26, 10, 18, 5, 5, 6, -8, 36, -18, 29, 34, -33, 18, 23, -21, -15, -4, 5, 0, -21, 39, 7, -15, -8, -25, 14, -10, 41, 34, 34, -11, 13, -15, 7, -26, -14, -28, -24, -9, 42, -31, -34, 13, 16, 6, -18, -11, -2, 26, 22, -7, -41, -32, -19, -41, -42, -1, -37, -29, -33, -6, -18, -26, 23, -21, 11, -30, 17, -26, -26, 28, -24, -28, 10, 38, -16, -23, -39, 5, -1, -31, 1, 2, 38, 11, -6, 33, -28, -33, 16, 33, 45, -3, -35, -13, -6, 9, 20, 11, 26, -5, -36, 45, -12, -15, -2, 0, -25, 39, 9, 0, 0, -16, 5, -24, -38, -41, 11, -29, 10, 5, -38, -38, -28, 3, 28, -25, -6, -21, -12, -17, -27, -16, 37, 30, 42, -24, 31, 10, -22, 42, 12, 23, -18, -17, -1, -10, -38, -38, -23, -32, 33, 32, 37, -18, -6, -25, -5, -34, -14, 26, 26, -26, 5, -34, -30, -37, -25, -2, 24, 3, -33, 34, 17, -38, -6, 21, -13, 24, 28, -16, -9, 21, 14, -35, -9, 38, 14, -32, -32, -32, 7, 29, -4, 5, 25, -38, -2, 16, -31, 27, -35, -24, 31, 23, -20, -8, -11, 17, 10, -4, 42, -13, -1, 7, -6, -31, -36, 21, -37, -1, -3, -33, -32, 19, 24, 22, 34, 9, 4, 41, 21, -34, 15, 14, 14, -32, -18, 29, 36, -1, -22, 13, -32, 25, 40, 50, 9, -17, -23, -6, -36, 35, 24, 14, -27, 30, 18, 38, 7, -34, 5, 1, 37, 9, 24, 40, -2, 22, 26, -31, 9, 5, 4, -5, -33, 1, 36, -27, -3, 24, 30, -40, -21, -14, -16, -20, 7, -35, 21, -20, 26, -3, 35, 29, -8, -37, -22, -20, -8, 0, 16, -19, 12, -8, -23, 0, -30, -33, 18, 30, -5, -37, 30, 29, -20, 26, 8, 26, 36, 25, 3, 0, 0, 40, -20, -8, -38, -23, -43, 41, 5, 5, 8, 24, -32, 38, 24, -1, -21, 16, 27, 8, 24, -39, 7, 16, 34, -1, -11, 8, 11, 6, -11, 26, -31, 26, -1, -9, 27, 41, -8, 21, 3, 27, -6, 42, 11, 31, 17, -25, -5, 5, -20, -33, -14, 20, 30, 41, -9, 11, -39, -36, -1, -15, 43, 39, 32, 36, -6, 22, -33, -41, 5, -23, 20, 19, 46, -42, 7, 15, -19, 0, -33, -38, -30, -5, -21, 18, 11, -29, -20, 21, 0, -27, 22, -12, -20, -12, 11, 9, -32, 29, 49, -13, 28, -20, 22, 39, -15, -8, 0, -13, 37, 7, 31, -26, -11, 15, 23, 37, -17, -3, 16, 31, 28, 5, 42, -3, 32, 40, -15, -30, 1, -6, 9, 41, -26, 25, -25, 27, -14, 27, 5, 22, 18, 18, -11, 0, 43, 38, -8, -13, 0, 8, 28, 37, 28, 17, -30, -5, -16, 31, 36, -21, 1, -18, 7, 0, -12, 20, 36, -25, 44, 24, -35, 31, 6, 18, -38, 37, 2, 23, 4, -29, 37, 41, 12, 11, 19, 28, -41, -14, 23, 31, 42, -30, 38, -33, -19, -6, 18, -16, -8, -29, 28, 34, -14, 2, 18, -40, 39, 21, 33, -20, 7, -21, 9, 30, 27, 46, 6, 13, 8, -2, -4, 38, 39, 32, -39, 14, -33, 4, -4, 30, -24, -17, 35, -18, -31, -6, 18, -25, -2, -26, -6, 28, -19, -34, 28, 0, -4, 16, 32, 44, -14, 9, -33, 6, -10, 17, 17, 8, -41, 47, 29, 21, -23, 6, -31, 27, 4, -8, 30, 22, 27, 5, -32, -22, -4, 4, -14, -28, 19, -24, -12, -15, 42, 15, -8, 19, 26, -25, 41, 20, -34, 15, -33, 12, 26, -33, 17, 32, 22, 34, -3, 32, -6, 1, -4, 22, 21, 29, 23, 20, 14, 34, 45, -32, -2, 36, 2, 1, 1, -8, -21, 20, 12, 5, 41, 27, -33, 20, 11, 18, -36, 21}
, {31, 15, 37, 21, 29, -30, -4, 11, 43, 31, 38, -19, 39, 0, 33, -39, -31, 3, 34, 16, 21, -24, -27, 0, 0, 6, -6, -36, -32, 5, -24, 20, -16, -22, -42, 17, 21, -25, 14, -42, -43, 32, -22, 39, -7, -7, -16, 30, -31, -10, 23, -31, 8, 32, 30, -13, -12, 40, 17, 31, -34, -18, -28, -4, -33, -2, 29, 20, -7, 29, 10, -14, -27, 26, 11, 34, -2, 17, 23, -23, 5, 10, 14, 39, -28, 34, 25, 40, 21, 21, -21, -9, -20, -30, -10, -13, -1, 31, 15, -15, -24, 26, -18, -22, 31, 17, 3, 28, -8, -11, -30, -35, 11, 20, -11, 26, 36, -31, -10, 17, 20, 28, -13, 30, 21, 42, 1, -7, -16, -2, 34, 36, 31, -20, 20, -18, -33, 12, -40, 3, 6, -23, -29, 38, 16, -32, 0, 36, 30, -30, -29, 24, 1, -6, 42, 32, 3, -4, 16, 10, 39, -29, 22, -11, 0, 36, 11, 37, 20, 42, -17, -5, 32, 18, 9, -4, 12, 4, 38, 3, -5, -16, -33, 40, 25, 18, 1, -37, -19, 2, -26, 23, -31, 27, 1, -24, 10, 37, 5, 4, 32, 25, 1, 14, -3, 7, -27, -20, -42, -42, -2, -33, 39, 30, 29, -13, -16, 17, 35, 13, -14, 41, -16, -6, 27, -33, -25, 38, -40, -37, 41, -24, -38, 23, 40, -37, 30, 0, -9, 25, 34, -36, 10, 27, -1, -10, -27, 19, -24, 3, -16, -10, 18, 26, 26, -7, 35, -31, 27, -32, -39, -34, -6, 43, 38, 20, -16, 0, -6, 4, 33, -26, -40, -20, 39, 0, 38, 7, -18, 48, 32, -22, 30, 20, 29, 4, 10, -15, 37, 27, -25, 17, 18, -18, -3, 26, 30, -13, 22, 40, -8, -29, -21, 11, 18, -15, -29, -13, 3, 10, 38, -36, -23, -27, -35, 19, 41, 39, 3, -40, 20, -10, 7, 5, 7, 6, 18, -8, -7, -41, 21, -6, -6, 26, -31, -14, -3, 30, 29, 20, -11, -9, 5, 39, 22, -3, -36, 41, 36, 18, 35, 32, 0, -13, -6, 21, -18, -29, -23, 17, -41, -18, 33, -13, -8, -38, -38, 18, 20, -35, -39, 11, 18, 34, -42, 33, -21, 42, 8, -9, -22, 41, 43, 41, 2, 38, -37, -7, 23, -30, -25, 25, 31, -41, 11, -37, 36, -39, -32, -2, -5, -14, -34, -24, 39, -23, -7, -7, 21, 7, -26, 0, -2, 2, 21, 1, 3, -19, 23, 42, 35, 16, 39, 12, 8, -16, -8, 17, 40, 34, -7, 34, -14, -7, 21, -10, -7, 25, -9, -27, 39, 16, -11, 34, 9, -21, -14, 27, -30, -39, 14, -10, -26, -3, -44, -41, -24, -13, -38, -24, 33, -5, 36, 39, 32, -5, 16, 25, -37, -24, -42, 38, 20, 14, -17, 17, 8, -22, -1, 2, -28, -6, 3, 36, 40, -11, -21, -16, 38, 42, 39, 10, 2, -4, 37, 40, -40, 33, -33, 2, -2, 10, 21, 38, 20, 25, -45, -10, 27, 6, 43, -27, 29, -33, -6, -4, 36, 23, -34, 12, 26, -3, 19, 47, -33, -8, 34, 17, 8, 4, -25, -16, -30, 38, 11, 45, -18, -18, 36, 24, -33, 13, 37, -14, 30, 35, -33, -38, 39, 31, -10, 36, 24, 1, 34, 21, -13, -20, -39, -2, -16, -20, 1, -33, 36, -35, 9, 44, 24, 7, 6, 14, -29, 9, -17, -12, 30, 14, 31, 3, 8, 16, -34, 4, 13, -31, 24, -6, -18, -7, -25, 21, 18, -4, -6, 37, 25, -20, -17, 0, -32, -3, -13, -27, 43, 0, -4, 33, 19, 35, 32, 24, -24, -37, 21, -37, 19, -18, 15, 12, -2, 43, 1, 37, -31, -20, 12, 34, 23, -10, 13, 41, 1, -27, 13, -24, -8, -4, 3, 38, 28, 25, -30, -31, -33, 37, 37, -10, -28, -17, 38, 31, 9, -17, 39, 2, 28, 45, 22, 18, -5, -32, -12, 7, -7, -24, 10, 45, 1, 36, 21, 7, -31, -5, 25, -15, 26, 2, 36, -39, 6, 19, -11, -40, 6, -12, -28, -34, 17, 1, 12, -2, -22, 18, -17, -32, -31, -6, -22, -34, 39, -23, -31, -20, -23, 20, -29, -25, -28, -1, -38, 3, -6, -29, -27, 14, 6, -42, 5, 21, 2, 7, 10, 6, -41, 18, 28, -17, -41, 26, 40, 11, 34, 16, 0, -1, -39, 38, 24, 29, -20, -20, -42, 31, -28, -6, -26, 36, -13, -6, 28, -20, -38, 6, 30, -29, 9, -18, -35, -12, 20, -23, -14, 40, 26, -1, -42, 41, -21, 47, -1, 45, 29, 18, 25, 29, 41, -17, 5, -16, 16, -18, 25, -16, 13, 25, 27, -7, -18, -1, 2, -40, 0, -31, 35, 42, 38, -25, 5, -36, 39, -15, -8, -17, -3, 38, -18, 28, 44, 20, 5, 19, 32, -24, -12, -7, -30, -38, 14, -29, -9, 16, -14, 46, -5, 25, -26, 35, -19, 35, -10, 8}
, {21, -24, 45, 17, -35, 12, 38, -7, 38, -26, 17, 28, 16, 19, 9, -16, -2, -3, 32, 0, -40, 38, 34, -3, 1, 0, 29, 23, 30, 30, 9, -25, 29, -29, 2, -32, -20, -25, -4, -37, 43, -5, -27, 35, -14, -16, -26, -17, -3, 5, 10, 12, 27, 38, 9, 3, 11, 23, 28, 34, -2, 4, 28, -20, -19, 48, 40, 22, 27, -36, 20, -44, 0, 15, -20, 21, 43, -12, 2, -6, 4, -41, 5, -12, 37, 22, -41, 30, 16, 1, -18, 18, 37, 44, -10, -5, 4, 40, 10, -18, -4, -12, -2, 25, 13, -21, -32, 47, 4, 14, 18, -30, -7, 10, 22, -4, -38, -5, 2, 29, -37, 12, 22, -22, -25, 3, -26, -18, -31, -31, -15, 23, 11, 30, -39, 16, 31, -11, -17, 13, -5, -10, -16, 25, -20, 34, 5, -23, -26, -25, -31, -6, 34, 29, 14, -1, 44, -32, -13, 44, 28, 41, 30, 37, 13, 42, 39, 15, -18, 22, -17, 45, -5, -3, 0, -18, 37, 8, 0, -7, -28, -28, -14, -34, 21, -26, -39, -6, -1, 31, -6, -25, 4, -2, -18, -31, -2, -5, 43, -32, 9, 6, 16, -20, -8, -38, -34, 39, 22, -31, -32, 8, -37, 44, 39, -1, -17, -38, -41, -9, -7, 0, -18, -38, 8, -12, 28, 19, 32, 3, -17, 35, 40, -17, 9, 46, 4, -30, 37, -18, -15, 9, -28, 25, -20, 43, -23, 0, -9, 7, 42, -27, 30, -43, 34, -38, -1, -25, 6, -4, 31, 39, 48, 32, -19, -36, 21, -15, 40, 46, 5, 7, -13, 6, 49, -13, -16, 43, 37, -8, 44, -3, -20, 18, 2, 41, -26, -7, 22, 7, -41, 1, 26, 23, -8, -27, 16, 34, -32, -6, 2, -6, -13, 37, -36, -23, -35, -28, 4, -23, 0, 46, 31, -32, 12, -7, 33, -15, -35, -38, -27, 36, -30, -3, 26, 16, 37, -28, -15, 0, 19, -12, -31, 15, -8, -26, 0, -5, 32, -31, -2, -16, 0, 13, 25, 40, -14, 13, 8, -34, 1, 34, -38, 30, 32, 7, 14, -17, -37, -37, 14, -24, -31, 36, 36, -27, -24, 24, 32, -10, -13, 21, -38, -17, 16, -1, -2, 23, -12, -11, -39, -42, 12, -2, 5, -32, 37, -9, -20, -2, 50, 18, 15, -39, 8, 33, 7, 13, -40, -1, 41, 16, -39, 11, -25, -4, 30, 18, 40, -11, 32, 6, 31, 33, 8, -17, -31, 15, 39, 1, 18, -19, 42, 40, 15, -17, -18, -32, 42, 31, 14, 8, -23, 25, 27, 38, -8, 5, -41, -12, -15, 3, -2, -27, -4, -39, -27, -18, 7, 1, 0, -32, -3, -3, 2, -10, 15, 18, 16, -41, -44, -4, 14, 5, -36, -5, 43, 3, -18, 15, 0, 7, 36, -33, 15, -6, 42, -24, -40, 32, 6, 32, -13, -8, -20, -28, -11, -8, -26, 29, 3, -35, 22, -8, 37, 28, -17, -29, 30, -33, 13, 8, 38, 29, -13, 26, 12, -19, -12, -34, -23, -1, 41, -30, -10, 46, -38, -2, 0, -4, -10, 41, -15, 42, -3, 13, 25, 37, -17, 29, -21, -32, -5, 29, 14, -27, -12, -31, 13, 13, -21, 4, -32, -7, 6, 20, -26, 42, -16, 25, 32, 26, -19, 32, 0, -40, 9, 24, -22, -28, 25, -29, -7, -17, 12, -3, -42, 4, 25, -1, -29, 23, 21, -30, -8, -32, -40, 6, -8, 8, -8, 12, 21, 20, 24, -21, 15, 21, -4, 7, -11, -30, -8, 1, -19, 0, -36, 45, 18, 10, 9, 26, 25, -33, -32, -35, -25, -33, -32, 12, -29, 9, 10, 41, 44, 11, 24, -27, 33, -22, 37, 16, 44, -4, -8, 21, 41, 35, 19, 10, 37, 39, -30, 0, -1, 11, 27, 13, -29, -23, 49, -4, -21, -27, -5, 0, 20, 18, 1, -33, 23, 46, -35, 34, 47, -32, -20, 27, 4, 34, -3, -18, 35, 6, -32, -34, -6, -32, 7, 26, -4, 46, 45, 19, -28, 23, -31, 29, 11, -20, 21, 3, 39, 5, 22, 36, 25, -26, -34, -23, -9, -16, 38, -27, 8, -29, -11, -29, 17, -37, 38, 29, -13, -11, -32, 1, -12, -24, 27, 39, 8, -30, -25, -4, -19, 10, -21, -36, -21, -29, 25, 41, 21, -5, 3, -6, 37, 18, 9, -2, -9, 18, -10, 23, -17, 42, 40, 10, 4, -21, 5, 1, 3, -9, -5, -4, 8, -21, 30, 21, -34, 38, -34, 25, 28, 33, 21, 31, -18, 13, 16, 6, 23, -1, -7, -25, -23, 49, 18, -28, -14, -11, 32, -6, -7, 49, 10, -32, 43, 34, 20, 27, 14, -3, 24, 25, 19, 15, -12, 12, -31, 12, -1, 18, 31, -33, -12, -6, 44, -33, 9, 28, 30, 47, -11, -2, 35, -20, 12, -22, -32, 6, -3, 25, 12, -17, 39, -20, 16, -11, -2, 35, -8, -7, 46, 24, -5, -17, 15, -23, -24, 24}
, {30, -29, 40, 28, 41, 7, -44, -25, 22, 20, -16, -38, -18, 7, -25, 15, 4, -34, 41, -24, 15, -27, -4, -8, -6, -45, -40, 23, -23, 1, 0, 3, -4, -31, -38, 21, -46, -4, -40, 35, 30, 6, 42, 27, -8, -2, 2, -20, 0, -12, -27, 0, -35, -40, 34, -15, -6, 29, 22, -11, 4, 44, -1, 3, -12, -43, -45, 8, 29, -14, 19, 42, 11, 0, -35, 29, -28, -43, 6, -31, 10, -30, -1, -20, 13, -3, -6, 22, -41, -33, -4, -45, -46, 2, -26, -31, 27, -6, 21, -1, 26, -39, -30, -33, -7, 29, 13, -38, -35, 9, -8, 21, 41, -10, -33, 1, 23, -44, 34, 17, 18, 1, 5, -26, 20, 46, -6, 11, -30, -27, -28, -10, 22, -34, -33, -17, -39, 22, 13, 6, -2, 11, -45, 0, -2, 6, -32, 26, 41, -41, 40, 28, -36, 24, 30, -39, -11, -14, -36, -26, 42, 37, 17, 2, -13, -37, -32, -26, -8, -38, -17, -11, 14, -1, -32, 11, 12, -40, 16, 36, 27, 34, 24, 7, 26, 2, 38, 37, 30, -8, 22, 22, 30, 18, 15, -25, 19, 32, 45, 23, 3, 45, -22, 45, 27, 25, 18, 18, 17, 5, 31, 0, 4, 37, -16, 12, 7, -18, -43, -28, 29, -15, 5, 5, 11, 15, 30, 3, -19, 30, -27, -25, 19, -10, -38, 4, 25, 17, -24, 2, 20, 12, -11, 40, 10, 12, 11, -10, -33, 6, -22, -11, 16, 25, -11, 0, 21, -27, 20, 0, 32, -13, 10, 0, -12, -32, 10, 0, -36, -7, 10, 3, -3, -22, 19, -15, -42, -29, 13, 19, -41, -4, -18, -10, 12, 14, 31, -6, -14, -4, 40, -10, -40, 12, -22, -15, 41, -31, 5, -9, 11, 0, 35, 43, -35, -3, 9, -41, 36, -32, -36, -31, -18, 26, 40, -18, 24, 31, 40, 13, 26, 0, 15, -17, 18, 4, 1, 33, 16, 14, -11, -16, 7, 18, 26, -12, 8, 11, 10, -8, 32, -2, 19, -19, -30, 21, 11, -9, -21, -9, 7, 33, 31, 14, 41, 15, -33, -6, 0, -1, -11, 20, -8, -23, 19, 0, 17, 16, 16, -34, -11, 36, -6, 37, 30, -33, -10, -14, -8, 0, 23, 29, 42, -27, -21, -17, 23, 6, 34, -39, -47, 31, -37, 31, 45, 0, 34, 1, 32, -29, 35, 20, 4, -35, -4, 16, -21, 46, 20, -35, 4, 14, -10, -4, 4, -34, 15, 37, -10, -4, 34, -20, -32, -35, -13, 11, -24, -44, 18, -44, 31, 3, 30, -19, 39, -18, -28, -21, -5, -37, -6, 3, 30, 2, 38, 14, -5, 49, -24, 15, -31, 8, -9, -7, 17, -32, 6, 13, 38, -1, 7, 44, -7, 17, -19, 6, 39, 0, -5, -38, 16, 23, 43, 48, 24, 0, 10, -20, -11, -38, 8, 13, 23, 3, 18, 16, 22, 7, 36, -23, -13, 20, 28, -32, -10, 8, -37, 6, -34, -11, -16, 19, 7, -3, -20, 8, -28, 33, 5, 13, 2, -38, 45, 33, 12, -15, 16, -31, 17, 42, -28, -2, 6, 10, 4, 0, 4, -34, -32, -26, 24, 34, -36, 8, 12, -37, 11, 35, 1, -11, 37, -11, 38, -16, -40, -14, -38, 0, -15, -5, -36, -32, 29, 31, 20, -1, -1, -9, -2, -20, -16, 36, -2, 21, 35, 48, 14, 45, 15, -4, -8, 19, -4, 7, 29, 4, -32, 27, -34, 32, 11, -25, -15, 28, 39, -18, 2, 5, -10, 42, 17, -4, 8, -26, -27, -16, 12, 2, -41, 41, 30, -16, 0, -15, -11, -24, 43, 24, -20, -40, 38, -38, -4, -29, 41, -22, -6, 15, 8, -28, 6, -6, -33, 14, -30, 0, -4, 30, -39, -10, 19, 40, -21, -4, -6, 28, 14, -13, -2, -17, 30, 2, -8, -12, 15, 3, 8, 7, 13, 20, 15, -36, 31, -17, -37, 45, 19, 2, 28, 13, -40, 5, -23, 9, 2, -1, 8, -9, 14, -22, 35, -34, 35, 34, -41, 15, -39, -39, -25, -13, 31, -11, 3, 1, 5, -6, -24, -36, -34, -9, 18, -31, 22, -3, -33, -1, 23, -15, -8, 33, 28, 32, 26, 37, 10, -27, 3, -21, 44, -9, -11, -4, -23, 36, -1, -12, -19, 0, -9, 30, 25, 8, -26, 44, 20, -16, 37, 38, 3, 3, 39, -38, -10, -38, 6, -2, -37, -8, -43, 18, -17, 15, -15, -16, -36, -10, 12, -39, -6, 38, -15, 6, 9, -23, -5, -6, -40, 34, 7, 38, -36, -37, 24, -24, -34, -15, -21, -21, -41, 19, 8, -11, -36, -19, 32, 27, 1, -11, -22, -28, -32, -9, 21, 0, 40, -4, -14, -39, 18, 35, -39, -28, -17, -16, 32, 30, 38, 27, -35, 26, -1, 11, 34, 30, -12, -9, -48, 19, -37, 4, 36, 15, 6, -18, -36, 7, 33, 5, 35, -36, -9, 18, -1, -19, -3, 20, -43, -8, 41, 11, 15, 28}
, {31, 19, -27, 11, 8, 17, 6, 38, -21, 5, 9, -38, 35, -8, -32, -33, -30, 0, 3, 22, -21, -40, -27, 31, 38, -18, -1, -16, -14, 33, 4, -3, 21, -16, 0, 18, -41, 5, -38, -5, -7, 13, -25, -9, 7, -11, -6, 26, -14, 23, -33, 12, -9, 41, -17, -15, 3, 13, 13, -32, 23, -31, -8, 26, 37, -26, -40, -30, 0, -13, -15, -15, 2, 11, -29, -7, 37, -16, 7, -23, 40, -12, -28, -9, 18, -26, -7, -32, 7, -32, -8, -31, -7, 24, -4, 7, 32, 16, 27, 20, 7, 0, -38, 21, 20, 24, -31, -15, 12, 12, 20, 0, 30, 26, 35, 14, 18, -17, -37, -25, 15, -6, 30, 2, 0, -19, -36, 32, -37, 12, 22, -42, -38, 34, 7, 16, -15, 21, -24, -8, 9, -36, 1, 5, 22, 30, -14, -5, 12, 27, -27, 33, 37, -25, 34, -22, 26, -25, 21, -4, -21, 4, 34, -26, -12, 27, -18, -30, 40, 0, -42, 6, 32, -23, -14, -23, 22, 13, -23, 31, -9, 25, -19, 24, 39, 0, -23, -32, -12, -33, 14, 36, 20, 19, 23, 28, -31, -20, 7, -44, -15, -12, -4, 28, -20, 12, 6, -16, 42, -36, -10, 3, 1, 8, -32, -25, 27, -16, -11, 0, 11, -39, -37, 4, 9, -10, 30, -36, 17, 1, 26, -21, 25, 38, 25, -15, -15, 3, 18, -12, 38, 22, 26, -18, 33, 32, -1, 12, -37, 13, -8, 8, 12, 18, -20, -2, 4, 18, -44, 31, -42, -19, -27, 11, 25, 0, 15, -18, -38, -41, -41, 19, 40, 28, 24, -3, -22, -34, 18, 14, -21, 40, -18, 32, -33, 11, -35, 30, 15, -23, 12, 30, -22, -3, -15, -2, -18, -2, -8, -33, 37, 6, -2, 32, -35, -2, -17, 0, 23, -15, -38, -7, -36, -7, -40, -23, -22, 36, -14, 39, -30, -41, -18, 34, 36, -22, -42, 3, 7, 30, -17, 6, 2, 9, -2, 4, 41, -39, 21, 15, -43, 33, 7, -7, -41, 16, 29, -40, -25, 14, -6, 3, -35, 28, 11, 20, -28, -33, -14, 19, -1, -18, 29, 29, -25, -40, 3, 38, -9, 32, 35, -41, 1, -36, 26, -19, 0, 37, 28, 10, -21, 4, -13, 24, -14, -36, 19, -34, 6, 5, 36, -42, -25, 26, -33, -18, -39, -3, -12, -40, -26, 12, 2, 11, 3, -12, 11, 20, 2, -41, 6, -5, -15, 16, -10, 32, -20, 30, -3, -12, -33, 8, -21, -32, -42, -16, 10, 13, 13, -37, -4, -27, 5, -10, -4, 13, -44, 36, -30, 11, -26, 21, -42, 25, -23, 2, 13, 21, -27, 15, -7, -29, -31, 30, -5, 42, -2, 32, -24, 25, -7, 40, -41, 37, 22, -45, 9, 5, -17, -21, -39, 32, -2, 8, -2, 36, 0, 37, -11, 10, -23, -43, 6, -6, 24, -32, -28, -16, 29, -21, 25, -29, -25, -40, -9, -17, -3, 25, -12, 16, 13, 11, -2, -25, 23, 24, 1, -21, -3, -40, -39, 4, -9, -13, 7, 7, 32, -29, -3, -11, 5, -16, 29, -3, -1, -5, -30, 8, 24, -14, 33, -25, -22, -18, 33, -10, -28, -5, -2, -31, -6, 5, -32, -24, 3, -24, 5, 34, -28, 10, 27, -29, -41, 0, -13, -14, 29, 2, -9, -8, 12, -38, 6, 20, 7, -33, 12, 4, 38, 19, 12, 0, -18, -30, -32, -45, 0, 4, -9, -28, 26, -30, -26, -13, -44, 24, 8, -4, -13, -41, 31, -24, -7, 35, 0, 21, -20, 14, -40, -5, -38, 37, -3, -30, 3, 23, -37, 15, -34, -19, 0, -22, -5, 19, 2, 6, -27, 29, -3, -7, -32, 13, -38, 0, -23, -38, -21, -21, -1, 19, 37, 35, 8, -5, 8, -8, 37, -34, -11, -30, -10, -29, 20, 16, -3, 15, 34, 18, -22, -24, 11, 3, -25, 30, -12, -23, 22, 39, 30, -29, -10, 5, 1, -24, 10, -40, -40, -19, 0, -19, 22, -40, 11, -38, -36, 10, -40, -2, 18, -8, -41, 41, -25, 32, -17, 17, -8, -10, -24, -20, -10, -11, 6, -11, -21, 4, 39, -39, -9, -42, -14, 14, -24, -30, 38, 15, 24, 16, -37, -12, 9, -37, -33, -24, 4, 40, 38, -42, 37, -4, -34, -21, 5, 37, 34, -11, 13, 6, -34, -42, -42, -3, -3, 8, 21, 33, 18, -2, 37, -10, -30, 15, -4, -2, 14, 10, -8, 15, -8, 6, -23, -10, -22, -39, -26, 35, 40, -12, 28, 3, 1, 28, 36, 4, -36, -33, 1, -7, 33, -9, -20, -41, 1, -17, -11, -23, -7, -38, -17, -24, -21, 29, 13, -32, 0, -26, 6, -3, 41, 22, 4, 19, 15, -40, 22, -26, 34, 24, 0, -33, -10, -39, -23, -1, -19, 36, -19, 42, -42, 36, -32, -17, -1, -22, 20, 5, 11, 9, 17, 23, 31, 16, 13, 42, 37, 40, 9, -18, 19, 36, -33, 8}
, {2, 47, 9, 6, 23, -1, 40, 32, 36, 18, 44, 38, 35, 39, -37, 10, -14, -35, 7, -26, 31, -35, -2, -9, -34, 4, -38, -22, 37, -19, -7, -39, -1, -12, -21, -39, 31, -36, 12, 4, -19, -39, -30, 36, 28, -27, -27, 20, -24, -33, -19, -40, 16, -34, 39, -28, 36, -34, 40, 35, 32, -29, 33, 27, 29, -27, 13, -39, -39, -25, 34, 41, 32, 30, 13, 14, 30, -43, -26, -20, 14, 13, 36, -15, 38, -27, -6, -15, 35, 21, -14, 1, -33, 22, 37, 21, -30, -35, 2, -26, 26, 19, 41, 5, 37, 30, -17, 30, 13, 40, 25, 21, -7, -15, -42, 19, -23, 5, 30, 25, 33, -31, -9, -39, 12, 22, -26, 28, -36, -6, -33, 19, -14, -27, -32, -37, 28, -1, 5, -42, -19, -27, 30, 25, 28, -39, -11, 7, 3, -8, -11, 39, 0, 16, 29, -3, 44, 35, -16, 11, -31, -13, 19, 36, 17, -34, -7, -11, 27, 41, 28, -31, 0, 41, -2, 35, 5, 5, 14, 16, 24, -40, -13, 12, 18, -6, 36, 18, 14, -40, 18, 2, -38, 27, -33, 1, -17, -36, -13, -40, 4, -35, 38, 22, 10, 21, -39, -9, 12, 37, 29, 37, 41, -5, -37, -31, 7, -40, 36, 40, 28, -31, -33, 38, 37, 0, 21, -2, -39, -25, 39, -26, -2, 22, 23, 25, 42, 4, 15, 29, 35, 32, -11, 11, 24, -20, -5, 19, -4, -1, 4, -28, -14, -14, 14, -32, 33, 4, 29, -13, 9, -14, -4, -14, -3, 38, 26, -33, -8, -23, 2, 2, 16, -32, -18, 19, -36, 31, 21, -18, 44, 39, -4, -38, 30, -8, 9, -21, 40, 19, -14, -41, -26, -4, -18, -37, 17, 20, -5, -35, 5, 27, 18, -34, -29, -15, 40, 11, -8, 13, 4, -20, 28, 2, -30, -35, -17, -6, -15, -31, -29, 32, 37, -39, 23, 39, -16, -39, -36, -33, 19, -24, 19, -5, 13, 12, 15, -11, 29, -5, 36, 7, 39, 44, -26, 2, -19, -41, 36, -22, 12, -29, 40, -37, -19, 25, -34, -22, -40, -29, 38, -37, 0, 24, 31, 26, -14, -29, 40, -4, -33, 32, 37, -27, -13, -27, -8, 15, 14, 35, 4, -38, -10, -16, 14, 31, 26, -3, -16, 27, -6, -14, -18, -6, 23, 33, -39, 6, 9, -32, -19, 39, 25, 36, 11, -32, 0, -36, -18, -8, 7, -24, -23, 26, -7, -29, 48, 38, 18, 11, 38, -34, 24, 22, 38, 38, -10, 23, 30, 35, 6, 13, 42, -37, 16, -29, -17, 27, -1, -19, 40, 16, 32, -2, 34, 20, -14, -22, -17, 41, 35, -35, 25, 12, -16, -42, 29, -18, -24, -21, -10, -10, -10, 35, -37, -42, 33, -23, -1, -8, -8, -28, -17, -17, -25, 11, 8, -7, -2, 0, -3, 24, -14, 31, -34, -28, 19, -4, 16, -40, 11, 12, -42, -16, -13, -13, 17, -17, 19, 21, 34, 0, 42, 3, 34, 31, 7, -10, -22, 35, 12, 12, 17, 11, -32, 5, -13, 22, 8, 41, 24, -5, -15, 14, -28, -30, 34, -9, 42, 11, 38, 30, 10, 13, -21, 31, 40, 46, 30, -33, -34, -22, -34, 39, 24, 0, -39, -19, 8, 36, -12, 8, 32, -13, 25, 34, 11, 0, 14, -23, 44, -11, 0, -15, 12, 44, 5, 40, 17, -29, 10, 15, -18, 17, -14, -35, -19, -8, -9, -5, -28, 4, 36, -43, -6, 27, 14, -3, 32, -16, -35, -42, 6, -34, 5, -28, -40, 41, -1, 20, -24, -16, 42, -4, 32, -26, 43, 23, 2, 18, -17, -9, 11, -15, 2, 14, 19, 24, 21, 38, -4, 35, 9, 12, -22, 16, -28, 23, -26, 11, 2, 13, 21, -31, -32, 12, 36, 14, -20, -35, -15, 21, 4, -25, 0, -23, 17, -35, 31, 16, -28, 10, -7, -25, -16, -7, 29, -31, -6, -11, -34, -27, 19, 7, -19, 6, 40, 1, 40, -31, 13, -32, 35, -13, 9, -17, 30, -7, -27, -13, -3, 42, -35, -15, -36, -17, 29, -21, 22, 5, 7, 15, 20, -32, 35, 1, -20, -22, 36, -20, 32, -41, 37, 14, -43, -29, -6, 18, -23, -5, 41, -6, 35, -24, 23, 36, 36, 6, -15, 24, -10, 32, -22, 24, 23, -41, 6, 37, -5, -12, 31, -29, 2, 0, -35, 29, -34, -17, 25, 30, 7, -3, 42, 37, -16, -25, -26, -16, -39, 37, -24, -5, -3, 30, -34, -36, -41, 32, 27, -28, 31, -19, -31, 28, 1, -38, -24, 13, 50, -36, 45, -26, 32, 17, 12, -28, 0, 2, 0, 29, 20, -27, -37, -24, -19, 28, -8, 41, 7, 7, 26, -29, 6, 44, 23, 37, 38, -26, 29, -29, 1, 2, 4, 39, -35, -35, 6, 30, -36, 37, -20, -32, 15, 3, -28, -35, -2, 2, -4, -10, 43, -13, -17, 24, -9, 31, -15, 34, -35, 33, -33, 29}
, {23, 1, -23, 9, 31, 26, -19, -27, -39, 2, -10, 22, 34, 30, -26, -33, 22, -29, 8, 33, 36, 29, -42, -38, -12, -37, -31, -40, -10, 20, 7, 18, 7, 43, -18, 35, 10, -12, -17, 33, 27, 3, 23, 20, 16, -7, 30, 36, 32, -38, 31, -19, -4, 8, 23, 37, 11, -35, -28, -30, -41, -5, 11, 24, 34, -32, 14, -39, 33, 12, 22, 9, 11, 4, 19, 39, -36, 19, -10, 0, -16, 38, -40, 8, -31, 29, -13, -26, -12, 22, 3, 9, 8, 12, -35, 36, 35, 21, -15, 34, 37, -18, 0, 35, -4, -33, -2, -14, 40, 35, -5, -32, -40, 20, 3, 2, 37, -18, 3, -29, -27, -20, -13, -8, 22, -41, 13, -9, -1, 12, -17, -1, -17, 9, 28, -19, 0, 0, -38, -2, 22, 27, 0, -14, -32, -6, -10, 39, 23, 22, -16, -34, 30, 35, -4, -34, 0, -18, 33, 20, -39, -3, -5, 25, 38, 26, -25, -29, -38, 32, -6, -29, -43, -24, 33, -39, 34, -22, 20, -14, -42, -42, -29, -4, -35, -20, -27, -28, 17, -22, -42, 7, 25, -33, -34, -20, 1, 0, -33, -28, -5, 6, 11, -26, 24, -20, -2, 21, 6, 23, -3, -17, -12, -21, 28, -28, 3, -42, 8, 41, -29, 29, -30, -31, 27, 37, 29, -17, -7, -16, -24, 27, -40, -20, -35, -15, -9, -2, 27, 23, -14, -6, -18, -10, 6, -5, 35, -14, -32, -6, -34, 16, -26, -9, 27, -2, -39, -42, -20, 24, 9, -4, 34, 7, -5, -35, -40, -42, -5, -16, -12, 36, -31, 12, 12, 14, -12, -8, -36, -34, 21, -22, 1, -25, -26, -31, -41, 3, -16, 5, 6, 37, -19, -18, 24, -34, 11, 34, 32, -24, -36, -9, -34, 12, 27, 35, 0, 43, 4, 2, -35, -14, -19, 26, 0, 32, 5, 28, 31, 12, 39, -8, -42, 37, 25, 27, 23, -21, -21, 30, -7, -26, -29, 32, -31, 37, 37, -27, 20, -31, 36, -27, -25, -30, 11, -34, 31, 36, 13, 35, -26, -3, -17, 0, 3, -39, -32, -22, -20, 0, 9, -25, 30, 22, 6, 21, -42, 35, -42, -13, 12, 25, 5, 19, 20, -26, 38, 25, 1, 17, -26, 16, -9, -21, 18, 34, -25, -7, -22, -22, -12, 33, 34, -19, -23, -25, 39, 28, -4, -22, -1, 3, -16, 16, 0, -32, 14, -41, 23, 20, -11, -30, 18, -40, 15, -26, 14, -35, 37, -30, -10, 28, -13, -31, -23, 27, 34, -24, -5, -20, -17, -19, -7, 31, -15, -8, 41, 22, 11, -35, -41, 1, -38, 1, -16, 19, -29, 14, -38, -4, 26, 17, -27, -29, -40, -17, -10, 32, 21, 29, -24, 5, 0, 27, 24, -21, 6, -25, -35, -19, 32, -29, 19, 35, -14, -24, -19, -31, -42, -9, 28, 40, 39, 34, 35, 18, 8, -4, -22, -21, 23, 24, -27, -38, 15, -12, -13, -38, -13, -8, -33, 40, 6, 25, -33, -34, 33, -19, 3, -24, -22, 7, -37, -12, -32, -37, -1, -41, 30, 29, 12, 31, -23, 19, -35, 28, 6, -17, 39, 34, 10, -35, 3, 12, -7, -38, 27, 5, -4, -23, -25, 23, -43, -22, -19, 0, 30, 3, 31, 33, -9, -27, 8, -23, 30, -4, -30, 29, -35, -28, 12, 19, 2, -8, 29, -34, -2, 19, 26, -41, -38, 15, -7, 18, -25, 2, -17, 36, 30, -39, 9, 8, 14, -2, 0, 25, -22, 39, -41, -37, -35, -9, 35, -34, 35, 38, 29, -26, -27, -41, 41, 22, 34, 5, 14, 38, -17, 7, -7, 33, -28, -17, 36, -27, -32, 35, -18, 4, -15, 0, -39, -7, -27, 40, -21, 36, -12, -24, -25, -13, -20, -25, -30, -26, 28, 5, -4, -39, -36, -24, 24, 26, 22, 28, -9, 26, 11, -1, 20, 7, -3, 41, 19, -10, -40, 36, 15, -5, 10, 20, 30, -42, -39, -18, -40, 30, -15, 40, 27, -11, -3, 34, 10, -41, 8, 32, -30, 23, 36, 23, 9, -6, 25, 35, -22, -32, 38, 37, -21, -24, 19, -2, -34, 35, 8, 29, 41, -17, -33, -35, -38, 22, 27, -11, 27, 37, -5, -17, -38, -8, 26, 23, -30, 35, -30, 0, 27, -1, -19, 18, 25, -7, -11, -33, 2, 6, 42, -41, 34, 36, -41, 17, 11, -37, 17, 25, -9, 8, 15, 42, -7, 8, -40, 23, -30, 28, 34, 31, -1, -35, 14, -6, 25, 19, -10, -37, 31, -33, -7, 1, 38, -19, -27, 16, -11, 18, -23, 25, 12, -36, -26, 22, -35, -40, 28, -10, -19, 31, 17, -16, -16, 21, -14, 21, 6, -26, 31, 14, 8, -33, -16, 33, 10, -5, 31, 35, 16, -16, 29, -9, 3, 29, -1, 1, -27, 25, 37, 20, 36, -3, 0, -24, -27, 30, -3, -35, 0, 30, -4, -29, 35, 15, 18, 0, -36, 15, -22, 19, 34, 30, -1, 10}
, {-36, 36, -8, -27, -22, -21, 30, -17, -13, -27, -2, -15, -29, -29, 22, -36, 34, 12, -25, -21, 0, 1, 5, -36, -20, -22, -38, -26, -18, 8, -38, -18, 29, 30, -44, -18, 13, -26, -11, 3, -38, 3, 16, 0, -21, 39, -28, 32, 20, -30, 10, 17, -50, 4, 27, 39, 11, 34, 22, 29, 33, -6, 28, -32, -16, -5, -6, -37, -14, -18, 12, -30, 44, 44, 0, 32, 40, 11, -5, -4, 19, -6, -2, 7, -24, -40, -40, -6, 28, 8, 0, -34, 31, -21, -22, -34, 0, -1, 9, -37, -28, -8, 27, 5, -50, 18, -28, 17, -11, -31, -6, 21, -39, -33, -33, 8, 27, -11, 16, -35, -34, 12, -5, -11, 39, -12, 31, -5, 14, -21, -31, 3, -2, 37, -33, -13, 3, 6, 5, 0, 27, 0, -44, -40, -22, -18, 35, 26, 14, -13, 23, 12, 14, 4, -42, 27, 13, -28, 32, 4, 12, 7, -23, 7, 36, 6, -7, 24, 26, -26, 31, -25, 19, -15, 32, -7, -46, 32, 30, 18, 15, -45, 6, -6, 5, -42, 36, -26, 25, -4, -2, -27, -39, -25, -15, 9, -40, 10, -27, -16, -32, 46, 14, -18, 41, 28, 1, 19, 33, -10, -32, -23, -22, -16, 11, -31, -13, 32, -24, -16, 23, -1, -22, 33, -43, 12, 30, -26, 16, -9, 10, -26, 42, -30, 8, -49, 0, -38, -22, -34, 0, 18, 9, -36, 33, -26, -12, 10, -49, 15, 26, 8, 12, -12, 25, 23, 36, -16, -16, 8, 8, -32, 13, -3, -16, 3, -6, -9, -19, -36, 0, -26, 8, -15, -13, -41, 34, -3, -11, -7, 15, 41, 40, 25, 3, 47, 29, -9, -7, -10, -6, 30, -7, -21, 18, 0, -38, 26, -10, 3, 24, -25, -11, 25, -34, -31, -32, -44, -48, 1, 19, 24, -49, 30, -20, 29, -14, -10, -1, 15, -29, 6, -28, 41, -7, 24, 9, -14, 23, 3, 24, 4, -33, 10, -42, 11, -38, 31, -53, -11, -25, 38, -12, 9, -27, -33, 42, -22, -35, 21, 4, -28, -13, -4, -18, 34, 42, 39, 40, 13, 33, 31, 8, -26, 16, -50, -29, -24, 20, 7, 27, 13, 7, 46, 39, 13, -29, -42, 1, -34, -23, 38, -26, -9, 30, -26, -13, 34, 42, -39, -37, -1, 1, 10, 41, 1, -3, 45, 24, 46, 36, -39, 18, 16, 2, 16, -15, 44, 41, 29, 29, 13, -25, 19, -40, -28, 6, -21, -11, 18, 0, 8, -38, 8, -20, 4, -13, 18, -39, -46, -29, 24, 11, 42, -2, -3, -15, -10, 6, 14, -5, -17, -3, 2, -11, 17, 19, 25, 22, -10, -7, -18, -14, 34, -7, 0, -29, -1, 27, -4, -8, 43, 0, 28, -21, 14, 11, 9, 29, -47, 31, 14, -34, 48, 48, 44, -5, -22, -2, 20, 31, -12, -32, 35, -16, 23, 34, 47, 24, -8, 19, -36, -8, -11, 30, -28, 0, 38, 33, -30, 27, 0, 39, -29, -11, 20, -15, -30, 23, 23, -7, 17, 13, -11, -25, 2, -19, 8, 3, 1, -39, 25, -18, 26, -39, 14, 16, -26, 2, -28, 9, 26, -34, -20, -2, -41, -2, -12, -22, 38, 18, 37, 42, -2, 26, 18, -3, -30, -29, 12, -46, 38, -18, -46, 0, -14, -23, 27, -25, 23, 15, -21, -34, -17, 9, -27, -7, 1, 39, 2, 27, -38, -24, -26, 12, -21, -33, 1, 4, 42, -28, 11, -36, 2, -27, 10, 35, -3, 34, -1, -39, 22, -21, -28, 28, -11, 30, -36, -20, 14, -17, -7, -3, -5, -40, -29, -8, 20, -4, 20, -12, -48, 16, 34, -8, 30, 20, -34, 0, -8, -14, 18, -28, -35, -12, 23, 26, 0, 8, -29, -30, -50, -9, -35, -26, -24, -32, -2, -48, 13, 1, -9, -35, -11, -39, -43, 2, 12, -1, -12, -41, 36, -28, -18, 13, 29, -28, -27, 18, 8, 12, -36, 28, -5, -22, 19, -19, 37, 27, 26, 20, 29, 10, -34, 24, -49, 15, -43, -15, 14, -26, 9, -1, -35, -35, 3, -24, 10, -28, 32, 30, -17, -3, 22, -17, -36, 0, -33, 0, -30, -14, -1, 0, -35, 32, 7, -9, 0, 36, 18, 32, 19, -11, 0, 12, -34, -2, -17, -22, -39, 26, -37, -39, 16, 17, -8, 38, -32, 27, -48, 11, -33, 33, -18, -34, -3, -3, -23, -18, -4, 26, -38, 39, -22, 5, -31, 7, 2, -22, -36, -41, -16, 16, 4, -41, -27, -39, -44, 5, -44, -31, -15, 3, -23, -23, 23, -37, 24, -4, 21, -49, 13, -45, -13, -35, -49, -8, -40, 22, 1, 30, -2, -42, -37, 4, 25, -16, 21, 36, 28, 16, -13, -16, 29, -47, -40, -9, -23, 20, -16, 1, -11, 6, -10, 8, -24, -26, -55, -43, 11, 7, -16, -23, -23, -15, -18, 12, 19, 20, -26, -9, 10, -37, 7, 35, -7, -6, 11, -14, -12, 0, -7}
, {20, -5, 30, 21, 11, 14, 37, 3, 7, -33, -21, 5, 35, 3, 0, -38, -8, -7, 40, -21, 7, -26, 27, 42, 16, 38, -13, 17, 6, -20, -12, 37, 0, -1, -22, 24, -42, 0, -25, -5, -2, -19, -19, -7, -8, 26, -21, -7, -24, 6, -35, -19, 12, 20, 2, 28, 15, -35, -4, -35, -15, 6, 22, -39, 4, -33, 38, 0, -35, -36, -16, 18, 40, 21, -36, -4, -3, -12, -27, -17, -28, -21, 0, 13, 39, -9, -17, -33, 8, 19, -38, 17, -10, -8, 27, 20, 21, 41, 27, -26, 3, -4, 31, 4, -37, 26, 28, -1, -4, 5, 5, -24, -11, 32, -42, 9, -12, 24, -6, -2, -5, 35, 2, 2, 25, -14, -26, 9, -21, -32, -26, 40, -28, -30, -31, -37, -30, 32, -31, -26, 9, 31, -9, -4, 44, -24, -19, -34, -39, 4, 27, 24, -29, -35, 16, -18, 16, 44, 0, -28, 35, -20, 28, 33, 8, 36, -7, 1, -39, 24, 42, -5, -27, 12, 27, 21, -26, 5, 4, -1, -23, 5, 11, 0, -17, 44, 6, -31, 34, -36, -26, 4, -12, -32, -33, 39, -37, -24, 15, 1, -18, -13, 27, -17, -8, -17, 18, -24, 24, 13, 3, -32, -34, 34, 22, -38, 41, -38, -1, 31, 41, -33, -2, -9, 35, -39, -29, -4, -3, -5, 5, -26, 24, -3, -38, 38, 28, -37, 17, 16, 12, 11, -34, -15, 8, -2, 3, 17, 20, -26, 45, -29, 41, 15, 3, -26, -17, 41, 0, -19, 46, -1, 36, -14, 10, -21, -5, -39, -38, 40, -2, -9, -39, -2, -5, -18, 43, -22, 36, 14, -32, 36, -42, -38, -18, 31, 0, -9, -15, 30, -41, -29, 3, 3, 31, -2, 21, -8, 6, -8, -25, -32, 33, -1, -6, 0, -9, 31, 6, -25, -18, -44, 21, 38, 31, 1, -1, -14, -30, -40, -12, 5, 18, 0, -6, -12, 13, 37, -14, 35, 17, -31, -10, -7, 35, 6, 15, 7, 6, -24, -2, -5, 14, 12, 22, -26, 0, 30, 36, -7, -35, 16, 18, -28, -12, 36, -3, 1, 12, -38, 4, -39, -8, 24, -29, 11, 1, 7, 10, -31, 25, -13, 42, -16, -34, -5, -16, -25, 37, 9, -31, -13, 44, -12, 36, -38, 3, 18, -1, -38, -22, -9, 6, -4, -13, 30, -32, 3, 9, 11, -36, 36, -3, -6, 29, -34, 8, -33, 36, 3, 33, 41, -37, -41, -28, 35, -37, 27, -7, -9, 5, 22, -12, -37, 16, 22, -24, 30, 10, -35, 31, 13, -2, -40, -25, -21, 13, -33, -38, 11, -3, 19, 30, -34, 19, 4, -40, -30, -27, 20, -15, 37, -36, 5, 17, 5, 23, -34, 4, -32, 41, -5, 1, 18, -16, 18, -20, -37, 28, -1, -15, -30, -34, 10, 2, -36, 2, -7, -1, -35, 20, 25, -26, 7, -27, -23, 3, 18, 31, 2, 12, 13, 24, -28, -32, 25, -33, 44, -22, -21, -4, 4, -23, -35, -13, 40, 20, 11, -35, 20, -5, -30, -25, 41, 5, -27, -15, 35, -36, 3, 40, 39, -27, 30, 10, -19, 31, 22, -21, 38, 7, -20, 9, 19, 43, -8, -12, -9, -14, 4, 8, 44, 19, 33, -28, 9, 0, 30, -11, 19, 24, -24, -7, -16, 7, -39, 34, -11, 1, 0, 12, -15, -6, 1, -6, -32, 34, -6, -12, -15, -32, -40, -26, -23, -38, 16, -35, -29, -33, -15, -32, -9, 40, -10, 6, 7, -37, 12, 14, -40, -4, -2, 1, -28, 1, 37, 14, 3, 14, -27, -11, -29, -2, -42, 0, 25, -40, -19, -29, 40, -29, 35, -4, 40, 44, -10, -11, -21, 27, 6, 10, -9, -38, 22, -26, 40, 8, -37, 9, -10, 0, -16, -25, 12, 21, -29, 23, 0, -7, -25, 40, -25, -22, 0, -20, 16, -13, 11, 40, 12, 12, -30, 8, -28, -13, 13, 19, 14, -8, -19, 36, -38, 30, -15, -21, -13, 35, -24, 43, -23, 29, -16, 31, 12, -15, -12, -23, 33, -38, 6, -33, 12, 3, -10, -7, -4, -31, -31, -25, 41, 6, -35, -34, 29, 24, -40, 21, -23, -11, 14, -2, 26, 28, 44, 12, -6, 15, 13, -13, -7, 19, 35, 41, -35, -2, 44, -37, -29, -2, -42, 34, 18, -28, -40, -11, -21, -3, -33, 25, -11, 16, 9, 37, -29, -36, -35, 22, -7, 4, -12, 34, -38, 35, 19, 30, -8, -14, -37, -30, -23, -43, -1, -38, 24, -31, 15, 30, 44, -21, -25, -7, -22, -2, -31, 15, -43, -34, -1, -17, -31, 22, -1, 21, -27, -6, 0, 14, 24, 6, -37, -17, -22, -17, -12, 14, 12, 41, -24, -12, 19, -14, 38, -12, -14, 21, 38, 42, 12, 42, 33, -36, 1, -5, -6, -31, -6, -2, 44, -18, 13, 0, -3, -28, 32, -6, -23, 33, -31, 24, 44, -31, -3, 27, 43, -28, -16, -6, -34, -13, 33, 26, -24}
, {38, 26, -22, -2, -30, 32, -31, 31, -8, -39, -42, 8, -33, 6, 29, -31, -14, 39, -4, -29, -5, -13, -34, -10, -11, 33, -13, -3, -2, 9, -42, -37, -35, 6, -14, 31, 5, 1, -31, -28, 32, -12, 29, -37, -36, 35, -5, 38, -1, -15, -36, 14, -2, 7, 33, -39, 15, 15, -21, -7, -35, -29, 43, -6, -1, 12, -10, 44, 2, 38, -17, 44, 42, 46, 4, -8, 17, 30, -31, 36, 14, -5, 22, -17, 38, 30, -11, 21, 37, -1, -26, 10, -1, -3, 35, 42, -35, -17, -19, 20, 34, -8, -1, 6, -1, -27, -23, 18, 16, 13, -1, -18, -30, 19, 12, -37, -38, -3, -5, 0, -21, -10, -27, -36, 37, 0, 5, -35, 35, 2, -23, -41, -31, -18, 19, -32, 29, 0, -26, 23, -13, -4, 5, 13, 19, -21, 2, 17, -1, -2, 10, 23, -38, 0, 22, -26, -40, -25, 23, 10, -2, 10, -12, -2, -39, 27, 19, 2, -18, 9, -41, 36, 13, -34, -21, 8, -34, -35, -44, -2, 20, -29, -10, 16, -13, -3, -31, -31, 25, 0, 25, -5, -41, -34, 9, 3, -41, 17, -37, 34, 46, 31, -11, -16, 13, -9, 6, 5, -37, -24, 40, -37, 19, -40, -40, -38, 36, 5, 23, -42, -11, -33, -29, -14, 23, 16, -20, 34, 46, 5, 7, 24, 31, -10, -9, 14, 30, -26, 27, -8, 30, -24, 8, 33, 0, -22, -28, -20, -6, -6, -17, -25, 43, -28, 22, -6, 4, 39, -2, 2, -37, -28, -42, 37, 34, -12, -26, -35, 21, 3, 26, 24, -40, 29, -43, -39, 31, -32, -30, 24, -13, -26, -1, 43, 23, -41, -13, 17, 13, -30, 0, 20, 35, 39, -18, -17, 34, 13, 11, 27, -5, 39, -14, 23, 26, -10, 26, 21, -27, -14, 36, -31, 14, -15, -15, -7, -25, 42, -11, 7, 37, 0, 22, 20, -23, -2, -6, 32, -41, 38, 34, 39, 0, 26, -32, 38, -3, 2, -33, -10, -30, 3, 38, 7, -12, 9, -27, 29, 4, -24, -26, 25, -14, 0, -26, -5, -26, -20, -38, 44, 10, 12, -35, 4, 36, 22, -6, 21, 8, 21, 39, 36, -15, 44, -25, 27, 41, -10, -1, 19, 48, -12, 0, 49, 47, 1, -6, -14, 44, -8, 6, 15, 13, 8, 33, -17, 22, 44, -7, -32, -6, 6, 16, -34, -9, -20, -36, -20, 9, 42, -22, 24, 1, -16, -6, -23, 35, 1, -35, 29, 21, -15, 3, -16, -21, -36, -19, 0, 39, -32, -41, -4, 30, -26, 13, 30, 12, -7, 22, 38, 30, 9, -39, 35, 0, 29, 46, -34, 38, 32, -37, 29, -8, 44, 0, 18, 17, -29, -9, -7, 37, -25, 30, -9, -9, 38, 41, 46, -29, 11, -22, 42, -27, -1, -21, 22, 10, 24, 9, 11, 4, -6, -23, 20, 0, -6, -7, -30, -20, -24, 0, -2, 0, -12, 15, 34, 9, 42, -24, 0, 29, 38, 44, 6, -24, 42, -10, -32, -2, -23, 7, 0, -2, -41, -13, -37, 16, 16, 2, 44, -41, -40, -31, 28, -27, 33, -20, -30, -37, -2, -45, 22, -16, 1, 16, -16, -33, 16, -27, -13, 28, -29, -22, 34, -37, 7, 37, -31, 10, 35, -33, -42, 14, 7, -31, 0, 17, -4, 31, -16, 10, -1, 5, 13, 24, -18, 27, 0, 11, -6, 5, 3, 6, 16, 10, 45, 17, 22, 28, -31, 2, -24, 7, -36, -19, -37, -41, -11, -35, 38, 12, -8, -36, 40, 37, 21, -2, 39, -18, 12, -23, 18, -39, 37, 12, -43, -33, -39, 26, 14, -13, 4, -19, 41, -14, 40, 26, -33, -13, 33, 24, 29, 8, 43, 22, -16, -26, -42, -13, 0, 5, 17, -38, -43, -36, 21, -11, 32, 0, -7, -34, -40, 0, 15, -43, 23, -9, -12, 5, 5, 17, -28, -10, 37, 11, 35, -26, -18, 16, -9, -1, 34, -9, 14, -39, 37, -35, 27, -16, 35, -4, 39, -24, 15, 3, 27, 19, 38, 8, 14, 14, 4, -25, -19, -22, 13, 27, 28, -4, 38, 9, -6, -32, 0, -31, -27, 16, -32, -28, -14, 26, -33, -28, -14, 14, 18, 30, -18, -28, -30, 17, 23, -1, 21, 9, 38, -34, -15, -25, -32, 30, -13, 37, -26, -3, 5, 4, 35, 9, -33, -11, 26, -27, -42, -17, 17, -22, -30, -40, -23, -23, -36, -28, -28, 19, -39, -22, -43, -10, -7, 8, 3, -19, -13, -41, -13, 24, -20, 43, -25, -26, 2, 33, -30, 28, 12, -30, -41, -8, -45, 6, 13, 35, 26, 0, -15, -20, -14, 15, -32, 12, 37, -31, -30, 12, 36, -29, 0, -28, 8, -8, -40, -1, -39, -29, -16, 12, 35, -22, 31, -31, -37, -37, 38, -46, 40, 25, 36, 6, 4, 36, 12, 14, -44, -31, 20, 8, 10, -40, 7, 17, -23, 7, -33, -35, 4, -20, 41, 25, 37, 32, 4}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 16
#define FC_UNITS 6
#define ACTIVATION_RELU

typedef number_t dense_51_output_type[FC_UNITS];

static inline void dense_51(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 16
#define FC_UNITS 6


const int16_t dense_51_bias[FC_UNITS] = {-2, 3, 3, 8, -10, -2}
;

const int16_t dense_51_kernel[FC_UNITS][INPUT_SAMPLES] = {{-223, 27, 93, -234, 45, 241, -244, -165, 4, -179, -113, 66, 144, -128, 38, 192}
, {230, 46, 188, -139, -56, 30, -201, 73, -166, 248, -252, -70, -45, 151, -129, 152}
, {-100, -110, 92, -188, 110, 138, 175, 171, 93, 45, -205, -95, 229, -203, -92, -134}
, {-253, 62, 161, -113, -100, 152, 255, 58, 171, 123, -19, 263, 184, -5, 225, -137}
, {-52, -145, 217, -80, 13, 250, -150, -135, -63, -48, 249, -245, 63, -75, 236, 96}
, {-239, 8, 248, -23, 97, -261, -203, 157, -114, -44, 2, 75, -11, -77, -124, 98}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 6
#define MODEL_INPUT_SAMPLES 16000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_51_output_type dense_51_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d_141.c" // InputLayer is excluded
#include "conv1d_154.c"
#include "weights/conv1d_154.c" // InputLayer is excluded
#include "max_pooling1d_142.c" // InputLayer is excluded
#include "conv1d_155.c"
#include "weights/conv1d_155.c" // InputLayer is excluded
#include "max_pooling1d_143.c" // InputLayer is excluded
#include "conv1d_156.c"
#include "weights/conv1d_156.c" // InputLayer is excluded
#include "max_pooling1d_144.c" // InputLayer is excluded
#include "conv1d_157.c"
#include "weights/conv1d_157.c" // InputLayer is excluded
#include "conv1d_158.c"
#include "weights/conv1d_158.c" // InputLayer is excluded
#include "conv1d_159.c"
#include "weights/conv1d_159.c" // InputLayer is excluded
#include "max_pooling1d_145.c" // InputLayer is excluded
#include "flatten_20.c" // InputLayer is excluded
#include "dense_50.c"
#include "weights/dense_50.c" // InputLayer is excluded
#include "dense_51.c"
#include "weights/dense_51.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_51_output_type dense_51_output) {

  // Output array allocation
  static union {
    max_pooling1d_141_output_type max_pooling1d_141_output;
    max_pooling1d_142_output_type max_pooling1d_142_output;
    max_pooling1d_143_output_type max_pooling1d_143_output;
    max_pooling1d_144_output_type max_pooling1d_144_output;
    conv1d_158_output_type conv1d_158_output;
    max_pooling1d_145_output_type max_pooling1d_145_output;
    flatten_20_output_type flatten_20_output;
  } activations1;

  static union {
    conv1d_154_output_type conv1d_154_output;
    conv1d_155_output_type conv1d_155_output;
    conv1d_156_output_type conv1d_156_output;
    conv1d_157_output_type conv1d_157_output;
    conv1d_159_output_type conv1d_159_output;
    dense_50_output_type dense_50_output;
  } activations2;


  //static union {
//
//    static input_21_output_type input_21_output;
//
//    static max_pooling1d_141_output_type max_pooling1d_141_output;
//
//    static conv1d_154_output_type conv1d_154_output;
//
//    static max_pooling1d_142_output_type max_pooling1d_142_output;
//
//    static conv1d_155_output_type conv1d_155_output;
//
//    static max_pooling1d_143_output_type max_pooling1d_143_output;
//
//    static conv1d_156_output_type conv1d_156_output;
//
//    static max_pooling1d_144_output_type max_pooling1d_144_output;
//
//    static conv1d_157_output_type conv1d_157_output;
//
//    static conv1d_158_output_type conv1d_158_output;
//
//    static conv1d_159_output_type conv1d_159_output;
//
//    static max_pooling1d_145_output_type max_pooling1d_145_output;
//
//    static flatten_20_output_type flatten_20_output;
//
//    static dense_50_output_type dense_50_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_141(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_141_output
  );
 // InputLayer is excluded 
  conv1d_154(
    
    activations1.max_pooling1d_141_output,
    conv1d_154_kernel,
    conv1d_154_bias,
    activations2.conv1d_154_output
  );
 // InputLayer is excluded 
  max_pooling1d_142(
    
    activations2.conv1d_154_output,
    activations1.max_pooling1d_142_output
  );
 // InputLayer is excluded 
  conv1d_155(
    
    activations1.max_pooling1d_142_output,
    conv1d_155_kernel,
    conv1d_155_bias,
    activations2.conv1d_155_output
  );
 // InputLayer is excluded 
  max_pooling1d_143(
    
    activations2.conv1d_155_output,
    activations1.max_pooling1d_143_output
  );
 // InputLayer is excluded 
  conv1d_156(
    
    activations1.max_pooling1d_143_output,
    conv1d_156_kernel,
    conv1d_156_bias,
    activations2.conv1d_156_output
  );
 // InputLayer is excluded 
  max_pooling1d_144(
    
    activations2.conv1d_156_output,
    activations1.max_pooling1d_144_output
  );
 // InputLayer is excluded 
  conv1d_157(
    
    activations1.max_pooling1d_144_output,
    conv1d_157_kernel,
    conv1d_157_bias,
    activations2.conv1d_157_output
  );
 // InputLayer is excluded 
  conv1d_158(
    
    activations2.conv1d_157_output,
    conv1d_158_kernel,
    conv1d_158_bias,
    activations1.conv1d_158_output
  );
 // InputLayer is excluded 
  conv1d_159(
    
    activations1.conv1d_158_output,
    conv1d_159_kernel,
    conv1d_159_bias,
    activations2.conv1d_159_output
  );
 // InputLayer is excluded 
  max_pooling1d_145(
    
    activations2.conv1d_159_output,
    activations1.max_pooling1d_145_output
  );
 // InputLayer is excluded 
  flatten_20(
    
    activations1.max_pooling1d_145_output,
    activations1.flatten_20_output
  );
 // InputLayer is excluded 
  dense_50(
    
    activations1.flatten_20_output,
    dense_50_kernel,
    dense_50_bias,
    activations2.dense_50_output
  );
 // InputLayer is excluded 
  dense_51(
    
    activations2.dense_50_output,
    dense_51_kernel,
    dense_51_bias, // Last layer uses output passed as model parameter
    dense_51_output
  );

}
