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

typedef number_t max_pooling1d_25_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_25(
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

typedef number_t conv1d_30_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_30(
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


const int16_t conv1d_30_bias[CONV_FILTERS] = {-3, -1}
;

const int16_t conv1d_30_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-280, -37, -212}
}
, {{-222, -172, -95}
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

typedef number_t max_pooling1d_26_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_26(
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

typedef number_t conv1d_31_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_31(
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


const int16_t conv1d_31_bias[CONV_FILTERS] = {-5, -1, 2, -4}
;

const int16_t conv1d_31_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-278, -284, 111}
, {-180, -295, -97}
}
, {{75, 90, -198}
, {-139, -209, 74}
}
, {{-190, -212, -208}
, {106, -18, -215}
}
, {{17, 258, 223}
, {84, -278, 239}
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

typedef number_t max_pooling1d_27_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_27(
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

typedef number_t conv1d_32_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_32(
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


const int16_t conv1d_32_bias[CONV_FILTERS] = {-1, 7, -7, 2, 0, 0, 0, 1}
;

const int16_t conv1d_32_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{99, 119, -84}
, {-143, -120, 96}
, {93, 138, 110}
, {-107, -209, -37}
}
, {{-81, -117, -172}
, {-176, 134, 32}
, {-201, 72, -119}
, {50, -194, 107}
}
, {{61, -45, 30}
, {-122, 122, -128}
, {-13, -116, 106}
, {180, 14, 182}
}
, {{65, 141, -32}
, {3, -193, 24}
, {174, -138, -195}
, {108, 199, 160}
}
, {{180, 192, 131}
, {-177, 3, 165}
, {131, 121, -50}
, {-4, -87, -127}
}
, {{-38, 159, 14}
, {148, 59, -70}
, {38, -170, 130}
, {-204, -98, -91}
}
, {{165, 15, 0}
, {203, -145, 95}
, {-88, 24, -184}
, {-45, 67, 180}
}
, {{-133, 15, 59}
, {196, -49, -172}
, {141, 141, 55}
, {98, 16, -27}
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

typedef number_t max_pooling1d_28_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_28(
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

typedef number_t conv1d_33_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_33(
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


const int16_t conv1d_33_bias[CONV_FILTERS] = {5, 7, 8, -4, 3, 3, 8, -8, 7, -6, 0, 2, -7, 0, -3, 9}
;

const int16_t conv1d_33_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-134, 49, -103}
, {-8, -66, -29}
, {46, -61, -48}
, {0, -20, 58}
, {15, -108, 19}
, {-135, 91, 107}
, {-9, -46, 14}
, {39, 75, 86}
}
, {{87, 21, 124}
, {-38, -86, -1}
, {-21, -29, -52}
, {41, 88, -50}
, {-20, 109, -102}
, {48, -35, -109}
, {34, -46, 112}
, {42, -8, 69}
}
, {{120, -54, -3}
, {-91, -31, -30}
, {-9, 47, 124}
, {27, 101, 40}
, {119, 70, -18}
, {86, 32, 37}
, {-134, 112, -99}
, {-1, 122, -79}
}
, {{-86, -125, -73}
, {-51, 8, -145}
, {-10, 44, -2}
, {56, -88, 136}
, {18, -78, 42}
, {-115, 27, -27}
, {83, -14, -98}
, {107, -87, -114}
}
, {{-13, 12, 7}
, {55, -44, -51}
, {-126, 24, 83}
, {65, 146, 107}
, {-86, 65, -125}
, {75, 10, -12}
, {105, -143, 124}
, {89, 69, -50}
}
, {{135, 110, 9}
, {72, -30, 133}
, {-31, 117, 135}
, {56, 99, -31}
, {-79, -14, -106}
, {109, 136, 97}
, {-19, 21, -50}
, {-116, -108, -130}
}
, {{71, -143, -16}
, {29, -4, -32}
, {48, 81, -54}
, {127, -4, -37}
, {59, -78, 44}
, {12, 42, 63}
, {13, 1, 59}
, {-72, -29, -13}
}
, {{-120, -65, 88}
, {129, -137, -112}
, {108, 129, -103}
, {-16, 128, 66}
, {139, 53, 145}
, {118, -13, 97}
, {108, -87, 23}
, {-119, -104, -21}
}
, {{-137, 141, 77}
, {-53, -9, 110}
, {-56, -138, 77}
, {137, 31, 39}
, {73, -88, -27}
, {-112, -83, 47}
, {83, -68, -99}
, {-18, 106, -17}
}
, {{-5, -54, 137}
, {83, -117, -121}
, {-87, 116, 151}
, {121, 77, 79}
, {-29, 85, 66}
, {-66, -35, 82}
, {-101, 29, -62}
, {103, 47, -79}
}
, {{119, 118, 8}
, {93, 123, -20}
, {93, 99, -28}
, {-42, -97, -107}
, {95, -75, 67}
, {-107, 56, 5}
, {-20, -84, -112}
, {-44, -67, 116}
}
, {{-118, 126, -105}
, {75, 143, 134}
, {-32, -27, -6}
, {-52, 31, -126}
, {139, -43, 72}
, {91, 145, -108}
, {150, 12, 144}
, {145, -90, 4}
}
, {{86, 75, -58}
, {-20, -44, 95}
, {-94, 71, -18}
, {138, -113, -125}
, {-125, 81, -3}
, {-125, 72, 131}
, {-85, -10, 127}
, {-48, -132, 29}
}
, {{105, -41, -3}
, {127, -99, -77}
, {-85, 87, -106}
, {-6, -146, -12}
, {-141, -101, 28}
, {-49, -75, 135}
, {-105, -90, 89}
, {96, -113, 56}
}
, {{-80, 120, -49}
, {42, -124, 35}
, {76, -27, -53}
, {-116, 67, -40}
, {41, 72, 87}
, {104, -60, -136}
, {32, -42, 10}
, {-63, 63, 74}
}
, {{-4, 61, 61}
, {-121, 132, 116}
, {-112, -89, 90}
, {32, 76, 44}
, {2, -45, -115}
, {35, 124, -26}
, {-137, 33, -53}
, {-76, -129, -2}
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

typedef number_t conv1d_34_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_34(
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


const int16_t conv1d_34_bias[CONV_FILTERS] = {3, 5, -5, 0, 0, -4, 8, -5, 7, 0, -1, 3, 1, 9, 8, 1, 7, 3, -4, 0, 0, 6, 4, -1, 6, -2, 2, 5, 2, -2, -7, -8}
;

const int16_t conv1d_34_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-59, -95, 34}
, {-83, 34, 101}
, {-90, -28, -39}
, {-12, 52, -84}
, {-35, 73, 38}
, {37, -92, 27}
, {-36, -92, -31}
, {-85, -16, -68}
, {-9, 38, 4}
, {-21, -51, 106}
, {41, 48, 83}
, {75, 97, 103}
, {-23, -86, 33}
, {-26, 31, -91}
, {-92, -60, 58}
, {44, -87, 93}
}
, {{17, 33, -44}
, {94, -14, 81}
, {70, -80, -37}
, {46, -54, -61}
, {32, 35, 88}
, {-93, -90, -3}
, {-92, 96, 12}
, {4, 9, -50}
, {-95, 38, 32}
, {-69, -11, 99}
, {18, -39, -42}
, {27, 100, 32}
, {36, -42, -54}
, {-52, -46, 68}
, {47, 90, -40}
, {102, -97, 78}
}
, {{69, 85, -33}
, {-52, -62, 56}
, {32, -23, -53}
, {91, 84, -51}
, {-70, 48, -40}
, {69, -37, -78}
, {-94, 37, 70}
, {-33, -80, 84}
, {71, 52, 75}
, {-6, -63, -97}
, {-25, 83, -94}
, {76, 60, -11}
, {21, -12, 72}
, {101, 65, -38}
, {-89, 102, -77}
, {-52, 102, -62}
}
, {{-48, -100, 40}
, {76, -77, 65}
, {102, 17, -7}
, {-65, 16, -21}
, {66, -87, -65}
, {-68, -103, 8}
, {-69, -94, -78}
, {-100, 14, 35}
, {29, -1, -41}
, {16, 14, 78}
, {101, 47, 40}
, {29, 71, 76}
, {-63, 97, 76}
, {77, -61, -37}
, {-68, 43, 33}
, {-52, -83, -58}
}
, {{91, -63, 2}
, {47, -50, 83}
, {51, -88, 65}
, {98, 43, 8}
, {91, 21, -11}
, {-45, 7, -48}
, {-62, 84, 46}
, {-23, 32, -76}
, {-13, -39, -55}
, {59, 87, 73}
, {-73, 11, -20}
, {-77, -10, 99}
, {-29, -54, -20}
, {-39, 22, 98}
, {1, -75, 11}
, {-89, -60, 17}
}
, {{-54, -80, 38}
, {2, -87, 36}
, {69, -74, 0}
, {77, 39, -5}
, {-87, 19, 106}
, {-45, 33, 79}
, {36, -90, -85}
, {10, 75, 104}
, {-74, -62, -66}
, {86, 107, -89}
, {6, 60, -77}
, {-37, 68, 57}
, {84, 17, -84}
, {-52, 33, -4}
, {5, -73, -50}
, {-21, -73, -22}
}
, {{84, 26, 107}
, {-47, 32, 26}
, {-68, 32, -104}
, {-70, -44, 70}
, {103, 42, 43}
, {80, -78, 56}
, {19, -30, 61}
, {-106, -77, 53}
, {68, 48, 63}
, {-86, 17, 76}
, {-93, 60, -32}
, {-9, -84, -28}
, {66, -51, 99}
, {100, -13, 26}
, {37, 99, 78}
, {-87, -61, 66}
}
, {{59, -76, 97}
, {26, 46, -66}
, {-66, -2, 70}
, {-12, -6, -67}
, {44, 1, 66}
, {-16, -97, -12}
, {12, 78, -88}
, {11, 32, 35}
, {-42, -68, -19}
, {104, 20, 105}
, {30, 70, -3}
, {41, 70, 87}
, {-33, 51, -23}
, {41, -27, 9}
, {-23, -99, -94}
, {-104, -31, 35}
}
, {{-57, -5, 50}
, {-26, -81, 98}
, {23, 102, 50}
, {14, 46, 2}
, {-10, 81, -3}
, {75, -81, 91}
, {-69, -5, 40}
, {-32, -96, -74}
, {-35, 59, -99}
, {-73, -62, 42}
, {67, -52, 50}
, {52, 35, -37}
, {-28, 102, -77}
, {81, -36, 34}
, {-17, -71, -98}
, {98, -27, 29}
}
, {{75, 69, 59}
, {69, 98, -61}
, {78, -86, 56}
, {62, 72, 79}
, {31, 79, 0}
, {-19, -9, -82}
, {32, 20, 64}
, {-23, 0, -98}
, {-41, -6, -101}
, {-10, 0, 32}
, {9, -4, 57}
, {35, 21, 7}
, {86, -90, 50}
, {13, 25, 21}
, {26, -22, -1}
, {-14, -91, 11}
}
, {{88, 70, 79}
, {39, 3, 101}
, {-96, 96, -88}
, {20, 31, 13}
, {-5, 7, 7}
, {-68, 87, -30}
, {76, -82, 9}
, {87, -37, 50}
, {-77, 2, -37}
, {-68, 3, 1}
, {44, 85, -16}
, {-13, -66, -85}
, {83, 5, 15}
, {89, 56, -87}
, {-49, 62, -12}
, {-83, -31, -93}
}
, {{-83, -51, 62}
, {-41, 79, 7}
, {38, -69, 69}
, {31, -19, -50}
, {71, 72, 77}
, {-82, 86, 28}
, {-53, -46, 86}
, {-86, -4, -77}
, {26, 55, 9}
, {-9, 89, -17}
, {-83, 64, -54}
, {-92, -7, -60}
, {-15, -71, 23}
, {71, 77, -16}
, {15, 80, -35}
, {-25, -87, 62}
}
, {{40, -77, 0}
, {-61, -15, -2}
, {-40, -78, -58}
, {-86, 9, -25}
, {-61, 27, -103}
, {56, -44, -47}
, {24, 60, 70}
, {85, 94, 72}
, {-82, 70, -73}
, {-75, -85, -20}
, {92, -83, 82}
, {-107, -98, -45}
, {-41, -31, -27}
, {-70, 90, -40}
, {-13, 44, 15}
, {72, 25, 78}
}
, {{75, 30, 4}
, {-71, 102, 66}
, {36, -3, -47}
, {-19, 12, -94}
, {-51, 97, 8}
, {34, -67, -108}
, {81, 50, -24}
, {51, 102, -90}
, {-12, 36, -39}
, {-88, -40, 43}
, {-72, -55, 28}
, {33, 105, 38}
, {47, -54, 63}
, {-81, -19, -93}
, {-58, 24, -5}
, {-21, -61, -26}
}
, {{42, 51, 66}
, {-19, 69, 50}
, {55, -8, 67}
, {-93, -98, -46}
, {-67, 38, 48}
, {3, 81, 82}
, {94, -82, 17}
, {15, 14, -85}
, {-13, 31, 100}
, {-47, 17, 59}
, {44, -67, 59}
, {-24, -34, 57}
, {-84, -45, -82}
, {93, -47, 68}
, {-14, 2, -6}
, {70, 12, -3}
}
, {{4, -28, 38}
, {88, -43, -88}
, {-96, 37, 37}
, {-4, -57, -91}
, {96, 55, 42}
, {78, -32, -54}
, {67, -37, 27}
, {98, 7, 23}
, {-12, 57, -43}
, {-43, -102, 101}
, {-24, -26, 39}
, {-42, 14, 53}
, {15, -50, 9}
, {74, -3, -79}
, {-72, 20, 80}
, {27, 19, -48}
}
, {{-31, 37, 95}
, {81, -12, 45}
, {56, 101, -25}
, {-62, -14, -38}
, {15, -96, -14}
, {54, 74, 24}
, {97, 75, 98}
, {14, -22, 83}
, {-44, -3, -76}
, {77, 42, -47}
, {-67, -19, 44}
, {63, 9, 42}
, {-73, 87, -24}
, {-39, -104, 22}
, {88, -37, -6}
, {47, -7, 30}
}
, {{-1, -88, 79}
, {-29, -2, -35}
, {85, -103, -36}
, {25, -32, 38}
, {-85, -11, -58}
, {99, 52, -10}
, {14, 68, 66}
, {59, 37, 40}
, {-1, 82, -68}
, {-44, -93, 62}
, {-77, -70, 83}
, {-73, 16, -29}
, {-10, -72, -65}
, {8, 49, 38}
, {50, 47, -74}
, {19, 55, -13}
}
, {{-36, 82, 23}
, {37, 0, 89}
, {58, -33, -29}
, {28, -50, -20}
, {16, 3, -7}
, {-11, 75, 33}
, {33, 36, 44}
, {41, -72, 24}
, {-65, 26, 103}
, {51, 103, 44}
, {-64, -24, 100}
, {29, 31, 30}
, {-91, -49, 38}
, {43, -96, -68}
, {95, 85, -18}
, {98, -34, -2}
}
, {{-31, 63, -11}
, {24, 8, 21}
, {-6, 22, 77}
, {-57, 88, 16}
, {53, 17, -47}
, {-11, -45, 45}
, {-79, -73, 22}
, {21, -4, 4}
, {100, -27, -40}
, {60, 96, 0}
, {-80, -82, 8}
, {44, -72, -56}
, {-50, 106, 27}
, {-6, 9, 3}
, {-81, 5, 82}
, {50, 83, -106}
}
, {{87, -79, -21}
, {-14, 67, -38}
, {-88, 8, -55}
, {79, -90, 17}
, {-50, 18, 106}
, {85, -91, 97}
, {-4, -28, -33}
, {-71, 99, 40}
, {-31, 17, 21}
, {26, -29, 82}
, {-72, -54, -94}
, {61, 1, -26}
, {94, -36, -53}
, {93, -36, -93}
, {-51, 30, -74}
, {79, 100, 66}
}
, {{-68, 43, -46}
, {-62, 105, -17}
, {90, -4, -34}
, {46, -81, 54}
, {26, 35, -83}
, {-26, -61, -43}
, {-81, 97, 101}
, {-61, -86, -37}
, {100, -79, 16}
, {25, 106, 87}
, {93, 19, 36}
, {0, 0, 98}
, {-21, -23, 10}
, {-41, 61, 1}
, {49, -77, -12}
, {107, 79, -88}
}
, {{46, 59, 95}
, {17, -94, 26}
, {67, -87, 97}
, {31, -6, 73}
, {-35, -7, 25}
, {-59, -63, 90}
, {97, 89, 31}
, {84, -50, -91}
, {-98, 20, 31}
, {24, -93, -78}
, {45, 103, -49}
, {-65, 74, 68}
, {9, 56, -95}
, {20, 61, -2}
, {-49, -9, -32}
, {58, -10, -13}
}
, {{46, 26, 64}
, {-8, -57, -89}
, {-4, -98, -58}
, {-71, 57, 64}
, {21, -13, -84}
, {74, -23, 63}
, {-16, -40, -11}
, {90, -88, 66}
, {-98, 79, 58}
, {-72, -44, -81}
, {102, 45, 13}
, {-93, -26, 29}
, {10, -54, -79}
, {73, -48, 1}
, {-87, 50, 73}
, {102, 77, 43}
}
, {{-13, -86, 55}
, {41, 102, 57}
, {-69, 46, 64}
, {68, 34, 14}
, {-3, 99, -1}
, {-89, 29, 57}
, {-53, -95, 50}
, {12, 2, -39}
, {-68, -15, -68}
, {-59, -39, -49}
, {39, 75, 41}
, {-20, -56, -58}
, {102, -50, -27}
, {-14, 81, -33}
, {74, -14, -100}
, {79, 102, -24}
}
, {{-71, 72, -100}
, {-55, -43, -58}
, {-30, 7, -97}
, {102, -47, -46}
, {-56, 94, 57}
, {-21, 98, 82}
, {-95, 39, -81}
, {-43, -78, 22}
, {14, 75, -86}
, {35, 26, -49}
, {66, -45, 42}
, {-69, -67, -79}
, {-45, 58, -21}
, {-92, 67, -92}
, {15, -73, -91}
, {-24, 38, -69}
}
, {{76, -93, -8}
, {47, -11, 79}
, {-9, 54, 28}
, {-77, -100, 71}
, {23, -78, -6}
, {98, -51, -83}
, {97, 82, -89}
, {-87, 0, -50}
, {87, 72, 77}
, {88, 46, -37}
, {30, 99, -41}
, {84, 36, -28}
, {-4, -95, 94}
, {27, 21, -82}
, {16, 43, -16}
, {5, -71, -37}
}
, {{73, 41, -58}
, {85, -61, -84}
, {46, 71, -61}
, {24, 63, 86}
, {-74, 9, 94}
, {-38, 11, -2}
, {22, -13, 53}
, {19, -96, -16}
, {3, -53, -6}
, {-90, -60, 64}
, {107, 12, -44}
, {-28, -53, 87}
, {74, -19, -95}
, {-10, -102, 81}
, {-46, 43, -15}
, {100, -47, -65}
}
, {{-100, -16, 41}
, {-75, -48, 48}
, {-82, 55, 36}
, {5, 92, 102}
, {33, 57, -54}
, {32, -101, 3}
, {63, -49, 49}
, {60, -38, 92}
, {51, 53, -12}
, {29, 48, -46}
, {-3, 82, -59}
, {42, 25, -6}
, {21, 81, 52}
, {35, -65, 37}
, {91, -79, -16}
, {93, -33, 93}
}
, {{-86, -60, -91}
, {53, 83, -43}
, {-9, 79, -59}
, {-17, -33, -85}
, {-26, 20, 63}
, {-71, -46, -2}
, {40, -52, 95}
, {-47, 25, -77}
, {-81, -42, 53}
, {67, -28, 43}
, {33, 72, -85}
, {-93, -18, -37}
, {-73, 7, -43}
, {-35, 40, 102}
, {43, 96, -33}
, {54, 18, 96}
}
, {{-25, -105, 69}
, {42, -91, 96}
, {-39, 105, -13}
, {-29, -70, 71}
, {105, -72, 87}
, {84, -81, -25}
, {-93, -41, -19}
, {-16, -38, -42}
, {-42, 41, 37}
, {47, 103, 85}
, {-52, -15, -102}
, {28, -1, 31}
, {-56, 72, 34}
, {73, 31, 20}
, {-39, 7, 28}
, {72, -59, 50}
}
, {{17, -45, 84}
, {61, -64, 41}
, {-100, 17, 85}
, {-60, 64, -91}
, {48, 76, 93}
, {0, 12, -14}
, {92, -13, 101}
, {95, -46, -11}
, {-84, -75, -39}
, {83, 66, 77}
, {-61, 56, -103}
, {-65, -50, -5}
, {-11, 80, 73}
, {-22, -80, -43}
, {100, 100, 33}
, {-111, 4, 67}
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

typedef number_t conv1d_35_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_35(
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


const int16_t conv1d_35_bias[CONV_FILTERS] = {8, -6, -3, -7, 2, -5, -3, 3, 1, -6, -3, 0, 11, -3, 3, 6, -3, -5, 1, 0, 4, 8, 0, -2, 6, 11, 0, 3, 8, 5, 3, 1, -4, -6, 0, -3, 7, -1, 8, 3, 0, -4, 4, -3, 6, 5, -3, 3, 1, -1, 7, 3, 7, -4, 7, 6, 9, -1, 3, 9, -1, 7, 5, -2}
;

const int16_t conv1d_35_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-2, -47, 56}
, {37, -22, 12}
, {47, -59, 19}
, {-15, -50, 26}
, {-54, -24, 33}
, {12, 25, -4}
, {4, -56, -66}
, {-7, -20, 29}
, {37, 34, 48}
, {-33, -35, 49}
, {-32, 30, 48}
, {38, 63, -23}
, {-15, 44, 48}
, {54, -29, -26}
, {14, 54, 67}
, {-28, 0, -28}
, {7, 48, -45}
, {-61, 73, -54}
, {-30, -41, -59}
, {71, -71, 7}
, {52, -55, -22}
, {64, 5, 32}
, {6, 65, 52}
, {65, -59, -71}
, {16, 27, -57}
, {23, -71, -57}
, {71, -19, 25}
, {-9, 5, -6}
, {35, 28, -53}
, {-18, 11, -30}
, {-11, 11, -36}
, {-9, -40, -1}
}
, {{13, -39, -48}
, {-64, -43, -22}
, {-66, 64, -59}
, {-69, 21, -6}
, {-28, 68, 54}
, {54, -10, -68}
, {33, -14, 18}
, {-24, 66, -56}
, {-19, -33, -30}
, {-29, -68, 7}
, {-42, -53, 18}
, {2, -5, 49}
, {-39, 41, 31}
, {3, -58, 66}
, {-45, -19, -62}
, {33, 58, -45}
, {42, 25, 18}
, {-54, -66, 60}
, {44, 55, 72}
, {49, -32, -24}
, {30, -19, 58}
, {64, -49, 39}
, {-19, -13, -49}
, {-50, 27, -31}
, {67, 43, -11}
, {-9, -66, -6}
, {-42, 20, -73}
, {52, -61, -24}
, {-27, -67, -51}
, {29, 16, -32}
, {7, -18, -10}
, {11, -8, -26}
}
, {{-41, -5, -65}
, {58, -63, -22}
, {34, 38, 12}
, {-6, 56, -67}
, {-38, -21, 9}
, {60, -63, 19}
, {-40, -53, 37}
, {-1, -72, 38}
, {54, -67, 2}
, {59, -57, -5}
, {60, -5, -64}
, {-3, 66, 0}
, {5, 50, -20}
, {-55, -40, -77}
, {15, 19, 70}
, {13, 21, -8}
, {-11, -60, 16}
, {22, -60, 53}
, {-41, 44, 37}
, {4, -11, -8}
, {32, -1, -22}
, {55, -29, -27}
, {68, 41, 54}
, {-43, 7, 42}
, {12, 39, 43}
, {51, -45, 0}
, {-24, -14, -24}
, {-67, 10, 49}
, {-49, 52, -46}
, {-29, -23, -14}
, {58, -58, -4}
, {63, 53, 18}
}
, {{43, -36, 16}
, {42, 35, -54}
, {48, -1, 49}
, {37, 44, 28}
, {-6, 32, 17}
, {-39, -6, 6}
, {-30, 38, -61}
, {-51, 27, 61}
, {43, -19, -56}
, {-36, -58, 0}
, {-44, 54, 58}
, {-29, -41, 41}
, {-71, -23, -3}
, {-49, 43, -40}
, {3, 30, -34}
, {36, 44, 31}
, {14, 69, -31}
, {-38, -42, -32}
, {-74, -22, -77}
, {46, 65, -39}
, {-41, -18, -10}
, {23, 10, 46}
, {-65, 6, -59}
, {-74, 32, 42}
, {-39, -52, -41}
, {38, 47, 23}
, {9, 29, -7}
, {27, -17, -52}
, {69, 7, 56}
, {-20, 17, -3}
, {41, 54, -49}
, {-6, 2, -48}
}
, {{-14, 58, -43}
, {-4, 45, 0}
, {-61, -21, -4}
, {52, 23, 21}
, {-14, -23, 40}
, {64, -28, -64}
, {-46, -31, 3}
, {12, -59, -24}
, {8, 1, 35}
, {40, -11, -5}
, {72, -38, -44}
, {-36, -12, 23}
, {-21, 33, -40}
, {26, 34, -63}
, {-41, -32, 27}
, {-18, -47, -44}
, {-13, -31, -71}
, {9, 28, 25}
, {71, 70, 57}
, {50, 50, 14}
, {16, -46, 65}
, {46, -69, -71}
, {73, 56, -14}
, {-27, 27, -55}
, {-23, 46, -53}
, {6, 62, 22}
, {56, 32, -33}
, {23, -3, 0}
, {25, 66, -22}
, {-63, -60, -64}
, {-22, -26, 25}
, {-71, 8, 61}
}
, {{30, 30, -35}
, {-66, 55, 49}
, {6, 70, -75}
, {-56, 22, 17}
, {-65, 60, -65}
, {22, -10, -64}
, {1, 18, -44}
, {66, -61, 24}
, {-20, 18, 1}
, {-47, 5, -38}
, {-58, -28, -72}
, {-65, -67, 54}
, {-67, -30, 34}
, {51, -63, -4}
, {47, 0, 8}
, {-18, 28, -45}
, {-59, 5, -20}
, {46, -41, 63}
, {-5, -26, 44}
, {-1, 68, 33}
, {29, -68, -37}
, {-58, 0, 63}
, {25, -52, 60}
, {-31, 26, 58}
, {-55, 56, 17}
, {-76, -65, -57}
, {-13, 51, 11}
, {68, -6, 28}
, {-74, -15, 42}
, {9, -8, 41}
, {-18, 56, -47}
, {-10, 25, -57}
}
, {{49, -75, -4}
, {-68, -55, 9}
, {-38, -60, 24}
, {-30, -61, -32}
, {-26, -48, -42}
, {-7, 56, 4}
, {45, 12, -48}
, {53, -6, 13}
, {54, -2, -38}
, {-25, 34, 40}
, {-7, 4, 42}
, {-29, 46, -67}
, {2, -19, 5}
, {36, -65, -2}
, {-23, 7, 15}
, {41, 1, -65}
, {-74, 24, -66}
, {12, 47, 58}
, {48, 21, -46}
, {-62, 56, -72}
, {34, 18, -6}
, {-76, 29, -12}
, {67, -37, 4}
, {-11, -6, -28}
, {10, -38, -41}
, {23, 38, -74}
, {41, -35, -65}
, {59, -24, -62}
, {-35, 70, -58}
, {-37, 50, -68}
, {-2, -35, -59}
, {16, -4, -46}
}
, {{0, 43, -48}
, {-2, -70, -66}
, {0, 75, -52}
, {16, -15, -66}
, {47, -40, 7}
, {-68, 55, 31}
, {15, -46, 9}
, {34, -52, 21}
, {-30, 29, -39}
, {-67, -1, 44}
, {1, -28, 0}
, {48, -65, 69}
, {-18, -5, -13}
, {21, -36, 73}
, {56, -22, -11}
, {12, -51, -27}
, {-62, -49, -42}
, {-62, 22, -8}
, {37, -54, -25}
, {-27, 12, 36}
, {20, -57, 23}
, {-1, 17, -39}
, {39, -9, 1}
, {2, -30, -55}
, {-61, -23, 60}
, {64, -58, 49}
, {-23, -66, 22}
, {72, 64, -4}
, {64, -67, 67}
, {-35, 1, 27}
, {9, 18, 27}
, {30, -20, -70}
}
, {{47, 22, 35}
, {-55, -15, -43}
, {61, -51, 49}
, {13, -69, -71}
, {-14, 34, -15}
, {59, -9, -68}
, {-54, 60, -52}
, {-38, 55, -73}
, {19, 22, -20}
, {-2, 41, 38}
, {-25, 41, -2}
, {68, -10, 12}
, {0, 22, 39}
, {-7, 57, -18}
, {36, -25, -36}
, {56, -66, 35}
, {0, 1, 18}
, {10, 66, -22}
, {14, -73, -12}
, {28, -68, 32}
, {23, 54, -23}
, {-24, 70, -73}
, {33, -16, 4}
, {-37, -10, -66}
, {48, 63, -60}
, {31, -10, -43}
, {-65, 29, 20}
, {49, 54, -47}
, {-70, -1, -5}
, {-71, -36, -37}
, {18, -26, 7}
, {67, -11, -34}
}
, {{-4, 53, -10}
, {10, 8, -14}
, {-28, 6, 69}
, {43, 23, 42}
, {-10, 4, 73}
, {-30, -19, 21}
, {1, -16, 73}
, {-7, 58, 17}
, {67, -14, 29}
, {60, 12, 42}
, {24, 38, 5}
, {38, 48, 65}
, {0, 50, 17}
, {25, -26, -42}
, {-48, -69, -17}
, {9, -18, -17}
, {-49, 46, -1}
, {-40, 12, 62}
, {-41, -55, -56}
, {-24, 50, -11}
, {3, 7, -38}
, {-5, -53, 17}
, {54, 13, 38}
, {58, -49, 21}
, {-21, -65, 35}
, {49, 40, 59}
, {-25, -11, -33}
, {-55, -42, 52}
, {23, -14, 43}
, {-48, 19, 0}
, {10, 56, 10}
, {61, -32, 43}
}
, {{62, -11, 28}
, {-11, -7, -38}
, {-60, -41, -7}
, {72, 36, 25}
, {32, 55, 23}
, {61, 53, 72}
, {18, 44, -44}
, {32, 43, -58}
, {72, 37, -47}
, {-24, 14, 19}
, {-51, -56, -70}
, {-4, 24, 14}
, {-57, 38, -22}
, {5, -7, -25}
, {12, 43, -32}
, {70, -60, -35}
, {10, -47, -42}
, {53, -13, 44}
, {-50, 53, 45}
, {32, 43, 5}
, {0, 30, 46}
, {-33, -24, 13}
, {16, 0, -26}
, {58, 4, -57}
, {-55, -56, -49}
, {-34, -1, -35}
, {-19, 47, 33}
, {-29, -18, 45}
, {-2, 52, -49}
, {14, -26, 33}
, {-22, 55, -34}
, {-33, 3, -28}
}
, {{-13, 36, 13}
, {-61, -53, 63}
, {70, -12, 12}
, {30, -32, 8}
, {-24, 22, 12}
, {30, -29, 22}
, {-65, 67, -15}
, {-38, -56, -10}
, {63, -21, 19}
, {56, -52, 0}
, {51, 18, 62}
, {-38, 23, 44}
, {60, 3, -46}
, {-37, -10, -16}
, {-70, -14, -21}
, {-24, 43, 45}
, {-58, -13, 12}
, {48, 23, 10}
, {-56, 34, -16}
, {-18, 29, -17}
, {2, 74, -7}
, {-12, -16, 13}
, {72, 63, -14}
, {-58, 7, -25}
, {-21, 54, 50}
, {71, -43, 47}
, {-7, -30, -1}
, {23, -22, 53}
, {-43, 68, 22}
, {-19, 62, -48}
, {-42, 48, -16}
, {23, -49, -9}
}
, {{-68, 30, 32}
, {2, 65, 6}
, {-30, -23, -34}
, {-41, -41, -5}
, {-10, 39, 39}
, {50, 44, -33}
, {47, 35, 67}
, {-64, -36, 18}
, {71, -34, -48}
, {14, -3, -18}
, {-24, 10, 7}
, {-55, -5, -44}
, {17, -40, 2}
, {-42, 2, 70}
, {57, 1, 5}
, {23, -35, 29}
, {15, 65, -33}
, {17, 22, -50}
, {55, 68, -25}
, {-5, 12, -67}
, {-16, 32, 17}
, {11, -31, 6}
, {50, -26, 35}
, {33, -9, 52}
, {34, -9, 8}
, {-32, -12, -30}
, {77, 78, 41}
, {34, -52, -64}
, {-44, -40, -51}
, {-15, -55, 43}
, {-35, -56, -37}
, {-26, -59, -9}
}
, {{-66, -57, 36}
, {-68, 74, 65}
, {-47, 47, 67}
, {-43, 57, -33}
, {68, 14, -19}
, {33, -31, 53}
, {54, -61, -48}
, {17, 27, -58}
, {58, 7, 53}
, {53, -56, -38}
, {56, 18, 65}
, {28, 22, 36}
, {53, 1, -69}
, {22, 10, 53}
, {38, -63, 14}
, {-18, -6, -4}
, {5, 39, 61}
, {-6, -9, 21}
, {-26, 43, 33}
, {-4, 1, -24}
, {-3, 0, -20}
, {-61, 43, -65}
, {53, 65, 63}
, {-74, -31, 49}
, {-43, -41, 47}
, {-63, 50, -71}
, {-1, -64, -11}
, {-20, 1, -30}
, {-38, -31, 40}
, {56, 57, 70}
, {-22, 62, 64}
, {-39, -9, 72}
}
, {{60, -54, -55}
, {73, 59, 27}
, {33, 9, 20}
, {32, 51, 37}
, {36, 46, 3}
, {-9, 23, -34}
, {-19, -6, -23}
, {-39, -47, -28}
, {-42, -44, 58}
, {66, -1, 4}
, {68, 64, 45}
, {-72, 58, 9}
, {44, 70, -14}
, {-1, 4, -17}
, {-17, -1, -51}
, {-35, -4, -56}
, {-55, 38, 1}
, {31, 4, -9}
, {15, 42, -72}
, {0, -59, -14}
, {-37, -53, 58}
, {-5, 30, -2}
, {34, -18, 27}
, {-44, 3, -28}
, {36, -44, -70}
, {-3, 56, -58}
, {66, 63, 36}
, {-62, 11, 53}
, {71, 25, -65}
, {1, 5, 18}
, {38, 6, -72}
, {-37, 6, 17}
}
, {{69, -3, 3}
, {-41, -38, -52}
, {-21, -64, 42}
, {12, -37, 14}
, {60, -68, 32}
, {-63, 62, 6}
, {0, 10, 63}
, {-49, -30, -72}
, {-16, 50, -6}
, {15, 51, -42}
, {-31, -54, -28}
, {44, -47, -69}
, {69, -61, -69}
, {-50, 35, 42}
, {-51, -30, 58}
, {52, -33, 1}
, {-60, -1, 33}
, {-53, 6, 19}
, {-61, -3, 44}
, {15, 67, -43}
, {-36, 68, -11}
, {-45, 62, -37}
, {-39, 42, -26}
, {-17, -28, -5}
, {-31, -32, 6}
, {-62, -22, -64}
, {-32, 33, 22}
, {5, -57, -1}
, {14, -17, 9}
, {29, 22, 19}
, {56, -35, 8}
, {-58, -41, -19}
}
, {{-35, 53, -30}
, {-66, 37, -40}
, {27, 24, 57}
, {53, 65, -48}
, {-38, -64, -18}
, {62, -66, -2}
, {37, -22, -46}
, {32, 33, 19}
, {-1, -53, -35}
, {-72, 10, -20}
, {-52, -54, 45}
, {-2, 7, -21}
, {-16, -25, 36}
, {-18, -53, -47}
, {-13, 12, -55}
, {-8, -58, 49}
, {4, -63, -6}
, {-27, -40, -16}
, {0, 32, 39}
, {67, -52, -26}
, {63, -27, 70}
, {41, 0, -63}
, {-2, 31, -13}
, {-25, 55, 6}
, {-36, -10, 24}
, {24, -1, -29}
, {42, 18, 64}
, {4, 7, 2}
, {-5, 62, -21}
, {-52, 57, 18}
, {44, -52, 41}
, {-68, 16, 59}
}
, {{-71, -11, 54}
, {-46, 70, 36}
, {-54, 17, -60}
, {-2, 73, -60}
, {24, -32, 30}
, {36, 36, 10}
, {47, -50, -14}
, {48, -12, 20}
, {41, -37, 63}
, {27, -73, 39}
, {44, -39, -59}
, {-56, 64, 5}
, {-68, -8, -2}
, {-3, 37, 12}
, {-38, 22, -31}
, {-51, 24, -32}
, {-19, 3, -57}
, {-56, -75, -57}
, {64, -24, 55}
, {43, 75, -35}
, {-2, 67, 66}
, {-20, -1, 27}
, {-75, 39, 38}
, {12, 19, 37}
, {-26, -28, -63}
, {-27, 38, 28}
, {-1, 47, 1}
, {-42, 1, -23}
, {45, -38, 65}
, {16, 42, -68}
, {56, -22, 34}
, {-33, 8, 9}
}
, {{-74, 18, -51}
, {-42, -53, 63}
, {-18, 56, 60}
, {-58, 9, -1}
, {24, -60, 17}
, {67, -25, -59}
, {-50, -29, -56}
, {54, 57, 22}
, {32, -64, -52}
, {60, 15, 4}
, {26, 69, 72}
, {-49, 51, 24}
, {-69, 72, 60}
, {-19, 25, -14}
, {47, 68, 59}
, {6, -47, -49}
, {-28, 0, -63}
, {-42, 7, -58}
, {47, -16, -68}
, {58, 40, 55}
, {-30, -36, -64}
, {-69, 69, 73}
, {-30, 48, 26}
, {-36, 2, -40}
, {44, 32, 50}
, {74, 57, 39}
, {-40, 4, 46}
, {-71, 33, 7}
, {3, -61, 25}
, {65, 50, -37}
, {42, -39, 72}
, {-49, 55, 70}
}
, {{-28, 58, 52}
, {-70, 0, 1}
, {7, 38, -67}
, {-12, 24, 2}
, {24, -67, -3}
, {51, -1, -47}
, {-61, -58, -38}
, {-39, 28, -66}
, {52, 28, 63}
, {40, 28, -66}
, {-63, 12, 0}
, {19, -40, -40}
, {1, -49, -15}
, {17, 48, 67}
, {37, -50, -38}
, {-38, 56, 60}
, {-17, 1, -22}
, {-45, 67, -64}
, {-31, -68, 0}
, {25, -5, 71}
, {-66, 57, -26}
, {-52, 14, 15}
, {-51, 53, -73}
, {-18, -29, -38}
, {-11, 16, -13}
, {-21, -66, -43}
, {-14, 8, 9}
, {-44, 33, -15}
, {-20, 46, -61}
, {-8, 14, -62}
, {-9, 35, 26}
, {-14, -73, 52}
}
, {{53, 31, -58}
, {-66, 71, 48}
, {4, 7, 14}
, {-51, -16, -10}
, {-60, 55, -8}
, {-35, -17, -30}
, {47, -59, 54}
, {-48, -49, -22}
, {11, -52, 61}
, {-60, -65, -21}
, {26, 31, 4}
, {3, 3, -51}
, {-64, 39, 24}
, {-9, 10, 16}
, {64, 71, 1}
, {40, 3, -14}
, {-45, 67, -7}
, {-48, -13, -43}
, {62, 52, 37}
, {-4, -42, -73}
, {-20, 20, -54}
, {61, 65, 23}
, {62, 35, -72}
, {20, 26, 53}
, {53, -36, 6}
, {-60, -71, -61}
, {19, 36, -68}
, {-25, 54, 43}
, {49, -72, 34}
, {-65, -45, 60}
, {-70, 35, 42}
, {-67, 53, 19}
}
, {{-7, -16, -30}
, {-6, 70, 2}
, {27, 8, -59}
, {8, -53, 8}
, {55, 32, 62}
, {5, -48, -18}
, {-40, 7, -17}
, {34, 21, 62}
, {63, 60, 40}
, {10, -36, -39}
, {-46, 54, -59}
, {-54, 46, -13}
, {-3, -58, -7}
, {71, 30, -23}
, {12, 59, 15}
, {9, 25, -68}
, {-25, -63, 15}
, {-45, 44, -17}
, {-28, 11, -55}
, {63, 47, 56}
, {-52, -63, -25}
, {-52, 62, -23}
, {20, -50, -25}
, {51, -41, 66}
, {56, 63, -18}
, {-56, -5, -59}
, {-13, -24, 6}
, {65, -59, 26}
, {-14, 29, -11}
, {-16, -39, -52}
, {-70, 39, 55}
, {-51, 13, 33}
}
, {{-13, -8, 22}
, {54, 41, 34}
, {34, -57, 37}
, {-70, 48, 51}
, {-63, 32, 36}
, {-34, -13, -57}
, {74, 3, 33}
, {67, -5, -30}
, {74, -43, 33}
, {-66, 34, 71}
, {32, 34, 23}
, {-15, -20, 46}
, {-25, -38, 58}
, {68, 57, -26}
, {-29, 1, 21}
, {20, 16, -37}
, {-60, 11, 35}
, {23, 58, -38}
, {-7, -52, 44}
, {63, -59, 58}
, {33, -35, -31}
, {13, 0, 16}
, {71, 50, -44}
, {-27, 77, 64}
, {-23, -11, -53}
, {33, 48, 14}
, {-40, 20, 11}
, {-55, 21, 65}
, {28, 58, 19}
, {16, -50, -59}
, {7, 36, -37}
, {-48, 59, -34}
}
, {{-32, -4, 28}
, {3, -38, 24}
, {42, 53, 31}
, {3, -69, 29}
, {6, -49, -71}
, {-27, -6, -11}
, {-31, -13, 41}
, {-44, -31, -37}
, {62, 30, 53}
, {50, 42, -30}
, {54, -68, 37}
, {-47, -60, 66}
, {-27, 68, -24}
, {49, -49, 34}
, {70, 38, -10}
, {25, -59, -41}
, {0, -24, -42}
, {24, -73, -43}
, {8, 13, -14}
, {-73, 54, 31}
, {33, -25, -65}
, {-54, -34, -14}
, {64, -18, -68}
, {58, -28, -62}
, {-9, 36, -36}
, {-26, -31, 34}
, {-25, 64, 50}
, {32, 20, 37}
, {55, -53, -42}
, {49, -15, 48}
, {37, 50, -22}
, {-50, -54, 67}
}
, {{43, 73, -15}
, {22, 70, 21}
, {-53, 35, -54}
, {63, 37, 15}
, {9, -55, 67}
, {-29, 62, -17}
, {23, -19, -54}
, {-17, -26, 63}
, {-15, -26, 44}
, {48, -56, 33}
, {-14, -24, 66}
, {70, 17, 17}
, {16, 21, 68}
, {34, 70, -8}
, {8, -70, -11}
, {-65, 71, -66}
, {53, 60, -20}
, {0, -3, 67}
, {35, 56, 12}
, {1, -60, -56}
, {-10, 72, 49}
, {53, 1, 29}
, {55, -15, -67}
, {63, 7, -20}
, {-24, 28, 32}
, {63, -54, -32}
, {8, -64, 33}
, {-26, -34, 52}
, {24, -46, -19}
, {-32, -47, -8}
, {-12, -17, 10}
, {-37, 62, 71}
}
, {{21, 34, -41}
, {41, 64, -21}
, {-5, -44, -53}
, {-25, -66, -42}
, {-40, 1, 23}
, {-70, -7, -60}
, {53, 21, 34}
, {0, 41, 27}
, {67, -32, 67}
, {-54, 5, 38}
, {-34, -21, -63}
, {30, 16, -75}
, {-65, -5, -46}
, {34, 62, 55}
, {69, -1, 5}
, {26, -26, 0}
, {69, -9, 60}
, {22, 76, -9}
, {16, -48, -26}
, {42, -42, 19}
, {18, -55, 69}
, {37, -13, 43}
, {76, 66, 24}
, {74, 32, 6}
, {-55, -36, -19}
, {-39, -62, 31}
, {-51, 9, 25}
, {-47, -16, 72}
, {-10, -45, 41}
, {59, -45, -65}
, {-6, -69, -8}
, {-52, -69, 30}
}
, {{-60, 33, -21}
, {-8, -49, 16}
, {-46, -17, 2}
, {-38, -32, 25}
, {46, -25, -38}
, {14, -24, -2}
, {35, 63, -50}
, {-54, 9, -25}
, {-39, 31, 69}
, {9, 46, 8}
, {0, -69, 14}
, {-10, 26, -23}
, {-15, 40, -26}
, {-70, -35, 24}
, {42, 24, 26}
, {38, 71, -43}
, {-21, 11, -50}
, {-57, -32, -72}
, {-27, -42, 35}
, {2, 2, -60}
, {0, 17, 52}
, {41, -3, -35}
, {-52, 16, 58}
, {21, -28, 17}
, {-49, 9, -2}
, {-76, 24, -60}
, {50, 2, 66}
, {-24, -14, 33}
, {-70, 19, 35}
, {36, -11, -27}
, {68, 21, -10}
, {-40, 8, 0}
}
, {{-8, 45, 18}
, {8, -14, 11}
, {63, -50, 8}
, {51, 70, -57}
, {59, 43, 22}
, {-60, -36, -9}
, {-24, 13, -13}
, {74, -37, 35}
, {34, 41, 15}
, {37, 0, 62}
, {-15, 39, -30}
, {-70, 32, -50}
, {0, 33, 43}
, {61, -64, -40}
, {1, 9, 72}
, {-32, 54, 34}
, {14, -23, -70}
, {-47, -67, 29}
, {18, -39, 22}
, {-62, -41, -45}
, {-27, -34, -6}
, {-70, -36, -59}
, {43, -63, 71}
, {0, 13, 39}
, {-71, 31, -20}
, {59, -11, 47}
, {48, -16, -51}
, {-31, -71, 32}
, {-15, -63, -28}
, {50, -18, 35}
, {-71, -32, -55}
, {-12, -48, -31}
}
, {{-28, -17, 55}
, {-65, 37, 19}
, {18, -60, -11}
, {70, 0, -52}
, {59, 6, -45}
, {-13, 0, 2}
, {-53, 34, 16}
, {-10, -50, -45}
, {-56, 15, -8}
, {-18, -43, -59}
, {-59, 34, -30}
, {31, -21, 7}
, {-69, -33, 4}
, {-54, 63, -26}
, {-30, 36, 3}
, {18, -60, 64}
, {28, -4, 1}
, {13, 74, 19}
, {29, -15, 55}
, {-37, -22, -44}
, {-54, 26, 14}
, {67, 16, -53}
, {9, 34, -62}
, {34, 49, 1}
, {-18, 51, 47}
, {-10, -48, -50}
, {-28, 48, -61}
, {15, 75, 28}
, {-12, -13, -70}
, {32, -59, -72}
, {27, -18, -68}
, {-61, 6, -31}
}
, {{12, -44, -24}
, {39, -6, 70}
, {27, 42, 45}
, {16, 40, 10}
, {-33, 50, -48}
, {54, 21, -37}
, {53, 32, 20}
, {10, -44, -60}
, {62, 12, 47}
, {-54, -42, -27}
, {-12, 77, -68}
, {-57, 39, 20}
, {-57, -38, 57}
, {62, -71, 33}
, {37, -29, -10}
, {74, -10, 10}
, {-17, 74, -65}
, {-65, -68, -66}
, {-11, -72, 13}
, {-62, 31, 30}
, {-17, 45, -72}
, {9, -23, 23}
, {9, -44, 17}
, {-51, -64, 51}
, {19, 24, 65}
, {-43, 52, 22}
, {26, 41, 64}
, {19, 27, -38}
, {11, -64, -55}
, {-23, -9, -5}
, {11, -39, 72}
, {36, -24, 25}
}
, {{68, 4, 24}
, {62, -51, -54}
, {-48, 0, 24}
, {-48, 64, 4}
, {-44, 63, 42}
, {48, -64, 43}
, {-29, -37, 24}
, {-5, 52, 45}
, {62, -55, -65}
, {-74, -27, 26}
, {-56, -38, 13}
, {73, 1, 11}
, {22, 44, 1}
, {1, -39, 23}
, {1, -63, -66}
, {-33, 70, 71}
, {-50, -4, 27}
, {-62, -72, 31}
, {14, -71, -37}
, {-43, -67, -69}
, {26, 20, 48}
, {-70, 10, 1}
, {-40, -37, 19}
, {-29, 46, -35}
, {31, -19, 75}
, {-26, 25, -7}
, {44, 62, 3}
, {20, -37, 15}
, {-37, 54, 36}
, {40, 61, 33}
, {63, -63, -8}
, {-72, 36, -72}
}
, {{-4, 45, -68}
, {-64, -15, 35}
, {-17, -68, -39}
, {34, 40, -51}
, {-48, 11, -8}
, {-51, 34, -31}
, {4, 12, -10}
, {-66, 0, 72}
, {-68, 23, -41}
, {-67, -58, 75}
, {-34, -72, -12}
, {18, -42, 50}
, {5, -43, 53}
, {51, 1, -53}
, {-1, -62, 49}
, {12, -43, 26}
, {-43, 9, 4}
, {62, 50, -53}
, {-24, 30, 12}
, {59, -68, -37}
, {-39, 67, 20}
, {-5, -35, 28}
, {-56, 41, -15}
, {-31, -2, 20}
, {47, -61, -15}
, {-23, -9, 22}
, {15, 71, 34}
, {34, -3, -41}
, {55, -15, 10}
, {-8, 39, -12}
, {59, -8, -9}
, {73, -40, -18}
}
, {{0, 53, 15}
, {-65, 18, 11}
, {47, -33, 56}
, {-10, -54, 9}
, {62, -57, 48}
, {23, 66, -63}
, {-11, 0, 24}
, {-56, 25, 21}
, {-29, -25, -51}
, {71, 64, 8}
, {-54, -65, -70}
, {-52, 42, 56}
, {-19, -15, 44}
, {37, 14, 31}
, {10, -63, -44}
, {55, -27, -52}
, {60, 60, -13}
, {-33, -49, 21}
, {-31, -8, -28}
, {58, 48, 47}
, {27, 31, 68}
, {15, 44, -28}
, {35, -29, 56}
, {41, 70, -37}
, {-37, 71, 44}
, {-14, -47, -26}
, {-11, 63, 8}
, {-61, 29, -13}
, {-57, -61, 66}
, {71, -32, 12}
, {35, -26, -60}
, {-60, -47, -29}
}
, {{-66, 37, -22}
, {48, -33, 54}
, {-72, -40, 23}
, {-73, -68, 68}
, {-7, -28, -9}
, {-42, 72, 26}
, {-29, 68, -42}
, {52, 20, -18}
, {34, 68, 72}
, {7, 32, 35}
, {5, -48, -32}
, {-37, 68, -19}
, {1, -36, 73}
, {51, 7, -22}
, {-7, 69, -16}
, {42, 55, 34}
, {22, -14, -43}
, {56, 59, -66}
, {-47, 41, -15}
, {8, -20, 33}
, {53, -31, -61}
, {40, 66, 72}
, {-23, 37, 26}
, {-55, 20, 60}
, {3, -10, -6}
, {-1, 38, -24}
, {-48, 58, -73}
, {73, 29, 18}
, {37, 8, -66}
, {-59, 71, 41}
, {36, 25, 32}
, {-58, 46, 44}
}
, {{-35, -56, -62}
, {-25, 31, 71}
, {37, 11, -21}
, {66, -65, -4}
, {-63, 2, 63}
, {33, 43, -12}
, {-39, -51, -31}
, {6, -39, 3}
, {51, -53, 61}
, {-33, -24, 26}
, {-50, 11, 2}
, {-29, 42, 0}
, {4, 65, -20}
, {18, 70, 17}
, {-8, 14, -73}
, {-4, -18, 42}
, {-54, -51, -62}
, {60, -3, -2}
, {-43, 3, 63}
, {-45, -58, -46}
, {-22, -70, 1}
, {-12, 55, 72}
, {66, -41, 31}
, {74, 45, 31}
, {51, 25, -9}
, {-66, 58, 9}
, {5, 19, -54}
, {-39, -23, -65}
, {52, -13, 62}
, {55, -48, -68}
, {-50, 54, 47}
, {22, -36, -53}
}
, {{15, 0, 53}
, {54, -33, 50}
, {-1, 31, -14}
, {-32, -65, -38}
, {-35, 66, -21}
, {66, 5, 5}
, {15, 51, 18}
, {-51, -19, 33}
, {73, 0, 19}
, {16, 12, 51}
, {28, 13, -58}
, {8, 21, 31}
, {31, 65, 22}
, {-29, 24, -67}
, {-29, 34, 44}
, {-21, -24, 0}
, {48, -17, 20}
, {64, 31, -38}
, {-50, -35, -13}
, {-35, 25, -73}
, {-8, 17, -20}
, {44, 0, 13}
, {-50, -6, -72}
, {30, 74, 15}
, {-2, -71, 10}
, {-46, 15, 16}
, {-28, -51, -9}
, {-25, -28, -44}
, {28, 3, -2}
, {38, -70, 8}
, {48, 52, -30}
, {-33, -24, -71}
}
, {{56, 35, -28}
, {39, -26, -19}
, {65, -63, -32}
, {-11, 20, 48}
, {-62, 23, 5}
, {-74, -70, -3}
, {33, 61, -37}
, {59, -50, -63}
, {17, 61, 33}
, {68, 68, -67}
, {-41, -56, -40}
, {33, 63, 51}
, {51, -60, -43}
, {-27, -47, -2}
, {32, 28, -28}
, {57, 66, 36}
, {-21, -1, 27}
, {-5, 4, 1}
, {-26, 5, -69}
, {5, 30, -3}
, {-29, 44, -35}
, {24, 24, 39}
, {0, 35, -22}
, {68, -54, 22}
, {71, 70, 19}
, {32, -29, -8}
, {58, -49, -55}
, {-61, 44, 50}
, {-27, -42, 30}
, {-45, 49, 38}
, {29, 1, 56}
, {22, -37, -73}
}
, {{-68, 57, -70}
, {48, -58, -22}
, {16, -55, -46}
, {-49, -45, -46}
, {53, -2, 63}
, {-69, 59, 57}
, {-12, -61, 6}
, {72, -8, -40}
, {-50, -34, -14}
, {22, 24, 50}
, {49, 46, -68}
, {-40, 46, -20}
, {36, -70, 55}
, {-21, -64, 37}
, {66, -64, 52}
, {31, 58, 29}
, {-37, 27, 29}
, {-29, -4, -67}
, {-62, -44, -37}
, {-29, -10, 66}
, {29, 0, -63}
, {-43, -31, -34}
, {-33, 56, 24}
, {2, 56, 18}
, {-46, -53, -32}
, {34, -23, -15}
, {64, -3, 65}
, {-65, 56, -38}
, {32, 59, -5}
, {-6, 46, 53}
, {32, 62, 26}
, {-56, 5, -6}
}
, {{23, -42, 68}
, {-25, 24, 3}
, {9, -57, -29}
, {-51, 31, -56}
, {-44, 20, 3}
, {1, -55, -48}
, {0, 61, -51}
, {-11, 15, 47}
, {11, 67, 38}
, {14, 4, -9}
, {-56, 71, 9}
, {-27, -63, 60}
, {-42, -52, 31}
, {61, -27, 65}
, {33, 72, -51}
, {-56, 5, -12}
, {7, 58, -58}
, {-34, 53, -64}
, {-8, 48, -68}
, {51, 16, -70}
, {16, 53, 28}
, {24, 35, -8}
, {-63, 51, 0}
, {-50, -27, -59}
, {56, 24, 21}
, {32, -45, -13}
, {-34, -27, -69}
, {-7, -1, -68}
, {-66, 8, 47}
, {-7, 37, 74}
, {-2, -31, -45}
, {-15, -1, 22}
}
, {{-36, 39, -52}
, {-70, 24, 57}
, {-46, -50, -51}
, {-55, -50, 56}
, {-9, 33, 65}
, {-70, 5, -6}
, {60, 48, 15}
, {-9, -66, 67}
, {-16, 55, 10}
, {49, -48, 54}
, {24, -63, 16}
, {-39, -8, -31}
, {43, -56, -26}
, {52, -54, -1}
, {12, 14, 42}
, {-58, 16, 63}
, {-65, 5, -18}
, {46, -38, 34}
, {16, -21, 1}
, {-69, -70, -27}
, {-31, -45, 65}
, {57, -42, -68}
, {57, -60, -49}
, {4, -68, 70}
, {-13, -29, 20}
, {42, -12, 14}
, {-64, -22, 46}
, {66, 47, -10}
, {-71, -1, -2}
, {-20, 65, 60}
, {65, -51, 57}
, {-51, 71, -24}
}
, {{46, -59, -74}
, {-37, 53, 52}
, {-50, 49, -68}
, {44, 46, -57}
, {-20, 47, -56}
, {49, -38, -28}
, {30, 4, 4}
, {11, 41, 64}
, {50, 33, -3}
, {44, -19, -23}
, {58, 30, -58}
, {46, 71, -24}
, {6, -8, 0}
, {-1, 42, -24}
, {-21, 46, 13}
, {55, -23, -3}
, {15, 71, -28}
, {-72, 29, -18}
, {39, -23, -32}
, {-20, -59, -68}
, {6, 36, 54}
, {20, 15, 10}
, {-73, -65, 0}
, {4, 7, 21}
, {-50, -65, 21}
, {-6, 72, -36}
, {55, 61, 53}
, {-3, 6, 40}
, {25, 3, 24}
, {23, 20, -52}
, {40, -51, 47}
, {-25, 40, -34}
}
, {{-31, 14, -73}
, {26, -10, -26}
, {68, 30, -66}
, {70, -15, 50}
, {60, -46, -39}
, {20, -37, -20}
, {56, 40, -11}
, {74, -44, 48}
, {-72, -8, 29}
, {34, -69, -36}
, {-23, 55, 37}
, {26, -47, -36}
, {1, -44, 40}
, {68, -55, 61}
, {58, -71, 53}
, {-9, -59, 11}
, {8, 48, -36}
, {18, -46, -24}
, {48, -17, 74}
, {-2, 62, 69}
, {62, 39, -20}
, {26, 61, -68}
, {-3, -10, 0}
, {-13, 27, -69}
, {-38, 60, 0}
, {52, 9, 32}
, {20, 42, 20}
, {-68, -50, 50}
, {-15, -24, -24}
, {-17, 57, -17}
, {48, -10, 49}
, {41, 45, -15}
}
, {{-69, -37, 14}
, {51, 5, 16}
, {19, 17, 29}
, {10, -62, -39}
, {55, -16, 2}
, {-51, -22, 7}
, {-28, 59, -10}
, {37, -59, -15}
, {38, 8, -64}
, {-53, 4, 47}
, {-73, 68, 12}
, {-52, -21, 67}
, {28, 8, 47}
, {-50, 37, 46}
, {69, -30, 66}
, {73, 8, 71}
, {-45, 30, 0}
, {57, 5, -35}
, {-9, 5, 47}
, {-18, 62, -20}
, {-14, -73, -62}
, {67, -40, 72}
, {19, 0, 20}
, {59, 12, -23}
, {-35, 39, -5}
, {9, -31, -68}
, {68, -59, 10}
, {-70, 68, 14}
, {-50, 59, 8}
, {69, 57, 28}
, {-1, -65, 0}
, {-23, 46, 13}
}
, {{-73, 43, -11}
, {-3, 17, 1}
, {40, 54, -71}
, {40, 7, 9}
, {-45, -55, 19}
, {29, 60, 40}
, {74, 0, -57}
, {-6, 56, -21}
, {42, 38, 25}
, {-22, 40, -19}
, {-65, 47, 33}
, {-24, -25, 18}
, {38, -61, -73}
, {-58, 4, 9}
, {-33, -4, -64}
, {-40, 39, -51}
, {22, -55, -11}
, {60, -23, -17}
, {4, -5, -44}
, {16, 64, 33}
, {-34, -34, -39}
, {-16, 31, 62}
, {32, -40, -26}
, {20, -58, -56}
, {58, -62, 59}
, {-25, -58, 13}
, {18, 68, -13}
, {56, 63, -77}
, {28, -37, 45}
, {74, 35, 26}
, {-39, 3, -51}
, {52, -29, 48}
}
, {{3, 11, -11}
, {33, -9, -57}
, {-39, -35, -28}
, {-48, -42, 30}
, {65, 1, 14}
, {0, 36, -7}
, {28, 61, 44}
, {-11, -38, -53}
, {33, -41, 20}
, {10, -66, 43}
, {39, 61, -64}
, {-31, 11, 1}
, {27, -40, 60}
, {-60, 24, -22}
, {-62, 43, 74}
, {-48, 43, 2}
, {59, -57, 53}
, {-45, 63, -55}
, {20, 69, 19}
, {27, 52, -11}
, {62, -29, -20}
, {-48, 29, -71}
, {-70, 24, 13}
, {27, -27, 39}
, {-34, 73, 72}
, {15, 66, -63}
, {-12, 15, 0}
, {-61, -1, -30}
, {-71, -35, -1}
, {44, -70, 22}
, {11, 33, -3}
, {25, -32, 2}
}
, {{29, 38, -54}
, {-24, -55, 19}
, {-40, -63, -8}
, {-41, -40, 36}
, {-23, 27, -5}
, {73, 25, 78}
, {19, -15, 0}
, {-55, 60, 62}
, {-40, -41, -59}
, {28, -67, 72}
, {-46, 28, -4}
, {35, 60, 0}
, {-30, -16, -72}
, {-9, -23, 36}
, {19, -29, 47}
, {21, 13, 3}
, {70, -34, -43}
, {-63, 22, -71}
, {-42, -67, 74}
, {-41, -41, -4}
, {72, 24, 2}
, {69, -3, 74}
, {-5, 52, -24}
, {-67, -59, 36}
, {60, -30, -6}
, {47, 28, 65}
, {24, -39, 76}
, {-51, -69, 49}
, {77, -3, -25}
, {16, 52, -44}
, {-3, -67, 28}
, {60, -60, 70}
}
, {{42, 14, -43}
, {-70, -39, -40}
, {-18, 28, 17}
, {42, -56, 14}
, {-27, -46, -46}
, {68, 72, -23}
, {12, -1, -24}
, {63, 1, -2}
, {-48, -25, -73}
, {71, 0, 12}
, {2, 58, -31}
, {-69, 70, 16}
, {58, -58, 30}
, {-7, -18, 53}
, {43, 55, 41}
, {-52, 41, -42}
, {65, 0, -40}
, {16, -5, 9}
, {-22, 61, 55}
, {10, -54, 45}
, {30, -6, -39}
, {-3, 73, 61}
, {-45, 13, 16}
, {24, 22, 53}
, {-36, 62, 17}
, {73, 62, 41}
, {32, -29, 35}
, {-43, -1, 32}
, {-55, 33, 41}
, {33, -61, 52}
, {-7, 49, -36}
, {40, -21, -5}
}
, {{-32, -16, 60}
, {-56, 21, 18}
, {7, 21, 35}
, {-48, -64, -68}
, {-53, -19, -15}
, {-18, -62, 7}
, {16, 51, 24}
, {-23, 11, 55}
, {-76, 70, -64}
, {29, -59, -27}
, {-2, 6, -21}
, {-65, -3, -34}
, {21, 74, -18}
, {60, -4, -10}
, {70, -29, -32}
, {-7, -1, -67}
, {47, -55, 49}
, {28, -56, -60}
, {69, -65, 51}
, {0, 5, 29}
, {69, 49, -49}
, {38, -26, 47}
, {-5, 9, -12}
, {22, -46, -54}
, {-68, -47, 78}
, {-53, -4, 40}
, {-37, 31, 38}
, {10, 7, -9}
, {46, 27, 41}
, {58, -48, -52}
, {51, -61, 34}
, {-47, 9, 59}
}
, {{33, 1, -11}
, {55, -44, 62}
, {49, -8, -59}
, {-8, -22, 22}
, {52, 5, -44}
, {-26, -31, 24}
, {62, -67, -49}
, {-37, 8, -69}
, {-77, -50, 15}
, {-22, 68, 18}
, {-28, -5, 60}
, {-43, -21, 17}
, {-8, -16, -32}
, {-31, 50, -40}
, {31, 22, -8}
, {0, -13, 77}
, {-8, -66, -66}
, {24, -65, -75}
, {-47, 21, 2}
, {11, 60, 53}
, {67, -26, 79}
, {75, 44, 49}
, {27, -41, -57}
, {29, -34, 17}
, {32, 10, 18}
, {-44, -27, 35}
, {-25, 53, -6}
, {70, -4, -26}
, {-19, -44, 57}
, {38, -8, -55}
, {1, -42, -63}
, {-35, -26, -41}
}
, {{70, -48, -12}
, {54, 52, -40}
, {31, 35, -36}
, {-50, 20, -8}
, {51, 16, -43}
, {37, -18, 2}
, {71, 26, -61}
, {-26, -9, -50}
, {-32, -31, -47}
, {-53, 16, -21}
, {-66, 29, -11}
, {-43, -37, 0}
, {31, 36, 27}
, {18, -36, -1}
, {-64, -32, -60}
, {35, 17, -35}
, {-12, -59, 49}
, {18, -65, 19}
, {0, -8, -35}
, {65, -22, -38}
, {-33, 0, -61}
, {15, 6, -21}
, {51, -15, 46}
, {34, -53, -56}
, {68, 1, -32}
, {4, -2, -8}
, {-52, -67, 19}
, {13, 49, -58}
, {11, 6, -46}
, {67, -49, 69}
, {-46, -15, -26}
, {-20, 34, -33}
}
, {{-12, -40, 15}
, {26, 73, 41}
, {-22, -22, -61}
, {23, 6, 44}
, {30, 34, -45}
, {42, -34, -43}
, {37, -52, -52}
, {1, -10, -39}
, {31, 35, 45}
, {-1, -22, -49}
, {-46, -19, -42}
, {21, 53, 4}
, {-25, 4, 60}
, {43, -64, -62}
, {70, -6, 5}
, {39, 71, 69}
, {20, 65, -25}
, {42, -14, 22}
, {-60, -70, -60}
, {-52, -13, 63}
, {-48, -21, -50}
, {-5, 69, -26}
, {19, -47, -15}
, {-22, 31, -27}
, {47, 27, 41}
, {67, -36, 17}
, {-32, 11, -32}
, {-23, -42, 21}
, {58, 12, 67}
, {47, -56, 44}
, {20, 44, -73}
, {73, -10, -73}
}
, {{-29, -25, -44}
, {59, 56, 7}
, {-22, -2, 56}
, {-18, 23, 47}
, {5, -3, -27}
, {-28, -51, 56}
, {-74, -7, -2}
, {-37, 41, -20}
, {26, 41, 67}
, {38, 60, -8}
, {-10, -24, -2}
, {-24, 14, -29}
, {45, -30, 22}
, {-38, 40, 64}
, {59, -42, 14}
, {-13, -58, 59}
, {-37, -26, 58}
, {30, 30, 13}
, {38, 38, 2}
, {-63, 50, -28}
, {65, -12, -26}
, {55, 29, -62}
, {-34, -3, 69}
, {70, 28, -3}
, {57, -2, -18}
, {67, -73, 72}
, {-50, -34, -41}
, {-49, 11, -57}
, {61, 1, 62}
, {-46, -32, 24}
, {69, 21, -18}
, {56, -34, -40}
}
, {{0, 51, 30}
, {23, -50, 22}
, {-18, 44, -21}
, {64, 55, -42}
, {0, 41, -8}
, {62, -49, -15}
, {11, 6, -28}
, {37, -28, -71}
, {-29, 2, -36}
, {35, 51, 3}
, {-47, 72, -16}
, {29, -22, 39}
, {-13, 52, 66}
, {-26, -7, 18}
, {46, -71, 44}
, {-63, 59, 44}
, {27, 65, 36}
, {40, -66, 25}
, {-68, 18, -70}
, {32, 21, -61}
, {-72, -16, 42}
, {-25, -59, 47}
, {24, -67, -25}
, {51, 9, 68}
, {64, 32, -32}
, {-70, 30, -40}
, {13, 45, -43}
, {-9, 0, -44}
, {27, 54, 29}
, {-21, -1, 58}
, {41, 68, 58}
, {-25, -18, 51}
}
, {{-4, 46, 53}
, {1, -43, -10}
, {24, -43, 16}
, {8, 55, -5}
, {-38, 69, -47}
, {-9, 19, 16}
, {-55, 19, 7}
, {5, -51, 61}
, {54, 38, 40}
, {-71, -58, -45}
, {-35, -48, 36}
, {-54, -8, -10}
, {51, -40, 73}
, {-37, 26, 34}
, {-68, -17, 4}
, {-43, -50, -3}
, {-51, -22, -11}
, {-9, -30, 11}
, {59, -42, -35}
, {-14, 36, -5}
, {-14, -30, -40}
, {-68, -22, -24}
, {-68, 1, 52}
, {-78, 6, 38}
, {-49, -14, 57}
, {42, -59, 33}
, {45, -73, -46}
, {-17, 20, 69}
, {-10, 40, -40}
, {5, -66, 32}
, {-21, 3, 10}
, {-63, -59, 20}
}
, {{19, 60, 44}
, {71, 45, -56}
, {41, 21, -48}
, {12, 3, -43}
, {-50, -9, 36}
, {49, 67, -47}
, {73, 48, -22}
, {-44, -41, 36}
, {45, -57, 4}
, {-19, -58, 50}
, {5, 61, 48}
, {-58, 15, -25}
, {-46, 45, -64}
, {45, -61, 0}
, {25, -36, 42}
, {13, -43, 42}
, {-61, 40, 56}
, {-48, -20, 52}
, {5, -52, 29}
, {64, 37, -29}
, {-40, -47, -35}
, {65, -5, 3}
, {-42, -21, -6}
, {46, 47, -37}
, {12, -72, -38}
, {61, 71, 10}
, {13, 65, -62}
, {2, -12, -5}
, {54, 15, 58}
, {31, 45, -53}
, {-71, 72, -41}
, {-64, -30, -22}
}
, {{15, -37, -46}
, {44, -42, -59}
, {9, -7, 46}
, {65, 20, -19}
, {29, 7, -24}
, {-59, 22, 30}
, {-22, -17, -39}
, {-24, -41, -20}
, {15, 19, -35}
, {34, -1, -61}
, {59, -48, -19}
, {-6, -49, 55}
, {4, 6, -41}
, {-17, 72, -15}
, {35, -23, -17}
, {-3, -30, -39}
, {-46, 46, 39}
, {60, 36, 55}
, {26, -29, -38}
, {-24, -61, 73}
, {-38, 57, 32}
, {-56, 62, -4}
, {-47, 53, -40}
, {-29, 67, -67}
, {-19, 61, 9}
, {-44, 62, 20}
, {-7, 25, 26}
, {53, -56, -33}
, {54, -68, 49}
, {4, 3, 14}
, {60, 20, -63}
, {5, -74, 63}
}
, {{72, -23, -25}
, {39, 53, 5}
, {21, -22, 9}
, {25, 25, 59}
, {65, -57, -69}
, {43, -39, 7}
, {-12, -47, 13}
, {-30, 65, 44}
, {-50, 77, -4}
, {-29, 59, 1}
, {-7, 53, 6}
, {46, -71, 50}
, {-2, 15, -5}
, {-61, -53, -50}
, {15, 2, 35}
, {-32, -38, -27}
, {-74, -19, 66}
, {-38, -61, -45}
, {71, 62, 5}
, {46, 0, 33}
, {45, 61, 19}
, {-20, 43, 62}
, {64, 15, -65}
, {7, 40, -2}
, {57, 32, -11}
, {45, -52, -58}
, {-65, -1, -45}
, {72, 29, -7}
, {35, 26, 14}
, {7, -41, -28}
, {5, -48, -4}
, {-70, -16, 14}
}
, {{74, -29, 2}
, {3, 14, 4}
, {-23, 10, 14}
, {-68, -65, 39}
, {24, 26, 63}
, {0, -59, -25}
, {1, -47, 19}
, {14, 6, 23}
, {-32, 1, -7}
, {-29, 67, 49}
, {45, 73, 42}
, {-4, -52, 60}
, {34, -68, -48}
, {67, 56, 32}
, {70, 35, -27}
, {-16, -45, 63}
, {-49, 0, -56}
, {-38, 27, -7}
, {-32, -10, 41}
, {38, 3, 2}
, {-61, -26, 52}
, {39, -75, -59}
, {36, 53, 56}
, {-3, -35, 64}
, {8, 16, 38}
, {-5, -23, -27}
, {-21, 3, -30}
, {6, -21, -10}
, {-22, 17, 32}
, {-56, 67, 71}
, {-36, -8, -57}
, {31, 23, -61}
}
, {{-40, 69, -8}
, {13, 31, 39}
, {26, -18, 49}
, {-18, -5, -16}
, {-53, 4, 47}
, {-19, -76, 30}
, {69, 59, -24}
, {-47, -16, 23}
, {45, -12, 56}
, {-4, 71, -20}
, {16, -62, -34}
, {60, 0, 20}
, {-66, 65, 53}
, {52, 75, -51}
, {33, 1, 60}
, {-49, -60, 30}
, {22, -13, -50}
, {-62, -25, -26}
, {-45, 50, 8}
, {58, -46, 43}
, {47, 1, -3}
, {32, -43, 65}
, {75, -5, -60}
, {-7, 0, -3}
, {-33, 72, 41}
, {-2, -3, -16}
, {57, 52, -8}
, {12, 25, 12}
, {42, -3, -64}
, {-35, 10, -71}
, {51, -53, -16}
, {11, -72, 68}
}
, {{66, 7, -56}
, {0, -23, -32}
, {-13, 33, -7}
, {-8, 5, 21}
, {0, 13, 19}
, {63, 43, 12}
, {-10, 38, 18}
, {9, -51, 19}
, {-55, 9, -57}
, {2, 32, 41}
, {51, -36, -73}
, {-63, -68, 9}
, {-2, -16, -15}
, {25, -63, -24}
, {45, -61, 54}
, {53, 32, -49}
, {-56, 59, 70}
, {-32, 60, 30}
, {7, -32, -42}
, {-57, 59, 35}
, {41, -27, 20}
, {8, -66, -26}
, {18, 25, 49}
, {49, 12, 50}
, {27, 22, -26}
, {-44, -10, -63}
, {7, -19, -5}
, {46, -44, 43}
, {50, 28, 61}
, {60, -46, -15}
, {-40, 24, 28}
, {-1, 48, -44}
}
, {{19, 75, 34}
, {70, -30, -65}
, {-47, 3, 45}
, {-58, -68, 64}
, {8, -69, 22}
, {7, 73, -16}
, {-5, 7, 42}
, {35, 65, -20}
, {53, -2, -7}
, {-38, -22, 62}
, {-55, 67, -28}
, {65, 38, -37}
, {68, 48, 71}
, {-18, -68, -10}
, {-43, 36, -25}
, {77, 50, 49}
, {21, 65, 68}
, {21, -20, -52}
, {59, 39, -47}
, {-2, 72, -62}
, {72, 11, 20}
, {-30, 56, 5}
, {5, 48, -67}
, {34, 34, -27}
, {18, 60, 65}
, {24, 57, 55}
, {-39, 74, -70}
, {-63, -62, 37}
, {70, -50, 69}
, {62, -50, 29}
, {65, 53, -24}
, {-68, -39, 54}
}
, {{33, -26, 71}
, {45, 30, 63}
, {69, -21, 35}
, {-61, 35, 37}
, {-6, 17, -43}
, {7, 65, 73}
, {30, -30, 65}
, {-17, 58, -28}
, {20, 24, -41}
, {70, -35, 24}
, {0, -17, -37}
, {-48, 60, 11}
, {-42, 58, 71}
, {-51, 43, -25}
, {50, 42, -73}
, {-29, 29, -55}
, {-44, -19, -4}
, {-64, 38, 0}
, {0, -69, 9}
, {-7, 53, -34}
, {-69, -45, -45}
, {-14, 72, -33}
, {-11, -1, 9}
, {-55, -11, 33}
, {-13, -6, 3}
, {-30, 58, -16}
, {6, -44, 45}
, {-69, -10, 63}
, {33, -23, -17}
, {-16, 58, 42}
, {-29, -72, -29}
, {47, 44, -41}
}
, {{42, -35, -63}
, {44, 0, 20}
, {-2, -7, -46}
, {14, -55, 55}
, {-39, -29, -61}
, {-50, -7, 32}
, {-37, 37, -3}
, {-21, 49, 17}
, {54, 28, 51}
, {39, -4, -42}
, {-16, 11, 44}
, {63, 63, 9}
, {-61, 6, 43}
, {27, -47, -40}
, {52, 6, -35}
, {60, 41, 40}
, {42, -6, 58}
, {4, -41, -10}
, {19, 16, -66}
, {56, -6, 23}
, {-60, -24, 48}
, {-38, 15, 13}
, {8, -41, 63}
, {66, -22, 33}
, {-48, -31, 57}
, {-18, 46, -31}
, {53, -3, 57}
, {-26, 36, 73}
, {-7, -25, 20}
, {26, 55, -4}
, {-36, 36, -77}
, {-13, -76, -45}
}
, {{60, 7, -8}
, {-61, 2, -57}
, {-23, -35, 32}
, {28, -15, -72}
, {-23, 74, 49}
, {-31, 76, 18}
, {-30, 5, -70}
, {-1, 38, -27}
, {-58, 50, -40}
, {-17, -44, 51}
, {5, -24, 51}
, {-54, 40, -28}
, {-61, 52, 0}
, {11, -67, 69}
, {76, 72, -64}
, {4, 0, 36}
, {36, -55, 32}
, {-57, 1, -27}
, {41, -31, -36}
, {-1, -45, -45}
, {-36, -16, 10}
, {-73, -18, 50}
, {33, 29, 46}
, {46, -7, -27}
, {51, 3, -1}
, {-21, 17, -53}
, {37, 14, -12}
, {13, -46, -17}
, {52, -20, 7}
, {64, 74, 30}
, {-13, 2, -33}
, {66, -30, -42}
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

typedef number_t max_pooling1d_29_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_29(
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

//typedef number_t *flatten_5_output_type;
typedef number_t flatten_5_output_type[OUTPUT_DIM];

#define flatten_5 //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_15_output_type[FC_UNITS];

static inline void dense_15(
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


const int16_t dense_15_bias[FC_UNITS] = {6, 2, -1, 1, 5, 9, 2, 6, 0, -3, 4, -1, -4, -2, 3, 0}
;

const int16_t dense_15_kernel[FC_UNITS][INPUT_SAMPLES] = {{24, 45, -9, 37, -7, 24, 5, 25, 9, 29, -36, 39, 32, -41, 11, -13, -20, 28, -24, 12, -1, -8, -28, -20, -25, 13, -30, 41, 41, 27, -35, 40, -30, -28, 41, 39, -11, -15, -3, 22, -39, -37, 29, -38, 19, -4, -42, 6, 41, 7, 38, -25, 17, -23, 38, -35, 10, -11, 5, 2, -26, -25, -41, 40, -36, -32, 24, 40, -26, 17, -22, -2, 12, -22, -14, 44, 34, 15, 24, -16, 14, -23, 31, 4, -7, 18, -15, -24, -21, 18, -8, -34, 7, 10, -1, -11, -29, 22, 17, 18, 32, 38, 33, 27, -14, 24, 34, 9, -37, -17, 28, 14, -28, 40, -18, -11, 36, -42, -40, -25, -19, -7, 33, -41, -32, -32, -26, -36, -25, -37, -7, -23, 17, 19, -34, -12, -41, 24, -40, 17, -5, -37, -30, 17, 19, 13, -10, -28, 7, 6, -20, -4, -34, 8, 22, 39, -30, 40, 24, 4, 38, 12, -9, -43, 23, -10, -1, 5, -27, 39, 22, -31, -6, -11, 27, -24, 14, 39, -28, 21, -27, 5, -30, 19, 36, -3, -37, -36, -29, 39, 4, -8, 39, 20, 22, 28, -24, -29, 36, -38, -4, -5, -11, 0, -26, 3, 21, -12, 0, 10, 4, 38, -32, 38, 38, -33, 22, 19, -6, -35, -2, 29, -39, -34, -35, 38, -32, 29, 3, -40, -27, 9, 25, -14, -11, 20, 16, 7, 2, -2, -22, 11, 41, 12, -20, -31, -6, -37, -38, -2, 40, 34, -10, -5, 29, 19, -11, 26, 41, 9, 2, 20, 7, -34, 17, -14, -5, 20, -34, 25, -4, 4, 18, -21, -11, 24, 26, -14, 32, 41, 15, -23, -27, -20, -15, 39, -4, -13, -38, -6, -25, -18, 34, -43, -2, -10, -34, 22, -14, 25, 19, 6, -6, 30, 29, -2, -29, 30, 22, 28, -6, 14, -7, 10, 7, -4, 25, 23, 25, 37, 16, 4, -29, 22, 28, 15, 1, 35, -29, 33, -28, 3, 38, 21, -15, 27, -8, 16, 41, -39, -13, 4, -23, -30, -16, 15, -40, 25, -23, 21, -16, -35, 14, -2, 16, 35, 17, -39, 13, -2, 30, -24, 43, 12, 18, 43, -10, 16, 43, -38, -33, 14, -32, 27, 17, -17, -17, 11, 38, -2, -23, -36, -22, 20, 1, -13, -33, -3, 21, 40, 39, 23, 27, -31, 24, -10, 23, -2, 32, -7, 35, -10, 16, 35, 15, -40, -36, 4, 24, 15, -9, -39, -22, -29, 14, -8, 0, -38, -27, -26, 39, -28, 40, 12, -9, -5, -34, 0, -26, -35, 16, -27, -2, -2, -9, 2, 35, 4, -3, -32, -32, -39, 45, -4, -10, 37, -22, -25, -2, 21, -15, -43, 15, -39, -21, -29, -14, 38, 22, -40, 24, 29, 19, -25, 9, -26, -19, -19, -16, 44, 25, 0, 42, 6, -7, 9, -32, 32, -6, -20, -29, -13, -30, 35, 25, 0, -42, -34, 38, -23, -36, -33, 13, -13, 42, 34, 39, 8, -22, 27, -18, 25, 32, 16, -23, -39, 32, -24, -41, 38, -19, 26, 15, 6, 11, -28, -17, -39, 36, 5, -21, -25, 36, 3, -40, 31, -25, -1, 29, 38, 24, -17, -18, -6, 38, 39, 33, -38, -18, -23, -43, 32, 38, 42, -8, 20, -3, -1, 0, 30, -31, -1, 40, 14, -9, -15, 43, -1, 16, -37, -32, 9, -12, 3, -41, 23, 34, 39, -39, -30, -5, -15, -1, -1, -4, 23, 42, 41, -7, 21, -11, -14, -15, 30, 35, 0, -11, 24, 30, -24, -14, 23, 40, -39, 33, -6, -14, 1, -8, 11, 26, 11, -15, -4, 14, -22, 7, -6, -4, -8, -13, -20, 1, -17, 37, 26, -11, 17, 39, -31, 12, -5, 42, -1, -3, 34, 15, 12, 7, -41, 34, 14, -1, 1, -37, -1, 43, -35, -25, 32, 1, 27, 15, 13, -39, -6, 18, 4, 29, -26, -4, 40, -17, -29, -8, 34, 33, -8, 43, 14, 1, -1, 4, 29, 34, -1, -19, -32, -8, -1, -42, -16, -14, 31, 44, 23, -34, 40, -20, 29, 6, 19, 28, 0, -2, 30, -24, 36, 30, 17, -18, 39, 20, 19, -41, -23, 39, 0, -29, -33, 20, 35, -36, 29, 41, 28, -1, 15, -40, -34, 3, 22, 11, -27, -9, -21, 39, -16, -36, 24, 13, 28, 11, 32, -26, -37, -29, 27, 36, 31, -20, -22, -6, 3, 17, -15, -12, 11, 18, -26, -14, -22, -10, 36, -29, -26, -26, 7, 29, -12, -20, -9, -30, -40, 36, 1, 31, 1, 12, -7, 27, -24, 3, -16, 17, 9, 39, -37, -29, -17, -12, 41, 38, 8, 26, 43, -3, 17, -13, 33, 21, 23, 14, 32, 18, -21, -39, 12, 1, -38, -8, 37, 7, -36, 27, 21, -18, -16, 0, 1, 37, -3, 19, 24, -8, -21, 15, -27, 19, 2, -13, 22, 4, 4, 20, 34, 28, 39, 0, 30, 5, -36, 5, -31, -28, -29, -18, -24, 30, 21, -14, -25}
, {-13, -33, -2, 8, 41, -27, 26, -22, 35, -37, 23, -29, 24, 26, 31, -15, 27, 14, -7, -15, 18, -26, -39, -24, -13, -36, -3, -12, 31, -31, 1, -12, -8, -42, -32, 4, -38, 16, 2, -9, 13, -6, 6, 32, 3, -26, -22, -37, 25, -13, -13, 24, 33, -12, -20, 8, -35, -12, 0, 10, 39, -17, -10, -36, -26, 6, -2, 18, 17, 5, 33, 27, -15, -17, 1, -23, -5, -2, -12, 14, 9, 2, 14, 2, 12, -9, 7, -39, -41, 2, 29, 23, 34, -17, 31, 15, 19, 10, -38, -3, -36, 34, 22, -25, -10, 2, 32, -27, 19, 26, -24, 18, -24, 16, 28, 16, 14, 22, -8, 16, 22, 23, 7, -14, 39, -26, -34, -34, 22, -36, 6, 34, 9, 22, -31, -18, 5, -13, 23, 43, -34, -11, 40, -31, 29, 42, 25, 37, 7, 40, 12, 12, 29, 39, -36, -37, 2, -12, -2, -22, 40, 3, 15, -37, -21, -19, -7, 6, -20, -15, 10, -34, -8, -6, 9, 14, 33, 30, 37, 13, 3, 32, -14, 37, -40, -13, 11, -26, 13, 42, 9, 29, -8, -26, -33, -25, 5, -5, -9, 30, 16, -12, -21, -40, 35, 5, 17, -45, -25, 23, -25, -1, 32, 2, 16, -1, -37, -38, -21, -32, -33, -32, 24, -29, -22, -34, -32, -16, -18, -20, -5, -24, 12, -21, 3, -20, 5, 8, 39, 31, 11, 16, 12, 0, 37, -37, -34, -22, -37, -28, 3, -31, -27, -3, 21, -40, 0, -6, 9, 39, -35, -29, -1, 38, 29, 18, -40, -39, -29, -15, 38, 39, 37, -14, 41, 35, -27, 33, 8, 20, -20, -2, 5, 6, -30, 38, 2, 3, -32, -15, -7, -19, -39, 39, -36, 2, -23, 30, 37, 4, 13, 15, -22, 8, -1, -21, 28, -5, 22, -40, 3, 25, -36, 7, 17, 39, 19, 32, -31, 5, 17, 7, -16, 2, -1, 16, 17, 27, -25, -13, 29, 17, 15, -8, 9, 44, 6, -8, 35, -23, 8, 44, -31, -35, 4, -14, 6, 26, -23, -11, -17, 30, 7, 18, 21, -2, 44, -39, -20, 16, 12, 21, 30, 19, -38, 11, -7, 15, 39, -34, 17, -13, 25, 18, -21, 40, -30, 39, -27, -21, -22, -3, 26, 22, 42, 17, -30, -14, 25, 41, 0, 44, 38, -39, -37, -21, 17, -22, 8, -26, -9, 30, -36, -13, -34, 27, 25, 18, 15, 28, 27, 37, 11, 26, 35, 6, -33, 44, -22, -31, 5, 8, 0, 11, 2, -37, 20, 10, 8, -14, -6, -9, -1, 28, -4, 25, -3, 15, 36, 33, -28, -32, 40, 9, -14, -34, 21, 19, -33, 21, 30, 19, 15, 28, -5, 23, -23, -37, 4, 36, -25, 0, -17, 13, 25, -15, 18, 33, 32, 22, -38, 45, 8, 25, 9, 8, -18, 31, 43, -1, 39, 28, 44, -12, 23, -4, 39, -24, 18, -1, -36, 8, -37, 32, -21, -4, -12, -17, -35, 20, -6, 19, -4, 14, 42, 37, -31, 20, -41, -35, -9, 36, 43, 39, -8, -22, 38, -13, 0, -3, 35, -13, -6, -32, -34, 23, -27, -5, 10, -13, 34, -19, -6, -8, -20, 30, -3, 42, 0, -7, 4, 30, -31, 13, 12, 29, -16, 29, 41, -34, 42, 21, -7, 3, 28, -18, 43, 34, 23, 24, 7, -30, 3, 35, -18, 19, 13, 24, 29, -2, 19, 15, 23, -4, -15, 17, -16, -1, -31, -1, -6, -36, 39, -36, 41, 23, -28, -33, 38, 27, 41, 20, 42, 1, 30, -16, 37, 20, -22, -24, -19, -20, -6, -36, -22, -35, 2, -5, 13, -24, -37, 22, -18, 3, 9, 5, 9, -16, -27, 15, 43, 5, 38, -22, 43, 3, -27, -44, -37, -22, 37, -37, 37, -22, -18, 10, -5, 15, 15, 0, -23, 29, -21, -39, -24, -6, -2, 11, 36, 37, 11, 22, 18, 43, -4, -4, -7, 2, 37, -19, -40, 0, 3, 45, 33, -35, -8, 1, 22, 41, 29, 32, -9, -9, -18, 22, 26, -4, -27, -32, -2, -38, 19, 28, -30, 9, 13, -26, 15, -15, 31, -30, 12, -25, 25, 5, 11, 22, -39, -3, 4, -12, 0, -10, 20, 1, -19, -4, -10, 19, 0, -32, 25, -34, -33, -29, 0, 1, -4, -26, 37, -10, -28, -40, -8, 9, -30, 14, 11, 5, -2, 21, 40, 30, 2, 31, 42, 12, -2, 11, -2, 27, -25, 31, 36, 36, 29, -13, -31, -9, -35, 1, 5, 32, 33, -8, -39, 44, 41, -33, 9, 42, -14, -10, -6, 7, 0, -26, 0, 33, -18, 29, 29, 32, -18, -20, -9, 28, 35, 0, 1, -2, 20, -21, 10, 41, -12, -37, 1, -41, -37, 24, 9, -34, -17, 8, -33, -14, -8, -3, -19, 45, -34, -7, 1, -10, 20, -37, -21, 30, -8, -3, 17, 10, -9, -11, 38, -40, 12, 21, 12, 11, -23, -26, -3, -25, -32, 10, 0, 36, 29, -34}
, {30, 21, -24, 7, -28, 35, -1, -40, -21, -16, -9, 37, 39, 33, -24, -7, 38, 21, 13, -19, 31, 0, 22, 6, 39, 30, 37, 34, 34, 13, 9, 27, 33, 0, 15, 35, 33, -38, -39, 37, 14, -2, -37, 3, 39, -21, -26, -27, 14, -23, 40, 32, -27, 5, -42, 29, 13, -27, 16, 28, 13, 26, -4, -22, 28, -5, -9, -34, -16, 37, -12, 25, -8, 30, 0, 0, 26, 12, 22, 9, -20, 38, 28, -36, -8, 29, 20, -40, 36, 23, -28, -31, 8, 6, 31, -5, 40, -19, -28, 30, 17, 1, 23, 20, -40, 11, 29, 38, -40, -28, -23, -8, -25, 38, -1, -17, 31, -41, -27, -38, 7, 9, -37, 37, 26, -26, -31, 40, 1, -42, -2, -16, -42, 8, 25, 16, 31, 28, -18, -9, -37, 3, 30, 33, -1, -41, -42, -33, 31, 5, -16, -5, 32, -1, -42, 0, -30, -32, -25, -18, -31, -22, -29, 0, -9, -41, 36, -3, -15, -4, 38, -10, 38, 42, 40, 22, -24, 29, 18, -15, 32, 0, -27, 12, -21, 24, -1, -43, -3, 12, 30, 4, 37, 31, -3, 37, 55, -2, 40, -32, 45, -28, -42, 21, -31, -6, 42, 19, -17, -33, 17, 37, 30, 41, 26, 22, -37, 33, 9, -27, -9, 27, 31, 22, -40, -1, 30, 30, -30, 39, -9, -21, 18, -40, 14, -35, -25, 25, -7, -27, -30, 13, 1, -3, -35, -13, -1, -14, -12, 31, -38, 0, -8, -24, 1, -40, 28, 27, 38, 36, 36, 37, -32, -26, -35, 21, -11, 30, -6, -27, 23, 28, 12, -5, -17, 34, -23, -1, 9, 18, 3, 10, 3, 6, -10, -5, -12, 19, 0, -35, -12, -20, 14, 27, -4, 36, -22, -19, -21, -12, -28, -4, -21, 36, 32, -20, -17, -10, -8, 38, 35, -5, 9, -42, -11, 10, 12, 43, 4, 12, 17, 3, 21, -41, -23, 3, 5, 6, 4, -41, -15, 9, 1, -6, -31, -40, 22, -7, 20, -12, -27, 36, -23, -43, 7, -8, -10, -24, 41, -13, -36, 39, 41, -19, 40, -44, -13, -8, -2, 32, -17, -36, -32, -39, 27, 11, -17, 40, 11, -22, -33, -13, -38, -13, 33, -35, 13, -1, 9, 22, -45, -2, -1, 35, 18, -30, 23, -4, -21, -45, 36, -3, -6, -26, -34, -6, 11, -46, -40, -29, 17, -1, -21, -2, -37, -20, -40, 4, 38, 13, -31, 30, 2, -12, 30, -14, 16, -18, 35, -33, 22, 31, -24, 6, 39, 30, 21, -27, -5, 22, -13, 29, -2, 27, -9, 16, 7, -30, -4, 42, -9, 31, 31, -23, 0, -40, 18, -17, 37, 42, -5, 3, 20, 32, 2, -5, 40, -4, 16, 6, -24, 41, 25, 1, 0, 21, -40, -36, 16, 40, 6, 4, -4, -40, -24, -44, -30, 33, -38, -41, -6, -44, 16, 16, 25, 37, -1, 18, 42, 9, 14, 14, -25, 40, -13, 36, -10, -2, 24, -31, -26, 27, -23, 35, -21, -34, -7, -16, -8, 25, -28, 39, 25, 33, 29, -44, -32, 27, -6, 37, 31, 31, 41, -18, -4, 32, 33, 1, -14, -10, 25, 11, 21, -33, -1, 30, -19, 19, 31, 4, 34, 20, 2, 2, 17, -41, 13, 18, 38, 14, 20, 37, -8, 15, 3, 21, 6, 9, 16, 31, 29, -16, 24, 13, -29, -26, 8, -5, 36, -34, -18, -2, 26, 9, -4, -6, -5, -10, -5, 44, -34, -33, -13, 16, 18, -32, -12, -36, 21, 34, -11, 38, 7, -17, 13, -21, 11, 22, -28, 22, 22, 23, -7, -17, 32, 25, -1, 36, 33, -10, 17, -41, -39, -9, -33, -30, 25, 19, -32, -34, -35, 30, -38, -7, 0, 11, -40, -17, -34, -26, -37, 30, -5, 6, 32, -10, -35, 20, 42, 34, -29, -2, -39, -22, 25, 26, -17, 14, -38, 41, 29, 6, -26, 16, 30, 39, 40, 3, 40, 39, 21, -16, -31, 24, 36, 28, 36, 4, 37, -8, 41, -28, -9, -39, -31, 14, 27, -31, 29, -29, 6, -32, -17, 29, 33, 8, -22, 33, 37, -20, 49, 44, -34, -11, 36, 22, -14, -36, -18, -35, 20, -22, 19, 33, -12, 1, -21, -25, -23, -11, 2, 27, -44, -8, -38, 9, 39, 20, 29, 23, -26, -33, 29, -6, 8, -23, 33, -6, 24, -32, 8, -22, 5, 38, -29, -40, -4, 33, 6, -36, -8, -11, 39, 39, 31, 7, -28, -20, -29, -44, -3, -29, 35, -16, 36, -13, -37, -6, 12, -11, 0, -13, 21, 3, 14, -12, 36, -30, 13, -30, 33, -5, -8, 37, -12, 26, -24, -24, 21, -29, 0, 22, -40, 15, 36, -32, 19, 43, 8, 20, -3, 13, -31, -11, -24, -31, 9, -32, 7, 12, 15, -35, -11, -43, -16, -23, -12, -7, -1, -17, 30, 2, -9, -13, 38, -21, 4, -14, 36, -2, -16, -32, 39, -16, -2, -9, 0, 27, -41, -27, 0, -24}
, {-35, 14, -35, 32, -29, 10, -34, 20, -17, -16, -36, 31, -12, -21, 35, -15, 40, -10, -2, -25, 37, 1, 1, -40, -36, -37, 9, -15, -42, 2, -21, 15, 13, -15, -36, -33, -19, 16, 31, -33, 45, 36, -11, 30, -33, -7, -26, 11, -36, -33, 40, -41, 2, -18, -37, -29, 36, -4, 46, 38, 12, -4, 46, 18, 41, -9, 33, 2, -41, 26, 32, 14, -35, -27, -36, 10, -38, -26, 22, -22, -43, 28, 1, -20, -37, -34, -19, 26, 8, -40, 2, 3, 33, 36, 17, -1, -30, 15, -9, 31, 24, 27, 18, 18, 1, -28, -4, -1, 33, 11, -33, 9, -34, -20, -24, 4, 21, 31, 0, 1, 0, 5, -18, 29, -7, -12, 20, -21, 17, 44, 6, 27, 18, 26, 16, -22, 28, 32, 15, 13, -20, 0, -2, -7, 30, -34, -12, -26, -11, -4, 37, -27, -22, -16, -23, 19, -23, 46, -28, 39, 13, 43, -5, 9, 0, 46, 22, 19, -40, -18, 7, 23, 6, -20, -7, 0, -38, -9, 0, 31, -22, 40, -37, 32, -16, 35, 20, 1, 0, 29, 17, -26, 42, -40, 23, -33, -19, 21, 16, 33, 33, -13, 18, 33, -19, 16, 28, 33, 20, -41, -21, -19, 38, 29, -11, 32, 6, 40, 31, -18, 0, 22, -33, -20, -36, 9, -23, 43, 30, 27, 6, -26, -7, -37, 12, 25, 13, 26, 26, -21, 11, 7, 10, -18, 25, 27, -26, 35, -38, 9, -10, 41, 41, -31, 0, 22, -26, -17, -13, -42, -13, -29, -16, -16, -18, -13, 3, -23, -9, 22, 35, -17, -14, -4, 36, -33, 39, -20, 20, 12, -9, 28, 30, 39, -37, 22, 33, -18, 39, 16, -7, -17, -2, 27, 25, 12, 10, -6, -32, 37, 35, 41, -14, -4, 1, 9, 21, 18, 31, -28, -35, -4, 48, -30, 6, -34, -3, 3, -30, -38, 34, 37, -10, 37, 24, 25, -38, 10, 10, -35, 44, 8, 19, 27, 27, 5, 38, 19, -26, -10, 29, 38, 8, 42, 40, 46, 34, 12, 45, -25, 45, 26, -35, -35, 46, 16, 0, 37, 43, -35, 35, 33, -13, -13, -42, -17, -5, -13, 31, 14, 10, -14, -3, -38, 23, 19, 0, -39, 31, -5, -26, 14, -31, -23, -18, 12, 34, 14, -24, 47, 3, -32, 18, -5, 43, 18, -30, -21, -1, 26, -9, 0, 13, -13, 40, 2, 21, -36, -28, -7, 3, 30, 27, 44, -2, 0, -11, 24, -28, -13, 12, 14, 8, 30, 36, 3, 41, -29, 8, -3, 20, -21, 40, -38, -13, 7, 28, 41, -37, 28, -15, 44, -9, 34, 4, -28, 15, 35, -29, -14, -26, -18, 34, 30, -3, -3, 36, 9, 30, 29, 9, 33, -31, 30, 0, 7, 17, 0, -27, -28, 39, 6, 44, -15, 32, 14, -18, 9, -14, -22, 2, 35, 16, -17, 22, -11, 34, 21, 34, 30, -26, 5, 35, 30, -10, -15, -31, -11, 6, 28, 26, 19, 19, -5, -21, -1, 8, -8, -37, -6, -4, 32, -37, -15, 46, 10, 30, 39, -27, 0, -25, -6, -4, 5, 35, 17, 30, 11, 22, -40, -32, -32, 22, -12, -16, 17, -8, -8, -20, 12, -29, -22, -40, 23, 20, -13, 26, 27, -28, 32, -6, -22, -6, 23, -2, 41, -39, -11, -28, 12, 40, 21, 18, -15, -22, -25, -4, -18, 0, -12, -6, -1, -17, 35, 27, -13, 14, -15, -26, 5, -21, 36, -29, 5, -22, 38, 19, 11, 1, 2, 6, 27, -28, 29, 23, 9, -10, 39, 14, 35, -7, 11, -3, -31, 7, 9, -6, -5, -10, -11, -5, -20, -9, -30, -32, 31, 30, -15, -15, 14, -10, -30, -17, -29, 39, -8, 25, 28, 7, 26, 23, 18, 16, -1, 0, 9, 34, 38, -25, -37, -8, -16, 31, 36, -20, 42, 38, 20, 14, 41, -22, -34, 40, -20, 22, 3, -40, -3, 6, 14, 27, 37, 31, -8, 3, 4, -21, -10, -3, 19, 25, 8, 17, 16, 28, 28, -32, -34, 37, 24, -26, 18, 7, 24, -17, 20, -26, 15, 6, -31, -22, 17, 35, 37, 28, -18, -13, -29, 9, 6, -3, -31, -36, 16, -22, -31, 24, -31, 2, -22, -23, 25, 38, 19, 28, -26, 27, -39, 17, -2, -25, -5, -27, -11, 5, 3, -8, -10, 43, 9, -31, -33, 18, 23, 38, 26, 16, 38, 12, -20, -37, -9, 25, 11, -16, -2, 3, 23, -4, -36, 34, 39, 24, 33, 1, 23, 26, 0, 17, -33, 19, 6, 0, -39, -11, 5, 36, 20, 6, 28, -21, 36, 21, 2, 14, 16, 19, -3, -6, -39, 14, 17, 14, -15, -30, 31, -1, -19, -42, -3, -34, 7, 25, 38, -13, 1, 21, 5, 16, 17, 45, 4, -28, 37, 35, -39, 32, 11, 24, 39, 6, 19, -25, -1, 25, -35, 37, 38, 36, 0, -21, 33, 22, 9, -21, -15, -40, -38, 12, -6, -25, 16}
, {36, -20, 39, 45, 31, -31, -31, -33, 19, -3, 36, -5, 15, 21, -17, -4, 0, 6, -35, 19, -7, -4, -33, 30, 9, -29, -24, -14, 42, -2, -36, -12, -10, -7, -31, -31, -40, -17, 24, -38, -8, 36, -8, 1, -37, 25, 18, 0, 9, -25, -39, 14, -3, 42, -12, 4, -14, -22, 8, 6, 28, 31, 15, 19, -26, 15, -7, 27, 6, 2, 3, 22, -24, 4, -39, -35, 12, -31, 44, -6, -28, 41, 25, 37, 8, -21, 11, -11, -8, -21, -16, -7, 24, 29, -12, 20, -31, -35, -36, -36, -42, 0, -38, -16, 27, 10, -20, 13, -9, 0, 15, 42, -19, 7, 43, -17, 31, 28, -35, -15, -8, -8, 18, 9, 10, 27, -30, 15, -11, 24, 18, 19, 15, 2, 24, -8, -2, 0, -12, -13, -13, -33, -24, -23, -25, 36, 0, -38, 1, 37, 10, -7, 39, 26, -30, 17, -8, 40, 10, 25, 2, 49, 40, -36, -12, 18, 4, -5, 23, 37, -32, -21, 1, 17, -15, -15, 32, -1, 23, -24, 3, -35, -8, -38, -31, 36, 12, -29, -11, -10, 11, 24, -44, 32, 4, 7, -12, 14, 21, 0, 9, 32, -27, 36, 36, -35, 37, -24, -26, -3, 38, 9, -38, -37, 34, 6, 36, -5, -19, -32, -8, -24, 30, -28, -12, 11, -26, -35, -25, 21, -4, -18, 32, 37, -28, 24, -22, -23, -34, -23, 8, -7, 16, -24, 0, 8, -25, 41, -8, 33, 19, 9, 35, -18, 42, 15, 40, 13, -22, 1, 35, -1, -7, -41, -16, -32, -21, -23, -9, 32, -2, -21, 23, 39, 44, -33, 26, -28, -8, 10, -16, -2, -32, 25, 6, 15, -33, 40, -4, 16, 32, -6, -29, -29, -15, -6, -18, 20, -27, -42, -37, -21, -36, 38, 1, 32, -40, 1, -19, 15, -4, -14, 36, 31, -25, -1, -17, 34, -40, 9, -21, -38, 2, -10, 33, 17, 19, 12, -27, -18, 45, 19, 15, -11, 0, 17, 43, -40, -29, -34, 28, 5, 11, -19, -41, 13, 25, 37, 41, -23, 35, 16, -6, 18, -17, 41, 33, -37, -25, -45, -23, -46, -11, 37, -3, -7, 45, 0, -23, 8, 6, -40, 12, -38, -6, 30, 43, 24, -21, 32, -9, 4, -9, -14, 44, 34, 45, 22, -28, -6, -5, -9, 1, 33, -42, 2, 2, 25, 30, 36, 12, 28, 24, 42, 19, 2, 24, -33, 5, 19, -21, -34, -16, -12, -28, -34, 40, 26, 34, 20, 44, 7, 3, -27, -13, 25, 10, -23, -39, 40, 26, -27, 28, -17, 33, 15, 5, -15, -12, -10, -29, 45, 22, -26, -10, 38, 12, 30, -4, 29, -27, 36, 20, -34, 14, -23, 26, 6, 21, -44, 6, -28, -19, 38, -29, 40, -31, 40, -31, -28, 24, -12, -7, 35, -9, 46, -16, -5, 14, -29, 11, -24, 17, 29, 0, 41, -2, -41, 19, 33, 16, -33, 39, 13, 18, -38, -1, 32, -28, -17, 37, 40, 7, 16, -23, -2, -34, -43, 17, -24, 39, 15, -11, 15, 34, -14, 7, -39, -4, -20, -20, -16, -20, 36, 7, -22, -31, 3, -1, -34, 17, 37, -6, 25, 32, 7, -12, 5, -19, -4, 22, 2, 26, 38, -39, -20, 13, 3, 38, -37, 13, 42, -13, -29, -15, 21, -18, 17, -26, -17, 32, -7, 41, -29, 11, -10, -19, 44, -26, -2, 32, 18, -29, -3, 33, 23, -21, 23, 28, 21, -16, 12, -11, -8, 4, 6, 19, 30, 25, -9, -18, 34, -35, 5, -14, -31, -39, 32, 15, -34, 7, 10, -21, -25, 26, -19, 41, 16, 0, -29, 9, 19, -11, 4, -35, 8, 27, 39, 21, 43, -7, -15, 10, 11, 0, 3, 39, 7, 13, -14, -19, -1, -2, -32, -27, 41, 11, 26, -5, 12, -9, -32, 39, -30, 21, 28, -6, -8, -34, -18, 22, -17, 8, -10, 41, 19, 30, 31, 0, 38, -29, 1, 1, 15, 39, -34, -3, -10, -34, 30, 41, 8, 39, 42, 15, 5, 35, 35, 14, -28, -22, 9, -1, 1, 20, -39, 39, 13, 35, -38, 29, -35, 17, 28, -20, 31, -28, 3, 38, 7, -3, 34, 46, 38, 29, -2, -12, -15, -1, 45, 12, 18, 43, 44, 13, 10, -35, 4, 39, -26, -11, 30, -21, 39, -18, -15, -3, -38, 0, 33, -28, 5, 6, 13, 14, -20, -19, -35, 15, 15, -3, 21, -39, -15, 38, 40, -8, 34, 4, 4, -13, -4, 30, 29, 10, 31, 40, 0, 14, 32, -7, 28, 12, 42, -7, -15, 4, 39, 34, 36, -24, -38, -11, 15, -36, 40, -40, 30, 0, -1, 16, -20, -38, 39, -17, 10, 30, 6, 36, 10, 34, -19, 33, -2, 22, 36, -20, -28, -35, -12, 19, 33, -13, 34, -19, 36, 27, 40, 9, -17, 10, 23, -13, 29, 7, 29, -33, -27, -33, 4, -21, 2, -15, -5, 29, -32, 22, -30, 4, -40, 29, -21}
, {21, 18, 14, 38, -17, -27, 40, 33, 0, -13, 41, 5, -5, 42, -25, 20, 0, -9, -34, 1, 15, 5, -22, 39, 0, 35, -33, -21, -9, -21, 40, 13, -7, 21, -21, -5, 3, -33, 28, 22, 22, -36, -22, 3, 19, -11, -10, 22, -8, 11, -16, 6, 0, -31, 21, 13, -6, 0, 20, -7, 9, -22, 3, -19, 2, 0, -22, 25, 2, -17, 16, -35, 38, 6, -21, -21, -30, 43, -17, -23, 44, -39, 43, -37, -24, -36, -5, 45, -30, -14, -19, -11, 8, 41, -4, -9, -12, 37, 33, -3, 4, 8, 0, 33, 48, -9, 23, -35, 40, -34, -37, -22, -12, 28, -32, 28, 18, -2, 6, 16, 11, -13, -37, 36, -33, 16, -19, 11, 21, 1, -26, -37, -22, 32, -7, -11, 0, 10, -17, 6, -27, -2, 16, 15, -18, -18, 42, 12, -11, 30, -12, -23, -14, -17, 4, -7, -15, -5, 9, 18, -14, 6, 38, 46, 3, 24, -25, 33, 13, -3, 0, 28, 6, 27, 40, -8, 40, -4, -28, 10, 17, -25, -33, 43, -1, 38, 0, 16, 30, 2, 26, -10, -22, 8, 32, 4, 21, -14, 29, 9, 1, 22, 38, -33, -8, -5, -35, -23, 10, -12, -23, -31, 22, -21, 5, -15, 27, 32, 45, -38, -29, -30, 22, -4, -12, -2, 22, -13, -2, 37, 5, 3, 31, 9, 26, 24, -21, 41, -13, -15, 2, -20, 31, -33, -26, -18, 11, 35, -17, 26, 35, 34, 3, 34, -7, -4, 2, -2, -39, -42, 40, 38, 9, -39, -20, 41, 13, 41, -9, 0, 8, 19, 3, -12, 40, 4, 35, 37, 19, -30, 34, 4, -13, 25, -26, -7, -37, -9, -36, 46, 0, 46, -5, 44, 9, 23, -34, 3, -13, -11, 3, 34, -27, -25, 27, -28, -10, 13, 11, 32, -12, -9, 3, 38, 10, 25, 0, 29, 39, -31, 6, 5, -23, -11, 29, 43, 25, 35, 41, 7, -24, 43, 11, -20, -27, 13, 7, 20, -12, 34, -30, -16, 45, 41, 18, 0, 44, -37, -27, -33, -3, 30, 32, 42, -35, -32, -22, 21, -23, 25, 4, -41, 29, -19, 24, -7, 7, -6, 8, 24, 8, -34, 11, 42, 22, 18, 24, 43, 44, 7, -19, 26, -33, -16, -5, 2, -28, 7, 9, 23, 35, 36, 8, -33, 1, -22, 13, -12, -25, 33, 19, 19, -36, 31, -18, 39, 9, 38, 16, -1, -3, 26, -19, 39, 44, -14, -6, -21, 25, -34, -32, 45, -17, -1, 4, 6, -21, -27, 10, 22, 0, -22, -15, 42, 22, 16, -26, -12, -10, -33, 30, -36, -8, 21, 30, 3, -22, 1, 46, 40, -8, -41, 8, -1, -28, 8, -37, -19, -15, 40, 20, 25, -34, -30, 20, 42, 39, -10, 33, 38, 34, 32, 30, -1, 28, 22, -20, -21, -24, 36, -13, -34, -40, -19, -15, 28, 33, 30, -10, 13, 34, -33, 13, 13, 20, 25, 5, -18, 0, 15, 5, 0, 36, -29, 45, -33, 10, -21, -11, -15, 30, 17, 23, -34, -31, 32, -8, 1, -1, 41, -16, 27, -23, -12, 44, -35, 29, -5, -19, -18, -12, 6, 32, 11, -35, 13, -32, -22, 8, 25, -20, 26, -33, 7, -28, -16, 18, 36, 33, -2, -8, -12, -19, 35, -24, 40, 28, 5, 45, -29, -33, -25, 17, 25, -33, 4, 44, -12, -32, 9, 17, -7, 10, 5, -2, 36, 13, 16, -35, 17, 24, 0, -7, -25, 6, 31, -18, -20, 29, 27, -34, -28, 31, 30, 10, -6, 22, 20, 0, 16, -12, 10, 40, -4, 36, -7, 40, 35, 19, -8, -33, 17, 0, -5, -30, 9, 16, -7, -8, 11, 40, -16, 32, 3, -8, -11, -5, 15, 26, 30, -8, 5, 4, 9, -31, 27, -17, -36, -27, -21, -8, 8, 30, -29, 5, -11, 9, -6, 31, 26, 33, 33, 29, 5, 13, -3, -18, 20, -8, 0, 4, -30, 33, 25, -11, 17, 11, 33, 6, 21, 37, 1, -15, 9, -2, 11, 11, 29, 32, -6, 26, -15, 32, -39, -26, 23, -30, 36, -35, 2, -42, 13, -4, 15, -41, -8, -24, -25, 18, 10, -19, -3, 31, -16, 4, -7, -35, 16, 26, 4, -19, 1, 2, 12, 13, 0, 42, 10, 24, -13, 13, -11, 40, 16, 21, 35, -21, 29, 6, 34, 40, 33, 32, 19, 25, 44, -26, 32, 27, 27, 13, -19, 22, -35, 43, 4, -32, 12, 5, 33, 28, -28, 6, 0, 26, -8, 0, 31, -35, 28, 11, -30, 33, 19, 6, -28, 30, -27, 46, 42, 24, -10, -33, -13, 18, 42, 37, 25, -6, -13, -12, -11, -40, -33, 17, -6, -38, 20, 19, -28, 21, -12, 7, -9, -30, -37, 9, -35, 39, -34, 6, 36, 42, -11, 37, 17, -1, 1, -15, 40, 25, -29, -5, 32, 17, -10, 37, 41, 13, 14, 41, -13, -5, -31, 21, 38, -21, -34, 22, 14, 26, 22}
, {-20, -34, -19, -10, -25, -36, -6, 33, -9, -32, 3, 25, -24, 2, -10, 12, -23, -15, -33, -9, -14, 17, -16, -24, -19, -27, 13, 36, 19, 13, -6, 30, -11, -17, -10, 35, 42, -31, 26, -10, 40, -29, 27, -23, 6, -4, -37, -18, 40, 14, 31, -12, -16, 27, -14, -30, 40, -24, 1, -27, -25, 41, -10, 20, 5, 14, 23, 6, 37, 10, 7, -18, 2, -7, -38, 18, 33, -13, 7, 27, 42, 44, -32, -10, 37, 16, -26, -38, -24, -30, -10, -6, 20, -25, -18, 25, -21, -27, 2, 28, 22, -31, 1, -5, -1, -28, 8, 36, -25, 39, 2, -21, -35, -13, 14, 25, -22, 28, -31, 16, 38, -26, 35, -4, 23, 13, -40, 36, -6, -1, 6, 44, 29, 17, -7, 39, -6, 0, 29, -40, 9, 37, -17, -35, -22, 32, 26, 31, -9, -4, -3, -9, 32, 32, 21, 26, 11, -10, -20, -21, -28, 20, -27, 8, -26, -23, -19, -30, 41, -2, -39, -31, -22, 28, 21, -7, 27, -24, -39, -32, -22, -18, -40, 2, -17, -38, -8, -22, 21, -29, 17, 15, 40, -13, 37, 34, -19, -6, 16, -38, 13, -21, -30, 38, 8, 0, -14, 16, 12, -20, 9, 4, -20, -39, 6, 2, 10, 25, -1, -21, -4, 19, -15, 43, -4, 9, -26, 2, 30, -9, -31, 1, 26, 36, -8, -20, 19, 9, -26, 29, -21, 34, 24, -33, -16, -37, -39, 17, 3, 34, 39, -34, 5, 41, -18, -31, -2, -7, -39, 34, -17, -15, -31, 10, 0, 41, 13, -14, -12, -28, 21, 40, 40, 10, 0, 30, -35, 40, -29, 1, 41, 35, -24, -24, 36, 24, -19, -26, 42, -11, 15, 35, -25, -24, 35, -8, 24, 19, -2, 6, -26, 29, -1, 16, 44, 40, 33, -17, 26, 20, 0, 37, -24, 0, -26, 26, 21, 12, -6, -22, -21, -17, 7, 40, -7, -22, 17, 41, -1, -5, -35, -4, 32, 12, 41, 22, -29, 22, 30, -11, 12, 22, 42, 41, -34, -44, -18, -3, -5, 29, -1, 2, -31, 23, 34, 21, -33, 27, -39, -15, -39, 0, -14, 24, -1, -32, -14, -3, 14, 0, -30, 41, 6, 19, 6, 48, -16, -10, 37, 44, -19, -34, 17, -41, 21, -35, 0, 4, -26, -7, -1, -19, -12, 28, 2, 15, 41, -13, 15, -14, 33, 7, 33, -32, -2, -21, -10, -39, 34, 41, 13, -31, -30, -7, -36, 27, 45, -14, -10, 4, -37, 27, -29, -40, 30, -3, 1, 30, -30, -9, -35, -34, -12, -23, 15, 40, -24, -28, 4, 37, 6, 6, -2, -12, -21, -17, 32, 41, 6, 25, -14, 42, 27, 11, 19, -24, 32, -21, 28, 28, 11, 11, 3, -8, -10, 16, 12, -31, -19, -23, 6, -9, 0, 28, 27, -24, -19, -9, -33, 15, -5, -25, 8, 6, -23, -6, -35, -39, -6, -11, -19, -15, -18, -30, -18, 2, 3, -16, 21, -34, 10, -35, 2, 18, 35, 0, -27, -15, -19, -22, 18, 19, -21, -29, -24, 25, 20, -15, 42, 12, 6, 17, 29, 4, 15, 6, -21, -20, -40, -39, -9, -8, -7, 26, 37, -2, 1, -24, 0, 2, -35, -32, 10, 14, 15, -8, -17, -4, 38, 32, -8, -22, 43, -7, -11, -27, -39, 33, -31, 14, 34, -13, 26, -13, 0, -8, 0, -20, -32, 22, 28, 0, 2, -41, 6, -1, -37, 10, -29, 31, 23, -25, 11, -28, 25, 43, -38, 36, -13, 35, -10, -13, 41, -1, 8, -40, -38, -37, -3, 20, 21, 12, -36, -22, 21, 37, 27, -19, 6, 14, -32, -13, -25, 32, 1, -37, 23, 24, 0, 41, 10, 39, -30, -23, -31, 4, -10, -4, -3, -6, 17, -16, 31, 35, 7, -10, 7, -2, 26, -42, -20, 40, -37, -4, -33, -10, 24, -29, 0, -42, 0, -30, 32, -1, 16, 20, 36, -31, 6, 6, -20, -30, 30, -9, 21, -6, 12, -38, 34, 0, -3, 36, -43, 37, 0, 0, -4, 27, -23, 13, 7, 7, 43, 40, 23, 31, -8, -37, -3, 37, -29, -21, 23, -11, -16, 22, 36, 10, 30, -30, 10, -3, 31, 19, -38, -35, 7, 12, 36, -11, 5, 22, 37, -26, 40, -25, -7, -36, -26, -20, 3, 2, 24, 24, 28, 38, 26, 32, 33, -4, 43, 27, -8, -28, 29, -19, -2, 38, 39, 32, -12, 17, -35, -14, -15, -6, -34, 3, -9, -38, 10, 7, 0, -9, -31, -22, -15, -30, 33, 25, 2, -32, 39, -1, 8, 5, -23, 40, 29, -32, 22, 26, -26, 9, 29, 7, -1, 5, -34, -1, -29, 29, -4, -4, 6, -13, -31, -11, -43, 12, -31, -35, -1, -34, -11, 39, -6, 31, -39, -18, -41, -7, -1, 22, -30, -29, -16, -29, 31, -36, -1, 21, 38, -34, 6, -7, -30, 14, 5, -22, -10, -31, -21, -25, -31, -23, -23, 3, -10, 0, 7, 37}
, {-33, 5, -9, 9, -19, 49, 19, -15, -7, 32, 19, -27, 7, 38, -25, -23, -5, 30, 25, -35, 33, -17, -8, 27, -9, -25, 16, 26, -36, 21, -25, -30, -17, 37, 39, 35, 11, 6, -7, -8, -26, -16, 6, -20, 7, 23, 31, 36, 28, 13, 37, -15, -8, -36, 18, -22, 34, 33, -31, -35, 44, -17, 28, -30, 23, -20, 12, 25, 24, -34, 23, 14, 42, -38, -17, 9, 6, 19, 10, 32, 19, 16, 0, -32, 32, 14, -29, 18, 11, -6, 24, -40, 32, 40, 24, 2, 22, 2, 12, 45, -40, -27, 11, 0, 10, 36, 36, -4, -33, 31, 34, 19, -29, 17, 30, 4, 31, -6, -2, 31, -36, 16, 11, 42, 21, 5, 41, -30, 27, 39, 40, -4, -39, 5, -3, 45, 6, 2, 4, -31, 41, -9, -34, -39, -16, 9, 19, -36, 25, 45, 17, 31, 23, 28, -18, 21, 45, 18, -7, -4, 27, -23, -6, 21, 7, -1, 12, -40, 24, -2, 22, -3, -2, 38, 25, 22, -32, -34, -16, -17, 32, 9, -14, 13, -20, -15, 31, -24, 30, 3, 39, -22, -34, -36, 6, -4, -7, 40, 14, 36, -31, -23, 5, 44, -7, 48, 38, 30, -7, 27, -24, -9, 23, -17, -37, -16, -6, 13, 35, -24, 32, 12, 13, -15, 4, -20, -34, -23, -4, 27, 19, -41, -7, -14, 10, 23, 1, 10, -20, -21, -23, 20, 7, -37, 24, -33, -24, -26, 5, -38, -17, 14, -13, -38, 4, -2, -29, -26, -20, 13, 16, 35, -18, -6, -16, 21, 35, 0, -8, 15, -39, -40, -22, 27, 41, -20, -34, -20, 39, 33, 33, 43, 13, -33, -18, -28, -33, 3, -37, 36, 13, 19, -32, 30, 43, 31, 0, -10, 6, -12, 3, 29, 40, 46, -34, -23, -4, 38, 26, 26, -30, 1, -2, -6, -14, 22, 1, 4, -36, 22, 33, 41, -16, -28, 11, 24, 13, 50, 37, 11, -35, -33, -16, 10, 35, -22, -40, -34, -18, 43, 50, 39, 12, -3, 31, 29, -7, 33, -30, -37, 20, 42, -13, -3, -7, 40, 11, 28, 36, 4, 12, 28, 4, -17, 15, -13, 33, -4, 40, -15, 3, -23, -12, 41, -5, 43, 39, -37, 30, -33, 39, -35, 1, 20, -21, 43, 22, -25, 25, 21, 18, 7, 25, 25, -12, 36, -30, -27, 21, 15, 6, -23, -5, -3, 36, 28, -6, -29, 24, -7, -8, -7, 35, -35, 16, 21, 50, 3, 40, -5, 24, 46, -11, -7, 25, -23, 5, -1, -27, 24, -22, -27, -9, 19, -25, -22, 13, -36, 6, 1, 32, 7, -8, -2, -24, 40, 34, -10, 34, 34, 28, -16, 19, -23, 23, -39, 11, 12, -39, 20, -11, 9, -8, -14, 1, -25, 18, 28, 17, -35, 11, 30, 12, -12, 29, 42, -6, 23, 10, 6, -33, -17, 33, 42, 9, -1, 43, 19, -7, 21, -38, 9, 1, -2, -21, 19, -19, 39, 0, 27, 15, 0, 8, 0, 37, -18, -37, -31, 7, 25, 28, -30, -4, 36, 9, -37, 41, 22, -1, -34, -14, 12, -34, 45, -3, -32, -13, -2, 33, 42, 6, 39, 28, -31, 9, 17, -31, -2, 1, 1, -31, -8, 19, -4, 12, 2, 30, 14, -27, -38, 12, 2, 0, 0, -8, 33, -37, 14, 20, -30, 33, 20, -9, 41, 12, 42, -14, 0, 13, 8, -3, 22, 0, 22, 31, -35, 13, -22, 36, 11, -7, -13, 38, -18, 2, -29, 8, -17, -38, -34, 15, 1, -28, 6, -5, 3, -25, 0, 17, -19, 2, 44, -26, 30, 24, 2, 11, 30, -3, 4, -9, -16, 10, 7, 46, -26, -26, -12, 13, -36, -37, -13, 8, 36, 31, -42, -35, -8, 22, -15, -37, -15, 9, 23, 41, 35, 11, 13, 36, -10, 18, -33, 37, -17, -18, 7, -37, 18, 20, -4, 34, 5, -2, -41, -25, 22, 36, 32, 35, 26, -42, -18, -46, -19, 0, 32, -15, 4, -36, -20, -39, 13, 0, -24, -24, -12, 5, 10, -37, -6, -33, 39, 7, 21, -28, -33, 26, -15, -2, 2, -21, -27, 35, 37, -12, 28, -18, -12, 40, 16, 37, 17, 19, 39, 9, -31, -31, -2, -29, -25, -10, -30, -18, -17, 43, 0, -11, -18, 16, -27, 36, 46, -36, 45, 18, 28, -23, 12, 22, -17, -29, 41, 33, 27, 14, -37, -28, -35, -35, -38, 43, 13, -33, 3, 14, 2, -22, -32, 28, -22, 34, -7, 11, 17, -32, -7, 46, 1, -30, -26, 31, -29, -33, -10, 25, -6, 10, 39, 44, 15, -2, -33, 6, -22, 13, 28, 3, 6, 19, -38, 45, -39, -12, 10, 17, 10, 3, -4, -22, -40, -6, 12, 34, -16, 8, 37, 5, 1, 30, 19, 23, 5, -11, 0, 7, 34, 45, 15, -8, 1, 0, 8, 24, -13, -22, -38, -1, 16, 31, -27, 7, 7, -18, 44, 44, -28, -31, -41, -26, -20, 12, 15}
, {2, -43, 18, -4, -5, 14, -24, 19, -21, -12, 11, -22, 21, 0, 20, -39, 32, 21, 15, -16, 0, 4, -9, 22, 35, -35, -5, -26, 31, -6, 6, -10, -3, 4, 38, 24, 12, 25, -21, 8, -12, 26, 4, -14, -39, 17, -34, -25, 3, 37, -1, 27, 39, -33, 32, 5, -40, -19, 20, 0, -4, 1, 1, -27, -2, -38, 31, 37, -8, -33, 0, 27, 7, -11, -42, 2, 22, 28, -39, -9, -9, 26, 12, -9, -14, -10, 26, -2, -32, 38, 21, -38, -22, 2, -16, 16, -41, -27, -3, 28, 27, -38, -10, 9, -38, 11, -24, -4, 39, -42, 40, 0, -36, 4, 4, 12, 41, 24, 19, -31, -6, 37, 24, -35, -11, -25, -39, 22, -30, -34, 33, -38, 37, 27, 39, 38, -36, 17, -11, -13, 42, 0, 27, 30, 13, 10, -10, 20, -4, 18, 14, 36, 13, 1, 27, -38, 12, 3, 31, -32, -19, -30, 42, -5, -13, 34, -29, 6, 37, 35, -34, -14, -15, 26, 29, -30, 1, 5, -33, 38, -19, 39, 29, 27, -45, 8, -16, 17, -3, 39, 9, -8, -24, 16, -16, -25, -34, 19, -32, 33, 7, 0, -28, 5, -25, 17, 32, 22, 10, 21, -19, 25, 26, -27, 18, -42, 5, 28, 19, 19, -33, -4, -8, 9, -25, 15, -42, 19, -23, -6, -22, -14, -36, -28, 38, -44, -20, -8, -11, 34, -13, -20, -11, -27, -12, -42, 23, -37, 31, -34, -7, -4, 28, -8, 13, 35, 35, 14, -19, -2, -32, 12, -40, 22, 29, 19, -18, -24, -33, -14, -5, -8, 0, 0, 29, -25, 36, -32, 38, 36, 18, 40, -31, -27, 30, -36, 13, 38, -5, -2, -26, 40, -17, 12, -12, 20, 27, -2, -7, -13, -42, -12, 0, 11, 0, 21, -23, -22, 15, 6, 4, 20, -27, -17, -33, 9, -40, -34, 42, -21, 19, -38, -37, 37, 0, 10, -31, -24, 18, 35, 36, -1, -27, 29, -23, -19, 13, -28, -33, -9, 13, -31, 26, 39, -5, 9, -33, -38, -15, 31, 10, 27, 37, 0, 1, 13, -25, 25, 1, 10, 35, 0, 31, 32, -36, 33, -34, 14, -27, -7, -29, -20, -38, 9, 44, 8, -23, 10, 29, 12, 35, -10, 31, -29, -2, 23, -24, 31, 38, -41, -29, -16, -11, 36, -28, 36, -45, -9, 20, -45, 35, -43, 21, -40, 4, 21, 11, -12, -29, -37, 16, 11, 11, 21, 37, -8, -33, 4, 28, -40, -39, -30, -2, -21, 40, -37, -18, -6, 37, -6, -43, -20, -31, 5, 30, -41, -23, 33, 3, -14, -38, -20, 43, -2, -3, 20, -26, 22, 32, -7, -21, 27, 4, -44, -11, -21, 26, -17, -40, -26, 0, -21, -37, 17, 26, -32, -28, 26, -3, -45, -40, 39, -8, 13, 31, 3, 13, 32, -31, -22, -29, -29, -31, 36, 33, -33, 26, 22, 0, -16, 41, -4, 22, -31, 37, 0, -39, 7, -16, 38, -37, 24, -14, -1, 22, 18, 30, -20, 24, -6, -30, 34, -41, -7, 19, 9, -15, 38, -42, -22, -14, 38, -9, -32, 6, -1, -24, -40, 21, -23, 30, -34, 11, -42, 5, -42, -20, -39, 38, 7, -14, -37, 3, -22, -30, 39, -39, -25, 13, -39, -10, 4, -30, -15, 35, 15, 33, -36, -36, 38, 35, 25, -25, 27, -6, -20, -42, 8, -35, -22, 11, -3, -17, 18, -26, 3, 38, 15, 4, 41, 15, -2, 6, 11, 35, 22, 34, 19, -38, -13, -11, -19, -8, 24, -27, 2, 38, -25, 16, 38, -16, -20, 4, -28, -20, -33, 18, -40, 35, -28, -19, -44, -10, -2, -3, 8, 17, 7, 5, -10, 38, -17, -36, 4, 45, 10, -35, 1, 23, -6, 20, 35, 18, 4, -25, -13, 2, -15, 18, 28, 0, 12, 1, 6, -14, 11, 0, 39, 23, 4, 3, -2, -12, 29, 22, 36, -28, 42, 0, 10, 15, 39, 17, -10, 28, 36, 6, -33, 29, -23, -35, 18, 31, 2, 40, 43, 38, -24, -15, 5, 27, 0, 41, -34, 39, -8, -3, -11, 34, 30, 39, 40, -34, 38, 35, 19, 17, -18, 30, 30, -31, -24, 4, 22, -36, -24, 10, 38, -11, 29, -10, -18, 26, -15, 35, 23, -3, 30, 22, 28, 36, 19, -7, -21, -28, -1, -7, -32, -36, -6, -22, 29, -22, 33, 25, 0, 34, 6, 20, -11, 22, 18, -43, 7, 24, 17, 0, 4, 29, -40, 16, 26, 17, -3, 3, -15, 1, 32, 32, -8, 12, -28, -33, 25, -23, 41, 9, 0, -1, 32, 2, 0, -34, -38, 20, 32, -24, 25, 19, 38, -6, -5, -31, 26, 10, -2, 20, 17, 1, -36, -32, -12, -10, 20, -11, -25, -14, 33, 5, 24, -28, 18, 26, 9, -31, 1, 26, 25, 39, 39, -6, 23, 35, 2, -4, 17, 0, 19, 10, 5, -3, -38, -32, 38, -18, 19, 23, -14, -28, -16, -35, 26}
, {32, 19, -26, -41, 16, -35, -30, 16, -25, -40, 8, 17, -42, -34, 37, -10, 15, 17, -9, -22, 30, 22, 6, -28, 8, -3, 8, -9, -25, -8, 0, -7, 39, -26, -33, 23, -12, -25, 21, -33, 40, 20, -25, 5, -25, 4, 40, 22, -11, 21, 40, 0, 7, 17, 11, 19, 23, -8, -13, -37, 22, 2, -32, -37, -19, 35, 11, -18, -23, 37, 20, 19, -10, -40, -14, -16, 32, 21, -43, -12, -14, -40, -35, 37, -15, -36, 3, -5, -26, -40, 21, 7, 33, 24, 14, 21, 9, -21, 36, 2, 40, 0, 33, -39, -10, 42, -11, -25, 32, 28, -34, 15, 10, -14, 41, 15, 20, 11, 0, -37, -23, 12, -12, 20, 9, 11, -13, -17, -40, -19, -39, 42, 24, 0, 8, 0, -20, 33, -30, -25, -34, -3, 40, 12, 20, 38, 31, -4, 36, 30, -39, 14, 13, 8, -30, 2, 21, -34, 10, 39, -7, 23, 9, 38, -13, -2, -1, 11, -16, -13, 40, 41, 32, 28, 40, -10, 40, 0, -26, 42, 1, 37, 25, 33, 38, -17, -40, 5, 29, -9, -18, -31, -15, -41, 37, 15, 32, -13, 36, -1, -4, -25, 17, 21, -11, 29, 23, -24, -12, -7, -34, 32, 30, 38, 37, 23, 32, -20, -12, 0, 0, 8, -40, -2, 31, 27, 9, 7, 6, -17, -9, -10, 12, 30, 3, 42, -41, -10, -36, 6, 7, 7, 34, 4, -31, 14, 9, 35, 24, 8, 39, 37, -41, -33, 3, 26, 22, -23, 16, -2, -17, 7, 5, -34, -1, 31, -21, -1, 12, -29, -38, -13, 14, -17, 7, -36, -28, 1, 0, 17, -35, 19, 31, 0, 1, -24, -6, -29, -1, -25, 13, 17, -1, 15, 39, 27, 29, -5, -35, 43, -27, 20, 4, -28, -29, -23, -27, -4, -38, 5, 10, 23, 11, -15, -34, -37, -22, -5, 16, 25, -23, -10, 28, 5, 27, -10, 28, -21, -6, -5, -9, -1, -17, 4, 21, 38, 14, -41, -4, 23, -21, 33, 36, 38, -32, -20, -21, -4, -21, -3, -6, 7, 22, -37, 36, -30, -20, -15, -20, 46, 1, -9, 39, 8, -26, 18, -11, 17, -6, 2, 32, -12, 13, -35, -43, -23, -32, 25, 13, 41, -33, 18, 6, -32, 5, -31, 31, -2, -18, -19, 20, -13, -21, 36, 33, -3, 26, 4, 18, 25, 27, 0, 31, -4, 43, -8, -42, 26, 22, 19, -7, 13, -37, 16, 29, 31, -24, 29, 20, -26, 13, -28, 35, 2, 17, 18, 9, -7, -21, -8, 3, -39, 17, 25, 33, 44, -2, 20, -20, 25, 22, 16, 5, 29, -31, 1, 40, 17, -10, 43, -35, 33, 37, 23, -42, -8, 37, 8, 24, 1, 19, -38, 42, -41, -40, 35, -26, 24, -8, -20, -30, -15, 27, 19, -41, -17, 15, 32, -6, -30, -32, -34, 37, -17, 21, -15, -28, 10, 38, 6, -19, -30, -34, -22, -27, 32, 32, 19, -20, -10, -35, -22, -33, 5, -43, -36, -7, 17, -41, -10, -21, -30, 19, 27, -24, 11, -34, -31, 8, -4, -40, 32, 2, 21, -13, -13, -22, 44, 20, 13, -40, -29, 10, 38, 39, -39, 0, -34, 8, 43, -8, 39, -39, 30, 38, -16, -16, 12, -9, 1, -34, -12, -40, -39, 14, 26, 12, -21, -15, 17, -27, 22, 12, -10, 20, -20, 36, -1, 7, -37, 32, -8, -39, -5, -12, -25, 10, 20, 40, 26, 43, -12, 11, -26, 16, -19, -2, -16, 23, -18, 7, 27, 22, -5, 7, 12, 29, 33, -4, 25, 41, -12, 22, 5, 36, 42, 36, 14, 14, -30, 27, 22, -37, 21, 20, -23, 40, 33, -15, -17, 37, 29, 39, -4, 30, 23, -36, 25, 24, -26, 16, 31, -30, -33, 16, -32, -31, -21, 11, 2, 0, 38, 10, 24, -37, 7, -13, -33, -27, -9, -39, -11, 5, 6, -21, -32, -28, -23, -20, -17, -20, 23, -16, -5, -38, 5, 23, -36, -26, -6, -35, -29, 19, 21, 14, -33, 7, 28, 10, 21, 10, -6, -28, 8, 41, -12, -32, -29, 31, -16, -6, 28, 5, 10, -25, 7, -38, -13, -20, 4, -23, -24, -13, -25, 42, 4, 20, -37, 13, 6, 42, 1, -37, 0, 33, 7, 13, 18, 30, -41, -25, 3, 6, -9, 35, -26, -42, -13, -22, -2, -26, 15, -32, 28, 26, -4, -24, -20, 3, 29, 39, -28, -23, 19, 27, 18, 1, -23, -37, 0, 16, 41, -37, 23, -10, -40, 3, 22, -6, -41, -4, -37, -32, -3, 34, 21, 0, 3, 0, -7, 0, 28, 8, 32, 13, -22, -21, 33, -6, 24, 25, 28, -12, 7, 0, 14, -12, 31, 29, 27, -38, 21, -14, -38, 3, -34, -19, -35, -23, 42, 35, 23, -38, -23, 9, -27, 16, -29, -29, 21, -39, 20, 33, -1, -3, -31, -35, 12, 34, 34, -34, -35, 43, -25, 17, -20, 23, 27, -17, 37, 16, -20}
, {-30, -1, 35, -23, -18, -24, 16, -17, -22, 42, -3, -19, -6, 50, -39, -8, 27, -10, -2, 34, -36, -22, 1, 20, 17, -8, 33, -35, -28, -22, -18, 5, -13, -17, 35, -2, 42, 26, -23, 50, -37, -29, -1, 44, 41, 18, 36, -14, -41, 14, 34, 42, -27, 34, 0, 24, 20, -28, -9, 41, -14, 31, 15, 32, -20, 13, -34, -29, -15, 26, -36, 24, 29, -23, -17, -14, -13, 35, -3, 7, -2, -37, -31, 25, -23, 27, 9, -41, 10, 38, 37, -26, -8, -11, -36, 22, 40, -24, 16, 26, -21, -22, 24, 30, -9, -23, 5, -9, 32, -1, -4, -33, 47, -14, 46, -14, 24, 41, -15, 2, -21, -13, 3, 9, -32, -8, -40, -32, -21, 5, 17, -29, -10, 7, 1, -12, 12, 14, 0, 6, -1, 3, -7, 5, 12, 12, -17, 46, 9, 13, -6, -10, 16, 32, -8, -36, -20, 41, -22, -32, 42, -13, 11, 37, -34, 3, 22, 39, -9, -11, 33, -2, -6, -42, 25, -35, 34, -29, 36, 10, 19, -38, -26, 2, 38, 40, 43, 36, 42, 16, 0, 26, -5, -27, 5, 1, -41, -33, -28, -7, -31, 7, 15, 30, 36, -16, -22, 19, -23, 17, -29, 16, -4, 46, -18, -14, 15, 26, -23, -22, 2, -29, -19, -38, 0, 25, 29, -9, -12, 5, 37, -20, -18, 10, 13, 19, 38, -22, 25, -24, 5, 25, -13, -27, 0, -41, 24, -38, -33, 14, 19, -22, -11, 17, -6, 42, -42, 39, -3, 34, 13, 13, 0, 24, -12, 30, -12, -29, 37, -36, -2, -18, 34, 0, 2, 19, 9, -24, 9, -5, 13, 24, 25, -30, 44, 23, 27, 35, -31, 14, 34, 10, 38, 10, 17, -15, 34, -32, 22, 40, 20, 35, 2, -16, 4, 14, -27, -16, 5, -24, -44, 29, 1, 10, -12, 17, -40, -5, 0, -30, -2, -22, 35, 7, 42, -4, 17, -17, -10, 19, 41, -14, 29, -19, 15, 42, 5, 45, -15, 9, -4, 12, 13, 42, 18, 42, 44, -33, -21, -6, 40, -26, 30, -13, -32, 42, -10, -18, 40, 43, -23, 41, -43, -39, -2, 42, -27, 2, 1, 40, -13, 6, -29, 43, -4, 7, 31, -11, 0, -29, 33, 21, -6, 22, 11, 43, -27, 42, -37, 7, 36, -27, -3, -12, -28, 12, -22, -25, -11, -29, 16, -19, 6, 46, -17, 35, 32, 29, -12, -36, -12, 21, 25, 24, 0, -30, 9, -7, 5, 47, 12, 38, -27, 3, 5, 42, 31, -29, 8, -35, -30, -36, 2, 34, 14, -33, 24, 7, -1, -32, -8, 3, -14, -18, -41, -3, 0, 9, -40, 11, 24, 4, 0, 1, -25, -19, 21, 46, -5, -23, -16, -18, -36, 24, 21, 25, 36, 8, -7, -30, -6, -25, -33, 17, 48, 11, 26, 12, 29, 13, -14, 25, 17, 22, -32, -6, 5, -22, 38, 19, -43, -2, 14, -21, 12, -9, 25, 11, 0, -23, -22, 10, 1, -29, -28, -18, -36, 15, 0, 40, -27, 40, 36, -14, -24, -24, 2, 42, 5, 46, -4, 7, -22, 4, -35, 35, -1, 25, -33, 38, 5, -15, -8, 11, 31, 33, 34, -31, -38, -31, 30, -40, -40, -20, 9, -6, 0, 12, 8, -26, -23, -17, -5, -5, -17, -23, -20, -35, 38, -4, 2, 1, -10, -30, 4, -35, -3, 31, -1, -24, -9, 17, -12, 35, -36, 43, 16, 30, 1, -7, 29, 17, 39, -18, -31, 36, -33, -11, 1, -6, -6, -27, 0, -2, 35, -31, -19, -4, -18, -17, -18, -28, 27, 26, 34, -36, -27, -12, -41, 22, 11, 13, 37, 2, 30, -20, -2, 20, 43, 7, 19, 37, -15, -39, -17, -21, 0, 7, -8, -44, -36, -31, 0, 13, 23, -2, 32, 15, -22, 6, 0, 40, 24, 7, -14, -24, -6, -27, -22, 23, -22, -34, 0, 26, -17, 40, 18, -10, 35, 25, 28, 10, 36, -1, 0, 8, -2, 22, -30, 0, -8, 0, -11, -35, -10, -7, 31, 22, 10, 17, 17, -11, 3, -19, 33, 12, -4, 6, 9, 11, -25, 15, -3, 5, -9, 5, -14, 6, 32, -28, -7, 10, 48, -29, -24, 5, 19, 47, -6, -20, -17, -21, 1, 36, 36, 45, -20, 6, -25, -37, 0, 19, -22, 0, 18, -8, 37, 7, 11, 32, 45, -33, 12, -34, -11, 20, -18, 23, -37, 2, 16, -31, 39, 2, -8, -32, 17, 24, -31, 33, -25, -16, -20, 38, 24, 0, -13, 23, -23, -2, -17, 38, -7, 3, -33, 4, 1, -2, -22, 26, -39, 5, -39, 31, 3, -16, 25, 30, -11, -6, -20, -30, -9, 25, 29, 21, -39, 9, 34, 16, -33, 23, 24, 1, -1, -34, 16, -23, 20, -14, 38, 40, -15, -19, -37, -19, -10, 0, -20, -30, -18, 26, 2, 23, 4, -27, -26, -17, 15, 39, 18, 28, 3, 41, 12, -14, 11, -33, 35, -38, 13, 19}
, {-49, 19, 9, -32, -47, -48, -30, 5, 10, 4, -18, 39, 3, -7, 16, -7, 9, 39, 13, 3, -2, 0, -7, -4, -31, -34, -12, -5, -35, 22, -9, 26, 30, 18, 34, 8, 7, -12, -22, 17, 27, 27, -15, -15, 18, -3, 2, -41, -3, 34, -29, 36, -13, -44, 18, 14, 22, 15, 36, -25, -45, -20, -45, 0, -16, -7, -17, -29, 14, -41, 33, 12, 6, 12, -19, 40, 25, 38, 4, 21, -2, 29, -21, 29, -10, 3, 34, 18, 17, -18, -2, -28, -30, 38, 39, -12, 24, 24, -34, 15, 39, 19, 20, 19, 31, 21, -37, -26, 34, -47, -40, 14, -22, -26, 17, 39, -15, -19, 20, 24, 23, -7, 36, 10, 21, 36, -25, -30, -11, -10, 28, 29, -5, -20, 32, 9, 17, -40, -13, -26, -40, 17, -27, 6, -23, -41, -30, 1, -5, 0, 21, 22, -39, -29, 27, -47, 17, -17, 32, 24, 3, -33, 0, -41, -43, -11, -1, -10, 45, 21, -8, 26, 1, 9, -28, 7, 36, 5, -36, 11, 34, -9, -10, -27, -44, 12, 3, 16, 19, 16, -45, -1, -3, -1, 19, 5, 12, 23, -19, 34, 17, -33, 19, 2, 42, 5, 34, -11, 25, 6, -35, -10, 3, -31, 41, -7, -35, 7, 37, 10, -40, -1, -38, -40, 3, 31, 25, 3, 21, 23, 14, -24, 33, 15, 22, -19, -31, -33, -5, 32, 9, -29, 23, -17, -33, 33, -11, -25, 32, -25, 29, 36, -29, -39, -10, 35, 43, 41, -29, -33, 34, -12, 2, 17, 35, 7, -14, -34, 21, -29, -30, 9, 38, 7, -19, -23, -15, 5, -5, 27, 28, 35, -28, -21, 21, -20, 26, -20, -4, 15, -13, 1, 0, -30, -41, -2, 24, -33, -42, -9, 9, -25, 26, 30, 21, -35, 28, -11, 32, 7, 20, 27, -37, 39, -36, 39, 19, 11, -34, 16, 32, 15, 21, 28, -17, -8, -34, 8, -12, 27, -23, -36, -48, -36, 20, 31, -6, -42, 31, -8, 20, 4, -25, 3, 39, -32, 15, 5, -45, 15, 10, 12, -35, 0, -18, 7, 26, 13, -8, 12, 13, 13, -31, 21, 25, 14, 38, 34, 34, 39, -9, -1, -2, -40, 0, 17, -15, -9, -4, -37, 17, -41, -47, 6, -39, 28, -33, 4, -18, -34, 19, 14, 4, -43, -13, -2, 15, -44, -32, 7, 15, 20, -21, -3, 26, -21, 6, -38, 39, 16, -23, -2, 39, 17, -29, 1, 22, -1, -16, -12, -2, -49, 20, -44, 14, 20, 26, 5, 19, -33, 34, 3, -6, -37, 25, 20, -10, 35, -23, -23, 14, -3, 18, 35, 18, 26, 12, 14, -26, -5, -5, -33, 10, -10, -22, 5, 42, -18, -5, 24, -22, 24, -9, -8, 12, 37, -4, 32, -11, 13, -48, -25, -47, 21, 4, -16, 21, 2, 9, -10, -7, 12, 22, 36, -16, 32, -22, 26, 16, 40, 33, 11, 0, -5, 2, -18, -32, -36, -13, 32, 3, 8, -9, 2, -40, -18, 19, -37, -33, 3, 33, -41, 25, 27, 21, -13, 31, -1, -5, 27, -7, 29, 5, 31, 30, 40, 17, -9, 27, -5, -38, 9, -14, 3, 8, -23, 19, 20, -39, 33, -16, -9, 14, 1, -18, -40, 34, 7, 6, -39, 17, 30, -23, 35, -31, -31, 7, -12, 39, -29, -30, 26, 29, -20, 14, 15, -26, -10, -7, -20, -28, -38, 15, -19, 6, -4, -36, -5, 22, -21, 23, 28, -20, -39, 1, -8, 28, -2, 1, 5, -28, 26, 40, -1, -6, -21, -31, -38, -45, -23, 31, -36, 1, -24, -41, -6, 40, 26, -6, -39, 3, 29, -39, 12, 16, 30, -18, -40, -39, 10, 27, -3, -8, 37, -8, 25, 21, -9, -29, 15, -12, -23, -25, -9, -24, -27, -35, -41, -7, -33, 9, 20, 21, 28, 11, -19, 9, 37, 25, -23, -34, -16, 23, 29, 24, -16, 18, -17, 5, 15, -17, 37, -16, 29, -4, 9, -4, -23, -22, 15, 36, -10, -21, -43, -16, -1, 3, 22, -25, 36, 26, 1, 22, -1, -19, -23, 30, -7, -35, -27, -25, 42, 21, -30, 1, -33, 16, 22, 5, -19, -43, 30, 11, -38, 29, 28, 25, -21, -35, -13, 10, 15, -24, -1, 5, -8, 34, 36, 19, -46, 12, 28, -41, 18, -29, 7, -42, 35, 26, 28, 22, -16, 0, 23, -8, 16, -32, -17, -3, -4, -46, 13, 31, -39, -39, -42, -36, -18, -21, 2, 12, -29, 31, 15, 38, 29, -1, -23, -40, 19, 13, -28, 7, 6, 39, 16, -1, 32, -13, 0, -11, 12, -21, 12, -22, -37, -18, 40, 0, 8, 4, -20, -30, 22, 38, 4, -30, 13, 36, 42, -9, -37, -19, 30, 32, 30, -9, 5, -32, 25, -2, 8, -44, -43, -18, -25, 14, -10, -30, 27, 3, 27, -40, -27, -3, 18, 6, 15, -34, -38, -30, 25, 23, -22, -18, 16, 14, -2, 4, -9, 35, 6}
, {14, -29, -8, -36, 7, 32, -18, -23, 19, 0, -6, 3, 5, -19, -23, 8, -24, -22, -21, 12, 22, 10, 23, 18, 6, 41, 37, 23, 2, 22, 3, 24, -15, 20, -44, -13, -30, 22, 41, 37, 23, -21, 9, 6, -7, -19, 2, -7, 26, 37, -32, -22, 7, -14, -37, -28, 20, 31, -41, 21, -13, -16, -37, 28, -25, 22, 3, -7, 32, -21, 27, 17, -32, 34, -23, 37, -32, -21, 5, 38, -27, 15, 2, 11, 37, -34, -22, -24, -4, -29, -10, 17, 23, 22, -20, -7, 18, 16, 33, -37, 28, -34, 23, -32, -17, -1, 34, 11, -2, -13, -9, -34, 40, 7, -27, 40, 17, 35, -5, 33, 31, 6, 37, -31, -27, 13, -15, -34, 17, -13, 23, -5, 11, -27, -10, 5, 0, 12, -18, -41, -42, -10, -12, -45, -4, -32, -16, 34, -27, -4, 15, -19, 4, 19, -23, 0, 26, 10, 1, 1, -16, 35, -7, 26, 13, -39, 11, -4, 27, -2, -34, -2, -6, 12, 30, -25, 9, 33, -23, -36, -35, 15, -22, -1, 32, 11, 11, -43, 12, 30, 23, -38, 11, 5, 33, -28, -54, -29, 1, 7, -49, -43, -41, -27, 19, -40, 3, -11, 27, -43, -37, 8, -19, -35, 20, -27, 16, -23, -23, 5, 8, 23, 34, -24, 15, 34, -44, -16, -5, -43, 38, 19, 32, -25, -29, -11, -18, -7, 19, 3, 34, -11, 17, 12, 13, 8, -23, -8, 24, 0, -21, 0, 5, -26, -15, 37, 36, 30, -23, -30, -43, 27, 30, 4, 40, -14, 18, 12, -13, -31, -26, 8, 2, -3, 22, -6, 20, 24, -4, -13, 10, 8, -32, 13, 33, 37, -34, 3, -35, -1, 0, 21, -28, -33, 33, -14, -35, 11, 1, 12, -33, 34, -19, -38, 17, 31, -14, -2, -43, -24, -2, -36, -30, -38, -10, -43, 21, -27, 5, 0, 32, -22, 34, 31, -7, 1, -7, -37, -3, -32, 31, -39, 32, 15, 11, 37, 12, 4, 18, 0, -21, 30, -35, 15, 13, -6, -21, -41, 11, 12, 5, -20, 39, -6, 10, -5, -21, -2, -20, 32, 38, -4, 36, 18, -31, -8, 5, -4, -8, 7, -37, 33, -21, -8, 39, -7, -43, -41, -23, 35, -13, -35, -33, -29, 37, -5, -16, 14, -43, 28, -38, -38, -7, 7, -3, -42, -32, -15, -46, 11, 28, -35, -24, 23, -21, 39, -16, -18, -1, -25, -18, -14, 16, 31, 37, 25, 4, -22, 26, 41, 0, -25, -29, 30, 21, 16, -32, 39, -27, -3, 1, 4, -12, -34, 23, 11, 4, 32, -37, 38, -40, -14, -13, 2, -23, 20, 28, 30, 13, 32, -25, -16, 22, 9, -39, -6, -25, 9, 0, 39, 2, -7, 22, -23, 29, 23, -30, -29, -5, 25, -4, 14, 30, 10, 34, 16, -13, -33, 14, 31, 24, -37, 0, -41, 16, -2, -16, -4, 24, -42, 6, 14, -15, 26, -38, 23, -24, 10, 38, 42, 4, 12, -12, 21, 0, 14, -4, 36, -26, -11, -30, -13, 18, 42, -42, 38, 33, -18, 42, 30, 11, -9, 28, 35, 34, 32, 37, 13, -24, 15, -28, -40, 2, 39, -16, 14, 33, 20, -7, 34, 39, -26, 38, 2, -15, 4, 31, -20, -42, 31, 29, -24, 6, -38, -42, -32, 22, 36, 38, -18, 12, 27, 14, -32, 1, -24, 45, -23, -19, 21, 10, 27, 0, 17, -43, -25, -26, 20, 10, 16, 7, 16, 21, 7, 7, 20, -27, -44, 0, -15, 4, 30, -25, 12, 20, -18, -28, -24, -38, 0, 28, 30, -13, -37, -8, -32, 0, -35, 31, -5, 17, -11, 12, -6, 29, -30, -9, 25, 3, 19, -40, 1, -22, -37, 36, 18, 37, 34, -6, -21, 7, 25, -18, 18, -2, 21, 34, -42, 2, 12, -40, 26, 13, 39, 6, -21, 9, 35, -5, -21, 35, 17, -40, 28, -22, 25, -22, 17, -27, -25, 2, -2, -36, 18, 21, 29, -24, 4, 17, -20, -38, 16, 42, 37, -8, 39, 29, -41, -9, 20, 0, -22, -8, -14, -23, 9, 33, 11, -6, 22, -2, 26, 2, 13, -12, 27, -11, 35, 34, 13, -42, 28, 13, 33, 12, -24, -42, -6, 3, 32, -6, -11, -42, -41, 43, 32, 17, 31, -36, 31, 16, -15, -35, 1, 27, 19, -5, -11, 19, 15, 12, 30, -3, -8, 40, -25, 6, 21, -37, 31, -33, 7, -38, -10, -29, 1, -28, 39, 35, 9, -8, 20, -5, -9, 32, 11, -6, 38, 12, 16, 32, 0, 13, 0, -10, -37, 39, -39, -3, 41, -19, 37, -13, -41, -8, 18, 33, -16, -21, -22, -28, 4, -41, 6, 21, -23, -15, 23, 15, 9, -34, -40, -6, 28, -19, -2, 8, -26, -41, 31, 0, 6, 24, 22, -12, -8, 15, -36, 0, -33, -25, 37, 15, 26, 33, 30, 30, -21, 13, -35, -19, 29, -41, 2, -31, -12, -13, 33, 16, 0, -22, 36}
, {4, -16, -21, -34, 16, 29, -12, -6, -12, -23, -21, -22, -16, 3, 12, 3, 26, 16, -42, -39, 22, -29, -17, 36, 12, -11, 37, 10, 12, -7, 3, 34, -8, -1, 21, 4, -30, -23, -23, -10, 38, 35, 34, 0, 21, -22, -26, 8, 1, -24, -9, -35, -15, -38, -33, -48, 8, -16, 37, 25, 7, -33, 6, -35, 17, -1, -27, -1, 24, -20, 34, -5, -20, 19, 18, 36, -32, -39, -43, 25, 22, -2, 15, 41, 11, 38, -9, 10, -8, -9, 0, 6, -20, 14, 11, 11, 27, -24, 5, -29, 4, 0, 8, 16, -18, -2, 37, 13, -22, -25, 38, -34, 37, -10, -18, -41, 23, -21, -39, -1, -31, -2, -40, 40, 8, 4, 37, -44, -33, 26, -39, -38, -34, -12, 21, -6, 32, -38, 0, -27, -36, -31, 9, 34, 19, -8, -8, -8, -31, -26, 38, 0, 10, -39, -20, 8, 20, -18, 3, 26, 34, 0, -17, -1, 30, 36, 1, -37, -13, -4, -28, 34, 22, -21, 28, -32, 6, 7, -12, -21, -10, 23, -35, -19, 3, -23, -24, 18, -40, -37, -42, 23, 6, -25, -32, -16, -6, 38, 30, -37, -26, 16, -27, -24, 7, 35, -36, -32, 30, 14, -21, -5, -27, 16, 37, 20, 13, 26, -6, -41, 38, 7, 9, 36, 33, -11, -41, -18, 23, -9, 33, -6, 10, 5, 4, 28, 0, 26, 19, 39, -39, -6, -10, 3, -19, -17, -31, -13, 35, 14, 22, -23, -37, 31, 8, -11, -25, -37, -16, 39, -40, 26, 3, 24, -31, -3, 18, 13, 18, -32, 6, 17, 21, -6, 9, -15, -23, 34, 28, -18, 6, -9, 0, 26, 13, 13, -3, -24, -19, -20, 11, 22, -40, -19, 37, -22, 1, 28, -4, -21, 23, -23, 34, 38, -36, 21, 7, -3, -38, 22, -7, 38, -14, 18, -17, 25, -37, 4, 39, -7, 36, 18, 36, -27, -16, 9, -40, 31, -22, -22, 23, 38, -17, -41, -34, 1, -3, 7, 17, 31, -19, 13, -23, -4, -19, 10, 31, -13, 7, 20, 7, 13, 17, -44, -33, 13, 38, -17, -23, 5, 37, -26, 27, 17, -40, -27, -33, -40, 8, 11, -16, -36, -30, -20, 11, -28, -28, -9, 7, 33, 31, -9, 21, 17, 19, 31, -11, 19, 26, 34, 28, 28, -37, 39, -27, -29, -35, -21, -8, -6, 28, -8, 11, -32, -5, 14, 24, 17, -44, 34, -7, -2, -6, -27, -13, -26, 27, 33, -46, 5, -7, -45, -23, -27, 29, -19, 21, -17, 4, -15, -21, 22, 21, 37, -14, -30, -29, 36, 15, -16, 0, 10, 24, 1, 31, -21, 13, 44, 24, 28, 29, 42, 19, 33, -22, 28, -17, -14, 23, -40, -8, 0, 39, -24, -1, -20, -10, 15, -27, -22, 9, -21, 36, 7, 19, 36, 4, 10, 39, 10, 7, 34, 19, 36, 22, -33, 37, -28, -12, -44, 36, -14, 30, -20, -39, -41, 3, -33, 16, -35, 8, -23, 0, -41, 37, 10, -17, -22, -43, -13, -36, 6, -29, -6, 34, 31, 16, 10, 0, -36, -12, 12, -32, 9, -23, -8, -22, -18, 33, 3, 2, 36, -1, 15, 41, -7, -3, 38, -40, -43, -10, 21, 32, -5, 1, -5, 19, 30, -11, -8, 19, -27, 26, -33, 23, -43, -6, 26, 26, -37, -12, 33, 1, 35, -31, 28, -2, -16, -21, 9, -31, 17, 11, -4, 7, 40, 34, -4, -13, 26, -30, 31, -5, -8, 29, -19, -10, 35, -6, 33, -12, 20, -32, 34, -19, -40, -34, 32, 24, -34, 10, 5, -21, -5, 13, 32, -28, -18, 15, 21, -13, -18, 9, -11, 12, -34, -14, -23, -43, -5, 16, -33, -35, 18, 12, 39, -28, 26, -19, -36, -21, -38, 29, 19, -38, 32, -12, 18, 42, 10, -40, -33, 36, -25, -9, -3, 36, 11, -34, 24, -11, -25, 43, -34, 15, -23, -41, 30, 2, -8, -12, -27, -43, -18, -15, -9, 26, 16, 6, -21, -9, -30, 36, 27, 2, -12, 42, 13, -26, 32, 17, 5, -44, 3, 3, -7, 15, -36, 9, -35, 23, 29, -10, -33, 41, 23, 18, -21, 20, 31, -13, 38, -23, 22, -44, -21, -41, 29, -10, -24, -21, 1, -27, -36, -5, -34, 27, -31, 4, -8, 0, 1, 36, 10, 14, -15, -38, 24, 37, 29, 17, 18, 1, -36, -38, -41, 28, -33, 22, -31, 32, -16, -18, -32, 24, -47, 29, -9, 38, -30, -43, -19, 37, -14, 39, 31, -11, 2, -35, 20, 1, -42, 10, -2, -7, 31, 5, -3, -10, -19, -29, 12, 27, -6, -44, -17, 0, 29, -42, -11, 16, 43, -29, -6, 13, 37, -38, -5, -4, -25, -25, 13, -36, -32, -20, -22, 21, 28, -17, 31, -1, -33, -18, -34, 21, 17, 12, -45, -43, 18, -12, 19, -25, -11, -6, -11, -9, 38, -43, 2, -20, -2, 13, 24, -6, 15, -9, -25, -29, -39, 7, 38}
, {21, -6, 22, 28, 30, 1, 27, -10, 29, 30, -9, -19, -19, 11, -28, -15, -6, -33, -7, -3, -36, -16, 32, -4, 33, -6, -24, 6, -23, -26, 29, 29, -12, 25, 36, -38, -24, 38, -26, -30, -21, -38, 33, -21, 0, -36, 31, -27, -18, -34, -8, 46, 39, 30, 5, 23, -25, -24, -23, -41, 35, -5, -35, 6, -18, 39, 32, -31, -24, 21, -41, 26, -12, 40, -38, -32, -27, -34, 23, 45, 4, 1, 24, -13, -26, -38, 43, 3, 9, -31, -9, 14, -17, -34, 9, 29, -21, -4, -31, 43, -13, 32, 32, 43, -26, -8, -37, 21, 16, 36, 33, -29, 36, 5, 0, 9, 22, 20, 3, -3, -3, -10, -16, -41, -38, -1, -10, -16, 18, -13, -32, -28, -7, -1, 37, 21, 7, 12, -19, -36, 4, 27, -8, 39, 13, -10, 44, 0, -39, 39, -14, -40, -28, -39, 27, -27, 40, 23, 30, -19, -15, -12, -44, -2, 16, -25, 3, 3, 5, -4, 1, 18, -38, -3, 2, -12, 21, 5, -20, -23, 12, 32, -33, -17, 32, 3, 5, 38, 6, 9, 10, 37, -20, -40, 33, 31, 5, -21, 26, -40, 6, -22, -44, -18, 35, 17, 11, 47, 25, -27, 24, 43, -8, 36, -2, 16, -23, 3, -3, -32, -19, -34, 13, 17, 16, -7, -18, 23, 27, 6, 25, -6, -12, -8, 15, -6, -29, 1, 33, 26, 6, 39, -17, 29, 0, 40, 25, 8, 29, 19, 32, -1, 33, 21, 27, 0, -41, -10, 6, 23, 6, -30, -25, -18, 17, 26, 14, 18, 7, 23, 2, 15, 30, 39, 18, 26, 26, -26, 44, 8, 26, -28, 39, -25, -35, 0, 1, 2, 32, -10, 43, 20, 16, -12, 20, 19, -1, -18, 34, -28, -22, -2, -38, 9, 15, -30, -6, -4, 6, 11, -4, -5, -34, -8, -3, 24, 33, -20, 33, -14, -18, -11, -4, -25, 3, 31, 23, 35, -6, -34, -18, 37, -11, -29, -27, 21, -14, 2, -30, -26, 27, -25, 41, -42, -34, 26, -7, -19, -8, -41, 19, -31, 1, -22, 28, 32, 25, -24, -4, 7, -5, -35, 11, 6, 30, -10, 36, -20, 11, 11, 20, 22, 38, -3, -11, 20, -20, 16, -39, -42, -1, -21, -39, -38, 27, 39, -5, -38, -34, 4, -18, 26, -28, 15, 37, 30, 29, 26, 38, 31, -11, 19, -24, 37, -10, 2, -40, -29, 2, 22, 31, 11, 4, -16, -36, -8, -16, 28, 19, -38, -30, 13, 10, 0, 32, -29, -40, 3, -12, 14, -36, 24, -24, -18, -39, -20, -13, -38, 18, 2, 18, -10, 40, 29, 28, 4, -40, -26, -14, -1, -19, -9, -1, -27, 7, -20, -12, -12, -33, -14, 24, 34, 24, 7, 0, -12, 38, 3, -8, 13, 24, 23, 33, -14, 15, -45, 33, 5, -15, 25, -4, -41, -34, -43, -24, -38, -17, -7, 2, 40, -27, 6, 7, -28, 43, 22, 7, 42, 2, 25, -21, -15, 11, 3, 19, 35, -31, 26, -3, -34, 29, 25, 3, -31, 15, -33, -9, -25, -28, 21, -39, -23, -22, 25, 39, 30, 45, 39, 45, 35, 6, 10, 33, -6, -9, -27, 33, -37, 39, -14, 28, -30, 16, 32, 32, -35, 23, 2, -41, -11, 4, -38, 1, -11, -16, -14, 39, -17, -22, -4, 27, -43, 13, -18, -35, -40, -26, 2, 21, 4, -11, 39, -41, -25, -31, 41, 38, 38, 6, 43, 11, 1, 9, 17, 20, -4, -39, 38, 21, 43, 30, 11, 21, -22, -23, 11, -16, 35, -40, -33, -31, 27, 41, -31, 11, 43, -19, 1, -25, -4, -33, -40, -36, 13, -11, 41, -2, 17, -10, -5, -18, 8, 12, -28, -20, 37, -1, 42, -10, 35, -28, -26, 46, 28, 22, -34, -8, 10, 37, -34, 33, 35, -9, 34, 20, -29, 22, -36, -15, -19, -7, 6, 23, -43, -9, -29, 36, -40, -29, -22, -2, 35, -23, -2, 11, -36, -39, 30, 29, 21, 6, 3, -32, 27, -40, 19, 16, -12, -24, 12, -9, 19, 45, 1, -38, -30, 15, -11, -30, -17, 50, -9, -30, 26, 39, 8, -10, -42, -27, 12, -20, -4, 0, 39, 28, 8, 36, -44, -26, 24, -15, 38, -37, -25, 15, 36, -12, 5, -16, 40, 19, -9, -39, -36, -15, 0, -2, 6, 19, -6, -2, 14, -15, 41, -31, -24, 44, -34, 4, 0, 10, -5, -12, 30, 38, 8, -4, -24, -31, 26, -31, 34, 5, 28, -32, 27, 40, -17, -30, -26, 11, -40, 18, 41, -37, 28, -23, 13, -44, -15, -22, -28, -17, 4, 6, 25, 32, 40, -16, 22, -42, -3, 13, -31, 2, 6, 33, 21, -26, 39, -39, 5, 4, 17, 38, 34, 32, -41, -10, 15, 21, -2, -21, -27, 4, -1, 8, 30, -6, 15, -27, -38, -6, -17, 7, -29, -40, -14, 12, 31, -41, 33, -7, 7, -16, 18, 36, 21, 40, -20, 7, -33}
, {0, 18, 29, -1, -11, 36, -20, 29, -11, 40, 8, 23, -23, 31, -26, -39, -44, -15, 8, -24, 36, -13, 24, 22, -6, -1, 5, 35, -14, 27, 37, 0, -8, -36, -36, -22, 35, 10, 25, -17, 27, 23, -21, -37, -2, 0, 6, 1, -38, -28, 38, 7, 28, -37, -25, 0, -39, 3, -31, 31, 13, 2, -38, 3, 2, 12, -33, -9, -8, -38, -11, 21, 11, 4, 42, 32, 42, -20, -31, 36, -38, -39, -40, -21, -21, -8, -10, -29, 33, -33, 11, -1, 9, 41, -21, 9, -37, -7, 35, -16, -8, -28, 18, 11, -7, -18, 33, 23, -15, 1, -5, 3, 28, 39, 35, -4, -30, 19, 29, -20, -23, -15, 4, 39, -29, 26, -1, 20, -3, -42, -29, 17, 37, 20, 14, -27, 24, -34, -19, -11, 14, 26, 36, 28, 20, 36, 4, -26, -17, 8, -33, 15, 30, -25, 38, -40, 37, 0, -26, -26, -31, -18, 21, 11, 13, 35, 20, -13, -16, -13, 10, 6, 21, 2, -31, -27, 34, -18, -35, 37, -41, 32, -11, -21, 7, -38, 32, 43, 16, -35, 21, 29, 26, 13, 8, 7, -36, 8, -33, 0, -21, 8, 8, -42, 33, 0, 10, -14, 5, -1, 14, -6, -17, -30, 26, -35, 42, -2, 12, -2, -21, 6, 30, 31, 2, -28, 30, -38, -6, -1, 0, 38, -13, 22, 4, -42, -6, -6, 11, -28, 19, 5, -33, -21, 6, 34, 7, -27, -11, -14, -34, 5, -11, -18, 34, 31, -14, -11, 15, -26, 29, 7, -34, 30, -15, 10, -37, 13, 40, -27, -4, 9, 17, 1, 1, 38, -7, 24, -13, -36, 29, -22, -11, -29, 4, 2, -38, -1, 0, -37, -10, -26, -27, 7, -11, -7, 7, -34, 41, -21, -24, 34, -33, 16, 10, -28, 7, -33, -6, -42, -23, 30, 8, -1, 10, -26, 14, 11, -28, -15, -26, 0, -25, -21, 10, 39, 19, 25, 20, -40, 11, 16, -10, 6, 0, -18, -21, -32, 30, -17, -29, -13, 39, -28, -34, -40, 29, -34, -8, -42, 38, 3, -24, 35, -2, -31, 0, 10, 38, -8, 10, 38, 40, 29, 0, 15, -30, 26, 9, -19, 2, -33, 40, -18, 11, 22, 44, 42, 27, 18, -21, 35, -2, -30, -40, 39, 12, 12, -33, 18, 35, -25, 0, 21, -36, 11, -39, -39, 40, -32, -23, 40, -21, 27, 31, -42, -37, 7, -41, -8, 4, 38, 41, -2, 2, 5, 40, -13, -40, 13, 12, -13, -11, -30, 31, 13, -1, 0, 36, -10, -28, 3, 17, 2, -29, 22, 6, -6, 33, -19, 21, 12, 6, -9, 37, -38, 14, -33, 40, -16, -45, 29, 35, -5, 38, -18, 38, 7, -35, -7, 36, -27, 13, 1, -25, -22, 21, 2, 32, -16, 39, 24, -40, -10, -3, -20, 2, 31, 18, 31, -1, -29, 4, -6, 14, -8, 16, 2, -18, -25, -35, -26, -18, -12, 12, 42, 5, -14, -30, 32, -9, 16, 35, 24, 4, -42, -22, 38, 33, 37, 41, 25, -3, 18, -30, 34, 34, 0, -34, 30, 25, -17, -5, 39, 3, -3, 39, -12, -19, 1, 0, -22, -28, 2, 7, 19, -22, 3, 4, -15, 17, -13, -4, -39, 3, 34, -39, 39, -3, -19, -24, 28, 3, -29, 1, -22, -31, -4, 26, 32, 19, -10, 18, 32, -22, -1, 16, 15, 10, 32, -6, 40, 30, -19, -25, 4, 7, -26, -14, -23, 29, -17, -7, 11, -29, 6, -10, -26, -46, -21, -41, -5, -28, -29, 41, 6, 12, -20, -22, -16, -22, -31, 18, 1, 12, 31, 2, -32, -34, -27, 0, -36, 39, 10, 24, -25, -41, 7, 14, -35, 13, -15, -8, -40, 45, 26, -4, 44, 13, -15, 16, -33, 10, 19, 15, -43, 24, 35, -43, -17, -18, 21, 27, 34, -30, -35, 17, 34, 3, 41, 0, -34, 37, 9, 26, -41, 11, 13, 7, 7, 16, 0, -13, -8, 42, 27, -12, 42, -20, 7, 33, 6, 17, 0, 2, 4, -17, 10, 35, 27, -29, -24, 37, 32, -7, 23, 35, -10, 42, -42, 30, 18, -7, -41, -27, -31, -25, -14, 26, -20, 28, -20, -16, 29, -14, 1, 24, 6, -39, 40, 10, -19, 25, -3, -26, -23, -12, 14, -23, -6, 14, 13, 6, 40, -29, 15, -14, 4, -5, -40, -38, -32, -43, 10, 23, -21, 3, -27, -21, -24, 17, 0, -23, 11, 8, 32, 3, -6, 22, -17, -22, -6, 42, -7, 10, 4, -15, -22, -34, 25, -37, 21, -20, -25, 26, 31, 17, 22, -21, -25, 32, -8, 20, -23, -7, -14, 23, 26, 12, -37, -25, -1, 33, 36, 8, 16, 12, 37, 14, -14, -39, -32, 12, -13, 3, 21, -17, 32, 1, -15, -10, 34, 2, 2, -3, 12, -9, 31, 40, -27, 19, 16, -36, 7, -40, -4, -25, 33, -2, -13, 28, 21, -22, 15, 33, -30, -12, -18, -25, 13, -18, -27}
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

typedef number_t dense_16_output_type[FC_UNITS];

static inline void dense_16(
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


const int16_t dense_16_bias[FC_UNITS] = {5, 0, -5, 13, 6, 1}
;

const int16_t dense_16_kernel[FC_UNITS][INPUT_SAMPLES] = {{232, -174, 125, -131, 152, 94, -162, 71, -94, 189, -17, -43, 147, -205, -157, -10}
, {60, -25, 180, -115, -187, -43, 16, -126, -237, 124, -198, 195, -175, -155, 211, -65}
, {-151, 135, 44, 113, 59, -120, 71, -13, 173, -191, -14, 89, 237, 70, -149, 150}
, {-172, 229, -15, 180, 25, 87, -157, -23, 165, -113, 214, -234, 177, -105, -230, 36}
, {193, 88, -51, 17, 115, 113, 135, 52, -260, -261, -15, 2, -167, -110, 238, 264}
, {215, -207, 82, -87, 118, 68, 56, -83, -41, -128, -157, 16, 139, 142, 173, -143}
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
  //dense_16_output_type dense_16_output);
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
#include "max_pooling1d_25.c" // InputLayer is excluded
#include "conv1d_30.c"
#include "weights/conv1d_30.c" // InputLayer is excluded
#include "max_pooling1d_26.c" // InputLayer is excluded
#include "conv1d_31.c"
#include "weights/conv1d_31.c" // InputLayer is excluded
#include "max_pooling1d_27.c" // InputLayer is excluded
#include "conv1d_32.c"
#include "weights/conv1d_32.c" // InputLayer is excluded
#include "max_pooling1d_28.c" // InputLayer is excluded
#include "conv1d_33.c"
#include "weights/conv1d_33.c" // InputLayer is excluded
#include "conv1d_34.c"
#include "weights/conv1d_34.c" // InputLayer is excluded
#include "conv1d_35.c"
#include "weights/conv1d_35.c" // InputLayer is excluded
#include "max_pooling1d_29.c" // InputLayer is excluded
#include "flatten_5.c" // InputLayer is excluded
#include "dense_15.c"
#include "weights/dense_15.c" // InputLayer is excluded
#include "dense_16.c"
#include "weights/dense_16.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_16_output_type dense_16_output) {

  // Output array allocation
  static union {
    max_pooling1d_25_output_type max_pooling1d_25_output;
    max_pooling1d_26_output_type max_pooling1d_26_output;
    max_pooling1d_27_output_type max_pooling1d_27_output;
    max_pooling1d_28_output_type max_pooling1d_28_output;
    conv1d_34_output_type conv1d_34_output;
    max_pooling1d_29_output_type max_pooling1d_29_output;
    flatten_5_output_type flatten_5_output;
  } activations1;

  static union {
    conv1d_30_output_type conv1d_30_output;
    conv1d_31_output_type conv1d_31_output;
    conv1d_32_output_type conv1d_32_output;
    conv1d_33_output_type conv1d_33_output;
    conv1d_35_output_type conv1d_35_output;
    dense_15_output_type dense_15_output;
  } activations2;


  //static union {
//
//    static input_6_output_type input_6_output;
//
//    static max_pooling1d_25_output_type max_pooling1d_25_output;
//
//    static conv1d_30_output_type conv1d_30_output;
//
//    static max_pooling1d_26_output_type max_pooling1d_26_output;
//
//    static conv1d_31_output_type conv1d_31_output;
//
//    static max_pooling1d_27_output_type max_pooling1d_27_output;
//
//    static conv1d_32_output_type conv1d_32_output;
//
//    static max_pooling1d_28_output_type max_pooling1d_28_output;
//
//    static conv1d_33_output_type conv1d_33_output;
//
//    static conv1d_34_output_type conv1d_34_output;
//
//    static conv1d_35_output_type conv1d_35_output;
//
//    static max_pooling1d_29_output_type max_pooling1d_29_output;
//
//    static flatten_5_output_type flatten_5_output;
//
//    static dense_15_output_type dense_15_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_25(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_25_output
  );
 // InputLayer is excluded 
  conv1d_30(
    
    activations1.max_pooling1d_25_output,
    conv1d_30_kernel,
    conv1d_30_bias,
    activations2.conv1d_30_output
  );
 // InputLayer is excluded 
  max_pooling1d_26(
    
    activations2.conv1d_30_output,
    activations1.max_pooling1d_26_output
  );
 // InputLayer is excluded 
  conv1d_31(
    
    activations1.max_pooling1d_26_output,
    conv1d_31_kernel,
    conv1d_31_bias,
    activations2.conv1d_31_output
  );
 // InputLayer is excluded 
  max_pooling1d_27(
    
    activations2.conv1d_31_output,
    activations1.max_pooling1d_27_output
  );
 // InputLayer is excluded 
  conv1d_32(
    
    activations1.max_pooling1d_27_output,
    conv1d_32_kernel,
    conv1d_32_bias,
    activations2.conv1d_32_output
  );
 // InputLayer is excluded 
  max_pooling1d_28(
    
    activations2.conv1d_32_output,
    activations1.max_pooling1d_28_output
  );
 // InputLayer is excluded 
  conv1d_33(
    
    activations1.max_pooling1d_28_output,
    conv1d_33_kernel,
    conv1d_33_bias,
    activations2.conv1d_33_output
  );
 // InputLayer is excluded 
  conv1d_34(
    
    activations2.conv1d_33_output,
    conv1d_34_kernel,
    conv1d_34_bias,
    activations1.conv1d_34_output
  );
 // InputLayer is excluded 
  conv1d_35(
    
    activations1.conv1d_34_output,
    conv1d_35_kernel,
    conv1d_35_bias,
    activations2.conv1d_35_output
  );
 // InputLayer is excluded 
  max_pooling1d_29(
    
    activations2.conv1d_35_output,
    activations1.max_pooling1d_29_output
  );
 // InputLayer is excluded 
  flatten_5(
    
    activations1.max_pooling1d_29_output,
    activations1.flatten_5_output
  );
 // InputLayer is excluded 
  dense_15(
    
    activations1.flatten_5_output,
    dense_15_kernel,
    dense_15_bias,
    activations2.dense_15_output
  );
 // InputLayer is excluded 
  dense_16(
    
    activations2.dense_15_output,
    dense_16_kernel,
    dense_16_bias, // Last layer uses output passed as model parameter
    dense_16_output
  );

}
