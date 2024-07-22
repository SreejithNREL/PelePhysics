#include "mechanism.H"
const int rmap[NUM_REACTIONS] = {
  16,  18,  42,  43,  44,  50,  61,  67,  79,  95,  104, 110, 118, 126, 137,
  141, 151, 155, 173, 187, 30,  11,  12,  13,  14,  15,  39,  40,  0,   1,
  2,   3,   4,   5,   6,   7,   8,   9,   10,  17,  19,  20,  21,  22,  23,
  24,  25,  26,  27,  28,  29,  31,  32,  33,  34,  35,  36,  37,  38,  41,
  45,  46,  47,  48,  49,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
  62,  63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
  78,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
  94,  96,  97,  98,  99,  100, 101, 102, 103, 105, 106, 107, 108, 109, 111,
  112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 127, 128,
  129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 142, 143, 144, 145,
  146, 147, 148, 149, 150, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162,
  163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 175, 176, 177, 178,
  179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194,
  195, 196, 197, 198, 199, 200, 201};

// Returns 0-based map of reaction order
void
GET_RMAP(int* _rmap)
{
  for (int j = 0; j < NUM_REACTIONS; ++j) {
    _rmap[j] = rmap[j];
  }
}

// Returns a count of gas species in a gas reaction, and their indices
// and stoichiometric coefficients. (Eq 50)
void
CKINU(const int i, int& nspec, int ki[], int nu[])
{
  const int ns[NUM_GAS_REACTIONS] = {
    9, 12, 11, 12, 12, 12, 12, 4, 4, 4, 3, 2, 2, 3, 3, 2, 3, 4, 2, 3, 4, 3, 3,
    4, 4,  4,  4,  4,  4,  4,  3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 3, 3, 4,
    4, 4,  4,  4,  3,  4,  4,  4, 4, 5, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 4,
    4, 4,  4,  4,  4,  4,  4,  4, 4, 4, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4,
    4, 4,  4,  3,  4,  4,  4,  4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4,
    4, 4,  4,  3,  4,  3,  4,  4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
    4, 4,  5,  3,  4,  4,  4,  4, 4, 5, 3, 3, 2, 3, 4, 5, 4, 3, 4, 4, 5, 4, 4,
    4, 4,  4,  4,  4,  4,  3,  4, 4, 4, 4, 4, 3, 4, 4, 5, 3, 3, 4, 4, 4, 4, 4,
    4, 4,  3,  3,  4,  4,  4,  4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 5};
  const int kiv[NUM_GAS_REACTIONS * 12] = {
    0,  1,  3,  5,  11, 9,  12, 19, 4,  0,  0,  0,  19, 0,  1,  3,  5,  11, 9,
    12, 2,  19, 6,  4,  12, 0,  1,  3,  5,  11, 9,  12, 2,  19, 4,  0,  15, 0,
    1,  3,  5,  11, 9,  12, 2,  19, 17, 4,  13, 0,  1,  3,  5,  11, 9,  12, 2,
    19, 16, 4,  16, 0,  1,  3,  5,  11, 9,  12, 2,  19, 18, 4,  14, 0,  1,  3,
    5,  11, 9,  12, 2,  19, 15, 4,  19, 13, 14, 15, 0,  0,  0,  0,  0,  0,  0,
    0,  6,  14, 19, 15, 0,  0,  0,  0,  0,  0,  0,  0,  6,  15, 19, 17, 0,  0,
    0,  0,  0,  0,  0,  0,  15, 17, 14, 0,  0,  0,  0,  0,  0,  0,  0,  0,  19,
    6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  19, 6,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  19, 15, 17, 0,  0,  0,  0,  0,  0,  0,  0,  0,  19, 14, 15,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  14, 13, 0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  19, 13, 16, 0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  13, 19, 16, 0,
    0,  0,  0,  0,  0,  0,  0,  15, 18, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    19, 16, 15, 0,  0,  0,  0,  0,  0,  0,  0,  0,  16, 14, 13, 15, 0,  0,  0,
    0,  0,  0,  0,  0,  16, 18, 13, 0,  0,  0,  0,  0,  0,  0,  0,  0,  16, 18,
    13, 0,  0,  0,  0,  0,  0,  0,  0,  0,  16, 15, 17, 13, 0,  0,  0,  0,  0,
    0,  0,  0,  16, 15, 17, 13, 0,  0,  0,  0,  0,  0,  0,  0,  16, 15, 17, 13,
    0,  0,  0,  0,  0,  0,  0,  0,  19, 18, 6,  16, 0,  0,  0,  0,  0,  0,  0,
    0,  19, 18, 17, 15, 0,  0,  0,  0,  0,  0,  0,  0,  18, 14, 16, 15, 0,  0,
    0,  0,  0,  0,  0,  0,  18, 15, 17, 16, 0,  0,  0,  0,  0,  0,  0,  0,  8,
    14, 25, 0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  15, 25, 19, 0,  0,  0,  0,
    0,  0,  0,  0,  8,  15, 25, 19, 0,  0,  0,  0,  0,  0,  0,  0,  8,  13, 25,
    14, 0,  0,  0,  0,  0,  0,  0,  0,  8,  16, 25, 15, 0,  0,  0,  0,  0,  0,
    0,  0,  19, 22, 8,  6,  0,  0,  0,  0,  0,  0,  0,  0,  22, 14, 8,  15, 0,
    0,  0,  0,  0,  0,  0,  0,  22, 14, 25, 19, 0,  0,  0,  0,  0,  0,  0,  0,
    22, 15, 8,  17, 0,  0,  0,  0,  0,  0,  0,  0,  22, 8,  19, 0,  0,  0,  0,
    0,  0,  0,  0,  0,  22, 8,  19, 0,  0,  0,  0,  0,  0,  0,  0,  0,  22, 13,
    8,  16, 0,  0,  0,  0,  0,  0,  0,  0,  8,  6,  23, 0,  0,  0,  0,  0,  0,
    0,  0,  0,  19, 22, 23, 0,  0,  0,  0,  0,  0,  0,  0,  0,  20, 19, 12, 0,
    0,  0,  0,  0,  0,  0,  0,  0,  20, 14, 19, 22, 0,  0,  0,  0,  0,  0,  0,
    0,  20, 15, 23, 19, 0,  0,  0,  0,  0,  0,  0,  0,  20, 6,  12, 19, 0,  0,
    0,  0,  0,  0,  0,  0,  20, 13, 22, 15, 0,  0,  0,  0,  0,  0,  0,  0,  20,
    13, 25, 19, 0,  0,  0,  0,  0,  0,  0,  0,  20, 8,  29, 0,  0,  0,  0,  0,
    0,  0,  0,  0,  21, 40, 20, 40, 0,  0,  0,  0,  0,  0,  0,  0,  21, 14, 8,
    6,  0,  0,  0,  0,  0,  0,  0,  0,  21, 15, 23, 19, 0,  0,  0,  0,  0,  0,
    0,  0,  21, 6,  12, 19, 0,  0,  0,  0,  0,  0,  0,  0,  21, 13, 8,  19, 15,
    0,  0,  0,  0,  0,  0,  0,  21, 13, 8,  17, 0,  0,  0,  0,  0,  0,  0,  0,
    21, 17, 20, 17, 0,  0,  0,  0,  0,  0,  0,  0,  21, 8,  20, 8,  0,  0,  0,
    0,  0,  0,  0,  0,  21, 25, 20, 25, 0,  0,  0,  0,  0,  0,  0,  0,  21, 25,
    23, 8,  0,  0,  0,  0,  0,  0,  0,  0,  23, 19, 24, 0,  0,  0,  0,  0,  0,
    0,  0,  0,  23, 19, 6,  22, 0,  0,  0,  0,  0,  0,  0,  0,  23, 14, 22, 15,
    0,  0,  0,  0,  0,  0,  0,  0,  23, 15, 17, 22, 0,  0,  0,  0,  0,  0,  0,
    0,  23, 13, 22, 16, 0,  0,  0,  0,  0,  0,  0,  0,  23, 16, 18, 22, 0,  0,
    0,  0,  0,  0,  0,  0,  12, 19, 2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  12,
    14, 23, 19, 0,  0,  0,  0,  0,  0,  0,  0,  12, 15, 20, 17, 0,  0,  0,  0,
    0,  0,  0,  0,  12, 15, 21, 17, 0,  0,  0,  0,  0,  0,  0,  0,  12, 13, 24,
    14, 0,  0,  0,  0,  0,  0,  0,  0,  12, 13, 23, 15, 0,  0,  0,  0,  0,  0,
    0,  0,  12, 16, 2,  13, 0,  0,  0,  0,  0,  0,  0,  0,  12, 16, 24, 15, 0,
    0,  0,  0,  0,  0,  0,  0,  12, 18, 2,  16, 0,  0,  0,  0,  0,  0,  0,  0,
    12, 22, 2,  8,  0,  0,  0,  0,  0,  0,  0,  0,  23, 12, 2,  22, 0,  0,  0,
    0,  0,  0,  0,  0,  20, 12, 1,  19, 0,  0,  0,  0,  0,  0,  0,  0,  12, 7,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  12, 27, 19, 0,  0,  0,  0,  0,  0,
    0,  0,  0,  12, 28, 1,  8,  0,  0,  0,  0,  0,  0,  0,  0,  24, 19, 23, 6,
    0,  0,  0,  0,  0,  0,  0,  0,  24, 19, 12, 15, 0,  0,  0,  0,  0,  0,  0,
    0,  24, 19, 21, 17, 0,  0,  0,  0,  0,  0,  0,  0,  24, 15, 23, 17, 0,  0,
    0,  0,  0,  0,  0,  0,  24, 13, 23, 16, 0,  0,  0,  0,  0,  0,  0,  0,  2,
    19, 12, 6,  0,  0,  0,  0,  0,  0,  0,  0,  2,  14, 12, 15, 0,  0,  0,  0,
    0,  0,  0,  0,  2,  15, 12, 17, 0,  0,  0,  0,  0,  0,  0,  0,  20, 2,  12,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  19, 28, 21, 8,  0,  0,  0,  0,  0,  0,
    0,  0,  28, 14, 8,  19, 0,  0,  0,  0,  0,  0,  0,  0,  28, 13, 8,  15, 0,
    0,  0,  0,  0,  0,  0,  0,  20, 28, 26, 8,  0,  0,  0,  0,  0,  0,  0,  0,
    26, 10, 19, 0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 14, 20, 8,  0,  0,  0,
    0,  0,  0,  0,  0,  10, 14, 19, 28, 0,  0,  0,  0,  0,  0,  0,  0,  10, 15,
    29, 19, 0,  0,  0,  0,  0,  0,  0,  0,  10, 15, 12, 8,  0,  0,  0,  0,  0,
    0,  0,  0,  10, 22, 26, 8,  0,  0,  0,  0,  0,  0,  0,  0,  10, 20, 31, 19,
    0,  0,  0,  0,  0,  0,  0,  0,  10, 21, 31, 19, 0,  0,  0,  0,  0,  0,  0,
    0,  10, 12, 32, 0,  0,  0,  0,  0,  0,  0,  0,  0,  29, 19, 30, 0,  0,  0,
    0,  0,  0,  0,  0,  0,  29, 19, 6,  28, 0,  0,  0,  0,  0,  0,  0,  0,  29,
    19, 12, 8,  0,  0,  0,  0,  0,  0,  0,  0,  29, 14, 28, 15, 0,  0,  0,  0,
    0,  0,  0,  0,  29, 14, 20, 25, 0,  0,  0,  0,  0,  0,  0,  0,  29, 15, 17,
    28, 0,  0,  0,  0,  0,  0,  0,  0,  26, 19, 1,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  26, 19, 10, 6,  0,  0,  0,  0,  0,  0,  0,  0,  26, 14, 29, 19, 0,
    0,  0,  0,  0,  0,  0,  0,  26, 14, 12, 8,  0,  0,  0,  0,  0,  0,  0,  0,
    26, 15, 10, 17, 0,  0,  0,  0,  0,  0,  0,  0,  26, 13, 10, 16, 0,  0,  0,
    0,  0,  0,  0,  0,  26, 13, 30, 14, 0,  0,  0,  0,  0,  0,  0,  0,  26, 13,
    23, 22, 0,  0,  0,  0,  0,  0,  0,  0,  26, 12, 3,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  26, 12, 19, 32, 0,  0,  0,  0,  0,  0,  0,  0,  30, 12, 8,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  30, 19, 12, 22, 0,  0,  0,  0,  0,  0,  0,
    0,  30, 19, 29, 6,  0,  0,  0,  0,  0,  0,  0,  0,  30, 14, 29, 15, 0,  0,
    0,  0,  0,  0,  0,  0,  30, 15, 29, 17, 0,  0,  0,  0,  0,  0,  0,  0,  30,
    13, 29, 16, 0,  0,  0,  0,  0,  0,  0,  0,  1,  19, 27, 0,  0,  0,  0,  0,
    0,  0,  0,  0,  1,  19, 26, 6,  0,  0,  0,  0,  0,  0,  0,  0,  1,  14, 26,
    15, 0,  0,  0,  0,  0,  0,  0,  0,  1,  14, 12, 22, 0,  0,  0,  0,  0,  0,
    0,  0,  1,  14, 20, 23, 0,  0,  0,  0,  0,  0,  0,  0,  1,  15, 26, 17, 0,
    0,  0,  0,  0,  0,  0,  0,  1,  22, 27, 8,  0,  0,  0,  0,  0,  0,  0,  0,
    1,  20, 19, 32, 0,  0,  0,  0,  0,  0,  0,  0,  1,  21, 19, 32, 0,  0,  0,
    0,  0,  0,  0,  0,  1,  12, 26, 2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  13,
    26, 16, 0,  0,  0,  0,  0,  0,  0,  0,  27, 19, 7,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  27, 14, 23, 12, 0,  0,  0,  0,  0,  0,  0,  0,  27, 13, 1,  16,
    0,  0,  0,  0,  0,  0,  0,  0,  27, 16, 23, 12, 15, 0,  0,  0,  0,  0,  0,
    0,  26, 27, 5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  19, 27, 6,  0,  0,
    0,  0,  0,  0,  0,  0,  7,  14, 27, 15, 0,  0,  0,  0,  0,  0,  0,  0,  7,
    15, 27, 17, 0,  0,  0,  0,  0,  0,  0,  0,  7,  12, 27, 2,  0,  0,  0,  0,
    0,  0,  0,  0,  31, 13, 29, 22, 0,  0,  0,  0,  0,  0,  0,  0,  31, 16, 26,
    8,  15, 0,  0,  0,  0,  0,  0,  0,  10, 31, 34, 0,  0,  0,  0,  0,  0,  0,
    0,  0,  31, 35, 19, 0,  0,  0,  0,  0,  0,  0,  0,  0,  31, 9,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  19, 32, 3,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    16, 32, 3,  13, 0,  0,  0,  0,  0,  0,  0,  0,  16, 32, 26, 23, 15, 0,  0,
    0,  0,  0,  0,  0,  22, 32, 3,  8,  0,  0,  0,  0,  0,  0,  0,  0,  12, 32,
    5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  19, 1,  12, 0,  0,  0,  0,  0,
    0,  0,  0,  3,  19, 6,  32, 0,  0,  0,  0,  0,  0,  0,  0,  3,  14, 29, 12,
    19, 0,  0,  0,  0,  0,  0,  0,  3,  14, 27, 22, 0,  0,  0,  0,  0,  0,  0,
    0,  3,  14, 15, 32, 0,  0,  0,  0,  0,  0,  0,  0,  3,  15, 17, 32, 0,  0,
    0,  0,  0,  0,  0,  0,  3,  12, 2,  32, 0,  0,  0,  0,  0,  0,  0,  0,  5,
    19, 1,  27, 0,  0,  0,  0,  0,  0,  0,  0,  5,  19, 3,  12, 0,  0,  0,  0,
    0,  0,  0,  0,  19, 4,  3,  12, 0,  0,  0,  0,  0,  0,  0,  0,  14, 4,  29,
    12, 0,  0,  0,  0,  0,  0,  0,  0,  35, 12, 11, 0,  0,  0,  0,  0,  0,  0,
    0,  0,  11, 13, 36, 16, 0,  0,  0,  0,  0,  0,  0,  0,  11, 15, 36, 17, 0,
    0,  0,  0,  0,  0,  0,  0,  11, 19, 36, 6,  0,  0,  0,  0,  0,  0,  0,  0,
    11, 19, 9,  12, 0,  0,  0,  0,  0,  0,  0,  0,  11, 12, 36, 2,  0,  0,  0,
    0,  0,  0,  0,  0,  36, 19, 11, 0,  0,  0,  0,  0,  0,  0,  0,  0,  36, 19,
    35, 12, 0,  0,  0,  0,  0,  0,  0,  0,  36, 14, 39, 19, 0,  0,  0,  0,  0,
    0,  0,  0,  36, 16, 39, 19, 15, 0,  0,  0,  0,  0,  0,  0,  35, 22, 39, 0,
    0,  0,  0,  0,  0,  0,  0,  0,  39, 38, 19, 0,  0,  0,  0,  0,  0,  0,  0,
    0,  39, 13, 38, 16, 0,  0,  0,  0,  0,  0,  0,  0,  39, 15, 38, 17, 0,  0,
    0,  0,  0,  0,  0,  0,  39, 19, 38, 6,  0,  0,  0,  0,  0,  0,  0,  0,  39,
    19, 9,  22, 0,  0,  0,  0,  0,  0,  0,  0,  39, 14, 38, 15, 0,  0,  0,  0,
    0,  0,  0,  0,  39, 12, 38, 2,  0,  0,  0,  0,  0,  0,  0,  0,  38, 18, 39,
    16, 0,  0,  0,  0,  0,  0,  0,  0,  38, 35, 8,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  35, 19, 9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  15, 35, 17, 0,
    0,  0,  0,  0,  0,  0,  0,  9,  14, 37, 19, 0,  0,  0,  0,  0,  0,  0,  0,
    9,  14, 34, 22, 0,  0,  0,  0,  0,  0,  0,  0,  35, 6,  9,  19, 0,  0,  0,
    0,  0,  0,  0,  0,  35, 13, 37, 14, 0,  0,  0,  0,  0,  0,  0,  0,  35, 14,
    34, 8,  0,  0,  0,  0,  0,  0,  0,  0,  35, 15, 37, 19, 0,  0,  0,  0,  0,
    0,  0,  0,  35, 16, 9,  13, 0,  0,  0,  0,  0,  0,  0,  0,  35, 2,  9,  12,
    0,  0,  0,  0,  0,  0,  0,  0,  37, 34, 8,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  34, 14, 33, 19, 0,  0,  0,  0,  0,  0,  0,  0,  34, 13, 33, 15, 0,  0,
    0,  0,  0,  0,  0,  0,  33, 10, 8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  33,
    14, 31, 8,  22, 0,  0,  0,  0,  0,  0,  0};
  const int nuv[NUM_GAS_REACTIONS * 12] = {
    -1, 1,  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 2,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
  if (i < 1) {
    // Return max num species per reaction
    nspec = 12;
  } else {
    if (i > NUM_GAS_REACTIONS) {
      nspec = -1;
    } else {
      nspec = ns[i - 1];
      for (int j = 0; j < nspec; ++j) {
        ki[j] = kiv[(i - 1) * 12 + j] + 1;
        nu[j] = nuv[(i - 1) * 12 + j];
      }
    }
  }
}

// Returns the progress rates of each reactions
// Given P, T, and mole fractions
void
CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real q_f[],
  amrex::Real q_r[])
{
  amrex::Real c[41]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 41; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  progressRateFR(q_f, q_r, c, T);

  // convert to chemkin units
  for (int id = 0; id < 202; ++id) {
    q_f[id] *= 1.0e-6;
    q_r[id] *= 1.0e-6;
  }
}

// compute the progress rate for each reaction
// USES progressRate : todo switch to GPU
void
progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);
  // compute the Gibbs free energy
  amrex::Real g_RT[41];
  gibbs(g_RT, T);

  amrex::Real sc_qss[1];
  comp_qfqr(q_f, q_r, sc, sc_qss, T, invT, logT);
}

// save atomic weights into array
void
atomicWeight(amrex::Real* awt)
{
  awt[0] = 15.999000; // O
  awt[1] = 1.008000;  // H
  awt[2] = 12.011000; // C
  awt[3] = 14.007000; // N
  awt[4] = 39.950000; // Ar
  awt[5] = 4.002602;  // He
}

// get atomic weight for all elements
void
CKAWT(amrex::Real* awt)
{
  atomicWeight(awt);
}

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
void
CKNCF(int* ncf)
{
  int kd = 6;
  // Zero ncf
  for (int id = 0; id < kd * 41; ++id) {
    ncf[id] = 0;
  }

  // POSF10325
  ncf[0 * kd + 2] = 11; // C
  ncf[0 * kd + 1] = 22; // H

  // C2H4
  ncf[1 * kd + 2] = 2; // C
  ncf[1 * kd + 1] = 4; // H

  // CH4
  ncf[2 * kd + 2] = 1; // C
  ncf[2 * kd + 1] = 4; // H

  // C3H6
  ncf[3 * kd + 2] = 3; // C
  ncf[3 * kd + 1] = 6; // H

  // iC4H8
  ncf[4 * kd + 2] = 4; // C
  ncf[4 * kd + 1] = 8; // H

  // C4H81
  ncf[5 * kd + 2] = 4; // C
  ncf[5 * kd + 1] = 8; // H

  // H2
  ncf[6 * kd + 1] = 2; // H

  // C2H6
  ncf[7 * kd + 2] = 2; // C
  ncf[7 * kd + 1] = 6; // H

  // CO
  ncf[8 * kd + 2] = 1; // C
  ncf[8 * kd + 0] = 1; // O

  // C6H6
  ncf[9 * kd + 2] = 6; // C
  ncf[9 * kd + 1] = 6; // H

  // C2H2
  ncf[10 * kd + 2] = 2; // C
  ncf[10 * kd + 1] = 2; // H

  // C6H5CH3
  ncf[11 * kd + 2] = 7; // C
  ncf[11 * kd + 1] = 8; // H

  // CH3
  ncf[12 * kd + 2] = 1; // C
  ncf[12 * kd + 1] = 3; // H

  // O2
  ncf[13 * kd + 0] = 2; // O

  // O
  ncf[14 * kd + 0] = 1; // O

  // OH
  ncf[15 * kd + 1] = 1; // H
  ncf[15 * kd + 0] = 1; // O

  // HO2
  ncf[16 * kd + 1] = 1; // H
  ncf[16 * kd + 0] = 2; // O

  // H2O
  ncf[17 * kd + 1] = 2; // H
  ncf[17 * kd + 0] = 1; // O

  // H2O2
  ncf[18 * kd + 1] = 2; // H
  ncf[18 * kd + 0] = 2; // O

  // H
  ncf[19 * kd + 1] = 1; // H

  // CH2
  ncf[20 * kd + 2] = 1; // C
  ncf[20 * kd + 1] = 2; // H

  // CH2*
  ncf[21 * kd + 2] = 1; // C
  ncf[21 * kd + 1] = 2; // H

  // HCO
  ncf[22 * kd + 2] = 1; // C
  ncf[22 * kd + 1] = 1; // H
  ncf[22 * kd + 0] = 1; // O

  // CH2O
  ncf[23 * kd + 2] = 1; // C
  ncf[23 * kd + 1] = 2; // H
  ncf[23 * kd + 0] = 1; // O

  // CH3O
  ncf[24 * kd + 2] = 1; // C
  ncf[24 * kd + 1] = 3; // H
  ncf[24 * kd + 0] = 1; // O

  // CO2
  ncf[25 * kd + 2] = 1; // C
  ncf[25 * kd + 0] = 2; // O

  // C2H3
  ncf[26 * kd + 2] = 2; // C
  ncf[26 * kd + 1] = 3; // H

  // C2H5
  ncf[27 * kd + 2] = 2; // C
  ncf[27 * kd + 1] = 5; // H

  // HCCO
  ncf[28 * kd + 2] = 2; // C
  ncf[28 * kd + 1] = 1; // H
  ncf[28 * kd + 0] = 1; // O

  // CH2CO
  ncf[29 * kd + 2] = 2; // C
  ncf[29 * kd + 1] = 2; // H
  ncf[29 * kd + 0] = 1; // O

  // CH2CHO
  ncf[30 * kd + 2] = 2; // C
  ncf[30 * kd + 1] = 3; // H
  ncf[30 * kd + 0] = 1; // O

  // C3H3
  ncf[31 * kd + 2] = 3; // C
  ncf[31 * kd + 1] = 3; // H

  // aC3H5
  ncf[32 * kd + 2] = 3; // C
  ncf[32 * kd + 1] = 5; // H

  // C5H4O
  ncf[33 * kd + 2] = 5; // C
  ncf[33 * kd + 1] = 4; // H
  ncf[33 * kd + 0] = 1; // O

  // C5H5
  ncf[34 * kd + 2] = 5; // C
  ncf[34 * kd + 1] = 5; // H

  // C6H5
  ncf[35 * kd + 2] = 6; // C
  ncf[35 * kd + 1] = 5; // H

  // C6H5CH2
  ncf[36 * kd + 2] = 7; // C
  ncf[36 * kd + 1] = 7; // H

  // C6H5O
  ncf[37 * kd + 2] = 6; // C
  ncf[37 * kd + 1] = 5; // H
  ncf[37 * kd + 0] = 1; // O

  // C6H5CO
  ncf[38 * kd + 2] = 7; // C
  ncf[38 * kd + 1] = 5; // H
  ncf[38 * kd + 0] = 1; // O

  // C6H5CHO
  ncf[39 * kd + 2] = 7; // C
  ncf[39 * kd + 1] = 6; // H
  ncf[39 * kd + 0] = 1; // O

  // N2
  ncf[40 * kd + 3] = 2; // N
}

// Returns the vector of strings of element names
void
CKSYME_STR(amrex::Vector<std::string>& ename)
{
  ename.resize(6);
  ename[0] = "O";
  ename[1] = "H";
  ename[2] = "C";
  ename[3] = "N";
  ename[4] = "Ar";
  ename[5] = "He";
}

// Returns the vector of strings of species names
void
CKSYMS_STR(amrex::Vector<std::string>& kname)
{
  kname.resize(41);
  kname[0] = "POSF10325";
  kname[1] = "C2H4";
  kname[2] = "CH4";
  kname[3] = "C3H6";
  kname[4] = "iC4H8";
  kname[5] = "C4H81";
  kname[6] = "H2";
  kname[7] = "C2H6";
  kname[8] = "CO";
  kname[9] = "C6H6";
  kname[10] = "C2H2";
  kname[11] = "C6H5CH3";
  kname[12] = "CH3";
  kname[13] = "O2";
  kname[14] = "O";
  kname[15] = "OH";
  kname[16] = "HO2";
  kname[17] = "H2O";
  kname[18] = "H2O2";
  kname[19] = "H";
  kname[20] = "CH2";
  kname[21] = "CH2*";
  kname[22] = "HCO";
  kname[23] = "CH2O";
  kname[24] = "CH3O";
  kname[25] = "CO2";
  kname[26] = "C2H3";
  kname[27] = "C2H5";
  kname[28] = "HCCO";
  kname[29] = "CH2CO";
  kname[30] = "CH2CHO";
  kname[31] = "C3H3";
  kname[32] = "aC3H5";
  kname[33] = "C5H4O";
  kname[34] = "C5H5";
  kname[35] = "C6H5";
  kname[36] = "C6H5CH2";
  kname[37] = "C6H5O";
  kname[38] = "C6H5CO";
  kname[39] = "C6H5CHO";
  kname[40] = "N2";
}

// compute the sparsity pattern of the chemistry Jacobian
void
SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 42; k++) {
    for (int l = 0; l < 42; l++) {
      if (Jac[42 * k + l] != 0.0) {
        nJdata_tmp = nJdata_tmp + 1;
      }
    }
  }

  *nJdata = NCELLS * nJdata_tmp;
}

// compute the sparsity pattern of the system Jacobian
void
SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 42; k++) {
    for (int l = 0; l < 42; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[42 * k + l] != 0.0) {
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
  }

  *nJdata = NCELLS * nJdata_tmp;
}

// compute the sparsity pattern of the simplified (for preconditioning) system
// Jacobian
void
SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  int nJdata_tmp = 0;
  for (int k = 0; k < 42; k++) {
    for (int l = 0; l < 42; l++) {
      if (k == l) {
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[42 * k + l] != 0.0) {
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
  }

  nJdata[0] = nJdata_tmp;
}

// compute the sparsity pattern of the chemistry Jacobian in CSC format -- base
// 0
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int nc = 0; nc < NCELLS; nc++) {
    int offset_row = nc * 42;
    int offset_col = nc * 42;
    for (int k = 0; k < 42; k++) {
      for (int l = 0; l < 42; l++) {
        if (Jac[42 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l + offset_row;
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
      colPtrs[offset_col + (k + 1)] = nJdata_tmp;
    }
  }
}

// compute the sparsity pattern of the chemistry Jacobian in CSR format -- base
// 0
void
SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtrs[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 42;
      for (int l = 0; l < 42; l++) {
        for (int k = 0; k < 42; k++) {
          if (Jac[42 * k + l] != 0.0) {
            colVals[nJdata_tmp - 1] = k + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
        rowPtrs[offset + (l + 1)] = nJdata_tmp;
      }
    }
  } else {
    rowPtrs[0] = 0;
    int nJdata_tmp = 0;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 42;
      for (int l = 0; l < 42; l++) {
        for (int k = 0; k < 42; k++) {
          if (Jac[42 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k + offset;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
        rowPtrs[offset + (l + 1)] = nJdata_tmp;
      }
    }
  }
}

// compute the sparsity pattern of the system Jacobian
// CSR format BASE is user choice
void
SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtr, const int* consP, int NCELLS, int base)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 42;
      for (int l = 0; l < 42; l++) {
        for (int k = 0; k < 42; k++) {
          if (k == l) {
            colVals[nJdata_tmp - 1] = l + 1 + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[42 * k + l] != 0.0) {
              colVals[nJdata_tmp - 1] = k + 1 + offset;
              nJdata_tmp = nJdata_tmp + 1;
            }
          }
        }
        rowPtr[offset + (l + 1)] = nJdata_tmp;
      }
    }
  } else {
    rowPtr[0] = 0;
    int nJdata_tmp = 0;
    for (int nc = 0; nc < NCELLS; nc++) {
      int offset = nc * 42;
      for (int l = 0; l < 42; l++) {
        for (int k = 0; k < 42; k++) {
          if (k == l) {
            colVals[nJdata_tmp] = l + offset;
            nJdata_tmp = nJdata_tmp + 1;
          } else {
            if (Jac[42 * k + l] != 0.0) {
              colVals[nJdata_tmp] = k + offset;
              nJdata_tmp = nJdata_tmp + 1;
            }
          }
        }
        rowPtr[offset + (l + 1)] = nJdata_tmp;
      }
    }
  }
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian
// on CPU BASE 0
void
SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  colPtrs[0] = 0;
  int nJdata_tmp = 0;
  for (int k = 0; k < 42; k++) {
    for (int l = 0; l < 42; l++) {
      if (k == l) {
        rowVals[nJdata_tmp] = l;
        indx[nJdata_tmp] = 42 * k + l;
        nJdata_tmp = nJdata_tmp + 1;
      } else {
        if (Jac[42 * k + l] != 0.0) {
          rowVals[nJdata_tmp] = l;
          indx[nJdata_tmp] = 42 * k + l;
          nJdata_tmp = nJdata_tmp + 1;
        }
      }
    }
    colPtrs[k + 1] = nJdata_tmp;
  }
}

// compute the sparsity pattern of the simplified (for precond) system Jacobian
// CSR format BASE is under choice
void
SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base)
{
  amrex::GpuArray<amrex::Real, 1764> Jac = {0.0};
  amrex::GpuArray<amrex::Real, 41> conc = {0.0};
  for (int n = 0; n < 41; n++) {
    conc[n] = 1.0 / 41.000000;
  }
  aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);

  if (base == 1) {
    rowPtr[0] = 1;
    int nJdata_tmp = 1;
    for (int l = 0; l < 42; l++) {
      for (int k = 0; k < 42; k++) {
        if (k == l) {
          colVals[nJdata_tmp - 1] = l + 1;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[42 * k + l] != 0.0) {
            colVals[nJdata_tmp - 1] = k + 1;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  } else {
    rowPtr[0] = 0;
    int nJdata_tmp = 0;
    for (int l = 0; l < 42; l++) {
      for (int k = 0; k < 42; k++) {
        if (k == l) {
          colVals[nJdata_tmp] = l;
          nJdata_tmp = nJdata_tmp + 1;
        } else {
          if (Jac[42 * k + l] != 0.0) {
            colVals[nJdata_tmp] = k;
            nJdata_tmp = nJdata_tmp + 1;
          }
        }
      }
      rowPtr[l + 1] = nJdata_tmp;
    }
  }
}
