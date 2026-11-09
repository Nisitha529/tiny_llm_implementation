// Planned checks for common/qweights.hpp:
// 1. Code order: a word 0x76543210 gives codes 0,1,...,7 for columns 0..7.
// 2. weight_at = code * scale + bias, with the scale and bias of the right group
//    (columns 127 and 128 fall in different groups).
