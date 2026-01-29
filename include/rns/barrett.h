#ifndef RNS_BARRETT_H
#define RNS_BARRETT_H

#include "config.h"

namespace rns {

RNS_HOST_DEVICE inline u32 barrett_reduce_u64(u64 x, u32 p, u64 mu) {
  u64 q = (u64)(((unsigned __int128)x * mu) >> 64);
  u64 r = x - q * p;
  return (u32)(r >= p ? r - p : r);
}

RNS_HOST_DEVICE inline u32 barrett_reduce_u32(u32 x, u32 p) {
  return x >= p ? x - p : x;
}

inline u64 compute_barrett_mu(u32 p) {
  unsigned __int128 one = ((unsigned __int128)1 << 64);
  return (u64)(one / (u64)p);
}

} // namespace rns

#endif // RNS_BARRETT_H
