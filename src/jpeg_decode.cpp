#include "png_image.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace pngout {
namespace {

// ---------------------------------------------------------------------------
// JPEG (baseline, extended sequential and progressive)
//
// PNGOUT embeds Ken Silverman's KPLIB JPEG decoder. The decoder is reproduced
// here as a direct port because its arithmetic is visible in the final PNG:
// the integer IDCT, the YCbCr colour transform with its 22-bit fixed point
// rounding and the clipping tables all decide the exact RGB bytes. Every table
// value and rounding step below matches the reference binary.
//
// The reference decodes into a 32-bit-per-pixel frame buffer where each word is
// 0xAABBGGRR (alpha is always 255), then reads it back as RGBA bytes.
// ---------------------------------------------------------------------------

const unsigned int kPow2mask[32] = {
   0x00000000u, 0x00000001u, 0x00000003u, 0x00000007u, 0x0000000fu,
   0x0000001fu, 0x0000003fu, 0x0000007fu, 0x000000ffu, 0x000001ffu,
   0x000003ffu, 0x000007ffu, 0x00000fffu, 0x00001fffu, 0x00003fffu,
   0x00007fffu, 0x0000ffffu, 0x0001ffffu, 0x0003ffffu, 0x0007ffffu,
   0x000fffffu, 0x001fffffu, 0x003fffffu, 0x007fffffu, 0x00ffffffu,
   0x01ffffffu, 0x03ffffffu, 0x07ffffffu, 0x0fffffffu, 0x1fffffffu,
   0x3fffffffu, 0x7fffffffu,
};

const unsigned int kPow2long[32] = {
   0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u, 0x00000010u,
   0x00000020u, 0x00000040u, 0x00000080u, 0x00000100u, 0x00000200u,
   0x00000400u, 0x00000800u, 0x00001000u, 0x00002000u, 0x00004000u,
   0x00008000u, 0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
   0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u, 0x01000000u,
   0x02000000u, 0x04000000u, 0x08000000u, 0x10000000u, 0x20000000u,
   0x40000000u, 0x80000000u,
};

// cosqr16[i] = (cos(PI*i/16)*sqrt(2))<<24
const int kCosQr16[8] = {23726566, 23270667, 21920489, 19727919,
                         16777216, 13181774, 9079764, 4628823};

int g_hufmaxatbit[8][20];
int g_hufvalatbit[8][20];
int g_hufcnt[8];
unsigned char g_hufnumatbit[8][20];
unsigned char g_huftable[8][256];
int g_hufquickval[8][1024];
int g_hufquickbits[8][1024];
int g_hufquickcnt[8];
int g_quantab[4][64];
int g_dct[12][64];
int g_lastdc[4];
int g_unzig[64];
int g_zigit[64];
unsigned char g_dcflagor[64];
unsigned char g_gnumcomponents;
int g_gcompid[4], g_gcomphsamp[4], g_gcompvsamp[4], g_gcompquantab[4];
int g_gcomphsampshift[4], g_gcompvsampshift[4];
int g_lnumcomponents;
int g_lcompid[4], g_lcompdc[4], g_lcompac[4];
int g_lcomphsamp[4], g_lcompvsamp[4], g_lcompquantab[4];
int g_lcomphvsamp0, g_lcomphsampshift0, g_lcompvsampshift0;
int g_colclip[1024], g_colclipup8[1024], g_colclipup16[1024];
int g_crmul[4096], g_cbmul[4096];
int g_clipxdim, g_clipydim;
int g_bytesperline, g_xres, g_yres, g_globxoffs, g_globyoffs;
std::int32_t* g_frameplace;
int g_coltype;
bool g_inited;

int mulshr24(int a, int b) {
   return static_cast<int>((static_cast<std::int64_t>(a) *
                            static_cast<std::int64_t>(b)) >> 24);
}

int mulshr32(int a, int b) {
   return static_cast<int>((static_cast<std::int64_t>(a) *
                            static_cast<std::int64_t>(b)) >> 32);
}

void initkpeg() {
   int i, x, y;

   // Back-and-forth diagonal pattern, aligned for best compression.
   x = 0;
   for (i = 0; i < 16; i += 2) {
      for (y = 8 - 1; y >= 0; --y) {
         if (static_cast<unsigned>(i - y) < 8u) {
            g_unzig[x++] = (y << 3) + i - y;
         }
      }
      for (y = 0; y < 8; ++y) {
         if (static_cast<unsigned>(i + 1 - y) < 8u) {
            g_unzig[x++] = (y << 3) + i + 1 - y;
         }
      }
   }
   for (i = 64 - 1; i >= 0; --i) g_zigit[g_unzig[i]] = i;
   for (i = 64 - 1; i >= 0; --i) {
      g_dcflagor[i] = static_cast<unsigned char>(1 << (g_unzig[i] >> 3));
   }

   for (i = 0; i < 128; ++i) g_colclip[i] = i + 128;
   for (i = 128; i < 512; ++i) g_colclip[i] = 255;
   for (i = 512; i < 896; ++i) g_colclip[i] = 0;
   for (i = 896; i < 1024; ++i) g_colclip[i] = i - 896;
   for (i = 0; i < 1024; ++i) {
      g_colclipup8[i] = g_colclip[i] << 8;
      g_colclipup16[i] = static_cast<int>(
         static_cast<std::uint32_t>(g_colclip[i] << 16) + 0xff000000u);
   }

   for (i = 0; i < 2048; ++i) {
      g_crmul[(i << 1) + 0] = (i - 1024) * 1470104;
      g_crmul[(i << 1) + 1] = (i - 1024) * -748830;
      g_cbmul[(i << 1) + 0] = (i - 1024) * -360857;
      g_cbmul[(i << 1) + 1] = (i - 1024) * 1858077;
   }

   std::memset(&g_dct[10][0], 0, 64 * 2 * sizeof(g_dct[0][0]));
}

// Canonical Huffman lookup. `num` holds the pending bits in its low `curbits`
// positions, most significant bit first.
void huffgetval(int index, int curbits, std::uint32_t num, int* daval,
                int* dabits) {
   int b, v, pow2;
   const int* hmax = &g_hufmaxatbit[index][0];

   pow2 = static_cast<int>(kPow2long[curbits - 1]);
   v = (num & static_cast<std::uint32_t>(pow2)) ? 1 : 0;
   for (b = 1; b <= 16; ++b) {
      if (v < hmax[b]) {
         *dabits = b;
         *daval = g_huftable[index][g_hufvalatbit[index][b] + v];
         return;
      }
      pow2 >>= 1;
      v <<= 1;
      if (num & static_cast<std::uint32_t>(pow2)) ++v;
   }
   *dabits = 16;
   *daval = 0;
}

void invdct8x8(int* dc, unsigned char dcflag) {
   const int kSqrt2 = 23726566;   // (sqrt(2))<<24
   const int kC182 = 31000253;    // (cos(PI/8)*2)<<24
   const int kC18S22 = 43840978;  // (cos(PI/8)*sqrt(2)*2)<<24
   const int kC38S22 = 18159528;  // (cos(PI*3/8)*sqrt(2)*2)<<24
   int* edc;
   int t0, t1, t2, t3, t4, t5, t6, t7;

   edc = dc + 64;
   do {
      if (dcflag & 1) {
         t3 = dc[2] + dc[6];
         t2 = (mulshr32(dc[2] - dc[6], kSqrt2 << 6) << 2) - t3;
         t4 = dc[0] + dc[4];
         t5 = dc[0] - dc[4];
         t0 = t4 + t3;
         t3 = t4 - t3;
         t1 = t5 + t2;
         t2 = t5 - t2;
         t4 = mulshr32(dc[5] - dc[3] + dc[1] - dc[7], kC182 << 6) << 2;
         t7 = dc[1] + dc[7] + dc[5] + dc[3];
         t6 = (mulshr32(dc[3] - dc[5], kC18S22 << 5) << 3) + t4 - t7;
         t5 = (mulshr32(dc[1] + dc[7] - dc[5] - dc[3], kSqrt2 << 6) << 2) - t6;
         t4 = (mulshr32(dc[1] - dc[7], kC38S22 << 6) << 2) - t4 + t5;
         dc[0] = t0 + t7;
         dc[7] = t0 - t7;
         dc[1] = t1 + t6;
         dc[6] = t1 - t6;
         dc[2] = t2 + t5;
         dc[5] = t2 - t5;
         dc[4] = t3 + t4;
         dc[3] = t3 - t4;
      }
      dc += 8;
      dcflag >>= 1;
   } while (dc < edc);
   dc -= 32;
   edc -= 24;
   do {
      t3 = dc[2 * 8 - 32] + dc[6 * 8 - 32];
      t2 = (mulshr32(dc[2 * 8 - 32] - dc[6 * 8 - 32], kSqrt2 << 6) << 2) - t3;
      t4 = dc[0 * 8 - 32] + dc[4 * 8 - 32];
      t5 = dc[0 * 8 - 32] - dc[4 * 8 - 32];
      t0 = t4 + t3;
      t3 = t4 - t3;
      t1 = t5 + t2;
      t2 = t5 - t2;
      t4 = mulshr32(dc[5 * 8 - 32] - dc[3 * 8 - 32] + dc[1 * 8 - 32] -
                       dc[7 * 8 - 32],
                    kC182 << 6)
           << 2;
      t7 = dc[1 * 8 - 32] + dc[7 * 8 - 32] + dc[5 * 8 - 32] + dc[3 * 8 - 32];
      t6 = (mulshr32(dc[3 * 8 - 32] - dc[5 * 8 - 32], kC18S22 << 5) << 3) + t4 -
           t7;
      t5 = (mulshr32(dc[1 * 8 - 32] + dc[7 * 8 - 32] - dc[5 * 8 - 32] -
                        dc[3 * 8 - 32],
                     kSqrt2 << 6)
            << 2)
           - t6;
      t4 = (mulshr32(dc[1 * 8 - 32] - dc[7 * 8 - 32], kC38S22 << 6) << 2) - t4 +
           t5;
      dc[0 * 8 - 32] = t0 + t7;
      dc[7 * 8 - 32] = t0 - t7;
      dc[1 * 8 - 32] = t1 + t6;
      dc[6 * 8 - 32] = t1 - t6;
      dc[2 * 8 - 32] = t2 + t5;
      dc[5 * 8 - 32] = t2 - t5;
      dc[4 * 8 - 32] = t3 + t4;
      dc[3 * 8 - 32] = t3 - t4;
      ++dc;
   } while (dc < edc);
}

// Render one MCU from the dequantised block array into the RGBA frame buffer.
void yrbrend(int x, int y, int* ldct) {
   int i, j, ox, oy, xx, yy, xxx, yyy, xxxend, yyyend, yv, cr = 0, cb = 0;
   int *odc, *dc, *dc2;
   std::uint8_t* p;
   std::uint8_t* pp;

   odc = ldct;
   dc2 = &ldct[10 << 6];
   for (yy = 0; yy < (g_lcompvsamp[0] << 3); yy += 8) {
      oy = y + yy + g_globyoffs;
      if (static_cast<unsigned>(oy) >= static_cast<unsigned>(g_clipydim)) {
         odc += g_lcomphsamp[0] << 6;
         continue;
      }
      pp = reinterpret_cast<std::uint8_t*>(g_frameplace) + oy * g_bytesperline +
           ((x + g_globxoffs) << 2);
      for (xx = 0; xx < (g_lcomphsamp[0] << 3); xx += 8, odc += 64) {
         ox = x + xx + g_globxoffs;
         if (static_cast<unsigned>(ox) >= static_cast<unsigned>(g_clipxdim)) {
            continue;
         }
         p = pp + (xx << 2);
         dc = odc;
         if (g_lnumcomponents > 1) {
            dc2 = &ldct[(g_lcomphvsamp0 << 6) +
                        ((yy >> g_lcompvsampshift0) << 3) +
                        (xx >> g_lcomphsampshift0)];
         }
         xxxend = (g_clipxdim - ox) < 8 ? (g_clipxdim - ox) : 8;
         yyyend = (g_clipydim - oy) < 8 ? (g_clipydim - oy) : 8;
         if ((g_lcomphsamp[0] == 1) && (xxxend == 8)) {
            for (yyy = 0; yyy < yyyend; ++yyy) {
               for (xxx = 0; xxx < 8; ++xxx) {
                  yv = dc[xxx];
                  cr = (dc2[xxx + 64] >> (20 - 1)) & ~1;
                  cb = (dc2[xxx] >> (20 - 1)) & ~1;
                  reinterpret_cast<std::int32_t*>(p)[xxx] =
                     g_colclipup16[static_cast<unsigned>(
                                      yv + g_crmul[cr + 2048]) >>
                                   22] +
                     g_colclipup8[static_cast<unsigned>(
                                     yv + g_crmul[cr + 2049] +
                                     g_cbmul[cb + 2048]) >>
                                  22] +
                     g_colclip[static_cast<unsigned>(
                                  yv + g_cbmul[cb + 2049]) >>
                               22];
               }
               p += g_bytesperline;
               dc += 8;
               if (!((yyy + 1) & (g_lcompvsamp[0] - 1))) dc2 += 8;
            }
         } else if ((g_lcomphsamp[0] == 2) && (xxxend == 8)) {
            for (yyy = 0; yyy < yyyend; ++yyy) {
               for (xxx = 0; xxx < 8; xxx += 2) {
                  yv = dc[xxx];
                  cr = (dc2[(xxx >> 1) + 64] >> (20 - 1)) & ~1;
                  cb = (dc2[(xxx >> 1)] >> (20 - 1)) & ~1;
                  i = g_crmul[cr + 2049] + g_cbmul[cb + 2048];
                  cr = g_crmul[cr + 2048];
                  cb = g_cbmul[cb + 2049];
                  reinterpret_cast<std::int32_t*>(p)[xxx] =
                     g_colclipup16[static_cast<unsigned>(yv + cr) >> 22] +
                     g_colclipup8[static_cast<unsigned>(yv + i) >> 22] +
                     g_colclip[static_cast<unsigned>(yv + cb) >> 22];
                  yv = dc[xxx + 1];
                  reinterpret_cast<std::int32_t*>(p)[xxx + 1] =
                     g_colclipup16[static_cast<unsigned>(yv + cr) >> 22] +
                     g_colclipup8[static_cast<unsigned>(yv + i) >> 22] +
                     g_colclip[static_cast<unsigned>(yv + cb) >> 22];
               }
               p += g_bytesperline;
               dc += 8;
               if (!((yyy + 1) & (g_lcompvsamp[0] - 1))) dc2 += 8;
            }
         } else {
            for (yyy = 0; yyy < yyyend; ++yyy) {
               i = 0;
               j = 1;
               for (xxx = 0; xxx < xxxend; ++xxx) {
                  yv = dc[xxx];
                  --j;
                  if (!j) {
                     j = g_lcomphsamp[0];
                     cr = (dc2[i + 64] >> (20 - 1)) & ~1;
                     cb = (dc2[i] >> (20 - 1)) & ~1;
                     ++i;
                  }
                  reinterpret_cast<std::int32_t*>(p)[xxx] =
                     g_colclipup16[static_cast<unsigned>(
                                      yv + g_crmul[cr + 2048]) >>
                                   22] +
                     g_colclipup8[static_cast<unsigned>(
                                     yv + g_crmul[cr + 2049] +
                                     g_cbmul[cb + 2048]) >>
                                  22] +
                     g_colclip[static_cast<unsigned>(
                                  yv + g_cbmul[cb + 2049]) >>
                               22];
               }
               p += g_bytesperline;
               dc += 8;
               if (!((yyy + 1) & (g_lcompvsamp[0] - 1))) dc2 += 8;
            }
         }
      }
   }
}

int kpegrend(const std::uint8_t* kfilebuf, int kfilength,
             std::int32_t* daframeplace, int dabytesperline, int daxres,
             int dayres, int daglobxoffs, int daglobyoffs) {
   int i, j, v, leng = 0, xdim = 0, ydim = 0, index, prec, restartcnt;
   int restartinterval;
   int x, y, z, xx, yy, zz, *dc = nullptr, curbits, c, daval, dabits;
   int *hqval, *hqbits, hqcnt, *quanptr = nullptr;
   int passcnt = 0, ghsampmax = 0, gvsampmax = 0;
   int glhsampmax = 0, glvsampmax = 0, glhstep, glvstep;
   int eobrun, Ss, Se, Ah, Al, Alut[2];
   int dctx[12], dcty[12], ldctx[12], lshx[4], lshy[4];
   std::unique_ptr<short[]> dctstorage;
   short* dctbuf = nullptr;
   short* dctptr[12] = {};
   short* ldctptr[12] = {};
   short* dcs = nullptr;
   unsigned char ch, marker, dcflag;
   std::uint32_t num;
   const std::uint8_t* kfileptr;
   const std::uint8_t* kfileend;

   if (!g_inited) {
      g_inited = true;
      initkpeg();
   }

   kfileptr = kfilebuf;
   kfileend = kfilebuf + kfilength;

   if (kfilength < 2 || kfileptr[0] != 0xff || kfileptr[1] != 0xd8) {
      return -1;
   }
   kfileptr += 2;

   restartinterval = 0;
   for (i = 0; i < 4; ++i) g_lastdc[i] = 0;
   for (i = 0; i < 8; ++i) g_hufcnt[i] = 0;

   g_coltype = 0;

   // The reference's entropy reader guards against running past the buffer: a
   // stuffed 0xff at the very end contributes one extra zero byte.
   const auto getbits = [&](int minbits) {
      while (curbits < minbits) {
         ch = *kfileptr++;
         num = (num << 8) + ch;
         curbits += 8;
         if (ch == 255) {
            ++kfileptr;
            if (kfileptr >= kfileend) {
               num <<= 8;
               curbits += 8;
            }
         }
      }
   };

   do {
      ch = *kfileptr++;
      if (ch != 255) continue;
      do {
         marker = *kfileptr++;
      } while (marker == 255);
      if (marker != 0xd9) { // don't read past the end of the buffer
         leng = (static_cast<int>(kfileptr[0]) << 8) +
                static_cast<int>(kfileptr[1]) - 2;
         kfileptr += 2;
      }
      switch (marker) {
      case 0xc0:
      case 0xc1:
      case 0xc2: {
         ++kfileptr; // precision
         ydim = (static_cast<int>(kfileptr[0]) << 8) |
                static_cast<int>(kfileptr[1]);
         xdim = (static_cast<int>(kfileptr[2]) << 8) |
                static_cast<int>(kfileptr[3]);

         g_frameplace = daframeplace;
         g_bytesperline = dabytesperline;
         g_xres = daxres;
         g_yres = dayres;
         g_globxoffs = daglobxoffs;
         g_globyoffs = daglobyoffs;

         g_gnumcomponents = kfileptr[4];
         kfileptr += 5;
         ghsampmax = gvsampmax = glhsampmax = glvsampmax = 0;
         for (z = 0; z < g_gnumcomponents; ++z) {
            g_gcompid[z] = kfileptr[0];
            g_gcomphsamp[z] = kfileptr[1] >> 4;
            g_gcompvsamp[z] = kfileptr[1] & 15;
            g_gcompquantab[z] = kfileptr[2];
            for (i = 0; i < 8; ++i) {
               if (g_gcomphsamp[z] == static_cast<int>(kPow2long[i])) {
                  g_gcomphsampshift[z] = i;
                  break;
               }
            }
            for (i = 0; i < 8; ++i) {
               if (g_gcompvsamp[z] == static_cast<int>(kPow2long[i])) {
                  g_gcompvsampshift[z] = i;
                  break;
               }
            }
            if (g_gcomphsamp[z] > ghsampmax) {
               ghsampmax = g_gcomphsamp[z];
               glhsampmax = g_gcomphsampshift[z];
            }
            if (g_gcompvsamp[z] > gvsampmax) {
               gvsampmax = g_gcompvsamp[z];
               glvsampmax = g_gcompvsampshift[z];
            }
            kfileptr += 3;
         }
         break;
      }
      case 0xc4: // Huffman table
         do {
            ch = *kfileptr++;
            --leng;
            index = ch >= 16 ? ch - 12 : ch;
            std::memcpy(&g_hufnumatbit[index][1], kfileptr, 16);
            kfileptr += 16;
            leng -= 16;

            v = 0;
            g_hufcnt[index] = 0;
            g_hufquickcnt[index] = 0;
            for (i = 1; i <= 16; ++i) {
               g_hufmaxatbit[index][i] = v + g_hufnumatbit[index][i];
               g_hufvalatbit[index][i] = g_hufcnt[index] - v;
               std::memcpy(&g_huftable[index][g_hufcnt[index]], kfileptr,
                           g_hufnumatbit[index][i]);
               if (i <= 10) {
                  for (c = 0; c < g_hufnumatbit[index][i]; ++c) {
                     for (j = 1 << (10 - i); j > 0; --j) {
                        g_hufquickval[index][g_hufquickcnt[index]] =
                           g_huftable[index][g_hufcnt[index] + c];
                        g_hufquickbits[index][g_hufquickcnt[index]] = i;
                        ++g_hufquickcnt[index];
                     }
                  }
               }
               kfileptr += g_hufnumatbit[index][i];
               leng -= g_hufnumatbit[index][i];
               g_hufcnt[index] += g_hufnumatbit[index][i];
               v = (v + g_hufnumatbit[index][i]) << 1;
            }
         } while (leng > 0);
         break;
      case 0xdb: // quantisation tables
         do {
            ch = *kfileptr++;
            --leng;
            index = ch & 15;
            prec = ch >> 4;
            for (z = 0; z < 64; ++z) {
               v = static_cast<int>(*kfileptr++);
               if (prec) v = (v << 8) + static_cast<int>(*kfileptr++);
               v <<= 19;
               if (g_unzig[z] & 7) v = mulshr24(v, kCosQr16[g_unzig[z] & 7]);
               if (g_unzig[z] >> 3) {
                  v = mulshr24(v, kCosQr16[g_unzig[z] >> 3]);
               }
               g_quantab[index][g_unzig[z]] = v;
            }
            leng -= 64;
            if (prec) leng -= 64;
         } while (leng > 0);
         break;
      case 0xdd:
         restartinterval = (static_cast<int>(kfileptr[0]) << 8) |
                           static_cast<int>(kfileptr[1]);
         kfileptr += leng;
         break;
      case 0xda: {
         if ((xdim <= 0) || (ydim <= 0)) return -1;

         g_lnumcomponents = static_cast<int>(*kfileptr++);
         if (!g_lnumcomponents) return -1;
         if (g_lnumcomponents > 1) g_coltype = 2;
         for (z = 0; z < g_lnumcomponents; ++z) {
            g_lcompid[z] = kfileptr[0];
            g_lcompdc[z] = kfileptr[1] >> 4;
            g_lcompac[z] = kfileptr[1] & 15;
            kfileptr += 2;
         }

         Ss = kfileptr[0];
         Se = kfileptr[1];
         Ah = kfileptr[2] >> 4;
         Al = kfileptr[2] & 15;
         kfileptr += 3;

         if ((!passcnt) && ((Ss) || (Se != 63) || (Ah) || (Al))) {
            for (z = zz = 0; z < g_gnumcomponents; ++z) {
               dctx[z] = ((xdim + (ghsampmax << 3) - 1) >> (glhsampmax + 3))
                         << g_gcomphsampshift[z];
               dcty[z] = ((ydim + (gvsampmax << 3) - 1) >> (glvsampmax + 3))
                         << g_gcompvsampshift[z];
               zz += dctx[z] * dcty[z];
            }
            dctstorage = std::make_unique<short[]>(
               static_cast<std::size_t>(zz) * 64);
            dctbuf = dctstorage.get();
            for (z = zz = 0; z < g_gnumcomponents; ++z) {
               dctptr[z] = &dctbuf[zz * 64];
               zz += dctx[z] * dcty[z];
            }
         }

         glhstep = glvstep = 0x7fffffff;
         for (z = 0; z < g_lnumcomponents; ++z) {
            for (zz = 0; zz < g_gnumcomponents; ++zz) {
               if (g_lcompid[z] == g_gcompid[zz]) {
                  ldctptr[z] = dctptr[zz];
                  ldctx[z] = dctx[zz];
                  g_lcomphsamp[z] = g_gcomphsamp[zz];
                  g_lcompvsamp[z] = g_gcompvsamp[zz];
                  g_lcompquantab[z] = g_gcompquantab[zz];
                  if (!z) {
                     g_lcomphsampshift0 = g_gcomphsampshift[zz];
                     g_lcompvsampshift0 = g_gcompvsampshift[zz];
                  }
                  lshx[z] = glhsampmax - g_gcomphsampshift[zz] + 3;
                  lshy[z] = glvsampmax - g_gcompvsampshift[zz] + 3;
                  if (g_gcomphsampshift[zz] < glhstep) {
                     glhstep = g_gcomphsampshift[zz];
                  }
                  if (g_gcompvsampshift[zz] < glvstep) {
                     glvstep = g_gcompvsampshift[zz];
                  }
               }
            }
         }
         glhstep = ghsampmax >> glhstep;
         g_lcomphsamp[0] = g_lcomphsamp[0] < glhstep ? g_lcomphsamp[0]
                                                     : glhstep;
         glhstep <<= 3;
         glvstep = gvsampmax >> glvstep;
         g_lcompvsamp[0] = g_lcompvsamp[0] < glvstep ? g_lcompvsamp[0]
                                                     : glvstep;
         glvstep <<= 3;
         g_lcomphvsamp0 = g_lcomphsamp[0] * g_lcompvsamp[0];

         g_clipxdim = (xdim + g_globxoffs) < g_xres ? (xdim + g_globxoffs)
                                                    : g_xres;
         g_clipydim = (ydim + g_globyoffs) < g_yres ? (ydim + g_globyoffs)
                                                    : g_yres;

         if ((g_globxoffs > 0 ? g_globxoffs : 0) >= g_xres ||
             (g_clipxdim <= 0) ||
             (g_globyoffs > 0 ? g_globyoffs : 0) >= g_yres ||
             (g_clipydim <= 0)) {
            return 0;
         }

         Alut[0] = 1 << Al;
         Alut[1] = -Alut[0];

         restartcnt = restartinterval;
         eobrun = 0;
         marker = 0xd0;
         num = 0;
         curbits = 0;
         for (y = 0; y < ydim; y += glvstep) {
            for (x = 0; x < xdim; x += glhstep) {
               if (kfileptr - 4 - kfilebuf >= kfilength) {
                  goto kpegrend_break2;
               }

               if (!dctbuf) dc = g_dct[0];
               for (c = 0; c < g_lnumcomponents; ++c) {
                  hqval = &g_hufquickval[g_lcompac[c] + 4][0];
                  hqbits = &g_hufquickbits[g_lcompac[c] + 4][0];
                  hqcnt = g_hufquickcnt[g_lcompac[c] + 4];
                  if (!dctbuf) quanptr = &g_quantab[g_lcompquantab[c]][0];
                  for (yy = 0; yy < (g_lcompvsamp[c] << 3); yy += 8) {
                     for (xx = 0; xx < (g_lcomphsamp[c] << 3); xx += 8) {
                        if (dctbuf) {
                           dcs = &ldctptr[c][(((y + yy) >> lshy[c]) *
                                                 ldctx[c] +
                                              ((x + xx) >> lshx[c]))
                                             << 6];
                        }

                        // DC coefficient
                        if (!Ss) {
                           getbits(16);
                           if (!Ah) {
                              i = (num >> (curbits - 10)) & 1023;
                              if (i < g_hufquickcnt[g_lcompdc[c]]) {
                                 daval = g_hufquickval[g_lcompdc[c]][i];
                                 curbits -= g_hufquickbits[g_lcompdc[c]][i];
                              } else {
                                 huffgetval(g_lcompdc[c], curbits, num, &daval,
                                            &dabits);
                                 curbits -= dabits;
                              }

                              if (daval) {
                                 getbits(daval);
                                 curbits -= daval;
                                 v = static_cast<int>(
                                        (num >> curbits) &
                                        kPow2mask[daval]);
                                 if (v <= static_cast<int>(
                                             kPow2mask[daval - 1])) {
                                    v -= static_cast<int>(kPow2mask[daval]);
                                 }
                                 g_lastdc[c] += v;
                              }
                              if (!dctbuf) {
                                 dc[0] = g_lastdc[c];
                              } else {
                                 dcs[0] = static_cast<short>(g_lastdc[c] << Al);
                              }
                           } else if (num &
                                      kPow2long[--curbits]) {
                              dcs[0] |= static_cast<short>(Alut[0]);
                           }
                        }

                        // AC coefficients
                        if (!dctbuf) std::memset(&dc[1], 0, 63 * 4);
                        z = Ss > 1 ? Ss : 1;
                        dcflag = 1;
                        if (eobrun <= 0) {
                           for (; z <= Se; ++z) {
                              getbits(16);
                              i = (num >> (curbits - 10)) & 1023;
                              if (i < hqcnt) {
                                 daval = hqval[i];
                                 curbits -= hqbits[i];
                              } else {
                                 huffgetval(g_lcompac[c] + 4, curbits, num,
                                            &daval, &dabits);
                                 curbits -= dabits;
                              }

                              zz = daval >> 4;
                              daval &= 15;
                              if (daval) {
                                 if (Ah) {
                                    getbits(8);
                                    if (num & kPow2long[--curbits]) {
                                       daval = Alut[0];
                                    } else {
                                       daval = Alut[1];
                                    }
                                 }
                              } else if (zz < 15) {
                                 eobrun = static_cast<int>(kPow2long[zz]);
                                 if (zz) {
                                    getbits(zz);
                                    curbits -= zz;
                                    eobrun += static_cast<int>(
                                       (num >> curbits) & kPow2mask[zz]);
                                 }
                                 if (!Ah) --eobrun;
                                 break;
                              }
                              if (Ah) {
                                 do {
                                    if (dcs[z]) {
                                       getbits(8);
                                       if (num & kPow2long[--curbits]) {
                                          dcs[z] += static_cast<short>(
                                             Alut[dcs[z] < 0]);
                                       }
                                    } else if (--zz < 0) {
                                       break;
                                    }
                                    ++z;
                                 } while (z <= Se);
                                 if (daval) dcs[z] = static_cast<short>(daval);
                              } else {
                                 z += zz;
                                 if (z > Se) break;

                                 getbits(daval);
                                 curbits -= daval;
                                 v = static_cast<int>(
                                    (num >> curbits) & kPow2mask[daval]);
                                 if (daval >= 1 &&
                                     v <= static_cast<int>(
                                             kPow2mask[daval - 1])) {
                                    v -= static_cast<int>(kPow2mask[daval]);
                                 }
                                 dcflag |= g_dcflagor[z];
                                 if (!dctbuf) {
                                    dc[g_unzig[z]] = v;
                                 } else {
                                    dcs[z] = static_cast<short>(v << Al);
                                 }
                              }
                           }
                        } else if (!Ah) {
                           --eobrun;
                        }
                        if ((Ah) && (eobrun > 0)) {
                           --eobrun;
                           for (; z <= Se; ++z) {
                              if (!dcs[z]) continue;
                              getbits(8);
                              if (num & kPow2long[--curbits]) {
                                 dcs[z] += static_cast<short>(Alut[dcs[z] < 0]);
                              }
                           }
                        }

                        if (!dctbuf) {
                           for (z = 64 - 1; z >= 0; --z) dc[z] *= quanptr[z];
                           invdct8x8(dc, dcflag);
                           dc += 64;
                        }
                     }
                  }
               }

               if (!dctbuf) yrbrend(x, y, &g_dct[0][0]);

               --restartcnt;
               if (!restartcnt) {
                  kfileptr += 1 - (curbits >> 3);
                  curbits = 0;
                  if ((kfileptr[-2] != 255) || (kfileptr[-1] != marker)) {
                     --kfileptr;
                  }
                  ++marker;
                  if (marker >= 0xd8) marker = 0xd0;
                  restartcnt = restartinterval;
                  for (i = 0; i < 4; ++i) g_lastdc[i] = 0;
                  eobrun = 0;
               }
            }
         }
      kpegrend_break2:;
         if (!dctbuf) return 0;
         ++passcnt;
         kfileptr -= (curbits >> 3) + 1;
         break;
      }
      case 0xd9:
         break;
      default:
         kfileptr += leng;
         break;
      }
   } while (kfileptr - kfilebuf < kfilength);

   if (!dctbuf) return 0;

   // Progressive images: run the final IDCT and render pass now that every
   // coefficient has been accumulated.
   g_lnumcomponents = g_gnumcomponents;
   for (i = 0; i < g_gnumcomponents; ++i) {
      g_lcomphsamp[i] = g_gcomphsamp[i];
      g_gcomphsamp[i] <<= 3;
      g_lcompvsamp[i] = g_gcompvsamp[i];
      g_gcompvsamp[i] <<= 3;
      lshx[i] = glhsampmax - g_gcomphsampshift[i] + 3;
      lshy[i] = glvsampmax - g_gcompvsampshift[i] + 3;
   }
   g_lcomphsampshift0 = g_gcomphsampshift[0];
   g_lcompvsampshift0 = g_gcompvsampshift[0];
   g_lcomphvsamp0 = g_lcomphsamp[0] << g_lcompvsampshift0;
   for (y = 0; y < ydim; y += g_gcompvsamp[0]) {
      for (x = 0; x < xdim; x += g_gcomphsamp[0]) {
         dc = g_dct[0];
         for (c = 0; c < g_gnumcomponents; ++c) {
            for (yy = 0; yy < g_gcompvsamp[c]; yy += 8) {
               for (xx = 0; xx < g_gcomphsamp[c]; xx += 8, dc += 64) {
                  dcs = &dctptr[c][(((y + yy) >> lshy[c]) * dctx[c] +
                                    ((x + xx) >> lshx[c]))
                                   << 6];
                  quanptr = &g_quantab[g_gcompquantab[c]][0];
                  for (z = 0; z < 64; ++z) {
                     dc[z] = static_cast<int>(dcs[g_zigit[z]]) * quanptr[z];
                  }
                  invdct8x8(dc, 0xff);
               }
            }
         }
         yrbrend(x, y, &g_dct[0][0]);
      }
   }

   return 0;
}

// Locate the frame size by scanning for the first SOF0/SOF1/SOF2 marker,
// mirroring the reference's dimension query byte for byte.
bool jpeg_dimensions(const std::vector<std::uint8_t>& file, std::uint32_t& width,
                     std::uint32_t& height) {
   if (file.size() < 16) return false;
   std::size_t pos = 2;
   while (true) {
      if (file.size() - 8 <= pos) return false;
      while (file[pos] != 0xff || file[pos + 1] == 0xff) {
         ++pos;
         if (file.size() - 8 <= pos) return false;
      }
      const std::uint8_t marker = file[pos + 1];
      if (static_cast<unsigned>(marker - 0xc0) < 3) {
         height = (static_cast<std::uint32_t>(file[pos + 5]) << 8) |
                  file[pos + 6];
         width = (static_cast<std::uint32_t>(file[pos + 7]) << 8) |
                 file[pos + 8];
         return true;
      }
      const std::size_t length =
         (static_cast<std::size_t>(file[pos + 2]) << 8) | file[pos + 3];
      pos += length + 2;
   }
}

} // namespace

bool decode_jpeg(const std::vector<std::uint8_t>& file, Image& image,
                 std::string& error) {
   if (file.size() < 2 || file[0] != 0xff || file[1] != 0xd8) {
      error = "not a JPEG file";
      return false;
   }

   std::uint32_t width = 0, height = 0;
   if (!jpeg_dimensions(file, width, height)) {
      error = "invalid JPEG dimensions";
      return false;
   }

   // The reference always decodes into a 32-bit RGBA frame. Pad the input so
   // the decoder's byte-stuffing lookahead cannot run past the buffer.
   std::vector<std::uint8_t> padded(file);
   padded.resize(file.size() + 16, 0);
   std::vector<std::int32_t> frame(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

   const int result =
      kpegrend(padded.data(), static_cast<int>(file.size()), frame.data(),
               static_cast<int>(width) * 4, static_cast<int>(width),
               static_cast<int>(height), 0, 0);
   if (result < 0) {
      error = "JPEG decoding failed";
      return false;
   }

   image.width = width;
   image.height = height;
   image.bit_depth = 8;
   image.palette = false;
   image.filters.clear();
   image.chunks.clear();
   image.has_transparent_key = false;
   image.transparent_key = 0;

   const int channels = g_coltype == 0 ? 1 : 3;
   image.source_color_type = g_coltype == 0 ? 0 : 2;
   image.channels = channels;

   const std::size_t count =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
   image.samples.resize(count * static_cast<std::size_t>(channels));
   for (std::size_t i = 0; i < count; ++i) {
      // kpegrend stores 0xAARRGGBB words, which the reference reads back in
      // BGRA byte order.
      const std::uint32_t pixel = static_cast<std::uint32_t>(frame[i]);
      std::uint8_t* out = &image.samples[i * static_cast<std::size_t>(channels)];
      out[0] = static_cast<std::uint8_t>((pixel >> 16) & 0xff);
      if (channels == 3) {
         out[1] = static_cast<std::uint8_t>((pixel >> 8) & 0xff);
         out[2] = static_cast<std::uint8_t>(pixel & 0xff);
      }
   }

   return true;
}

} // namespace pngout
